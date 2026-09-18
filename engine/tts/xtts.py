"""XTTS-v2 — синтез ru/en с клонированием голоса (ARCHITECTURE.md 4.4, ~2.5 GB VRAM).

Используется пакет `coqui-tts` (поддерживаемый форк idiap от архивного
`coqui-ai/TTS`): ``TTS.tts.models.xtts.Xtts`` + ``TTS.tts.configs.xtts_config.XttsConfig``.
Чекпоинт берётся из локального кэша моделей (`scripts/download_models.py`,
репозиторий ``coqui/XTTS-v2``: ``config.json``, ``model.pth``, ``vocab.json``,
``speakers_xtts.pth``).

Особенности реализации:

* torch и TTS импортируются **только внутри методов** — модуль обязан
  импортироваться (и тесты — проходить) без тяжёлых пакетов;
* модель грузится один раз: singleton на процесс с ключом ``(каталог, устройство)``,
  повторный ``warmup`` и параллельные ``synthesize`` его переиспользуют;
* ``inference_stream`` — блокирующий генератор, поэтому он крутится в отдельном
  потоке (``asyncio.to_thread``), а чанки едут в корутину через ``asyncio.Queue``;
* латенты голоса (``gpt_cond_latent``, ``speaker_embedding``) считаются один раз
  при создании профиля и кладутся в ``<voices_dir>/<voice_id>/latents.pt``.

Лицензия XTTS-v2 — Coqui Public Model License (CPML), **некоммерческое
использование**; см. README модуля.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator
from concurrent.futures import TimeoutError as FutureTimeoutError
from pathlib import Path
from typing import Any, Final

import numpy as np

from engine.contracts.events import Lang
from engine.tts.audio import Pcm16, Rechunker, chunk_samples, float_to_int16
from engine.tts.base import (
    OUTPUT_SAMPLE_RATE,
    Device,
    TtsConfig,
    TtsError,
)
from engine.tts.voices import LATENTS_FILENAME, VoiceStore

__all__ = ["XTTS_LANGS", "XttsProvider", "clear_model_cache"]

log: Final[logging.Logger] = logging.getLogger(__name__)

#: Языки XTTS, которые нужны проекту (модель умеет больше, нам хватает двух).
XTTS_LANGS: Final[frozenset[Lang]] = frozenset({Lang.RU, Lang.EN})

#: Файлы, по которым узнаём распакованный чекпоинт XTTS-v2.
CHECKPOINT_FILES: Final[tuple[str, ...]] = ("config.json", "model.pth")

#: Размер очереди между потоком инференса и корутиной (в чанках модели).
QUEUE_MAXSIZE: Final[int] = 8

#: Как часто поток инференса проверяет, не ушёл ли потребитель, секунды.
PUT_POLL_SECONDS: Final[float] = 0.2

# Singleton: одна модель на (каталог, устройство) на весь процесс — 2.5 GB VRAM
# второй раз не выделяем (ARCHITECTURE.md раздел 6).
_MODEL_CACHE: dict[tuple[str, str], Any] = {}
_MODEL_LOCK: Final[threading.Lock] = threading.Lock()


def clear_model_cache() -> None:
    """Забыть загруженные модели (нужно только тестам и перезапуску движка)."""
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


def _find_checkpoint_dir(models_dir: Path, model_id: str) -> Path:
    """Найти каталог с распакованным чекпоинтом XTTS-v2.

    Проверяются раскладки: ``<models>/xtts`` (так кладёт scripts/download_models.py),
    ``<models>/coqui/XTTS-v2``, ``<models>/XTTS-v2`` и сам ``<models>``.

    Raises:
        TtsError: чекпоинт не найден.
    """
    candidates = [
        models_dir / "xtts",
        models_dir / model_id,
        models_dir / model_id.split("/")[-1],
        models_dir,
    ]
    for candidate in candidates:
        if all((candidate / name).is_file() for name in CHECKPOINT_FILES):
            return candidate
    tried = "\n  ".join(str(path) for path in candidates)
    raise TtsError(
        f"чекпоинт XTTS-v2 не найден. Проверены каталоги:\n  {tried}\n"
        "Скачайте его: python scripts/download_models.py --only xtts "
        "(каталог задаётся RT_MODELS_DIR)"
    )


class XttsProvider:
    """Провайдер XTTS-v2 для ru/en с клонированием голоса по профилю."""

    __slots__ = ("_config", "_latents_cache", "_lock", "_model", "_voices")

    def __init__(self, config: TtsConfig, voices: VoiceStore | None = None) -> None:
        self._config = config
        self._voices = voices if voices is not None else VoiceStore(config.voices_dir)
        self._model: Any | None = None
        self._lock = asyncio.Lock()
        self._latents_cache: dict[str, tuple[Any, Any]] = {}

    @property
    def supports_cloning(self) -> bool:
        """XTTS-v2 клонирует голос по сэмплу 15-30 секунд."""
        return True

    def supports(self, lang: Lang) -> bool:
        """ru и en; казахский уходит на отдельный провайдер."""
        return lang in XTTS_LANGS

    @property
    def voices(self) -> VoiceStore:
        """Хранилище профилей голосов, с которым работает провайдер."""
        return self._voices

    # --- загрузка модели ---------------------------------------------------

    def _device(self) -> Device:
        return self._config.resolve_device()

    def _load_model(self) -> Any:
        """Загрузить (или взять из кэша) модель. Блокирующая операция."""
        checkpoint_dir = _find_checkpoint_dir(self._config.models_dir, self._config.xtts_model_id)
        device = self._device()
        key = (str(checkpoint_dir), device.value)
        with _MODEL_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                return cached

            try:
                from TTS.tts.configs.xtts_config import XttsConfig
                from TTS.tts.models.xtts import Xtts
            except ImportError as exc:  # pragma: no cover — требует тяжёлых пакетов
                raise TtsError(
                    "не установлен пакет coqui-tts: pip install -r engine/tts/requirements-tts.txt"
                ) from exc

            log.info("загружаю XTTS-v2 из %s на %s", checkpoint_dir, device.value)
            config = XttsConfig()
            config.load_json(str(checkpoint_dir / "config.json"))
            model = Xtts.init_from_config(config)
            model.load_checkpoint(config, checkpoint_dir=str(checkpoint_dir), use_deepspeed=False)
            if device is Device.CUDA:
                model.cuda()
            model.eval()
            _MODEL_CACHE[key] = model
            return model

    async def _ensure_model(self) -> Any:
        """Гарантировать загруженную модель (двойная проверка под локом)."""
        if self._model is not None:
            return self._model
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
        return self._model

    async def warmup(self) -> None:
        """Загрузить модель заранее (при старте движка)."""
        await self._ensure_model()

    # --- латенты голоса ----------------------------------------------------

    def compute_latents(self, sample_wav: Path, voice_dir: Path) -> Path:
        """Посчитать латенты по сэмплу и сохранить в ``voice_dir/latents.pt``.

        Подходит как ``latents_fn`` для :meth:`~engine.tts.voices.VoiceStore.create_voice`.
        Блокирующая операция (грузит модель) — из корутины вызывать через
        ``asyncio.to_thread``.
        """
        import torch

        model = self._load_model()
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[str(sample_wav)]
        )
        path = voice_dir / LATENTS_FILENAME
        torch.save(
            {
                "gpt_cond_latent": gpt_cond_latent.cpu(),
                "speaker_embedding": speaker_embedding.cpu(),
            },
            str(path),
        )
        log.info("латенты голоса сохранены: %s", path)
        return path

    def _to_device(self, tensor: Any) -> Any:
        """Переложить тензор латентов на устройство инференса."""
        if self._device() is Device.CUDA and hasattr(tensor, "to"):
            return tensor.to("cuda")
        return tensor

    def _default_latents(self, model: Any) -> tuple[Any, Any]:
        """Встроенный голос XTTS из ``speakers_xtts.pth`` (``voice_id is None``)."""
        name = self._config.xtts_speaker
        manager = getattr(model, "speaker_manager", None)
        speakers = getattr(manager, "speakers", None)
        if not speakers:
            raise TtsError(
                "в чекпоинте нет speakers_xtts.pth — встроенный голос недоступен, "
                "передайте voice_id"
            )
        if name not in speakers:
            available = ", ".join(sorted(speakers)[:10])
            raise TtsError(f"встроенный голос {name!r} не найден. Доступны, например: {available}")
        entry = speakers[name]
        try:
            gpt_cond_latent, speaker_embedding = (
                (entry["gpt_cond_latent"], entry["speaker_embedding"])
                if hasattr(entry, "keys")
                else tuple(entry)
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TtsError(f"неожиданный формат встроенного голоса {name!r}: {exc}") from exc
        return self._to_device(gpt_cond_latent), self._to_device(speaker_embedding)

    def _voice_latents(self, model: Any, voice_id: str) -> tuple[Any, Any]:
        """Латенты профиля: из кэша, с диска или посчитать и положить на диск."""
        cached = self._latents_cache.get(voice_id)
        if cached is not None:
            return cached

        profile = self._voices.get(voice_id)  # VoiceNotFoundError раньше импорта torch

        import torch

        if not profile.latents_path.is_file():
            log.info("латентов у голоса %s нет — считаю по сэмплу", voice_id)
            self.compute_latents(profile.sample_path, profile.dir)
        data = torch.load(str(profile.latents_path), map_location="cpu")
        latents = (
            self._to_device(data["gpt_cond_latent"]),
            self._to_device(data["speaker_embedding"]),
        )
        self._latents_cache[voice_id] = latents
        return latents

    def _latents_for(self, model: Any, voice_id: str | None) -> tuple[Any, Any]:
        if voice_id is None:
            return self._default_latents(model)
        return self._voice_latents(model, voice_id)

    # --- синтез ------------------------------------------------------------

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        """Стриминговый синтез: PCM 24 kHz mono int16 чанками ``config.chunk_ms``.

        Блокирующий генератор модели крутится в отдельном потоке, чанки
        передаются в корутину через :class:`asyncio.Queue`.

        Raises:
            TtsError: язык не поддержан, нет чекпоинта/голоса или упал инференс.
        """
        if not self.supports(lang):
            raise TtsError(f"XTTS: язык {lang.value} не поддерживается (только ru/en)")
        if not text.strip():
            return

        model = await self._ensure_model()
        gpt_cond_latent, speaker_embedding = await asyncio.to_thread(
            self._latents_for, model, voice_id
        )

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Pcm16 | BaseException | None] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        stop = threading.Event()

        def put(item: Pcm16 | BaseException | None) -> bool:
            """Положить элемент в очередь из потока; False — потребитель ушёл."""
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not stop.is_set():
                try:
                    future.result(timeout=PUT_POLL_SECONDS)
                except FutureTimeoutError:
                    continue
                except RuntimeError:  # цикл событий закрыт
                    return False
                return True
            future.cancel()
            return False

        def produce() -> None:
            """Крутить блокирующий inference_stream и класть чанки в очередь."""
            try:
                stream = model.inference_stream(
                    text,
                    lang.value,
                    gpt_cond_latent,
                    speaker_embedding,
                    stream_chunk_size=self._config.stream_chunk_size,
                )
                for piece in stream:
                    if stop.is_set() or not put(float_to_int16(_to_numpy(piece))):
                        return
            except Exception as exc:  # ошибку отдаём в корутину, поток не роняем
                put(exc)
            finally:
                put(None)

        worker = asyncio.create_task(asyncio.to_thread(produce))
        rechunker = Rechunker(chunk_samples(OUTPUT_SAMPLE_RATE, self._config.chunk_ms))
        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise TtsError(f"XTTS: инференс не удался: {item}") from item
                for chunk in rechunker.push(item):
                    yield chunk
            for chunk in rechunker.flush():
                yield chunk
        finally:
            stop.set()  # поток инференса выйдет сам, даже если чанки не забрали
            await asyncio.gather(worker, return_exceptions=True)


def _to_numpy(piece: Any) -> np.ndarray[Any, np.dtype[Any]]:
    """Привести чанк модели (torch.Tensor или массив) к одномерному numpy."""
    if hasattr(piece, "detach"):
        piece = piece.detach().cpu().numpy()
    return np.asarray(piece, dtype=np.float32).reshape(-1)
