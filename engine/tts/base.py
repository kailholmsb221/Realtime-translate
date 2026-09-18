"""Интерфейс синтеза речи (ARCHITECTURE.md 4.4).

Единый контракт для всех провайдеров TTS:

* :data:`OUTPUT_SAMPLE_RATE` — 24 kHz mono int16, как требует контракт
  `tts.chunk` (ресемплинг под устройство делает orchestrator/audio_io);
* :class:`TtsProvider` — протокол провайдера (XTTS для ru/en, MMS для kk, фейк);
* :class:`TtsConfig` — конфигурация, собирается из переменных окружения.

Тяжёлые модели грузятся лениво и только внутри провайдеров (CLAUDE.md,
технический стандарт), поэтому этот модуль не импортирует ни torch, ни TTS,
ни transformers.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Protocol, TypeVar, runtime_checkable

from engine.contracts.events import Lang
from engine.tts.audio import Pcm16

__all__ = [
    "CHUNK_MS_MAX",
    "CHUNK_MS_MIN",
    "DEFAULT_CHUNK_MS",
    "OUTPUT_SAMPLE_RATE",
    "Backend",
    "Device",
    "KkBackend",
    "TtsConfig",
    "TtsError",
    "TtsProvider",
]

#: Частота дискретизации всего, что отдаёт TTS (контракт `tts.chunk`).
OUTPUT_SAMPLE_RATE: Final[int] = 24_000

#: Допустимая длительность одного чанка, мс (ARCHITECTURE.md 4.4: поток 24 kHz).
CHUNK_MS_MIN: Final[int] = 20
CHUNK_MS_MAX: Final[int] = 100
DEFAULT_CHUNK_MS: Final[int] = 40

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR: Final[Path] = REPO_ROOT / "models"
DEFAULT_VOICES_DIR: Final[Path] = REPO_ROOT / "voices"

#: Модели по умолчанию (ARCHITECTURE.md 4.4, раздел 6 — бюджет VRAM).
DEFAULT_XTTS_MODEL_ID: Final[str] = "coqui/XTTS-v2"
DEFAULT_XTTS_API_NAME: Final[str] = "tts_models/multilingual/multi-dataset/xtts_v2"
DEFAULT_XTTS_SPEAKER: Final[str] = "Ana Florence"
DEFAULT_KK_MODEL_ID: Final[str] = "facebook/mms-tts-kaz"

#: Имена переменных окружения (README модуля).
ENV_DEVICE: Final[str] = "RT_TTS_DEVICE"
ENV_MODELS_DIR: Final[str] = "RT_MODELS_DIR"
ENV_VOICES_DIR: Final[str] = "RT_VOICES_DIR"
ENV_KK_BACKEND: Final[str] = "RT_TTS_KK_BACKEND"
ENV_BACKEND: Final[str] = "RT_TTS_BACKEND"
ENV_XTTS_MODEL_ID: Final[str] = "RT_TTS_XTTS_MODEL_ID"
ENV_KK_MODEL_ID: Final[str] = "RT_TTS_KK_MODEL_ID"
ENV_XTTS_SPEAKER: Final[str] = "RT_TTS_XTTS_SPEAKER"


class TtsError(RuntimeError):
    """Ошибка синтеза речи (нет модели, язык не поддержан, битый чекпоинт)."""


class Device(StrEnum):
    """Устройство инференса XTTS."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"


class KkBackend(StrEnum):
    """Бэкенд казахского TTS (клонирования нет ни у одного)."""

    MMS = "mms"
    KAZAKHTTS2 = "kazakhtts2"


class Backend(StrEnum):
    """Режим модуля: реальные модели или фейк (тесты, файловый режим)."""

    REAL = "real"
    FAKE = "fake"


E = TypeVar("E", bound=StrEnum)


def _env_enum(env: Mapping[str, str], key: str, enum: type[E], default: E) -> E:
    """Прочитать enum-значение из окружения с понятной ошибкой."""
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return enum(raw.strip().lower())
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum)
        raise TtsError(f"{key}={raw!r}: допустимые значения — {allowed}") from exc


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    return Path(raw).expanduser()


@dataclass(frozen=True, slots=True)
class TtsConfig:
    """Конфигурация модуля tts.

    Attributes:
        device: устройство XTTS; ``auto`` — cuda, если torch видит GPU.
        models_dir: кэш моделей (`scripts/download_models.py`, ``RT_MODELS_DIR``).
        voices_dir: профили голосов на диске (``RT_VOICES_DIR``, по умолчанию ``./voices``).
        xtts_model_id: репозиторий XTTS-v2 на Hugging Face.
        xtts_api_name: имя модели для высокоуровневого ``TTS`` API (если чекпоинта нет локально).
        xtts_speaker: встроенный голос XTTS для случая ``voice_id is None``.
        kk_backend: ``mms`` (facebook/mms-tts-kaz) или ``kazakhtts2`` (ISSAI, адаптер-заготовка).
        kk_model_id: модель казахского TTS.
        backend: ``real`` — настоящие модели, ``fake`` — :class:`~engine.tts.fake.FakeTts`.
        chunk_ms: длительность чанка на выходе, 20-100 мс.
        stream_chunk_size: ``stream_chunk_size`` для ``Xtts.inference_stream``.
    """

    device: Device = Device.AUTO
    models_dir: Path = DEFAULT_MODELS_DIR
    voices_dir: Path = DEFAULT_VOICES_DIR
    xtts_model_id: str = DEFAULT_XTTS_MODEL_ID
    xtts_api_name: str = DEFAULT_XTTS_API_NAME
    xtts_speaker: str = DEFAULT_XTTS_SPEAKER
    kk_backend: KkBackend = KkBackend.MMS
    kk_model_id: str = DEFAULT_KK_MODEL_ID
    backend: Backend = Backend.REAL
    chunk_ms: int = DEFAULT_CHUNK_MS
    stream_chunk_size: int = 20

    def __post_init__(self) -> None:
        if not CHUNK_MS_MIN <= self.chunk_ms <= CHUNK_MS_MAX:
            raise TtsError(
                f"chunk_ms={self.chunk_ms}: допустимо {CHUNK_MS_MIN}-{CHUNK_MS_MAX} мс "
                "(ARCHITECTURE.md 4.4)"
            )
        if self.stream_chunk_size <= 0:
            raise TtsError(f"stream_chunk_size должен быть > 0, получено {self.stream_chunk_size}")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> TtsConfig:
        """Собрать конфигурацию из переменных окружения (см. README модуля)."""
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            device=_env_enum(source, ENV_DEVICE, Device, Device.AUTO),
            models_dir=_env_path(source, ENV_MODELS_DIR, DEFAULT_MODELS_DIR),
            voices_dir=_env_path(source, ENV_VOICES_DIR, DEFAULT_VOICES_DIR),
            xtts_model_id=source.get(ENV_XTTS_MODEL_ID) or DEFAULT_XTTS_MODEL_ID,
            xtts_speaker=source.get(ENV_XTTS_SPEAKER) or DEFAULT_XTTS_SPEAKER,
            kk_backend=_env_enum(source, ENV_KK_BACKEND, KkBackend, KkBackend.MMS),
            kk_model_id=source.get(ENV_KK_MODEL_ID) or DEFAULT_KK_MODEL_ID,
            backend=_env_enum(source, ENV_BACKEND, Backend, Backend.REAL),
        )

    def resolve_device(self) -> Device:
        """Во что превращается ``auto``: cuda при доступном GPU, иначе cpu.

        torch импортируется лениво — модуль обязан работать без него.
        """
        if self.device is not Device.AUTO:
            return self.device
        try:
            import torch
        except ImportError:
            return Device.CPU
        return Device.CUDA if torch.cuda.is_available() else Device.CPU


@runtime_checkable
class TtsProvider(Protocol):
    """Провайдер синтеза речи.

    Реализации — асинхронные генераторы: ``synthesize`` объявляется как
    ``async def ... yield`` и потому имеет тип ``AsyncIterator``; вызывающий
    код всегда пишет ``async for chunk in provider.synthesize(...)``.
    """

    @property
    def supports_cloning(self) -> bool:
        """Умеет ли провайдер клонировать голос по сэмплу (``voice_id``)."""
        ...

    def supports(self, lang: Lang) -> bool:
        """Поддерживается ли язык этим провайдером."""
        ...

    async def warmup(self) -> None:
        """Заранее загрузить модель в память (вызывается при старте движка)."""
        ...

    def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        """Синтезировать текст, отдавая чанки PCM 24 kHz mono int16 по 20-100 мс.

        Args:
            text: текст на языке ``lang``.
            lang: язык синтеза; если ``supports(lang)`` ложно — :class:`TtsError`.
            voice_id: профиль голоса из :class:`~engine.tts.voices.VoiceStore`;
                ``None`` — голос по умолчанию. Провайдеры без клонирования
                (kk) аргумент игнорируют.

        Returns:
            Асинхронный итератор чанков PCM.
        """
        ...
