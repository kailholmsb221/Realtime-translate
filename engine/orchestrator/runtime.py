"""Сборка компонентов движка и прогрев моделей (ARCHITECTURE.md 4.5, 6).

Здесь единственное место, где оркестратор решает, какие реализации взять:
живые модели и устройства Windows (``backend="real"``) или фейки того же
интерфейса (``backend="fake"`` — Linux, CI, проверка UI без железа). Все
фабрики — публичные функции чужих модулей (``create_engine``,
``create_provider``, ``create_tts``, ``create_source``/``create_sink``),
внутренности никто не импортирует (CLAUDE.md, технический стандарт).

Прогрев идёт в порядке из ARCHITECTURE.md раздела 6: сначала whisper, затем
XTTS (обе на GPU, 1.0 + 2.5 GB), NLLB — на CPU в отдельном потоке. После
каждого шага в лог пишется время и занятая VRAM, если доступен ``torch``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Final, cast

import numpy as np

from engine.audio_io import AudioSink, AudioSource, FakeSink, FakeSource
from engine.audio_io import create_sink as make_audio_sink
from engine.audio_io import create_source as make_audio_source
from engine.contracts.events import Lang, Stream
from engine.orchestrator.config import OrchestratorConfig
from engine.stt import create_engine, create_transcriber
from engine.stt.base import Transcriber
from engine.stt.engine import SttEngine
from engine.translate import create_provider
from engine.translate.base import TranslationProvider
from engine.translate.service import TranslationService
from engine.tts import create_tts
from engine.tts.router import TtsRouter
from engine.tts.voices import LatentsFn, VoiceStore

__all__ = ["FAKE_TRANSCRIPT", "Runtime", "cuda_memory_mb"]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.runtime")

#: Что «распознаёт» фейковый распознаватель (backend="fake").
FAKE_TRANSCRIPT: Final[str] = "проверочная фраза"

#: Сколько тишины скармливаем whisper при прогреве, мс.
WARMUP_SILENCE_MS: Final[int] = 1_000

#: Длительность «тишины» фейкового источника, если WAV не задан, мс.
FAKE_SILENCE_MS: Final[int] = 2_000


def cuda_memory_mb() -> float | None:
    """Сколько VRAM занято сейчас, МБ (``None``, если torch/CUDA недоступны)."""
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        return float(torch.cuda.memory_allocated()) / (1024 * 1024)
    except Exception:  # pragma: no cover — драйвер может отвалиться на целевой машине
        logger.debug("не удалось прочитать torch.cuda.memory_allocated()", exc_info=True)
        return None


def _log_step(name: str, started: float) -> None:
    """Записать в лог время шага прогрева и занятую VRAM."""
    elapsed = time.perf_counter() - started
    vram = cuda_memory_mb()
    if vram is None:
        logger.info("прогрев %s: %.1f с", name, elapsed)
    else:
        logger.info("прогрев %s: %.1f с, VRAM %.0f МБ", name, elapsed, vram)


@dataclass(slots=True)
class Runtime:
    """Общие на процесс компоненты движка.

    Тяжёлые модели живут здесь в одном экземпляре (ARCHITECTURE.md раздел 6:
    «модели загружаются один раз при старте и живут в памяти»), а состояние,
    привязанное к потоку (VAD, таймлайн, окно контекста), создаётся отдельно
    на каждый конвейер.
    """

    config: OrchestratorConfig
    transcriber: Transcriber
    provider: TranslationProvider
    tts: TtsRouter
    voices: VoiceStore

    @classmethod
    def create(cls, config: OrchestratorConfig) -> Runtime:
        """Собрать компоненты по конфигурации (модели ещё не загружаются)."""
        resolved = config.for_backend()
        if resolved.is_fake:
            transcriber = create_transcriber(resolved.stt, "fake", text=FAKE_TRANSCRIPT)
        else:
            transcriber = create_transcriber(resolved.stt, "faster_whisper")
        tts = create_tts(resolved.tts)
        return cls(
            config=resolved,
            transcriber=transcriber,
            provider=create_provider(resolved.translate),
            tts=tts,
            voices=tts.voices,
        )

    # --- компоненты конвейера ---------------------------------------------

    def stt_engine(self) -> SttEngine:
        """Новый :class:`SttEngine` (VAD и таймлайн — свои на каждый поток)."""
        return create_engine(
            self.config.stt,
            vad_backend="energy" if self.config.is_fake else "silero",
            transcriber=self.transcriber,
        )

    def translation_service(self) -> TranslationService:
        """Новый сервис перевода поверх общего провайдера."""
        return TranslationService(self.provider)

    def create_source(self, stream: Stream) -> AudioSource:
        """Источник потока: WASAPI loopback / микрофон или WAV в фейковом режиме.

        Raises:
            AudioBackendUnavailable: живое устройство запрошено не на Windows
                или без установленных аудио-пакетов.
        """
        if self.config.is_fake:
            path = self.config.fake_in if stream is Stream.IN else self.config.fake_out
            if path is not None:
                return FakeSource.from_wav(
                    path,
                    fmt=self.config.audio.format,
                    chunk_ms=self.config.audio.chunk_ms,
                    realtime=True,
                )
            logger.warning(
                "фейковый источник %s без WAV — поток будет молчать "
                "(--fake-in / --fake-out задают файлы)",
                stream.value,
            )
            frames = self.config.audio.format.frames_for_ms(FAKE_SILENCE_MS)
            return FakeSource(
                np.zeros(frames, dtype=np.int16),
                fmt=self.config.audio.format,
                chunk_ms=self.config.audio.chunk_ms,
                realtime=True,
            )
        kind = "loopback" if stream is Stream.IN else "mic"
        return make_audio_source(kind, self.config.audio)

    def create_sink(self, stream: Stream, session_id: int | None = None) -> AudioSink:
        """Приёмник перевода: наушники / VB-Cable или WAV в фейковом режиме."""
        if self.config.is_fake:
            name = f"{session_id if session_id is not None else 'session'}_{stream.value}_tts.wav"
            path = self.config.recordings_dir / name
            path.parent.mkdir(parents=True, exist_ok=True)
            return FakeSink(fmt=self.config.audio.format, path=path)
        kind = "headphones" if stream is Stream.IN else "cable"
        return make_audio_sink(kind, self.config.audio)

    def latents_fn(self) -> LatentsFn | None:
        """Чем считать латенты XTTS при создании голосового профиля.

        Берётся публичный ``compute_latents`` у провайдера ru/en — так же, как
        это делает ``scripts/tts_smoke.py --create-voice``. В фейковом режиме и
        у провайдеров без клонирования возвращается ``None``: тогда латенты
        посчитает сам XTTS при первом синтезе этого голоса.
        """
        if self.config.is_fake:
            return None
        provider = self.tts.provider_for(Lang.RU)
        compute = getattr(provider, "compute_latents", None)
        if callable(compute):
            return cast(LatentsFn, compute)
        logger.warning(
            "провайдер %s не умеет считать латенты — профиль создастся без latents.pt",
            type(provider).__name__,
        )
        return None

    # --- прогрев -----------------------------------------------------------

    async def warmup(self, langs: tuple[Lang, ...] = (Lang.RU, Lang.EN)) -> None:
        """Загрузить модели заранее: whisper → XTTS (GPU), затем NLLB (CPU).

        Порядок важен: XTTS занимает ~2.5 GB и должен вставать на уже занятую
        whisper память, а не наоборот (ARCHITECTURE.md раздел 6).
        """
        if self.config.is_fake:
            logger.info("backend=fake — прогрев не нужен, модели не загружаются")
            return

        started = time.perf_counter()
        await asyncio.to_thread(self._warmup_stt)
        _log_step("STT (faster-whisper small)", started)

        started = time.perf_counter()
        await self.tts.warmup(langs)
        _log_step("TTS (XTTS-v2)", started)

        started = time.perf_counter()
        await asyncio.to_thread(self.provider.warmup)
        _log_step("MT (NLLB-200, CPU)", started)

    def _warmup_stt(self) -> None:
        """Прогнать секунду тишины через распознаватель — это грузит модель."""
        silence = np.zeros(16 * WARMUP_SILENCE_MS, dtype=np.int16)
        self.transcriber.transcribe(silence, self.config.stt.language)

    async def aclose(self) -> None:
        """Освободить ресурсы моделей, если реализация это умеет."""
        close = getattr(self.transcriber, "close", None)
        if callable(close):
            await asyncio.to_thread(close)
