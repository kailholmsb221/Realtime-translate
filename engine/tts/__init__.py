"""Пакет tts: синтез речи (XTTS-v2 для ru/en с клоном голоса, MMS/KazakhTTS2 для kk).

Зона агента D. См. ARCHITECTURE.md раздел 4.4 и README этого пакета.

Быстрый старт::

    from engine.contracts.events import Lang
    from engine.tts import TtsConfig, create_tts

    tts = create_tts(TtsConfig.from_env())
    async for chunk in tts.synthesize("привет", Lang.RU, voice_id=None):
        ...  # PCM 24 kHz mono int16

Тяжёлые модели (torch, coqui-tts, transformers) импортируются лениво внутри
провайдеров: сам пакет можно импортировать без них.
"""

from __future__ import annotations

from engine.tts.audio import Pcm16, read_wav, resample_pcm, write_wav
from engine.tts.base import (
    CHUNK_MS_MAX,
    CHUNK_MS_MIN,
    OUTPUT_SAMPLE_RATE,
    Backend,
    Device,
    KkBackend,
    TtsConfig,
    TtsError,
    TtsProvider,
)
from engine.tts.fake import FakeTts
from engine.tts.router import TtsRouter, chunk_to_event, pcm_from_event
from engine.tts.voices import (
    MAX_SAMPLE_SECONDS,
    MIN_SAMPLE_SECONDS,
    RECOMMENDED_SAMPLE_SECONDS,
    VoiceError,
    VoiceNotFoundError,
    VoiceProfile,
    VoiceStore,
)

__all__ = [
    "CHUNK_MS_MAX",
    "CHUNK_MS_MIN",
    "MAX_SAMPLE_SECONDS",
    "MIN_SAMPLE_SECONDS",
    "OUTPUT_SAMPLE_RATE",
    "RECOMMENDED_SAMPLE_SECONDS",
    "Backend",
    "Device",
    "FakeTts",
    "KkBackend",
    "Pcm16",
    "TtsConfig",
    "TtsError",
    "TtsProvider",
    "TtsRouter",
    "VoiceError",
    "VoiceNotFoundError",
    "VoiceProfile",
    "VoiceStore",
    "chunk_to_event",
    "create_tts",
    "create_voice_store",
    "pcm_from_event",
    "read_wav",
    "resample_pcm",
    "write_wav",
]


def create_tts(config: TtsConfig | None = None) -> TtsRouter:
    """Фабрика модуля: собрать маршрутизатор TTS по конфигурации.

    Args:
        config: конфигурация; ``None`` — собрать из переменных окружения
            (``RT_TTS_BACKEND``, ``RT_TTS_DEVICE``, ``RT_TTS_KK_BACKEND``,
            ``RT_MODELS_DIR``, ``RT_VOICES_DIR``).

    Returns:
        :class:`~engine.tts.router.TtsRouter`; модели грузятся лениво —
        при первом ``synthesize`` или явном ``warmup``.
    """
    return TtsRouter(config if config is not None else TtsConfig.from_env())


def create_voice_store(config: TtsConfig | None = None) -> VoiceStore:
    """Хранилище голосовых профилей по конфигурации (``RT_VOICES_DIR``)."""
    resolved = config if config is not None else TtsConfig.from_env()
    return VoiceStore(resolved.voices_dir)
