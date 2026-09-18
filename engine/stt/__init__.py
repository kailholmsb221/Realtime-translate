"""stt — распознавание речи (ARCHITECTURE.md 4.2, зона агента B).

Публичный API пакета — фабрики :func:`create_vad`, :func:`create_transcriber`,
:func:`create_engine` и типы из :mod:`engine.stt.base`. Остальные модули движка
не должны знать, какой бэкенд используется.

Бэкенды:

* VAD — ``"silero"`` (Silero VAD, продакшн) или ``"energy"``
  (:class:`~engine.stt.fake.EnergyVad`, RMS-порог, чистый numpy);
* распознавание — ``"faster_whisper"`` (модель ``small`` int8_float16) или
  ``"fake"`` (:class:`~engine.stt.fake.FakeTranscriber` для тестов и файлового
  режима).

Тяжёлые модели грузятся лениво внутри реализаций, поэтому импорт пакета не
требует ``torch``/``faster-whisper`` (CLAUDE.md, технический стандарт).

Пример::

    from engine.stt import create_engine
    from engine.stt.base import SttConfig

    engine = create_engine(SttConfig.from_env())
    async for envelope in engine.run(source, stream="in", lang="ru"):
        ...  # envelope.type == "stt.partial" | "stt.final"

Запуск smoke-теста: ``python scripts/stt_smoke.py sample.wav --lang ru``.
"""

from __future__ import annotations

from typing import Final, Literal

from engine.stt.base import (
    SAMPLE_RATE,
    Device,
    Int16Array,
    SttConfig,
    SttError,
    Transcriber,
    TranscriptResult,
    TranscriptSegment,
    VadEvent,
    VadEventKind,
    VoiceActivityDetector,
)
from engine.stt.engine import SttEngine
from engine.stt.fake import EnergyVad, FakeTranscriber

__all__ = [
    "SAMPLE_RATE",
    "TRANSCRIBER_BACKENDS",
    "VAD_BACKENDS",
    "Device",
    "EnergyVad",
    "FakeTranscriber",
    "Int16Array",
    "SttConfig",
    "SttEngine",
    "SttError",
    "Transcriber",
    "TranscriberBackend",
    "TranscriptResult",
    "TranscriptSegment",
    "VadBackend",
    "VadEvent",
    "VadEventKind",
    "VoiceActivityDetector",
    "create_engine",
    "create_transcriber",
    "create_vad",
]

VadBackend = Literal["silero", "energy"]
TranscriberBackend = Literal["faster_whisper", "fake"]

VAD_BACKENDS: Final[tuple[str, ...]] = ("silero", "energy")
TRANSCRIBER_BACKENDS: Final[tuple[str, ...]] = ("faster_whisper", "fake")


def create_vad(
    config: SttConfig | None = None,
    backend: VadBackend | str = "silero",
) -> VoiceActivityDetector:
    """Создать детектор речи.

    Args:
        config: настройки модуля; по умолчанию :meth:`SttConfig.from_env`.
        backend: ``"silero"`` — Silero VAD (нужен ``torch``), ``"energy"`` —
            :class:`~engine.stt.fake.EnergyVad` без тяжёлых зависимостей.

    Returns:
        Объект, реализующий :class:`~engine.stt.base.VoiceActivityDetector`.

    Raises:
        ValueError: неизвестный бэкенд.
    """
    cfg = config or SttConfig.from_env()

    if backend == "energy":
        return EnergyVad.from_config(cfg)
    if backend == "silero":
        from engine.stt.vad_silero import SileroVad

        return SileroVad(cfg)
    raise ValueError(f"неизвестный backend VAD: {backend!r}; доступны {VAD_BACKENDS}")


def create_transcriber(
    config: SttConfig | None = None,
    backend: TranscriberBackend | str = "faster_whisper",
    **kwargs: object,
) -> Transcriber:
    """Создать распознаватель речи.

    Args:
        config: настройки модуля; по умолчанию :meth:`SttConfig.from_env`.
        backend: ``"faster_whisper"`` — реальная модель (грузится лениво),
            ``"fake"`` — :class:`~engine.stt.fake.FakeTranscriber`.
        **kwargs: параметры :class:`~engine.stt.fake.FakeTranscriber`
            (``text``, ``lang``, ``by_duration``, ``delay_s``).

    Returns:
        Объект, реализующий :class:`~engine.stt.base.Transcriber`.

    Raises:
        ValueError: неизвестный бэкенд.
    """
    cfg = config or SttConfig.from_env()

    if backend == "fake":
        text = str(kwargs.pop("text", "тестовая фраза"))
        lang = str(kwargs.pop("lang", cfg.language or cfg.fallback_lang))
        return FakeTranscriber(text, lang=lang, **kwargs)  # type: ignore[arg-type]
    if backend == "faster_whisper":
        from engine.stt.whisper_fw import FasterWhisperTranscriber

        return FasterWhisperTranscriber(cfg)
    raise ValueError(
        f"неизвестный backend распознавания: {backend!r}; доступны {TRANSCRIBER_BACKENDS}"
    )


def create_engine(
    config: SttConfig | None = None,
    *,
    backend: TranscriberBackend | str = "faster_whisper",
    vad_backend: VadBackend | str = "silero",
    vad: VoiceActivityDetector | None = None,
    transcriber: Transcriber | None = None,
) -> SttEngine:
    """Собрать :class:`~engine.stt.engine.SttEngine`.

    Args:
        config: настройки модуля; по умолчанию :meth:`SttConfig.from_env`.
        backend: бэкенд распознавания (``"faster_whisper"`` | ``"fake"``).
        vad_backend: бэкенд VAD (``"silero"`` | ``"energy"``).
        vad: готовый детектор — тогда ``vad_backend`` игнорируется.
        transcriber: готовый распознаватель — тогда ``backend`` игнорируется.

    Returns:
        Движок, готовый к :meth:`~engine.stt.engine.SttEngine.run`.
    """
    cfg = config or SttConfig.from_env()
    return SttEngine(
        vad=vad if vad is not None else create_vad(cfg, vad_backend),
        transcriber=transcriber if transcriber is not None else create_transcriber(cfg, backend),
        config=cfg,
    )
