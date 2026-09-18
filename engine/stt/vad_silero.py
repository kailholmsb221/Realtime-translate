"""Silero VAD (ARCHITECTURE.md 4.2): режет поток на фразы.

Модель загружается лениво — при первом :meth:`SileroVad.process`, — поэтому
импорт модуля не тянет ``torch`` и юнит-тесты живут без него
(CLAUDE.md, технический стандарт).

Silero работает окнами ровно по 512 сэмплов при 16 kHz, а audio_io отдаёт
чанки по 30 мс (480 сэмплов), поэтому обёртка копит входные чанки в буфер и
режет его на окна по 512.

Установка::

    pip install -r engine/stt/requirements-stt.txt
"""

from __future__ import annotations

import logging
from typing import Any, Final

import numpy as np

from engine.stt.base import (
    SAMPLE_RATE,
    Int16Array,
    SttConfig,
    SttError,
    VadEvent,
    samples_to_ms,
    to_float32,
)

__all__ = ["WINDOW_SAMPLES", "SileroVad"]

logger = logging.getLogger("engine.stt")

WINDOW_SAMPLES: Final[int] = 512
"""Обязательный размер окна Silero VAD при 16 kHz."""

_SPEECH_PAD_MS: Final[int] = 100
"""Запас вокруг границ речи, который Silero добавляет сам, мс."""


class SileroVad:
    """Обёртка над ``silero_vad.VADIterator`` под протокол ``VoiceActivityDetector``.

    Таймкоды событий считаются от начала потока: ``VADIterator`` возвращает
    абсолютный номер сэмпла, обёртка переводит его в миллисекунды.
    """

    def __init__(
        self,
        config: SttConfig | None = None,
        *,
        threshold: float | None = None,
        min_silence_ms: int | None = None,
        speech_pad_ms: int = _SPEECH_PAD_MS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """Создать детектор (модель ещё не грузится).

        Args:
            config: конфиг модуля; из него берутся порог и ``min_silence_ms``.
            threshold: явный порог вероятности речи, перебивает конфиг.
            min_silence_ms: явная длительность тишины, закрывающая фразу.
            speech_pad_ms: запас вокруг границ речи, мс.
            sample_rate: частота дискретизации (Silero поддерживает 8k и 16k).

        Raises:
            ValueError: неподдерживаемая частота дискретизации.
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"SileroVad рассчитан на {SAMPLE_RATE} Гц, получено {sample_rate}")

        cfg = config or SttConfig()
        self._threshold = cfg.vad_threshold if threshold is None else threshold
        self._min_silence_ms = cfg.min_silence_ms if min_silence_ms is None else min_silence_ms
        self._speech_pad_ms = speech_pad_ms
        self._sample_rate = sample_rate
        self._iterator: Any | None = None
        self._buffer: Int16Array = np.zeros(0, dtype=np.int16)

    @property
    def frame_samples(self) -> int:
        """Размер окна анализа в сэмплах (512 при 16 kHz)."""
        return WINDOW_SAMPLES

    def process(self, chunk_int16_16k: Int16Array) -> list[VadEvent]:
        """Скормить чанк PCM 16 kHz mono int16 и получить границы речи.

        Raises:
            SttError: пакет ``silero-vad`` не установлен.
        """
        iterator = self._ensure_iterator()

        samples = np.asarray(chunk_int16_16k, dtype=np.int16).reshape(-1)
        if samples.size:
            self._buffer = (
                samples if self._buffer.size == 0 else np.concatenate((self._buffer, samples))
            )

        events: list[VadEvent] = []
        while self._buffer.size >= WINDOW_SAMPLES:
            window = self._buffer[:WINDOW_SAMPLES]
            self._buffer = self._buffer[WINDOW_SAMPLES:]
            verdict = iterator(to_float32(window), return_seconds=False)
            event = self._to_event(verdict)
            if event is not None:
                events.append(event)
        return events

    def reset(self) -> None:
        """Сбросить состояние модели и накопленный буфер (новый поток)."""
        self._buffer = np.zeros(0, dtype=np.int16)
        if self._iterator is not None:
            self._iterator.reset_states()

    def _to_event(self, verdict: Any) -> VadEvent | None:
        if not verdict:
            return None
        if "start" in verdict:
            return VadEvent.speech_start(self._sample_to_ms(verdict["start"]))
        if "end" in verdict:
            return VadEvent.speech_end(self._sample_to_ms(verdict["end"]))
        logger.debug("VAD: неизвестный вердикт %r", verdict)
        return None

    def _sample_to_ms(self, sample_index: Any) -> int:
        return max(0, samples_to_ms(int(sample_index), self._sample_rate))

    def _ensure_iterator(self) -> Any:
        """Лениво загрузить модель Silero и создать итератор."""
        if self._iterator is not None:
            return self._iterator
        try:
            from silero_vad import VADIterator, load_silero_vad
        except ImportError as exc:  # pragma: no cover — зависит от окружения
            raise SttError(
                "не установлен пакет silero-vad; поставьте зависимости модуля: "
                "pip install -r engine/stt/requirements-stt.txt "
                "(или используйте backend='energy' — EnergyVad работает без torch)"
            ) from exc

        logger.info("Загружаю Silero VAD (threshold=%.2f)", self._threshold)
        model = load_silero_vad()
        self._iterator = VADIterator(
            model,
            threshold=self._threshold,
            sampling_rate=self._sample_rate,
            min_silence_duration_ms=self._min_silence_ms,
            speech_pad_ms=self._speech_pad_ms,
        )
        return self._iterator
