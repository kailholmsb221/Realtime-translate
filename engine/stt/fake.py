"""Лёгкие реализации интерфейсов stt: чистый numpy, без моделей.

Нужны в двух местах:

* юнит-тесты (CLAUDE.md: тяжёлые модели в тестах не поднимаем);
* аварийный режим — :class:`EnergyVad` работает без ``torch`` и годится как
  fallback, если Silero VAD недоступен.

Обе реализации удовлетворяют протоколам из :mod:`engine.stt.base`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping

import numpy as np

from engine.stt.base import (
    SAMPLE_RATE,
    Int16Array,
    SttConfig,
    TranscriptResult,
    TranscriptSegment,
    VadEvent,
    samples_to_ms,
)

__all__ = ["FRAME_MS", "EnergyVad", "FakeTranscriber"]

logger = logging.getLogger("engine.stt")

FRAME_MS: int = 30
"""Окно анализа EnergyVad, мс (480 сэмплов при 16 kHz — как чанк audio_io)."""


class EnergyVad:
    """Детектор речи по энергии (RMS) — простой, детерминированный, без torch.

    Поток режется на кадры по :data:`FRAME_MS`; кадр считается речевым, если
    его RMS (в долях полной шкалы int16) не ниже порога. Гистерезис:
    ``speech_start`` выдаётся, когда речь держится ``min_speech_ms``, а
    ``speech_end`` — когда тишина держится ``min_silence_ms``. Таймкоды
    считаются от начала потока и указывают на реальные границы речи, а не на
    момент срабатывания гистерезиса.

    Не заменяет Silero VAD на живом звуке (шум, музыка и дыхание дают ложные
    срабатывания), но полностью предсказуем на фикстурах.
    """

    def __init__(
        self,
        threshold: float = 0.02,
        *,
        min_speech_ms: int = 250,
        min_silence_ms: int = 500,
        frame_ms: int = FRAME_MS,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        """Создать детектор.

        Args:
            threshold: порог RMS в долях полной шкалы (0..1).
            min_speech_ms: минимальная длительность речи для ``speech_start``.
            min_silence_ms: длительность тишины, закрывающая фразу.
            frame_ms: окно анализа, мс.
            sample_rate: частота дискретизации, Гц.

        Raises:
            ValueError: некорректный порог или окно.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold должен быть в [0, 1], получено {threshold}")
        if frame_ms <= 0:
            raise ValueError(f"frame_ms должен быть > 0, получено {frame_ms}")

        self._threshold = threshold
        self._min_speech_ms = min_speech_ms
        self._min_silence_ms = min_silence_ms
        self._sample_rate = sample_rate
        self._frame = int(frame_ms * sample_rate // 1000)
        self._frame_ms = samples_to_ms(self._frame, sample_rate)
        self._buffer: Int16Array = np.zeros(0, dtype=np.int16)
        self._pos_samples = 0
        self._in_speech = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._candidate_start_ms = 0
        self._last_speech_end_ms = 0

    @classmethod
    def from_config(cls, config: SttConfig) -> EnergyVad:
        """Собрать детектор из :class:`SttConfig`."""
        return cls(
            threshold=config.energy_threshold,
            min_speech_ms=config.min_speech_ms,
            min_silence_ms=config.min_silence_ms,
        )

    @property
    def frame_samples(self) -> int:
        """Размер кадра анализа в сэмплах."""
        return self._frame

    def process(self, chunk_int16_16k: Int16Array) -> list[VadEvent]:
        """Скормить чанк PCM и получить найденные границы речи."""
        samples = np.asarray(chunk_int16_16k, dtype=np.int16).reshape(-1)
        if samples.size:
            self._buffer = (
                samples if self._buffer.size == 0 else np.concatenate((self._buffer, samples))
            )

        events: list[VadEvent] = []
        while self._buffer.size >= self._frame:
            frame = self._buffer[: self._frame]
            self._buffer = self._buffer[self._frame :]
            events.extend(self._feed_frame(frame))
        return events

    def reset(self) -> None:
        """Сбросить состояние и таймлайн."""
        self._buffer = np.zeros(0, dtype=np.int16)
        self._pos_samples = 0
        self._in_speech = False
        self._speech_run_ms = 0
        self._silence_run_ms = 0
        self._candidate_start_ms = 0
        self._last_speech_end_ms = 0

    def flush(self) -> list[VadEvent]:
        """Закрыть незавершённую фразу в конце потока.

        Returns:
            ``[speech_end]``, если речь ещё шла, иначе пустой список.
        """
        if not self._in_speech:
            return []
        self._in_speech = False
        self._silence_run_ms = 0
        self._speech_run_ms = 0
        return [VadEvent.speech_end(self._last_speech_end_ms)]

    def _feed_frame(self, frame: Int16Array) -> list[VadEvent]:
        start_ms = samples_to_ms(self._pos_samples, self._sample_rate)
        self._pos_samples += self._frame
        end_ms = samples_to_ms(self._pos_samples, self._sample_rate)

        is_speech = self._rms(frame) >= self._threshold
        events: list[VadEvent] = []

        if self._in_speech:
            if is_speech:
                self._silence_run_ms = 0
                self._last_speech_end_ms = end_ms
            else:
                self._silence_run_ms += self._frame_ms
                if self._silence_run_ms >= self._min_silence_ms:
                    events.append(VadEvent.speech_end(self._last_speech_end_ms))
                    self._in_speech = False
                    self._speech_run_ms = 0
                    self._silence_run_ms = 0
            return events

        if is_speech:
            if self._speech_run_ms == 0:
                self._candidate_start_ms = start_ms
            self._speech_run_ms += self._frame_ms
            self._last_speech_end_ms = end_ms
            if self._speech_run_ms >= self._min_speech_ms:
                events.append(VadEvent.speech_start(self._candidate_start_ms))
                self._in_speech = True
                self._silence_run_ms = 0
        else:
            self._speech_run_ms = 0
        return events

    @staticmethod
    def _rms(frame: Int16Array) -> float:
        values = frame.astype(np.float64) / 32768.0
        return float(np.sqrt(np.mean(values * values)))


class FakeTranscriber:
    """Распознаватель-заглушка: отдаёт заранее заданный текст.

    Варианты поведения:

    * фиксированный ``text`` на любой буфер;
    * ``by_duration`` — словарь ``порог_мс -> текст``: берётся текст с
      наибольшим порогом, не превышающим длительность буфера (так тест может
      различать partial и final по длине накопленного аудио).

    Считает вызовы в :attr:`calls` — удобно для проверки, что движок не
    запускает две транскрипции одновременно.
    """

    def __init__(
        self,
        text: str = "тестовая фраза",
        *,
        lang: str = "ru",
        by_duration: Mapping[int, str] | None = None,
        delay_s: float = 0.0,
    ) -> None:
        """Создать заглушку.

        Args:
            text: текст по умолчанию.
            lang: язык, возвращаемый при ``lang=None``.
            by_duration: карта ``порог_мс -> текст``.
            delay_s: искусственная задержка распознавания, с.
        """
        self._text = text
        self._lang = lang
        self._by_duration = dict(by_duration or {})
        self._delay_s = delay_s
        self.calls: list[tuple[int, str | None]] = []

    def transcribe(self, pcm_int16_16k: Int16Array, lang: str | None) -> TranscriptResult:
        """Вернуть заготовленный текст для переданного буфера."""
        samples = np.asarray(pcm_int16_16k, dtype=np.int16).reshape(-1)
        duration_ms = samples_to_ms(samples.size)
        self.calls.append((duration_ms, lang))
        if self._delay_s:
            time.sleep(self._delay_s)

        text = self._text_for(duration_ms)
        segments = (
            (TranscriptSegment(text=text, t_start_ms=0, t_end_ms=duration_ms),) if text else ()
        )
        return TranscriptResult(
            text=text,
            lang=lang or self._lang,
            segments=segments,
            duration_ms=duration_ms,
        )

    def _text_for(self, duration_ms: int) -> str:
        if not self._by_duration:
            return self._text
        applicable = [ms for ms in self._by_duration if ms <= duration_ms]
        if not applicable:
            return self._text
        return self._by_duration[max(applicable)]
