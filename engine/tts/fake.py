"""Фейковый синтезатор: тон вместо речи (тесты и файловый режим).

Не тянет ни torch, ни TTS — реализует тот же :class:`~engine.tts.base.TtsProvider`,
что и XTTS/MMS (CLAUDE.md: в юнит-тестах — фейковые реализации того же интерфейса).
Длительность пропорциональна длине текста: ``ms_per_word`` на слово.
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from collections.abc import AsyncIterator, Iterable
from typing import Final

import numpy as np

from engine.contracts.events import Lang
from engine.tts.audio import Pcm16, chunk_samples, float_to_int16
from engine.tts.base import CHUNK_MS_MIN, OUTPUT_SAMPLE_RATE, TtsError

__all__ = ["DEFAULT_MS_PER_WORD", "FakeTts"]

log: Final[logging.Logger] = logging.getLogger(__name__)

#: Сколько миллисекунд «речи» приходится на одно слово.
DEFAULT_MS_PER_WORD: Final[int] = 60
MIN_UTTERANCE_MS: Final[int] = 60
TONE_HZ: Final[float] = 220.0
AMPLITUDE: Final[float] = 0.3
NOISE: Final[float] = 0.02


class FakeTts:
    """Генератор тона с лёгким шумом — заглушка вместо реальной модели.

    Args:
        langs: какие языки «поддерживает» (по умолчанию все три).
        supports_cloning: что отвечать на вопрос о клонировании.
        chunk_ms: длительность чанка, по умолчанию 20 мс.
        ms_per_word: длительность «речи» на одно слово.
        freq_hz: частота тона; по ``voice_id`` слегка сдвигается, чтобы разные
            голоса звучали по-разному.
    """

    __slots__ = ("_amplitude", "_chunk_ms", "_cloning", "_freq_hz", "_langs", "_ms_per_word")

    def __init__(
        self,
        langs: Iterable[Lang] = (Lang.RU, Lang.EN, Lang.KK),
        *,
        supports_cloning: bool = False,
        chunk_ms: int = CHUNK_MS_MIN,
        ms_per_word: int = DEFAULT_MS_PER_WORD,
        freq_hz: float = TONE_HZ,
        amplitude: float = AMPLITUDE,
    ) -> None:
        self._langs = frozenset(langs)
        self._cloning = supports_cloning
        self._chunk_ms = chunk_ms
        self._ms_per_word = ms_per_word
        self._freq_hz = freq_hz
        self._amplitude = amplitude

    @property
    def supports_cloning(self) -> bool:
        """Фейк по умолчанию «не клонирует», но флаг настраивается в тестах."""
        return self._cloning

    def supports(self, lang: Lang) -> bool:
        """Поддерживается ли язык (настраивается конструктором)."""
        return lang in self._langs

    async def warmup(self) -> None:
        """Ничего не грузит — фейк всегда готов."""
        return None

    def duration_ms(self, text: str) -> int:
        """Сколько миллисекунд «речи» даст этот текст."""
        words = max(1, len(text.split()))
        return max(MIN_UTTERANCE_MS, words * self._ms_per_word)

    def render(self, text: str, voice_id: str | None = None) -> Pcm16:
        """Собрать весь сигнал целиком (удобно для тестов и файлового режима)."""
        n_samples = round(OUTPUT_SAMPLE_RATE * self.duration_ms(text) / 1000)
        t = np.arange(n_samples, dtype=np.float64) / OUTPUT_SAMPLE_RATE
        seed = zlib.crc32(f"{voice_id}|{text}".encode())  # стабильно между запусками
        freq = self._freq_hz * (1.0 + 0.1 * (seed % 5)) if voice_id else self._freq_hz
        signal = self._amplitude * np.sin(2.0 * np.pi * freq * t)
        rng = np.random.default_rng(seed)
        signal += NOISE * rng.standard_normal(n_samples)
        fade = min(chunk_samples(OUTPUT_SAMPLE_RATE, 10), n_samples // 2)
        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
            signal[:fade] *= ramp
            signal[-fade:] *= ramp[::-1]
        return float_to_int16(signal)

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        """Отдать тон чанками по ``chunk_ms`` мс (PCM 24 kHz mono int16)."""
        if not self.supports(lang):
            raise TtsError(f"FakeTts: язык {lang.value} не поддерживается")
        pcm = self.render(text, voice_id)
        size = chunk_samples(OUTPUT_SAMPLE_RATE, self._chunk_ms)
        for start in range(0, pcm.size, size):
            await asyncio.sleep(0)  # отдать управление циклу событий
            yield pcm[start : start + size]
