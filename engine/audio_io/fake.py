"""Фейковые источник и приёмник — тесты и файловый режим (этап 1).

:class:`FakeSource` играет WAV-файл или ``numpy``-массив чанками по 30 мс,
:class:`FakeSink` копит всё записанное и умеет сохранить это в WAV.
Оба класса реализуют те же протоколы, что и Windows-реализации
(:class:`~engine.audio_io.base.AudioSource` / ``AudioSink``), поэтому
оркестратор в файловом режиме работает с ними без изменений кода.

Пример::

    from engine.audio_io.fake import FakeSink, FakeSource

    src = FakeSource.from_wav("sample.wav")
    sink = FakeSink()
    async with src, sink:
        async for chunk in src:
            await sink.write(chunk)
    sink.save_wav("out.wav")
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from engine.audio_io.base import (
    CHUNK_MS,
    ENGINE_FORMAT,
    AudioChunk,
    AudioFormat,
    BaseAudioSink,
    BaseAudioSource,
)
from engine.audio_io.pcm import Int16Array, resample, to_mono, write_wav
from engine.audio_io.pcm import read_wav as _read_wav

__all__ = ["FakeSink", "FakeSource"]


class FakeSource(BaseAudioSource):
    """Источник, играющий заранее известный сигнал чанками фиксированной длины.

    Args:
        samples: int16-массив моно-сэмплов (или любой массив, приводимый к int16).
        fmt: формат сигнала; по умолчанию формат движка (16 kHz mono int16).
        chunk_ms: длительность чанка, мс.
        realtime: выдерживать реальную скорость воспроизведения
            (``asyncio.sleep`` между чанками) — для похожих на живые прогоны.
        pad_last: дополнить последний неполный чанк нулями (иначе отбросить).
        start_ts_ms: таймкод первого чанка.
    """

    def __init__(
        self,
        samples: npt.NDArray[Any] | Sequence[int],
        *,
        fmt: AudioFormat = ENGINE_FORMAT,
        chunk_ms: float = CHUNK_MS,
        realtime: bool = False,
        pad_last: bool = True,
        start_ts_ms: int = 0,
    ) -> None:
        super().__init__(fmt=fmt, chunk_ms=chunk_ms)
        if fmt.channels != 1:
            raise ValueError("FakeSource работает только с моно-сигналом")
        data = np.asarray(samples)
        if data.ndim != 1:
            raise ValueError(f"ожидался моно-сигнал, получен массив формы {data.shape}")
        self._samples: Int16Array = np.ascontiguousarray(data, dtype=np.int16)
        self._realtime = realtime
        self._pad_last = pad_last
        self._start_ts_ms = start_ts_ms

    @classmethod
    def from_wav(
        cls,
        path: str | Path,
        *,
        fmt: AudioFormat = ENGINE_FORMAT,
        chunk_ms: float = CHUNK_MS,
        realtime: bool = False,
        pad_last: bool = True,
    ) -> FakeSource:
        """Прочитать WAV, свести в моно и при необходимости ресемплить под ``fmt``."""
        samples, rate = _read_wav(path)
        if rate != fmt.sample_rate:
            samples = resample(samples, rate, fmt.sample_rate)
        return cls(
            samples,
            fmt=fmt,
            chunk_ms=chunk_ms,
            realtime=realtime,
            pad_last=pad_last,
        )

    @property
    def samples(self) -> Int16Array:
        """Исходный сигнал целиком."""
        return self._samples

    @property
    def total_frames(self) -> int:
        """Длина исходного сигнала во фреймах."""
        return int(self._samples.size)

    @property
    def n_chunks(self) -> int:
        """Сколько чанков отдаст источник."""
        frames = self.chunk_frames
        whole, rest = divmod(self.total_frames, frames)
        return whole + (1 if rest and self._pad_last else 0)

    async def _open(self) -> None:
        return None

    async def _close(self) -> None:
        return None

    async def _iter_chunks(self) -> AsyncIterator[AudioChunk]:
        frames = self.chunk_frames
        period = frames / self._fmt.sample_rate
        started = time.perf_counter()
        for index, offset in enumerate(range(0, self.total_frames, frames)):
            piece = self._samples[offset : offset + frames]
            if piece.size < frames:
                if not self._pad_last:
                    break
                padded = np.zeros(frames, dtype=np.int16)
                padded[: piece.size] = piece
                piece = padded
            ts_ms = self._start_ts_ms + round(self._fmt.ms_for_frames(offset))
            if self._realtime:
                deadline = started + (index + 1) * period
                delay = deadline - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await asyncio.sleep(0)  # не монополизировать цикл событий
            yield AudioChunk.from_array(piece, ts_ms, self._fmt)


class FakeSink(BaseAudioSink):
    """Приёмник, копящий всё записанное в памяти; умеет сохранить WAV.

    Чанки с другой частотой (например, 24 kHz с выхода TTS) ресемплятся
    к ``fmt.sample_rate``, многоканальные — сводятся в моно: так же, как это
    делает настоящий приёмник перед отправкой в устройство.

    Args:
        fmt: формат накопления (по умолчанию 16 kHz mono int16).
        path: если задан, WAV пишется автоматически при ``close()``.
    """

    def __init__(self, *, fmt: AudioFormat = ENGINE_FORMAT, path: str | Path | None = None) -> None:
        super().__init__()
        if fmt.channels != 1:
            raise ValueError("FakeSink накапливает только моно")
        self._fmt = fmt
        self._path = Path(path) if path is not None else None
        self._parts: list[Int16Array] = []
        self.chunks: list[AudioChunk] = []
        self.drains = 0

    @property
    def format(self) -> AudioFormat:
        """Формат накопленного сигнала."""
        return self._fmt

    @property
    def path(self) -> Path | None:
        """Куда будет сохранён WAV при ``close()`` (если задан)."""
        return self._path

    @property
    def samples(self) -> Int16Array:
        """Всё записанное как один int16-массив."""
        if not self._parts:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(self._parts)

    @property
    def n_frames(self) -> int:
        """Сколько фреймов записано."""
        return int(sum(part.size for part in self._parts))

    @property
    def duration_ms(self) -> float:
        """Длительность записанного, мс."""
        return self._fmt.ms_for_frames(self.n_frames)

    def clear(self) -> None:
        """Забыть всё записанное."""
        self._parts.clear()
        self.chunks.clear()

    async def _open(self) -> None:
        return None

    async def _write(self, chunk: AudioChunk) -> None:
        samples = chunk.samples
        if chunk.fmt.channels > 1:
            samples = to_mono(samples, chunk.fmt.channels).astype(np.int16)
        if chunk.fmt.sample_rate != self._fmt.sample_rate:
            samples = resample(samples, chunk.fmt.sample_rate, self._fmt.sample_rate)
        self._parts.append(np.ascontiguousarray(samples, dtype=np.int16))
        self.chunks.append(chunk)

    async def drain(self) -> None:
        """Ничего не ждём — данные уже в памяти (счётчик для тестов)."""
        self.drains += 1

    async def _close(self) -> None:
        if self._path is not None:
            self.save_wav(self._path)

    def save_wav(self, path: str | Path | None = None) -> Path:
        """Сохранить накопленное в WAV.

        Args:
            path: куда писать; по умолчанию — ``path`` из конструктора.

        Returns:
            Путь записанного файла.

        Raises:
            ValueError: путь не задан ни здесь, ни в конструкторе.
        """
        target = Path(path) if path is not None else self._path
        if target is None:
            raise ValueError("не задан путь для сохранения WAV")
        write_wav(target, self.samples, self._fmt.sample_rate)
        return target
