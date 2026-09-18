"""Аудио-утилиты модуля tts: WAV, ресемплинг, нарезка на чанки.

Внутри движка TTS отдаёт PCM 24 kHz mono int16 (CLAUDE.md, технический
стандарт; `pcm_base64` в контракте `tts.chunk.json`). Здесь собраны мелкие
чистые функции, которыми пользуются провайдеры, `VoiceStore` и smoke-скрипт:

* чтение/запись WAV стандартным модулем :mod:`wave` (без лишних зависимостей);
* сведение в моно и ресемплинг (scipy, если установлен, иначе numpy-интерполяция);
* нарезка длинного массива на чанки фиксированной длительности (`Rechunker`),
  чтобы провайдеры, синтезирующие фразу целиком, отдавали поток 20-100 мс.
"""

from __future__ import annotations

import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Final, TypeAlias, cast

import numpy as np
import numpy.typing as npt

__all__ = [
    "INT16_MAX",
    "Pcm16",
    "Rechunker",
    "chunk_samples",
    "float_to_int16",
    "int16_to_float",
    "iter_chunks",
    "read_wav",
    "resample_linear",
    "resample_pcm",
    "to_mono",
    "write_wav",
]

#: PCM 16 бит со знаком — единственный формат аудио на границах модуля.
Pcm16: TypeAlias = npt.NDArray[np.int16]

INT16_MAX: Final[int] = 32767
SAMPLE_WIDTH_BYTES: Final[int] = 2


def float_to_int16(samples: npt.ArrayLike) -> Pcm16:
    """Перевести float-сэмплы из диапазона [-1, 1] в PCM int16 (с клиппингом)."""
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    clipped = np.clip(array, -1.0, 1.0)
    return np.round(clipped * INT16_MAX).astype(np.int16)


def int16_to_float(samples: npt.ArrayLike) -> npt.NDArray[np.float32]:
    """Перевести PCM int16 в float32 [-1, 1]."""
    array = np.asarray(samples, dtype=np.float32).reshape(-1)
    return (array / INT16_MAX).astype(np.float32)


def to_mono(samples: npt.ArrayLike, channels: int) -> Pcm16:
    """Свести чересстрочный (interleaved) int16-буфер в моно усреднением каналов."""
    if channels < 1:
        raise ValueError(f"каналов должно быть >= 1, получено {channels}")
    array = np.asarray(samples).reshape(-1)
    if channels == 1:
        return cast(Pcm16, array.astype(np.int16))
    usable = (array.size // channels) * channels
    frames = array[:usable].astype(np.float64).reshape(-1, channels)
    return cast(Pcm16, np.round(frames.mean(axis=1)).astype(np.int16))


def resample_linear(samples: npt.ArrayLike, src_rate: int, dst_rate: int) -> Pcm16:
    """Линейная интерполяция int16-сигнала ``src_rate`` -> ``dst_rate`` (только numpy)."""
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"частоты должны быть > 0, получено {src_rate} -> {dst_rate}")
    array = np.asarray(samples).reshape(-1)
    if src_rate == dst_rate or array.size == 0:
        return array.astype(np.int16)

    n_dst = round(array.size * dst_rate / src_rate)
    if n_dst <= 0:
        return np.zeros(0, dtype=np.int16)
    src_index = np.arange(n_dst, dtype=np.float64) * (src_rate / dst_rate)
    resampled = np.interp(src_index, np.arange(array.size, dtype=np.float64), array.astype(float))
    return cast(Pcm16, np.clip(np.round(resampled), -INT16_MAX - 1, INT16_MAX).astype(np.int16))


def resample_pcm(samples: npt.ArrayLike, src_rate: int, dst_rate: int) -> Pcm16:
    """Ресемплинг int16-сигнала: scipy (polyphase), если доступен, иначе numpy.

    Оба пути дают одинаковую длину результата ``round(n * dst_rate / src_rate)``.
    """
    array = np.asarray(samples).reshape(-1)
    if src_rate == dst_rate or array.size == 0:
        return array.astype(np.int16)
    try:
        from math import gcd

        from scipy.signal import resample_poly
    except ImportError:  # scipy опционален — тогда честная линейная интерполяция
        return resample_linear(array, src_rate, dst_rate)

    divisor = gcd(int(src_rate), int(dst_rate))
    resampled = resample_poly(array.astype(np.float64), dst_rate // divisor, src_rate // divisor)
    expected = round(array.size * dst_rate / src_rate)
    if resampled.size > expected:
        resampled = resampled[:expected]
    elif resampled.size < expected:
        resampled = np.pad(resampled, (0, expected - resampled.size))
    return cast(Pcm16, np.clip(np.round(resampled), -INT16_MAX - 1, INT16_MAX).astype(np.int16))


def chunk_samples(sample_rate: int, chunk_ms: int) -> int:
    """Сколько сэмплов в чанке длительностью ``chunk_ms`` мс."""
    if chunk_ms <= 0:
        raise ValueError(f"длительность чанка должна быть > 0, получено {chunk_ms}")
    return max(1, round(sample_rate * chunk_ms / 1000))


def iter_chunks(samples: npt.ArrayLike, sample_rate: int, chunk_ms: int) -> Iterator[Pcm16]:
    """Нарезать int16-массив на чанки по ``chunk_ms`` мс (последний может быть короче)."""
    array = np.asarray(samples).reshape(-1).astype(np.int16)
    size = chunk_samples(sample_rate, chunk_ms)
    for start in range(0, array.size, size):
        yield array[start : start + size]


class Rechunker:
    """Буфер, превращающий произвольные куски PCM в чанки фиксированной длины.

    Нужен провайдерам, чей генератор отдаёт куски произвольного размера
    (XTTS `inference_stream` — сотни миллисекунд), а интерфейс требует 20-100 мс.

    Пример::

        rechunker = Rechunker(chunk_samples(24_000, 40))
        for piece in model_stream:
            for chunk in rechunker.push(piece):
                yield chunk
        for chunk in rechunker.flush():
            yield chunk
    """

    __slots__ = ("_buffer", "_size")

    def __init__(self, size: int) -> None:
        if size <= 0:
            raise ValueError(f"размер чанка должен быть > 0, получено {size}")
        self._size = size
        self._buffer: Pcm16 = np.zeros(0, dtype=np.int16)

    @property
    def pending(self) -> int:
        """Сколько сэмплов лежит в буфере и ещё не отдано."""
        return int(self._buffer.size)

    def push(self, samples: npt.ArrayLike) -> list[Pcm16]:
        """Добавить кусок и забрать готовые полные чанки."""
        array = np.asarray(samples).reshape(-1).astype(np.int16)
        if array.size:
            self._buffer = np.concatenate((self._buffer, array))
        ready: list[Pcm16] = []
        while self._buffer.size >= self._size:
            ready.append(self._buffer[: self._size])
            self._buffer = self._buffer[self._size :]
        return ready

    def flush(self) -> list[Pcm16]:
        """Забрать остаток (короткий последний чанк) и очистить буфер."""
        if not self._buffer.size:
            return []
        tail = self._buffer
        self._buffer = np.zeros(0, dtype=np.int16)
        return [tail]


def read_wav(path: Path) -> tuple[Pcm16, int]:
    """Прочитать PCM-WAV в моно int16.

    Returns:
        Кортеж ``(сэмплы int16 mono, частота дискретизации)``.

    Raises:
        wave.Error: файл не WAV или сжат (float/ADPCM — не поддерживаются).
        ValueError: неподдерживаемая разрядность сэмплов.
    """
    with wave.open(str(path), "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        sample_rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())

    if width == SAMPLE_WIDTH_BYTES:
        array: npt.NDArray[np.int16] = np.frombuffer(raw, dtype="<i2")
    elif width == 1:  # 8 бит — беззнаковые
        array = ((np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128) * 256).astype(np.int16)
    elif width == 3:  # 24 бита — три байта на сэмпл
        packed = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        array = packed[:, 2].astype(np.int8).astype(np.int16) * 256 + packed[:, 1].astype(np.int16)
    elif width == 4:
        array = (np.frombuffer(raw, dtype="<i4") >> 16).astype(np.int16)
    else:
        raise ValueError(f"неподдерживаемая разрядность WAV: {width * 8} бит")

    return to_mono(array, channels), sample_rate


def write_wav(path: Path, samples: npt.ArrayLike, sample_rate: int) -> None:
    """Записать int16-массив в WAV (mono, 16 бит)."""
    array = np.asarray(samples).reshape(-1).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(SAMPLE_WIDTH_BYTES)
        fh.setframerate(sample_rate)
        fh.writeframes(array.tobytes())
