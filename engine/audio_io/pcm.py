"""Утилиты работы с PCM: ресемплинг, даунмикс, конвертация типов, WAV.

Чистый ``numpy`` + стандартный модуль ``wave`` — никаких обязательных
системных зависимостей, поэтому модуль импортируется и работает на любой
платформе (в том числе в CI без звуковой карты).

Формат аудио внутри движка — PCM 16 kHz mono int16 (CLAUDE.md, технический
стандарт). TTS отдаёт 24 kHz mono int16, ресемплинг под устройство делает
``audio_io`` (ARCHITECTURE.md 4.1).

Ресемплинг:

* ``method="auto"`` (по умолчанию) — ``scipy.signal.resample_poly``, если
  ``scipy`` установлен, иначе линейная интерполяция на ``numpy``;
* ``method="poly"`` — только polyphase (ошибка, если нет ``scipy``);
* ``method="linear"`` — только линейная интерполяция.

Пример::

    from engine.audio_io.pcm import read_wav, resample

    samples, rate = read_wav("in.wav")
    pcm16k = resample(samples, rate, 16_000)
"""

from __future__ import annotations

import math
import wave
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt

__all__ = [
    "INT16_MAX",
    "INT16_SCALE",
    "Float32Array",
    "Int16Array",
    "ResampleMethod",
    "float32_to_int16",
    "has_scipy",
    "int16_to_float32",
    "read_wav",
    "resample",
    "resample_float",
    "target_length",
    "to_mono",
    "write_wav",
]

Int16Array = npt.NDArray[np.int16]
Float32Array = npt.NDArray[np.float32]

#: Методы ресемплинга (см. модульный docstring).
ResampleMethod = Literal["auto", "poly", "linear"]

INT16_MAX: Final[int] = 32767
INT16_SCALE: Final[float] = 32768.0

SAMPLE_WIDTH_BYTES: Final[int] = 2


def has_scipy() -> bool:
    """Доступен ли ``scipy`` (polyphase-ресемплинг более высокого качества)."""
    try:
        import scipy.signal  # noqa: F401  — проверка наличия
    except ImportError:
        return False
    return True


def to_mono(samples: npt.NDArray[Any], channels: int) -> npt.NDArray[Any]:
    """Свести interleaved-аудио к моно усреднением каналов.

    Args:
        samples: плоский массив сэмплов (interleaved: ``L R L R ...``)
            или уже двумерный ``(frames, channels)``.
        channels: число каналов в исходном массиве.

    Returns:
        Одномерный массив того же dtype (int16 считается через float64,
        чтобы не переполниться при суммировании).

    Raises:
        ValueError: ``channels < 1`` или длина не кратна числу каналов.
    """
    if channels < 1:
        raise ValueError(f"channels должно быть >= 1, получено {channels}")

    data = np.asarray(samples)
    if data.ndim == 2:
        if data.shape[1] != channels:
            raise ValueError(f"ожидалось {channels} каналов, в массиве {data.shape[1]}")
        frames = data
    else:
        if channels == 1:
            return np.ascontiguousarray(data)
        if data.size % channels:
            raise ValueError(f"длина {data.size} не кратна числу каналов {channels}")
        frames = data.reshape(-1, channels)

    if frames.shape[1] == 1:
        return np.ascontiguousarray(frames[:, 0])

    if np.issubdtype(frames.dtype, np.integer):
        mono = frames.astype(np.float64).mean(axis=1)
        info = np.iinfo(frames.dtype)
        clipped = np.clip(np.round(mono), info.min, info.max)
        return np.ascontiguousarray(clipped.astype(frames.dtype))

    return np.ascontiguousarray(frames.mean(axis=1, dtype=frames.dtype))


def int16_to_float32(pcm: Int16Array) -> Float32Array:
    """int16 PCM -> float32 в диапазоне [-1.0, 1.0)."""
    data = np.asarray(pcm, dtype=np.int16)
    return (data.astype(np.float32) / np.float32(INT16_SCALE)).astype(np.float32)


def float32_to_int16(samples: npt.NDArray[Any]) -> Int16Array:
    """float32/float64 в [-1.0, 1.0] -> int16 PCM с клиппингом."""
    data = np.asarray(samples, dtype=np.float32)
    scaled = np.clip(data, -1.0, 1.0) * np.float32(INT16_MAX)
    return np.ascontiguousarray(np.round(scaled).astype(np.int16))


def target_length(n_frames: int, src_rate: int, dst_rate: int) -> int:
    """Сколько фреймов должно получиться после ресемплинга ``src -> dst``."""
    if n_frames <= 0:
        return 0
    return round(n_frames * dst_rate / src_rate)


def _fit_length(data: Float32Array, n: int) -> Float32Array:
    """Подрезать или дополнить нулями до ровно ``n`` отсчётов."""
    if data.size == n:
        return data
    if data.size > n:
        return np.ascontiguousarray(data[:n])
    pad = np.zeros(n - data.size, dtype=np.float32)
    return np.concatenate([data, pad])


def _resample_poly(data: Float32Array, src_rate: int, dst_rate: int) -> Float32Array | None:
    """Polyphase-ресемплинг через scipy; ``None``, если scipy недоступен."""
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return None
    gcd = math.gcd(src_rate, dst_rate)
    out = resample_poly(data.astype(np.float64), dst_rate // gcd, src_rate // gcd)
    return np.asarray(out, dtype=np.float32)


def _resample_linear(data: Float32Array, n_out: int) -> Float32Array:
    """Линейная интерполяция до ``n_out`` отсчётов (fallback без scipy)."""
    if data.size == 1:
        return np.full(n_out, data[0], dtype=np.float32)
    src_index = np.arange(data.size, dtype=np.float64)
    step = (data.size - 1) / max(n_out - 1, 1) if n_out > 1 else 0.0
    dst_index = np.arange(n_out, dtype=np.float64) * step
    return np.interp(dst_index, src_index, data.astype(np.float64)).astype(np.float32)


def resample_float(
    samples: npt.NDArray[Any],
    src_rate: int,
    dst_rate: int,
    *,
    method: ResampleMethod = "auto",
) -> Float32Array:
    """Ресемплинг моно float32-сигнала.

    Args:
        samples: одномерный массив float32/float64.
        src_rate: исходная частота дискретизации, Гц.
        dst_rate: целевая частота дискретизации, Гц.
        method: см. модульный docstring.

    Returns:
        float32-массив длиной ``round(len * dst/src)``.

    Raises:
        ValueError: неположительная частота или многомерный вход.
        RuntimeError: ``method="poly"``, но ``scipy`` не установлен.
    """
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError(f"ожидался моно-сигнал, получен массив формы {data.shape}")
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError(f"частоты должны быть > 0: src={src_rate}, dst={dst_rate}")
    if data.size == 0:
        return np.zeros(0, dtype=np.float32)
    if src_rate == dst_rate:
        return np.ascontiguousarray(data)

    n_out = target_length(data.size, src_rate, dst_rate)
    if method in ("auto", "poly"):
        poly = _resample_poly(data, src_rate, dst_rate)
        if poly is not None:
            return _fit_length(poly, n_out)
        if method == "poly":
            raise RuntimeError("method='poly' требует scipy; поставьте scipy или method='linear'")

    return _fit_length(_resample_linear(data, n_out), n_out)


def resample(
    pcm_int16: npt.NDArray[Any],
    src_rate: int,
    dst_rate: int,
    *,
    method: ResampleMethod = "auto",
) -> Int16Array:
    """Ресемплинг моно int16 PCM (основная точка входа модуля).

    Args:
        pcm_int16: одномерный int16-массив.
        src_rate: исходная частота, Гц.
        dst_rate: целевая частота, Гц.
        method: см. модульный docstring.

    Returns:
        int16-массив длиной ``round(len * dst/src)``.
    """
    data = np.asarray(pcm_int16, dtype=np.int16)
    if src_rate == dst_rate:
        return np.ascontiguousarray(data)
    resampled = resample_float(int16_to_float32(data), src_rate, dst_rate, method=method)
    return float32_to_int16(resampled)


def read_wav(path: str | Path, *, mono: bool = True) -> tuple[Int16Array, int]:
    """Прочитать WAV (PCM 16 бит) в int16-массив.

    Args:
        path: путь к файлу.
        mono: свести многоканальный файл к моно (по умолчанию да —
            внутри движка всё моно).

    Returns:
        Кортеж ``(samples, sample_rate)``. Если ``mono=False``, массив
        остаётся interleaved.

    Raises:
        ValueError: разрядность отличается от 16 бит.
    """
    with wave.open(str(path), "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        raw = fh.readframes(fh.getnframes())

    if width != SAMPLE_WIDTH_BYTES:
        raise ValueError(f"{path}: ожидался PCM 16 бит, в файле {width * 8} бит")

    samples: Int16Array = np.frombuffer(raw, dtype="<i2").astype(np.int16)
    if mono and channels > 1:
        samples = to_mono(samples, channels).astype(np.int16)
    return np.ascontiguousarray(samples), rate


def write_wav(
    path: str | Path,
    samples: npt.NDArray[Any],
    rate: int,
    *,
    channels: int = 1,
) -> None:
    """Записать int16-массив в WAV (PCM 16 бит).

    Args:
        path: путь к файлу; недостающие каталоги создаются.
        samples: int16-массив (interleaved, если ``channels > 1``).
        rate: частота дискретизации, Гц.
        channels: число каналов.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.int16)
    with wave.open(str(target), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(SAMPLE_WIDTH_BYTES)
        fh.setframerate(rate)
        fh.writeframes(data.astype("<i2").tobytes())
