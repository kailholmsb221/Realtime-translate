"""Абстракции audio_io: формат, чанк, источник/приёмник, описание устройства.

Ключевые сущности (ARCHITECTURE.md 4.1):

* :class:`AudioFormat` — формат PCM; внутри движка всегда 16 kHz mono int16;
* :class:`AudioChunk` — чанк 30 мс (= 480 фреймов при 16 kHz) с таймкодом;
* :class:`AudioSource` / :class:`AudioSink` — протоколы захвата и вывода;
* :class:`BaseAudioSource` / :class:`BaseAudioSink` — общая реализация
  жизненного цикла (``open``/``close``, async context manager);
* :class:`ChunkAssembler` — режет произвольные куски потока на ровные чанки;
* :class:`AudioDeviceInfo` — описание устройства (вход/выход/loopback).

Модуль чистый: ``numpy`` и стандартная библиотека, никаких аудио-бэкендов,
поэтому импортируется на любой платформе.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any, Final, Protocol, Self, runtime_checkable

import numpy as np
import numpy.typing as npt

from engine.audio_io.pcm import Int16Array

__all__ = [
    "CHUNK_FRAMES",
    "CHUNK_MS",
    "ENGINE_FORMAT",
    "SAMPLE_RATE_ENGINE",
    "SAMPLE_RATE_TTS",
    "TTS_FORMAT",
    "VIRTUAL_CABLE_MARKER",
    "AudioBackendUnavailable",
    "AudioChunk",
    "AudioDeviceInfo",
    "AudioFormat",
    "AudioIOError",
    "AudioSink",
    "AudioSource",
    "BaseAudioSink",
    "BaseAudioSource",
    "ChunkAssembler",
    "DeviceKind",
    "DeviceNotFound",
    "StreamStats",
]

#: Частота дискретизации внутри движка (CLAUDE.md, технический стандарт).
SAMPLE_RATE_ENGINE: Final[int] = 16_000
#: Частота, в которой TTS отдаёт PCM (ARCHITECTURE.md 4.4).
SAMPLE_RATE_TTS: Final[int] = 24_000
#: Длительность чанка захвата, мс.
CHUNK_MS: Final[int] = 30
#: Подстрока в имени устройства VB-Audio Virtual Cable.
VIRTUAL_CABLE_MARKER: Final[str] = "CABLE"


class AudioIOError(RuntimeError):
    """Базовая ошибка подсистемы audio_io."""


class AudioBackendUnavailable(AudioIOError):
    """Нужный аудио-бэкенд недоступен (не та ОС или не установлен пакет)."""


class DeviceNotFound(AudioIOError):
    """Устройство с заданными признаками не найдено в системе."""


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Формат PCM-потока. По умолчанию — формат движка: 16 kHz mono int16."""

    sample_rate: int = SAMPLE_RATE_ENGINE
    channels: int = 1
    dtype: str = "int16"

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate должен быть > 0, получено {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"channels должен быть > 0, получено {self.channels}")
        if self.dtype != "int16":
            raise ValueError(f"поддерживается только int16, получено {self.dtype!r}")

    @property
    def bytes_per_frame(self) -> int:
        """Размер одного фрейма в байтах (все каналы)."""
        return 2 * self.channels

    def frames_for_ms(self, ms: float) -> int:
        """Сколько фреймов укладывается в ``ms`` миллисекунд."""
        return round(self.sample_rate * ms / 1000.0)

    def ms_for_frames(self, frames: int) -> float:
        """Длительность ``frames`` фреймов в миллисекундах."""
        return frames * 1000.0 / self.sample_rate


#: Формат движка: 16 kHz mono int16.
ENGINE_FORMAT: Final[AudioFormat] = AudioFormat()
#: Формат выхода TTS: 24 kHz mono int16.
TTS_FORMAT: Final[AudioFormat] = AudioFormat(sample_rate=SAMPLE_RATE_TTS)
#: Число фреймов в чанке 30 мс при 16 kHz.
CHUNK_FRAMES: Final[int] = ENGINE_FORMAT.frames_for_ms(CHUNK_MS)


@dataclass(frozen=True, slots=True)
class AudioChunk:
    """Кусок PCM-потока с таймкодом от начала потока.

    Конструктор принимает как ``bytes``, так и int16 ``numpy``-массив —
    массив сразу приводится к байтам, поэтому поле ``pcm`` всегда ``bytes``
    (сериализуемо и дёшево передаётся между задачами). Обратно — ``samples``.

    Attributes:
        pcm: PCM-данные (little-endian int16, interleaved при channels > 1).
        ts_ms: таймкод начала чанка от начала потока, мс.
        n_frames: число фреймов в чанке.
        fmt: формат данных (по умолчанию формат движка).
    """

    pcm: bytes
    ts_ms: int = 0
    n_frames: int = -1
    fmt: AudioFormat = ENGINE_FORMAT

    def __post_init__(self) -> None:
        pcm = self.pcm
        if isinstance(pcm, np.ndarray):
            pcm = np.ascontiguousarray(pcm, dtype=np.int16).astype("<i2").tobytes()
            object.__setattr__(self, "pcm", pcm)
        elif isinstance(pcm, memoryview | bytearray):
            object.__setattr__(self, "pcm", bytes(pcm))
        elif not isinstance(pcm, bytes):
            raise TypeError(f"pcm: ожидались bytes или numpy int16, получено {type(pcm).__name__}")

        expected = len(self.pcm) // self.fmt.bytes_per_frame
        if self.n_frames < 0:
            object.__setattr__(self, "n_frames", expected)
        elif self.n_frames != expected:
            raise ValueError(f"n_frames={self.n_frames}, а в pcm {expected} фреймов")

    @classmethod
    def from_array(
        cls,
        samples: npt.NDArray[Any],
        ts_ms: int = 0,
        fmt: AudioFormat = ENGINE_FORMAT,
    ) -> AudioChunk:
        """Собрать чанк из int16-массива (interleaved при channels > 1)."""
        data = np.ascontiguousarray(samples, dtype=np.int16)
        return cls(pcm=data.astype("<i2").tobytes(), ts_ms=ts_ms, fmt=fmt)

    @property
    def samples(self) -> Int16Array:
        """PCM как int16-массив (interleaved при channels > 1)."""
        return np.frombuffer(self.pcm, dtype="<i2").astype(np.int16)

    @property
    def duration_ms(self) -> float:
        """Длительность чанка в миллисекундах."""
        return self.fmt.ms_for_frames(self.n_frames)

    @property
    def end_ts_ms(self) -> int:
        """Таймкод конца чанка, мс."""
        return self.ts_ms + round(self.duration_ms)

    def __len__(self) -> int:
        return self.n_frames


class DeviceKind(StrEnum):
    """Роль устройства."""

    INPUT = "input"
    OUTPUT = "output"
    LOOPBACK = "loopback"


@dataclass(frozen=True, slots=True)
class AudioDeviceInfo:
    """Описание аудиоустройства, независимое от бэкенда.

    Attributes:
        id: индекс устройства внутри своего бэкенда.
        name: имя устройства как его показывает ОС.
        kind: вход / выход / WASAPI loopback.
        default_sample_rate: нативная частота устройства, Гц.
        channels: число каналов.
        backend: какой бэкенд отдал устройство (``pyaudiowpatch``,
            ``sounddevice``, ``fake``).
        is_default: устройство по умолчанию для своего направления.
        hostapi: имя host API (WASAPI, MME, ...), если бэкенд его знает.
    """

    id: int
    name: str
    kind: DeviceKind
    default_sample_rate: float
    channels: int
    backend: str = "unknown"
    is_default: bool = False
    hostapi: str = ""

    @property
    def is_virtual_cable(self) -> bool:
        """Похоже ли устройство на VB-Audio Virtual Cable (подстрока ``CABLE``)."""
        return VIRTUAL_CABLE_MARKER in self.name.upper()

    @property
    def is_loopback(self) -> bool:
        """Это WASAPI loopback (захват того, что играет в колонках/наушниках)."""
        return self.kind is DeviceKind.LOOPBACK

    def describe(self) -> str:
        """Однострочное описание для CLI-вывода."""
        tags = []
        if self.is_default:
            tags.append("default")
        if self.is_loopback:
            tags.append("loopback")
        if self.is_virtual_cable:
            tags.append("CABLE")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        return (
            f"#{self.id:<3} {self.kind.value:<8} {self.name} "
            f"({self.channels}ch, {self.default_sample_rate:.0f} Hz, {self.backend})" + suffix
        )


class ChunkAssembler:
    """Режет произвольные куски моно int16-потока на ровные чанки.

    Используется реализациями захвата: callback драйвера отдаёт блоки
    произвольной длины, а наружу поток обязан идти чанками ровно по
    ``chunk_ms`` (30 мс = 480 фреймов при 16 kHz).

    Пример::

        asm = ChunkAssembler()
        for chunk in asm.push(np.zeros(1000, dtype=np.int16)):
            ...  # чанки по 480 фреймов
        rest = asm.flush()  # хвост, дополненный нулями
    """

    __slots__ = ("_buffer", "_chunk_frames", "_emitted_frames", "_fmt", "_start_ts_ms")

    def __init__(
        self,
        fmt: AudioFormat = ENGINE_FORMAT,
        chunk_ms: float = CHUNK_MS,
        *,
        start_ts_ms: int = 0,
    ) -> None:
        if fmt.channels != 1:
            raise ValueError("ChunkAssembler работает только с моно-потоком")
        self._fmt = fmt
        self._chunk_frames = fmt.frames_for_ms(chunk_ms)
        if self._chunk_frames <= 0:
            raise ValueError(f"chunk_ms={chunk_ms} даёт нулевой чанк")
        self._buffer: Int16Array = np.zeros(0, dtype=np.int16)
        self._emitted_frames = 0
        self._start_ts_ms = start_ts_ms

    @property
    def chunk_frames(self) -> int:
        """Размер чанка во фреймах."""
        return self._chunk_frames

    @property
    def pending_frames(self) -> int:
        """Сколько фреймов лежит в буфере и ещё не отдано."""
        return int(self._buffer.size)

    def _next_ts_ms(self) -> int:
        return self._start_ts_ms + round(self._fmt.ms_for_frames(self._emitted_frames))

    def push(self, samples: npt.NDArray[Any]) -> list[AudioChunk]:
        """Добавить кусок потока; вернуть готовые целые чанки."""
        data = np.asarray(samples, dtype=np.int16).reshape(-1)
        if data.size:
            self._buffer = np.concatenate([self._buffer, data]) if self._buffer.size else data
        out: list[AudioChunk] = []
        while self._buffer.size >= self._chunk_frames:
            piece = self._buffer[: self._chunk_frames]
            self._buffer = self._buffer[self._chunk_frames :]
            out.append(AudioChunk.from_array(piece, self._next_ts_ms(), self._fmt))
            self._emitted_frames += self._chunk_frames
        return out

    def flush(self, *, pad: bool = True) -> list[AudioChunk]:
        """Отдать остаток буфера.

        Args:
            pad: дополнить хвост нулями до полного чанка (иначе — отбросить).
        """
        if not self._buffer.size:
            return []
        if not pad:
            self._buffer = np.zeros(0, dtype=np.int16)
            return []
        tail = np.zeros(self._chunk_frames, dtype=np.int16)
        tail[: self._buffer.size] = self._buffer
        self._buffer = np.zeros(0, dtype=np.int16)
        chunk = AudioChunk.from_array(tail, self._next_ts_ms(), self._fmt)
        self._emitted_frames += self._chunk_frames
        return [chunk]


@runtime_checkable
class AudioSource(Protocol):
    """Источник аудио: async-генератор чанков фиксированной длины."""

    @property
    def format(self) -> AudioFormat:
        """Формат отдаваемых чанков."""
        ...

    async def open(self) -> None:
        """Открыть устройство/файл. Повторный вызов безопасен."""
        ...

    async def close(self) -> None:
        """Закрыть источник и освободить ресурсы. Повторный вызов безопасен."""
        ...

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        """Итератор чанков; открывает источник при первом обращении."""
        ...

    async def __aenter__(self) -> AudioSource:
        """Async context manager: открывает источник."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context manager: закрывает источник."""
        ...


@runtime_checkable
class AudioSink(Protocol):
    """Приёмник аудио: воспроизведение или запись в файл."""

    async def open(self) -> None:
        """Открыть устройство/файл. Повторный вызов безопасен."""
        ...

    async def write(self, chunk: AudioChunk) -> None:
        """Отправить чанк на воспроизведение (ресемплинг — забота приёмника)."""
        ...

    async def drain(self) -> None:
        """Дождаться, пока всё записанное действительно проиграно."""
        ...

    async def close(self) -> None:
        """Закрыть приёмник. Повторный вызов безопасен."""
        ...

    async def __aenter__(self) -> AudioSink:
        """Async context manager: открывает приёмник."""
        ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Async context manager: дренит и закрывает приёмник."""
        ...


class BaseAudioSource(ABC):
    """Общая реализация жизненного цикла источника.

    Наследник реализует ``_open``/``_close``/``_iter_chunks``; ленивое
    открытие, идемпотентные ``open``/``close`` и context manager — здесь.
    """

    def __init__(self, fmt: AudioFormat = ENGINE_FORMAT, chunk_ms: float = CHUNK_MS) -> None:
        self._fmt = fmt
        self._chunk_ms = chunk_ms
        self._opened = False
        self._closed = False

    @property
    def format(self) -> AudioFormat:
        """Формат отдаваемых чанков."""
        return self._fmt

    @property
    def chunk_ms(self) -> float:
        """Длительность одного чанка, мс."""
        return self._chunk_ms

    @property
    def chunk_frames(self) -> int:
        """Длительность одного чанка во фреймах."""
        return self._fmt.frames_for_ms(self._chunk_ms)

    @property
    def is_open(self) -> bool:
        """Открыт ли источник."""
        return self._opened and not self._closed

    async def open(self) -> None:
        """Открыть источник (идемпотентно)."""
        if self._opened:
            return
        await self._open()
        self._opened = True
        self._closed = False

    async def close(self) -> None:
        """Закрыть источник (идемпотентно)."""
        if not self._opened or self._closed:
            self._closed = True
            return
        await self._close()
        self._closed = True

    def __aiter__(self) -> AsyncIterator[AudioChunk]:
        return self._open_then_iter()

    async def _open_then_iter(self) -> AsyncIterator[AudioChunk]:
        await self.open()
        async for chunk in self._iter_chunks():
            yield chunk

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    @abstractmethod
    async def _open(self) -> None:
        """Открыть устройство/файл."""

    @abstractmethod
    async def _close(self) -> None:
        """Освободить ресурсы."""

    @abstractmethod
    def _iter_chunks(self) -> AsyncIterator[AudioChunk]:
        """Асинхронный генератор чанков."""


class BaseAudioSink(ABC):
    """Общая реализация жизненного цикла приёмника."""

    def __init__(self) -> None:
        self._opened = False
        self._closed = False

    @property
    def is_open(self) -> bool:
        """Открыт ли приёмник."""
        return self._opened and not self._closed

    async def open(self) -> None:
        """Открыть приёмник (идемпотентно)."""
        if self._opened:
            return
        await self._open()
        self._opened = True
        self._closed = False

    async def write(self, chunk: AudioChunk) -> None:
        """Отправить чанк; при необходимости откроет приёмник."""
        await self.open()
        await self._write(chunk)

    async def write_many(self, chunks: Iterable[AudioChunk]) -> None:
        """Отправить несколько чанков подряд."""
        for chunk in chunks:
            await self.write(chunk)

    async def close(self) -> None:
        """Закрыть приёмник (идемпотентно)."""
        if not self._opened or self._closed:
            self._closed = True
            return
        await self._close()
        self._closed = True

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            await self.drain()
        await self.close()

    @abstractmethod
    async def _open(self) -> None:
        """Открыть устройство/файл."""

    @abstractmethod
    async def _write(self, chunk: AudioChunk) -> None:
        """Записать чанк."""

    @abstractmethod
    async def drain(self) -> None:
        """Дождаться фактического проигрывания/записи всего буфера."""

    @abstractmethod
    async def _close(self) -> None:
        """Освободить ресурсы."""


@dataclass(slots=True)
class StreamStats:
    """Счётчики потока — для логов и smoke-тестов."""

    chunks: int = 0
    frames: int = 0
    dropped_chunks: int = 0
    xruns: int = 0
    statuses: list[str] = field(default_factory=list)

    def note_chunk(self, chunk: AudioChunk) -> None:
        """Учесть отданный/принятый чанк."""
        self.chunks += 1
        self.frames += chunk.n_frames
