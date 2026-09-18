"""Windows-реализация захвата и вывода звука (WASAPI + sounddevice).

Что здесь есть:

* :class:`WasapiLoopbackSource` — захват того, что ОС играет в наушники
  (речь собеседника из Zoom) через ``pyaudiowpatch`` (WASAPI loopback);
* :class:`MicSource` — захват физического микрофона через ``sounddevice``;
* :class:`SoundDeviceSink` — вывод в наушники или во вход VB-Audio Virtual
  Cable через ``sounddevice``.

Захват идёт в нативном формате устройства (обычно 48 kHz stereo), затем
даунмикс в моно и ресемплинг в 16 kHz int16 — формат движка. Вывод принимает
чанки 16 kHz (речь) или 24 kHz (выход TTS) и ресемплит под частоту устройства.

Callback-и драйвера работают в своём потоке: они только конвертируют блок и
передают готовый чанк в ``asyncio.Queue`` через ``loop.call_soon_threadsafe``.

``pyaudiowpatch`` и ``sounddevice`` импортируются лениво
(:mod:`engine.audio_io.devices`), поэтому модуль импортируется и на Linux —
там любые попытки открыть устройство дают
:class:`~engine.audio_io.base.AudioBackendUnavailable` с инструкцией.
"""

from __future__ import annotations

import asyncio
import collections
import contextlib
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from engine.audio_io.base import (
    CHUNK_MS,
    ENGINE_FORMAT,
    AudioChunk,
    AudioDeviceInfo,
    AudioFormat,
    AudioIOError,
    BaseAudioSink,
    BaseAudioSource,
    ChunkAssembler,
    DeviceKind,
    StreamStats,
)
from engine.audio_io.config import AudioConfig
from engine.audio_io.devices import (
    find_default_mic,
    find_default_output,
    find_loopback_device,
    find_virtual_cable_output,
    import_pyaudiowpatch,
    import_sounddevice,
)
from engine.audio_io.pcm import (
    Float32Array,
    Int16Array,
    ResampleMethod,
    float32_to_int16,
    int16_to_float32,
    resample,
    resample_float,
    to_mono,
)

__all__ = [
    "MicSource",
    "SoundDeviceSink",
    "WasapiLoopbackSource",
    "make_cable_sink",
    "make_headphones_sink",
    "make_loopback_source",
    "make_mic_source",
]

log = logging.getLogger(__name__)

#: Сколько чанков держим в очереди захвата, прежде чем выбрасывать старые.
DEFAULT_QUEUE_CHUNKS: Final[int] = 200
#: Сколько миллисекунд аудио максимум копим в буфере воспроизведения.
DEFAULT_SINK_BUFFER_MS: Final[int] = 4000
#: Сколько ждать места в буфере воспроизведения, прежде чем отбросить чанк, с.
DEFAULT_SINK_OVERFLOW_WAIT_S: Final[float] = 2.0
#: Как часто проверять, освободилось ли место в буфере, с.
SINK_ROOM_POLL_S: Final[float] = 0.005


def _bytes_to_mono_int16(raw: bytes, channels: int, dtype: npt.DTypeLike) -> Int16Array:
    """Сырой блок драйвера -> моно int16 (без смены частоты)."""
    data = np.frombuffer(raw, dtype=dtype)
    if data.dtype == np.float32:
        mono = to_mono(data, channels)
        return float32_to_int16(mono)
    mono_i = to_mono(data.astype(np.int16), channels)
    return np.ascontiguousarray(mono_i, dtype=np.int16)


class _QueueCaptureSource(BaseAudioSource):
    """База для захвата: очередь чанков, наполняемая из потока драйвера."""

    def __init__(
        self,
        device: AudioDeviceInfo,
        *,
        fmt: AudioFormat = ENGINE_FORMAT,
        chunk_ms: float = CHUNK_MS,
        device_sample_rate: int | None = None,
        device_channels: int | None = None,
        queue_chunks: int = DEFAULT_QUEUE_CHUNKS,
        timeout_s: float = 5.0,
        resample_method: ResampleMethod = "auto",
    ) -> None:
        super().__init__(fmt=fmt, chunk_ms=chunk_ms)
        if fmt.channels != 1:
            raise ValueError("захват отдаёт только моно")
        self.device = device
        self.stats = StreamStats()
        self._device_rate = int(device_sample_rate or device.default_sample_rate or fmt.sample_rate)
        self._device_channels = int(device_channels or device.channels or 1)
        self._queue_chunks = queue_chunks
        self._timeout_s = timeout_s
        self._resample_method: ResampleMethod = resample_method
        self._queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue(maxsize=queue_chunks)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._assembler = ChunkAssembler(fmt, chunk_ms)
        self._error: BaseException | None = None

    @property
    def device_sample_rate(self) -> int:
        """Нативная частота устройства, Гц."""
        return self._device_rate

    @property
    def device_channels(self) -> int:
        """Число каналов, которые отдаёт устройство."""
        return self._device_channels

    @property
    def frames_per_buffer(self) -> int:
        """Размер блока драйвера во фреймах (≈ один чанк на нативной частоте)."""
        return max(round(self._device_rate * self._chunk_ms / 1000.0), 64)

    # --- поток драйвера --------------------------------------------------

    def _submit_block(self, mono_device_rate: Int16Array) -> None:
        """Принять моно int16 блок на частоте устройства (поток драйвера)."""
        if self._device_rate != self._fmt.sample_rate:
            mono_device_rate = resample(
                mono_device_rate,
                self._device_rate,
                self._fmt.sample_rate,
                method=self._resample_method,
            )
        for chunk in self._assembler.push(mono_device_rate):
            self._post(chunk)

    def _post(self, item: AudioChunk | None) -> None:
        """Передать чанк в цикл событий (поток драйвера)."""
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):  # RuntimeError: цикл уже остановлен
            loop.call_soon_threadsafe(self._enqueue, item)

    def _enqueue(self, item: AudioChunk | None) -> None:
        """Положить чанк в очередь, вытеснив самый старый при переполнении."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - гонка, практически не бывает
                pass
            else:
                self.stats.dropped_chunks += 1
                log.warning(
                    "%s: очередь захвата переполнена, чанк отброшен (всего %d)",
                    type(self).__name__,
                    self.stats.dropped_chunks,
                )
        self._queue.put_nowait(item)

    def _note_status(self, status: object) -> None:
        """Зафиксировать xrun/ошибку драйвера (поток драйвера)."""
        if not status:
            return
        self.stats.xruns += 1
        text = str(status)
        if text not in self.stats.statuses:
            self.stats.statuses.append(text)
        log.debug("%s: статус драйвера %s", type(self).__name__, text)

    # --- цикл событий ----------------------------------------------------

    async def _iter_chunks(self) -> AsyncIterator[AudioChunk]:
        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), self._timeout_s)
            except TimeoutError as exc:
                raise AudioIOError(
                    f"{type(self).__name__}: нет данных с устройства "
                    f"{self.device.name!r} дольше {self._timeout_s} с"
                ) from exc
            if item is None:
                if self._error is not None:
                    raise AudioIOError(f"{type(self).__name__}: поток остановлен") from self._error
                return
            self.stats.note_chunk(item)
            yield item


class WasapiLoopbackSource(_QueueCaptureSource):
    """Захват WASAPI loopback: то, что ОС играет в наушники (речь собеседника).

    Требует ``pyaudiowpatch`` и Windows. Запрашивает у драйвера int16, при
    отказе — float32 (конвертация в int16 на нашей стороне).

    Пример::

        from engine.audio_io.windows import make_loopback_source

        async with make_loopback_source() as src:
            async for chunk in src:
                ...  # чанки 30 мс / 16 kHz mono int16
    """

    def __init__(self, device: AudioDeviceInfo | None = None, **kwargs: Any) -> None:
        resolved = device if device is not None else find_loopback_device()
        if resolved.kind is not DeviceKind.LOOPBACK:
            log.warning("устройство %r не помечено как loopback", resolved.name)
        super().__init__(resolved, **kwargs)
        self._pa: Any | None = None
        self._stream: Any | None = None
        self._np_dtype: npt.DTypeLike = np.int16

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: object,
        status: int,
    ) -> tuple[bytes | None, int]:
        """Callback pyaudio (поток драйвера)."""
        pyaudio = import_pyaudiowpatch()
        self._note_status(status)
        if in_data:
            try:
                self._submit_block(
                    _bytes_to_mono_int16(in_data, self._device_channels, self._np_dtype)
                )
            except Exception as exc:  # pragma: no cover - железо
                self._error = exc
                log.exception("loopback: ошибка обработки блока")
                self._post(None)
                return (None, pyaudio.paAbort)
        return (None, pyaudio.paContinue)

    def _open_stream(self) -> None:
        """Открыть WASAPI loopback-поток (блокирующий вызов, идёт в to_thread)."""
        pyaudio = import_pyaudiowpatch()
        self._pa = pyaudio.PyAudio()
        last_error: Exception | None = None
        for flag, dtype in ((pyaudio.paInt16, np.int16), (pyaudio.paFloat32, np.float32)):
            try:
                self._stream = self._pa.open(
                    format=flag,
                    channels=self._device_channels,
                    rate=self._device_rate,
                    input=True,
                    input_device_index=self.device.id,
                    frames_per_buffer=self.frames_per_buffer,
                    stream_callback=self._callback,
                )
            except OSError as exc:
                last_error = exc
                continue
            self._np_dtype = dtype
            break
        if self._stream is None:
            self._pa.terminate()
            self._pa = None
            raise AudioIOError(
                f"не удалось открыть loopback {self.device.name!r} "
                f"({self._device_rate} Hz, {self._device_channels}ch): {last_error}"
            )
        self._stream.start_stream()

    async def _open(self) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._open_stream)
        log.info(
            "loopback открыт: %s (%d Hz, %dch) -> %d Hz mono",
            self.device.name,
            self._device_rate,
            self._device_channels,
            self._fmt.sample_rate,
        )

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            finally:
                self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    async def _close(self) -> None:
        await asyncio.to_thread(self._close_stream)
        for chunk in self._assembler.flush():
            self._enqueue(chunk)
        self._enqueue(None)


class MicSource(_QueueCaptureSource):
    """Захват физического микрофона через ``sounddevice`` (float32 -> int16 16 kHz)."""

    def __init__(self, device: AudioDeviceInfo | None = None, **kwargs: Any) -> None:
        resolved = device if device is not None else find_default_mic()
        super().__init__(resolved, **kwargs)
        self._stream: Any | None = None

    def _callback(
        self,
        indata: npt.NDArray[Any],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Callback sounddevice (поток драйвера)."""
        self._note_status(status)
        try:
            block = np.array(indata, copy=True)
            mono = to_mono(block, block.shape[1] if block.ndim == 2 else 1)
            pcm = float32_to_int16(mono) if mono.dtype != np.int16 else mono.astype(np.int16)
            self._submit_block(pcm)
        except Exception as exc:  # pragma: no cover - железо
            self._error = exc
            log.exception("mic: ошибка обработки блока")
            self._post(None)

    def _open_stream(self) -> None:
        sd = import_sounddevice()
        self._stream = sd.InputStream(
            device=self.device.id,
            channels=self._device_channels,
            samplerate=self._device_rate,
            dtype="float32",
            blocksize=self.frames_per_buffer,
            callback=self._callback,
        )
        self._stream.start()

    async def _open(self) -> None:
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self._open_stream)
        log.info(
            "микрофон открыт: %s (%d Hz, %dch) -> %d Hz mono",
            self.device.name,
            self._device_rate,
            self._device_channels,
            self._fmt.sample_rate,
        )

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    async def _close(self) -> None:
        await asyncio.to_thread(self._close_stream)
        for chunk in self._assembler.flush():
            self._enqueue(chunk)
        self._enqueue(None)


class SoundDeviceSink(BaseAudioSink):
    """Воспроизведение в устройство вывода (наушники или вход VB-Cable).

    Принимает чанки 16 kHz (речь) и 24 kHz (выход TTS) — частота берётся из
    ``chunk.fmt`` — и ресемплит их под нативную частоту устройства, размножая
    моно на нужное число каналов.

    Args:
        device: устройство вывода.
        device_sample_rate: частота устройства (по умолчанию нативная).
        channels: число каналов (по умолчанию 2 или сколько поддерживает устройство).
        buffer_ms: максимум аудио в буфере; при переполнении ``write`` ждёт,
            пока драйвер вычерпает место (обратное давление).
        overflow_wait_s: сколько ждать это место, прежде чем всё-таки отбросить
            чанк (поток не играет — ждать бессмысленно).
    """

    def __init__(
        self,
        device: AudioDeviceInfo,
        *,
        device_sample_rate: int | None = None,
        channels: int | None = None,
        buffer_ms: int = DEFAULT_SINK_BUFFER_MS,
        blocksize: int | None = None,
        resample_method: ResampleMethod = "auto",
        overflow_wait_s: float = DEFAULT_SINK_OVERFLOW_WAIT_S,
    ) -> None:
        super().__init__()
        self.device = device
        self.stats = StreamStats()
        rate = device_sample_rate or device.default_sample_rate or ENGINE_FORMAT.sample_rate
        self._rate = int(rate)
        self._channels = int(channels or min(device.channels or 2, 2) or 1)
        self._buffer_frames_max = int(self._rate * buffer_ms / 1000)
        self._blocksize = int(blocksize or max(int(self._rate * CHUNK_MS / 1000), 64))
        self._resample_method: ResampleMethod = resample_method
        self._overflow_wait_s = max(0.0, overflow_wait_s)
        self._lock = threading.Lock()
        self._buffer: collections.deque[Float32Array] = collections.deque()
        self._buffered_frames = 0
        self._stream: Any | None = None

    @property
    def device_sample_rate(self) -> int:
        """Частота устройства, Гц."""
        return self._rate

    @property
    def device_channels(self) -> int:
        """Число каналов устройства."""
        return self._channels

    @property
    def buffered_frames(self) -> int:
        """Сколько фреймов сейчас ждёт воспроизведения."""
        with self._lock:
            return self._buffered_frames

    def _callback(
        self,
        outdata: npt.NDArray[Any],
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """Callback sounddevice: отдать драйверу накопленные данные."""
        if status:
            self.stats.xruns += 1
        pos = 0
        with self._lock:
            while pos < frames and self._buffer:
                head = self._buffer[0]
                take = min(frames - pos, head.shape[0])
                outdata[pos : pos + take] = head[:take]
                if take == head.shape[0]:
                    self._buffer.popleft()
                else:
                    self._buffer[0] = head[take:]
                self._buffered_frames -= take
                pos += take
        if pos < frames:
            outdata[pos:] = 0.0

    def _open_stream(self) -> None:
        sd = import_sounddevice()
        self._stream = sd.OutputStream(
            device=self.device.id,
            channels=self._channels,
            samplerate=self._rate,
            dtype="float32",
            blocksize=self._blocksize,
            callback=self._callback,
        )
        self._stream.start()

    async def _open(self) -> None:
        await asyncio.to_thread(self._open_stream)
        log.info(
            "вывод открыт: %s (%d Hz, %dch)",
            self.device.name,
            self._rate,
            self._channels,
        )

    def _prepare(self, chunk: AudioChunk) -> Float32Array:
        """Чанк -> float32 блок (frames, channels) на частоте устройства."""
        samples = chunk.samples
        if chunk.fmt.channels > 1:
            samples = to_mono(samples, chunk.fmt.channels).astype(np.int16)
        mono = int16_to_float32(samples)
        if chunk.fmt.sample_rate != self._rate:
            mono = resample_float(
                mono, chunk.fmt.sample_rate, self._rate, method=self._resample_method
            )
        block = np.repeat(mono.reshape(-1, 1), self._channels, axis=1)
        return np.ascontiguousarray(block, dtype=np.float32)

    def _try_append(self, block: Float32Array) -> bool:
        """Положить блок в буфер, если там есть место (потокобезопасно)."""
        with self._lock:
            if self._buffered_frames + block.shape[0] > self._buffer_frames_max:
                return False
            self._buffer.append(block)
            self._buffered_frames += block.shape[0]
            return True

    async def _write(self, chunk: AudioChunk) -> None:
        block = self._prepare(chunk)
        deadline = time.monotonic() + self._overflow_wait_s
        while not self._try_append(block):
            # Буфер полон. Раньше чанк просто отбрасывался, и длинная фраза
            # обрывалась на середине: TTS отдаёт до 15 с речи (max_utterance_ms)
            # быстрее реального времени, а в буфере всего buffer_ms. Теперь
            # ждём, пока драйвер вычерпает место, — фраза доигрывает целиком.
            if (
                self._stream is None
                or block.shape[0] > self._buffer_frames_max
                or time.monotonic() >= deadline
            ):
                self.stats.dropped_chunks += 1
                log.warning(
                    "%s: буфер воспроизведения переполнен, чанк отброшен (всего %d)",
                    self.device.name,
                    self.stats.dropped_chunks,
                )
                return
            await asyncio.sleep(SINK_ROOM_POLL_S)
        self.stats.note_chunk(chunk)

    async def drain(self) -> None:
        """Дождаться, пока буфер опустеет (плюс задержка устройства)."""
        while self.buffered_frames > 0 and self._stream is not None:
            await asyncio.sleep(0.005)
        latency = 0.0
        stream = self._stream
        if stream is not None:
            try:
                latency = float(stream.latency)
            except (AttributeError, TypeError):  # pragma: no cover - зависит от драйвера
                latency = 0.0
        await asyncio.sleep(min(max(latency, 0.0), 0.5))

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None

    async def _close(self) -> None:
        await asyncio.to_thread(self._close_stream)
        with self._lock:
            self._buffer.clear()
            self._buffered_frames = 0


# --- фабрики по конфигу ----------------------------------------------------


def make_loopback_source(
    config: AudioConfig | None = None,
    **kwargs: Any,
) -> WasapiLoopbackSource:
    """Источник loopback по конфигу (речь собеседника из Zoom)."""
    cfg = config or AudioConfig.from_env()
    device = find_loopback_device(name=cfg.loopback_name)
    return WasapiLoopbackSource(
        device,
        fmt=cfg.format,
        chunk_ms=cfg.chunk_ms,
        timeout_s=cfg.device_timeout_s,
        **kwargs,
    )


def make_mic_source(config: AudioConfig | None = None, **kwargs: Any) -> MicSource:
    """Источник «микрофон пользователя» по конфигу."""
    cfg = config or AudioConfig.from_env()
    device = find_default_mic(name=cfg.mic_name)
    return MicSource(
        device,
        fmt=cfg.format,
        chunk_ms=cfg.chunk_ms,
        timeout_s=cfg.device_timeout_s,
        **kwargs,
    )


def make_headphones_sink(config: AudioConfig | None = None, **kwargs: Any) -> SoundDeviceSink:
    """Приёмник «наушники» — туда идёт перевод речи собеседника."""
    cfg = config or AudioConfig.from_env()
    device = find_default_output(name=cfg.headphones_name)
    return SoundDeviceSink(device, **kwargs)


def make_cable_sink(config: AudioConfig | None = None, **kwargs: Any) -> SoundDeviceSink:
    """Приёмник «виртуальный кабель» — туда идёт перевод речи пользователя."""
    cfg = config or AudioConfig.from_env()
    device = find_virtual_cable_output(name=cfg.cable_name)
    return SoundDeviceSink(device, **kwargs)
