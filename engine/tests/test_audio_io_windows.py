"""Тесты Windows-реализации audio_io на фейковых драйверах.

Настоящие ``pyaudiowpatch``/``sounddevice`` здесь не нужны: вместо них
подставляются фейковые модули с такими же API. Это позволяет проверить на
Linux ровно то, что иначе проверяется только руками на целевой машине:
конвертацию 48 kHz stereo -> 16 kHz mono, нарезку по 480 фреймов, передачу
из потока драйвера в asyncio-очередь и буфер воспроизведения.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
import types
from typing import Any, ClassVar

import numpy as np
import pytest

from engine.audio_io import pcm
from engine.audio_io.base import (
    TTS_FORMAT,
    AudioChunk,
    AudioDeviceInfo,
    DeviceKind,
)
from engine.audio_io.windows import MicSource, SoundDeviceSink, WasapiLoopbackSource

DEVICE_RATE = 48_000
DEVICE_CHANNELS = 2
BLOCK_FRAMES = 1440  # 30 мс при 48 kHz
TONE_HZ = 440.0


def loopback_device() -> AudioDeviceInfo:
    """Фейковое loopback-устройство 48 kHz stereo."""
    return AudioDeviceInfo(
        id=7,
        name="Наушники (Realtek(R) Audio) [Loopback]",
        kind=DeviceKind.LOOPBACK,
        default_sample_rate=float(DEVICE_RATE),
        channels=DEVICE_CHANNELS,
        backend="fake",
    )


def output_device(name: str = "CABLE Input (VB-Audio Virtual Cable)") -> AudioDeviceInfo:
    """Фейковое устройство вывода 48 kHz stereo."""
    return AudioDeviceInfo(
        id=3,
        name=name,
        kind=DeviceKind.OUTPUT,
        default_sample_rate=float(DEVICE_RATE),
        channels=DEVICE_CHANNELS,
        backend="fake",
    )


def stereo_tone(frames: int) -> np.ndarray:
    """Interleaved stereo int16: 440 Гц в обоих каналах."""
    t = np.arange(frames, dtype=np.float64) / DEVICE_RATE
    mono = pcm.float32_to_int16(0.5 * np.sin(2.0 * np.pi * TONE_HZ * t))
    return np.repeat(mono, DEVICE_CHANNELS)


class _FeederThread:
    """Поток «драйвера»: периодически отдаёт блоки в callback."""

    def __init__(self, blocks: list[Any], callback: Any) -> None:
        self._blocks = blocks
        self._callback = callback
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        for block in self._blocks:
            if self._stop.is_set():
                return
            self._callback(block)
            time.sleep(0.001)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None


def fake_pyaudio_module(blocks: list[bytes]) -> types.SimpleNamespace:
    """Мини-версия ``pyaudiowpatch``, отдающая заранее заданные блоки."""

    class FakeStream:
        def __init__(self, callback: Any) -> None:
            self.feeder = _FeederThread(
                blocks,
                lambda raw: callback(raw, BLOCK_FRAMES, None, 0),
            )
            self.started = False
            self.closed = False

        def start_stream(self) -> None:
            self.started = True
            self.feeder.start()

        def stop_stream(self) -> None:
            self.feeder.stop()

        def close(self) -> None:
            self.closed = True

    class FakePyAudio:
        instances: ClassVar[list[FakePyAudio]] = []

        def __init__(self) -> None:
            self.terminated = False
            self.stream: FakeStream | None = None
            FakePyAudio.instances.append(self)

        def open(self, **kwargs: Any) -> FakeStream:
            assert kwargs["input"] is True
            assert kwargs["rate"] == DEVICE_RATE
            assert kwargs["channels"] == DEVICE_CHANNELS
            self.stream = FakeStream(kwargs["stream_callback"])
            return self.stream

        def terminate(self) -> None:
            self.terminated = True

    return types.SimpleNamespace(
        PyAudio=FakePyAudio,
        paInt16=8,
        paFloat32=1,
        paContinue=0,
        paAbort=2,
        paWASAPI=13,
    )


def fake_sounddevice_module(blocks: list[np.ndarray]) -> types.SimpleNamespace:
    """Мини-версия ``sounddevice`` с InputStream и OutputStream."""

    class InputStream:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.feeder = _FeederThread(
                blocks,
                lambda block: kwargs["callback"](block, block.shape[0], None, None),
            )
            self.stopped = False

        def start(self) -> None:
            self.feeder.start()

        def stop(self) -> None:
            self.stopped = True
            self.feeder.stop()

        def close(self) -> None:
            pass

    class OutputStream:
        instances: ClassVar[list[OutputStream]] = []

        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.callback = kwargs["callback"]
            self.channels = kwargs["channels"]
            self.latency = 0.01
            self.stopped = False
            OutputStream.instances.append(self)

        def start(self) -> None:
            pass

        def stop(self) -> None:
            self.stopped = True

        def close(self) -> None:
            pass

        def pull(self, frames: int) -> np.ndarray:
            """Сымитировать запрос драйвера: забрать ``frames`` фреймов."""
            out = np.zeros((frames, self.channels), dtype=np.float32)
            self.callback(out, frames, None, None)
            return out

    return types.SimpleNamespace(InputStream=InputStream, OutputStream=OutputStream)


# --- loopback --------------------------------------------------------------


async def test_loopback_source_downmixes_resamples_and_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """48 kHz stereo int16 от WASAPI -> чанки 480 фреймов 16 kHz с ts по 30 мс."""
    n_blocks = 10
    tone = stereo_tone(BLOCK_FRAMES * n_blocks)
    blocks = [
        tone[i * BLOCK_FRAMES * DEVICE_CHANNELS : (i + 1) * BLOCK_FRAMES * DEVICE_CHANNELS]
        .astype("<i2")
        .tobytes()
        for i in range(n_blocks)
    ]
    fake = fake_pyaudio_module(blocks)
    monkeypatch.setattr("engine.audio_io.windows.import_pyaudiowpatch", lambda: fake)

    source = WasapiLoopbackSource(loopback_device())
    assert source.device_sample_rate == DEVICE_RATE
    assert source.frames_per_buffer == BLOCK_FRAMES

    collected: list[AudioChunk] = []
    async with source:
        async for chunk in source:
            collected.append(chunk)
            if len(collected) == n_blocks:
                break

    assert {c.n_frames for c in collected} == {480}
    assert [c.ts_ms for c in collected[:4]] == [0, 30, 60, 90]
    assert all(c.fmt.sample_rate == 16_000 for c in collected)

    joined = np.concatenate([c.samples for c in collected])
    data = pcm.int16_to_float32(joined).astype(np.float64)
    spectrum = np.abs(np.fft.rfft(data * np.hanning(data.size)))
    freqs = np.fft.rfftfreq(data.size, d=1.0 / 16_000)
    assert abs(float(freqs[int(np.argmax(spectrum))]) - TONE_HZ) <= 5.0
    assert source.stats.chunks == len(collected)


async def test_loopback_source_falls_back_to_float32(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если драйвер не отдаёт int16, используется float32 с конвертацией."""
    frames = BLOCK_FRAMES
    floats = (0.25 * np.ones(frames * DEVICE_CHANNELS, dtype=np.float32)).tobytes()
    fake = fake_pyaudio_module([floats] * 3)

    original_open = fake.PyAudio.open

    def open_reject_int16(self: Any, **kwargs: Any) -> Any:
        if kwargs["format"] == fake.paInt16:
            raise OSError("формат не поддерживается")
        return original_open(self, **kwargs)

    fake.PyAudio.open = open_reject_int16
    monkeypatch.setattr("engine.audio_io.windows.import_pyaudiowpatch", lambda: fake)

    source = WasapiLoopbackSource(loopback_device())
    async with source:
        async for chunk in source:
            assert chunk.n_frames == 480
            assert int(np.median(chunk.samples)) == pytest.approx(8192, abs=64)
            break


async def test_loopback_close_flushes_tail_and_terminates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """При закрытии поток завершается, PyAudio освобождается, итерация кончается."""
    tone = stereo_tone(BLOCK_FRAMES)
    fake = fake_pyaudio_module([tone.astype("<i2").tobytes()])
    monkeypatch.setattr("engine.audio_io.windows.import_pyaudiowpatch", lambda: fake)

    source = WasapiLoopbackSource(loopback_device())
    await source.open()
    await asyncio.sleep(0.05)
    await source.close()

    chunks = [chunk async for chunk in source]  # очередь дочитывается до сентинела
    assert len(chunks) == 1  # 1440 фреймов 48 kHz = 480 фреймов 16 kHz = ровно один чанк
    assert chunks[0].n_frames == 480
    assert fake.PyAudio.instances[-1].terminated


async def test_capture_times_out_without_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Молчащее устройство не вешает пайплайн: срабатывает таймаут."""
    fake = fake_pyaudio_module([])
    monkeypatch.setattr("engine.audio_io.windows.import_pyaudiowpatch", lambda: fake)

    source = WasapiLoopbackSource(loopback_device(), timeout_s=0.05)
    with pytest.raises(Exception, match="нет данных"):
        async with source:
            async for _ in source:
                pass


# --- микрофон --------------------------------------------------------------


async def test_mic_source_converts_float32_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """sounddevice отдаёт float32 (frames, channels) -> int16 mono 16 kHz."""
    block = np.full((BLOCK_FRAMES, DEVICE_CHANNELS), 0.5, dtype=np.float32)
    fake = fake_sounddevice_module([block] * 4)
    monkeypatch.setattr("engine.audio_io.windows.import_sounddevice", lambda: fake)

    device = AudioDeviceInfo(
        id=1,
        name="Микрофон (Realtek(R) Audio)",
        kind=DeviceKind.INPUT,
        default_sample_rate=float(DEVICE_RATE),
        channels=DEVICE_CHANNELS,
        backend="fake",
    )
    source = MicSource(device)
    async with source:
        async for chunk in source:
            assert chunk.n_frames == 480
            assert int(np.median(chunk.samples)) == pytest.approx(16384, abs=64)
            break


# --- вывод -----------------------------------------------------------------


async def test_sink_resamples_tts_chunk_and_feeds_driver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Чанк TTS 24 kHz ресемплится в 48 kHz устройства и раздаётся в callback."""
    fake = fake_sounddevice_module([])
    monkeypatch.setattr("engine.audio_io.windows.import_sounddevice", lambda: fake)

    sink = SoundDeviceSink(output_device())
    assert sink.device_sample_rate == DEVICE_RATE
    assert sink.device_channels == DEVICE_CHANNELS

    t = np.arange(2400, dtype=np.float64) / TTS_FORMAT.sample_rate  # 100 мс
    tone = pcm.float32_to_int16(0.5 * np.sin(2.0 * np.pi * TONE_HZ * t))
    await sink.write(AudioChunk.from_array(tone, 0, TTS_FORMAT))

    assert sink.buffered_frames == 4800  # 100 мс на 48 kHz
    stream = fake.OutputStream.instances[-1]

    played = stream.pull(1000)
    assert sink.buffered_frames == 3800
    assert np.max(np.abs(played)) > 0.1
    assert np.allclose(played[:, 0], played[:, 1])  # моно размножено по каналам

    stream.pull(4000)  # добираем остаток + недобор
    assert sink.buffered_frames == 0
    await sink.drain()
    await sink.close()
    assert stream.stopped


async def test_sink_drops_chunks_on_overflow(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если драйвер не забирает звук, переполнение не растит память, а считается."""
    fake = fake_sounddevice_module([])
    monkeypatch.setattr("engine.audio_io.windows.import_sounddevice", lambda: fake)

    # overflow_wait_s=0 — ждать место бессмысленно: никто не вычерпывает буфер.
    sink = SoundDeviceSink(output_device(), buffer_ms=100, overflow_wait_s=0.0)
    chunk = AudioChunk.from_array(np.zeros(1600, dtype=np.int16))  # 100 мс на 16 kHz
    await sink.write(chunk)
    await sink.write(chunk)

    assert sink.stats.dropped_chunks == 1
    assert sink.buffered_frames == 4800
    await sink.close()


async def test_sink_waits_for_room_instead_of_cutting_phrase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Регрессия: длинная фраза TTS не обрезается буфером воспроизведения.

    TTS отдаёт фразу быстрее реального времени, а буфер приёмника ограничен
    (``buffer_ms``). Раньше лишние чанки молча отбрасывались и перевод
    обрывался на середине; теперь ``write`` ждёт, пока драйвер вычерпает место.
    """
    fake = fake_sounddevice_module([])
    monkeypatch.setattr("engine.audio_io.windows.import_sounddevice", lambda: fake)

    sink = SoundDeviceSink(output_device(), buffer_ms=100)
    await sink.open()
    stream = fake.OutputStream.instances[-1]

    # 10 чанков по 100 мс (24 kHz TTS) в буфер на 100 мс: без обратного давления
    # доехал бы только первый.
    chunks = [
        AudioChunk.from_array(np.full(2400, 1000 + index, dtype=np.int16), 0, TTS_FORMAT)
        for index in range(10)
    ]

    async def consume() -> None:
        """Драйвер, забирающий буфер как настоящий — по чуть-чуть."""
        for _ in range(200):
            stream.pull(2400)  # 50 мс на 48 kHz
            await asyncio.sleep(0.001)

    consumer = asyncio.create_task(consume())
    for chunk in chunks:
        await sink.write(chunk)
    await sink.drain()
    consumer.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await consumer

    assert sink.stats.dropped_chunks == 0
    assert sink.stats.chunks == len(chunks)
    await sink.close()


async def test_sink_callback_pads_silence_on_underrun(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой буфер -> драйвер получает тишину, а не мусор."""
    fake = fake_sounddevice_module([])
    monkeypatch.setattr("engine.audio_io.windows.import_sounddevice", lambda: fake)

    sink = SoundDeviceSink(output_device("Наушники (Realtek(R) Audio)"))
    await sink.open()
    stream = fake.OutputStream.instances[-1]
    assert np.count_nonzero(stream.pull(256)) == 0
    await sink.close()
