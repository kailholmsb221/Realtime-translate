"""Тесты audio_io (зона агента A, ARCHITECTURE.md 4.1).

Железо не трогаем: всё проверяется на numpy-сигналах, WAV-фикстурах и
фейковых списках устройств. Тесты обязаны быть зелёными на Linux без
установленных ``sounddevice`` / ``pyaudiowpatch``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from engine.audio_io import (
    CHUNK_FRAMES,
    ENGINE_FORMAT,
    SAMPLE_RATE_ENGINE,
    TTS_FORMAT,
    AudioBackendUnavailable,
    AudioChunk,
    AudioConfig,
    AudioDeviceInfo,
    AudioFormat,
    AudioIOError,
    AudioSink,
    AudioSource,
    ChunkAssembler,
    DeviceKind,
    DeviceNotFound,
    FakeSink,
    FakeSource,
    create_sink,
    create_source,
    list_devices,
    pcm,
)
from engine.audio_io import devices as devices_mod
from engine.audio_io.config import (
    DEFAULT_CABLE_NAME,
    ENV_CABLE,
    ENV_CHUNK_MS,
    ENV_HEADPHONES,
    ENV_LOOPBACK,
    ENV_MIC,
    ENV_SAMPLE_RATE,
)

TONE_HZ = 440.0


def make_tone(
    freq_hz: float,
    rate: int,
    seconds: float = 1.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    """Синус как int16 PCM (для проверок ресемплинга по спектру)."""
    n = round(rate * seconds)
    t = np.arange(n, dtype=np.float64) / rate
    return pcm.float32_to_int16(amplitude * np.sin(2.0 * np.pi * freq_hz * t))


def peak_freq(samples: np.ndarray, rate: int) -> float:
    """Частота главного пика спектра, Гц."""
    data = pcm.int16_to_float32(np.asarray(samples, dtype=np.int16)).astype(np.float64)
    window = np.hanning(data.size)
    spectrum = np.abs(np.fft.rfft(data * window))
    freqs = np.fft.rfftfreq(data.size, d=1.0 / rate)
    return float(freqs[int(np.argmax(spectrum))])


# --- pcm: ресемплинг -------------------------------------------------------


@pytest.mark.parametrize("method", ["auto", "linear"])
@pytest.mark.parametrize(
    ("src_rate", "dst_rate"),
    [(16_000, 48_000), (48_000, 16_000), (24_000, 16_000)],
)
def test_resample_keeps_length_and_tone(method: str, src_rate: int, dst_rate: int) -> None:
    """Длина меняется по отношению частот, а тон 440 Гц остаётся 440 Гц."""
    tone = make_tone(TONE_HZ, src_rate)
    out = pcm.resample(tone, src_rate, dst_rate, method=method)  # type: ignore[arg-type]

    assert out.dtype == np.int16
    assert out.size == pcm.target_length(tone.size, src_rate, dst_rate)
    assert abs(peak_freq(out, dst_rate) - TONE_HZ) <= 2.0


def test_resample_same_rate_is_identity() -> None:
    """Одинаковые частоты — данные не меняются."""
    tone = make_tone(TONE_HZ, SAMPLE_RATE_ENGINE, seconds=0.1)
    assert np.array_equal(pcm.resample(tone, 16_000, 16_000), tone)


def test_resample_empty_and_short() -> None:
    """Пустой и односэмпловый вход не ломают ресемплер."""
    assert pcm.resample(np.zeros(0, dtype=np.int16), 16_000, 48_000).size == 0
    out = pcm.resample(np.array([1000], dtype=np.int16), 16_000, 48_000, method="linear")
    assert out.size == 3


def test_resample_poly_requires_scipy() -> None:
    """method='poly' работает при наличии scipy и внятно падает без него."""
    tone = make_tone(TONE_HZ, 48_000, seconds=0.2)
    if pcm.has_scipy():
        out = pcm.resample(tone, 48_000, 16_000, method="poly")
        assert abs(peak_freq(out, 16_000) - TONE_HZ) <= 2.0
    else:  # pragma: no cover - в CI scipy стоит
        with pytest.raises(RuntimeError, match="scipy"):
            pcm.resample(tone, 48_000, 16_000, method="poly")


def test_resample_from_wav_fixture(tone_wav: Path) -> None:
    """Фикстура 440 Гц 16 kHz -> 48 kHz -> обратно: тон сохраняется."""
    samples, rate = pcm.read_wav(tone_wav)
    assert rate == SAMPLE_RATE_ENGINE
    up = pcm.resample(samples, rate, 48_000)
    down = pcm.resample(up, 48_000, rate)
    assert abs(peak_freq(down, rate) - TONE_HZ) <= 2.0
    assert abs(down.size - samples.size) <= 1


# --- pcm: даунмикс и конвертация ------------------------------------------


def test_to_mono_downmix_interleaved() -> None:
    """Interleaved stereo сводится усреднением каналов."""
    stereo = np.array([100, 300, -200, 0, 32767, 32767], dtype=np.int16)
    mono = pcm.to_mono(stereo, 2)
    assert mono.dtype == np.int16
    assert mono.tolist() == [200, -100, 32767]


def test_to_mono_2d_and_passthrough() -> None:
    """Принимается и двумерный (frames, channels), моно проходит как есть."""
    block = np.array([[0.5, -0.5], [1.0, 0.0]], dtype=np.float32)
    assert pcm.to_mono(block, 2).tolist() == pytest.approx([0.0, 0.5])

    mono = np.array([1, 2, 3], dtype=np.int16)
    assert np.array_equal(pcm.to_mono(mono, 1), mono)


def test_to_mono_rejects_bad_shape() -> None:
    """Длина, не кратная числу каналов, — ошибка."""
    with pytest.raises(ValueError, match="не кратна"):
        pcm.to_mono(np.zeros(5, dtype=np.int16), 2)


def test_float_int_roundtrip_and_clipping() -> None:
    """float32 <-> int16 без заметной потери; выход за [-1, 1] клиппится."""
    source = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float32)
    back = pcm.int16_to_float32(pcm.float32_to_int16(source))
    assert back == pytest.approx(source, abs=1e-3)

    clipped = pcm.float32_to_int16(np.array([-2.0, 2.0], dtype=np.float32))
    assert clipped.tolist() == [-32767, 32767]


# --- pcm: WAV --------------------------------------------------------------


def test_wav_roundtrip(tmp_path: Path) -> None:
    """write_wav -> read_wav возвращает те же сэмплы и частоту."""
    tone = make_tone(TONE_HZ, SAMPLE_RATE_ENGINE, seconds=0.25)
    path = tmp_path / "nested" / "tone.wav"
    pcm.write_wav(path, tone, SAMPLE_RATE_ENGINE)

    samples, rate = pcm.read_wav(path)
    assert rate == SAMPLE_RATE_ENGINE
    assert np.array_equal(samples, tone)


def test_read_wav_downmixes_stereo(tmp_path: Path) -> None:
    """Стереофайл читается как моно (внутри движка всё моно)."""
    stereo = np.array([100, 300, -200, 0], dtype=np.int16)
    path = tmp_path / "stereo.wav"
    pcm.write_wav(path, stereo, 48_000, channels=2)

    samples, rate = pcm.read_wav(path)
    assert rate == 48_000
    assert samples.tolist() == [200, -100]


# --- base: формат, чанк, сборщик ------------------------------------------


def test_engine_format_defaults() -> None:
    """Формат движка — 16 kHz mono int16, чанк 30 мс = 480 фреймов."""
    assert (ENGINE_FORMAT.sample_rate, ENGINE_FORMAT.channels, ENGINE_FORMAT.dtype) == (
        16_000,
        1,
        "int16",
    )
    assert ENGINE_FORMAT.frames_for_ms(30) == CHUNK_FRAMES == 480
    assert ENGINE_FORMAT.bytes_per_frame == 2
    assert TTS_FORMAT.sample_rate == 24_000


@pytest.mark.parametrize("kwargs", [{"sample_rate": 0}, {"channels": 0}, {"dtype": "float32"}])
def test_audio_format_validates(kwargs: dict[str, object]) -> None:
    """Некорректный формат отклоняется сразу."""
    with pytest.raises(ValueError):
        AudioFormat(**kwargs)  # type: ignore[arg-type]


def test_audio_chunk_from_bytes_and_array() -> None:
    """Чанк принимает и bytes, и int16-массив; n_frames считается сам."""
    samples = np.arange(480, dtype=np.int16)
    from_array = AudioChunk.from_array(samples, ts_ms=30)
    from_bytes = AudioChunk(samples.astype("<i2").tobytes(), ts_ms=30)

    assert from_array.pcm == from_bytes.pcm
    assert from_array.n_frames == from_bytes.n_frames == 480
    assert len(from_array) == 480
    assert from_array.duration_ms == pytest.approx(30.0)
    assert from_array.end_ts_ms == 60
    assert np.array_equal(from_array.samples, samples)


def test_audio_chunk_rejects_wrong_n_frames() -> None:
    """Несовпадение n_frames и данных — ошибка."""
    with pytest.raises(ValueError, match="n_frames"):
        AudioChunk(b"\x00\x00", ts_ms=0, n_frames=7)


def test_chunk_assembler_emits_fixed_chunks() -> None:
    """Сборщик режет поток ровно по 480 фреймов с шагом ts 30 мс."""
    asm = ChunkAssembler()
    assert asm.chunk_frames == 480

    chunks = asm.push(np.ones(1000, dtype=np.int16))
    assert [c.n_frames for c in chunks] == [480, 480]
    assert [c.ts_ms for c in chunks] == [0, 30]
    assert asm.pending_frames == 40

    tail = asm.flush()
    assert len(tail) == 1
    assert tail[0].n_frames == 480
    assert tail[0].ts_ms == 60
    assert tail[0].samples[40:].tolist() == [0] * 440
    assert asm.flush() == []


def test_chunk_assembler_flush_without_pad() -> None:
    """flush(pad=False) выбрасывает неполный хвост."""
    asm = ChunkAssembler()
    asm.push(np.ones(100, dtype=np.int16))
    assert asm.flush(pad=False) == []
    assert asm.pending_frames == 0


# --- fake: источник и приёмник --------------------------------------------


async def test_fake_source_chunks_are_exactly_480_frames(tone_wav: Path) -> None:
    """FakeSource режет WAV ровно по 480 фреймов, ts_ms идут по 30 мс."""
    source = FakeSource.from_wav(tone_wav)
    chunks = [chunk async for chunk in source]

    assert len(chunks) == source.n_chunks == SAMPLE_RATE_ENGINE // CHUNK_FRAMES + 1
    assert {c.n_frames for c in chunks} == {480}
    assert [c.ts_ms for c in chunks[:4]] == [0, 30, 60, 90]
    assert all(c.fmt == ENGINE_FORMAT for c in chunks)
    # последний чанк дополнен нулями: 16000 = 33*480 + 160
    assert chunks[-1].samples[160:].tolist() == [0] * 320


async def test_fake_source_drop_incomplete_tail() -> None:
    """pad_last=False отбрасывает неполный хвост."""
    source = FakeSource(np.ones(500, dtype=np.int16), pad_last=False)
    chunks = [chunk async for chunk in source]
    assert len(chunks) == 1
    assert source.n_chunks == 1


async def test_fake_source_context_manager_and_realtime() -> None:
    """Async context manager открывает/закрывает источник; realtime тормозит поток."""
    source = FakeSource(np.zeros(CHUNK_FRAMES * 3, dtype=np.int16), realtime=True)
    started = time.perf_counter()
    async with source as src:
        assert src.is_open
        chunks = [chunk async for chunk in src]
    elapsed = time.perf_counter() - started

    assert len(chunks) == 3
    assert elapsed >= 0.05  # 3 чанка по 30 мс с реальной скоростью
    assert not source.is_open


async def test_fake_sink_roundtrip_wav(tmp_path: Path, tone_wav: Path) -> None:
    """Файл -> FakeSource -> FakeSink -> WAV: сигнал совпадает с исходным."""
    original, rate = pcm.read_wav(tone_wav)
    out = tmp_path / "out.wav"

    source = FakeSource.from_wav(tone_wav)
    sink = FakeSink(path=out)
    async with source, sink:
        async for chunk in source:
            await sink.write(chunk)

    assert out.is_file()
    written, written_rate = pcm.read_wav(out)
    assert written_rate == rate
    assert written.size >= original.size  # хвост дополнен нулями
    assert np.array_equal(written[: original.size], original)
    assert sink.drains == 1


async def test_fake_sink_resamples_tts_chunks() -> None:
    """Чанки TTS (24 kHz) приводятся к 16 kHz приёмника."""
    tone = make_tone(TONE_HZ, 24_000, seconds=0.5)
    sink = FakeSink()
    await sink.write(AudioChunk.from_array(tone, 0, TTS_FORMAT))
    await sink.close()

    assert sink.format.sample_rate == SAMPLE_RATE_ENGINE
    assert sink.n_frames == pcm.target_length(tone.size, 24_000, 16_000)
    assert abs(peak_freq(sink.samples, SAMPLE_RATE_ENGINE) - TONE_HZ) <= 2.0


async def test_fake_sink_save_wav_requires_path() -> None:
    """Без пути сохранить нельзя — явная ошибка."""
    sink = FakeSink()
    with pytest.raises(ValueError, match="путь"):
        sink.save_wav()


def test_fake_implementations_satisfy_protocols() -> None:
    """FakeSource/FakeSink реализуют протоколы AudioSource/AudioSink."""
    assert isinstance(FakeSource(np.zeros(10, dtype=np.int16)), AudioSource)
    assert isinstance(FakeSink(), AudioSink)


# --- конфиг ----------------------------------------------------------------


def test_config_defaults_without_env() -> None:
    """Пустое окружение -> дефолты: кабель CABLE Input, 16 kHz, 30 мс."""
    cfg = AudioConfig.from_env({})
    assert cfg.loopback_name is None
    assert cfg.mic_name is None
    assert cfg.headphones_name is None
    assert cfg.cable_name == DEFAULT_CABLE_NAME
    assert cfg.sample_rate == SAMPLE_RATE_ENGINE
    assert cfg.chunk_ms == 30
    assert cfg.chunk_frames == CHUNK_FRAMES
    assert cfg.format == ENGINE_FORMAT


def test_config_parses_env() -> None:
    """Все переменные RT_AUDIO_* разбираются; пробелы обрезаются."""
    cfg = AudioConfig.from_env(
        {
            ENV_LOOPBACK: " Наушники (Realtek ",
            ENV_MIC: "Microphone (USB",
            ENV_HEADPHONES: "Headphones",
            ENV_CABLE: "CABLE Input (VB-Audio",
            ENV_SAMPLE_RATE: "16000",
            ENV_CHUNK_MS: "20",
        }
    )
    assert cfg.loopback_name == "Наушники (Realtek"
    assert cfg.mic_name == "Microphone (USB"
    assert cfg.headphones_name == "Headphones"
    assert cfg.cable_name == "CABLE Input (VB-Audio"
    assert cfg.chunk_ms == 20
    assert cfg.chunk_frames == 320


def test_config_empty_values_fall_back_to_defaults() -> None:
    """Пустая строка = переменная не задана."""
    cfg = AudioConfig.from_env({ENV_MIC: "   ", ENV_CABLE: ""})
    assert cfg.mic_name is None
    assert cfg.cable_name == DEFAULT_CABLE_NAME


def test_config_rejects_non_numeric_env() -> None:
    """Нечисловое значение числовой переменной — понятная ошибка."""
    with pytest.raises(ValueError, match=ENV_CHUNK_MS):
        AudioConfig.from_env({ENV_CHUNK_MS: "тридцать"})


# --- устройства ------------------------------------------------------------


def device(
    id_: int,
    name: str,
    kind: DeviceKind,
    *,
    default: bool = False,
    rate: float = 48_000.0,
    channels: int = 2,
) -> AudioDeviceInfo:
    """Фейковое устройство для тестов поиска."""
    return AudioDeviceInfo(
        id=id_,
        name=name,
        kind=kind,
        default_sample_rate=rate,
        channels=channels,
        backend="fake",
        is_default=default,
    )


@pytest.fixture
def fake_devices() -> list[AudioDeviceInfo]:
    """Набор устройств, похожий на типичную Windows-машину с VB-Cable."""
    return [
        device(0, "Микрофон (Realtek(R) Audio)", DeviceKind.INPUT, default=True),
        device(1, "CABLE Output (VB-Audio Virtual Cable)", DeviceKind.INPUT),
        device(2, "Наушники (Realtek(R) Audio)", DeviceKind.OUTPUT, default=True),
        device(3, "CABLE Input (VB-Audio Virtual Cable)", DeviceKind.OUTPUT),
        device(4, "Наушники (Realtek(R) Audio) [Loopback]", DeviceKind.LOOPBACK, default=True),
    ]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("CABLE Input (VB-Audio Virtual Cable)", True),
        ("CABLE Output (VB-Audio Virtual Cable)", True),
        ("cable input (vb-audio)", True),
        ("Наушники (Realtek(R) Audio)", False),
        ("Microphone (USB Audio Device)", False),
    ],
)
def test_is_virtual_cable_by_name(name: str, expected: bool) -> None:
    """Виртуальный кабель определяется по подстроке CABLE (регистр не важен)."""
    assert device(0, name, DeviceKind.OUTPUT).is_virtual_cable is expected


def test_find_default_devices(fake_devices: list[AudioDeviceInfo]) -> None:
    """Без имени берутся устройства по умолчанию своего типа."""
    assert devices_mod.find_default_mic(fake_devices).id == 0
    assert devices_mod.find_default_output(fake_devices).id == 2
    assert devices_mod.find_loopback_device(fake_devices).id == 4


def test_find_by_name_substring(fake_devices: list[AudioDeviceInfo]) -> None:
    """Поиск по подстроке имени, регистр не важен."""
    assert devices_mod.find_default_mic(fake_devices, "cable output").id == 1
    assert devices_mod.find_default_output(fake_devices, "наушники").id == 2


def test_find_virtual_cable_output(fake_devices: list[AudioDeviceInfo]) -> None:
    """CABLE Input находится как приёмник перевода пользователя."""
    cable = devices_mod.find_virtual_cable_output(fake_devices)
    assert cable.id == 3
    assert cable.is_virtual_cable
    assert cable.kind is DeviceKind.OUTPUT


def test_find_virtual_cable_missing_hints_install() -> None:
    """Без кабеля — ошибка с подсказкой про установку VB-Cable."""
    only_headphones = [device(0, "Наушники (Realtek(R) Audio)", DeviceKind.OUTPUT, default=True)]
    with pytest.raises(DeviceNotFound, match="VB-Audio Virtual Cable"):
        devices_mod.find_virtual_cable_output(only_headphones)


def test_find_unknown_name_lists_available(fake_devices: list[AudioDeviceInfo]) -> None:
    """Сообщение об ошибке перечисляет, что вообще есть в системе."""
    with pytest.raises(DeviceNotFound, match="Наушники"):
        devices_mod.find_default_output(fake_devices, "Focusrite")


def test_find_loopback_without_devices_raises() -> None:
    """Пустой список устройств -> DeviceNotFound, а не падение."""
    with pytest.raises(DeviceNotFound):
        devices_mod.find_loopback_device([])


def test_device_describe_marks_loopback_and_cable(fake_devices: list[AudioDeviceInfo]) -> None:
    """describe() помечает loopback и CABLE — это видно в `audio_smoke.py --list`."""
    assert "loopback" in fake_devices[4].describe()
    assert "CABLE" in fake_devices[3].describe()


def test_list_devices_on_host_without_backends() -> None:
    """В контейнере без аудио-библиотек list_devices() просто пуст, без исключений."""
    result = list_devices()
    assert isinstance(result, list)
    assert all(isinstance(item, AudioDeviceInfo) for item in result)


def test_backend_report_mentions_every_backend() -> None:
    """Отчёт о бэкендах содержит статус обоих пакетов."""
    report = devices_mod.backend_report()
    assert set(report) == {"sounddevice", "pyaudiowpatch"}


def test_import_pyaudiowpatch_explains_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    """На не-Windows импорт WASAPI-бэкенда объясняет, что делать."""
    monkeypatch.setattr(devices_mod, "is_windows", lambda: False)
    with pytest.raises(AudioBackendUnavailable, match="Windows"):
        devices_mod.import_pyaudiowpatch()


# --- фабрика ---------------------------------------------------------------


@pytest.mark.parametrize("kind", ["loopback", "mic"])
def test_create_source_hardware_unavailable_on_linux(kind: str) -> None:
    """На Linux живой захват недоступен — ошибка с инструкцией."""
    if devices_mod.is_windows():  # pragma: no cover - на Windows проверять нечего
        pytest.skip("тест про поведение вне Windows")
    with pytest.raises(AudioBackendUnavailable, match="файловый режим"):
        create_source(kind)


@pytest.mark.parametrize("kind", ["headphones", "cable"])
def test_create_sink_hardware_unavailable_on_linux(kind: str) -> None:
    """На Linux живой вывод недоступен — ошибка с инструкцией."""
    if devices_mod.is_windows():  # pragma: no cover
        pytest.skip("тест про поведение вне Windows")
    with pytest.raises(AudioBackendUnavailable, match="Windows"):
        create_sink(kind)


def test_create_source_without_backend_even_if_platform_faked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Даже если платформа «windows», без пакетов/устройств ошибка остаётся внятной."""
    import engine.audio_io as audio_io

    monkeypatch.setattr(audio_io, "is_windows", lambda: True)
    monkeypatch.setattr(devices_mod, "is_windows", lambda: True)
    with pytest.raises(AudioIOError):
        create_source("loopback")


def test_create_source_file_requires_path() -> None:
    """Файловый режим без пути — ValueError."""
    with pytest.raises(ValueError, match="path"):
        create_source("file")
    with pytest.raises(ValueError, match="path"):
        create_sink("file")


@pytest.mark.parametrize(
    ("factory", "kind"),
    [(create_source, "speaker"), (create_sink, "microphone")],
)
def test_factory_rejects_unknown_kind(factory: Callable[..., object], kind: str) -> None:
    """Неизвестный kind — ValueError со списком допустимых."""
    with pytest.raises(ValueError, match="неизвестный"):
        factory(kind)


async def test_factory_file_mode_roundtrip(tmp_path: Path, tone_wav: Path) -> None:
    """Фабрика в файловом режиме: WAV -> чанки 480 -> WAV (этап 1 из ARCHITECTURE.md)."""
    out = tmp_path / "translated.wav"
    source = create_source("file", path=tone_wav)
    sink = create_sink("file", path=out)

    assert isinstance(source, AudioSource)
    assert isinstance(sink, AudioSink)

    async with source, sink:
        async for chunk in source:
            assert chunk.n_frames == CHUNK_FRAMES
            await sink.write(chunk)

    written, rate = pcm.read_wav(out)
    assert rate == SAMPLE_RATE_ENGINE
    assert abs(peak_freq(written, rate) - TONE_HZ) <= 2.0


async def test_factory_file_mode_respects_config_chunk_ms(tone_wav: Path) -> None:
    """chunk_ms из конфига управляет длиной чанка."""
    cfg = AudioConfig.from_env({ENV_CHUNK_MS: "20"})
    source = create_source("file", cfg, path=tone_wav)
    async with source:
        async for chunk in source:
            assert chunk.n_frames == 320
            break
