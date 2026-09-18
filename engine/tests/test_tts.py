"""Тесты модуля tts (зона агента D, ARCHITECTURE.md 4.4).

Без тяжёлых моделей: XTTS/MMS подменяются фейками того же интерфейса
(CLAUDE.md, технический стандарт). Тесты с реальными моделями помечены
``@pytest.mark.real_models`` и пропускаются без ``RT_REAL_MODELS=1``.
"""

from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine.contracts.events import EVENT_TTS_CHUNK, Lang, Stream, TtsChunk, parse_event, to_json
from engine.tts import (
    CHUNK_MS_MAX,
    CHUNK_MS_MIN,
    MAX_SAMPLE_SECONDS,
    MIN_SAMPLE_SECONDS,
    OUTPUT_SAMPLE_RATE,
    Backend,
    Device,
    FakeTts,
    KkBackend,
    Pcm16,
    TtsConfig,
    TtsError,
    TtsProvider,
    TtsRouter,
    VoiceError,
    VoiceNotFoundError,
    VoiceStore,
    chunk_to_event,
    create_tts,
    pcm_from_event,
)
from engine.tts.audio import (
    Rechunker,
    float_to_int16,
    iter_chunks,
    read_wav,
    resample_linear,
    resample_pcm,
    to_mono,
    write_wav,
)
from engine.tts.base import (
    ENV_BACKEND,
    ENV_DEVICE,
    ENV_KK_BACKEND,
    ENV_MODELS_DIR,
    ENV_VOICES_DIR,
)
from engine.tts.kk_kazakhtts2 import KazakhTts2Provider
from engine.tts.voices import META_FILENAME, SAMPLE_FILENAME
from engine.tts.xtts import XttsProvider

# --- вспомогательное -------------------------------------------------------


def tone(duration_s: float, sample_rate: int = OUTPUT_SAMPLE_RATE, freq_hz: float = 440.0) -> Pcm16:
    """Синтетический тон как PCM int16 mono."""
    n = round(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    return float_to_int16(0.5 * np.sin(2.0 * np.pi * freq_hz * t))


def write_tone(path: Path, duration_s: float, sample_rate: int = OUTPUT_SAMPLE_RATE) -> Path:
    """Записать тон в WAV и вернуть путь."""
    write_wav(path, tone(duration_s, sample_rate), sample_rate)
    return path


def dominant_hz(samples: Pcm16, sample_rate: int) -> float:
    """Частота с максимальной энергией (для проверки ресемплинга)."""
    spectrum = np.abs(np.fft.rfft(samples.astype(np.float64)))
    return float(np.fft.rfftfreq(samples.size, 1.0 / sample_rate)[int(np.argmax(spectrum))])


class RecordingTts:
    """Фейковый провайдер: помнит вызовы, отдаёт короткий тон."""

    def __init__(self, langs: frozenset[Lang], *, cloning: bool) -> None:
        self.langs = langs
        self.cloning = cloning
        self.calls: list[tuple[str, Lang, str | None]] = []
        self.warmups = 0

    @property
    def supports_cloning(self) -> bool:
        return self.cloning

    def supports(self, lang: Lang) -> bool:
        return lang in self.langs

    async def warmup(self) -> None:
        self.warmups += 1

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        self.calls.append((text, lang, voice_id))
        yield tone(0.02)


@pytest.fixture
def voices_dir(tmp_path: Path) -> Path:
    """Пустое хранилище голосов во временном каталоге."""
    path = tmp_path / "voices"
    path.mkdir()
    return path


@pytest.fixture
def store(voices_dir: Path) -> VoiceStore:
    return VoiceStore(voices_dir)


@pytest.fixture
def sample_10s(tmp_path: Path) -> Path:
    """WAV-сэмпл: тон 10 секунд, 24 kHz (короче рекомендованных 15 с)."""
    return write_tone(tmp_path / "sample_10s.wav", 10.0)


@pytest.fixture
def fake_config(voices_dir: Path) -> TtsConfig:
    return TtsConfig(backend=Backend.FAKE, voices_dir=voices_dir, chunk_ms=CHUNK_MS_MIN)


# --- аудио-утилиты ---------------------------------------------------------


def test_float_to_int16_clips() -> None:
    pcm = float_to_int16(np.array([0.0, 1.0, -1.0, 2.0, -2.0], dtype=np.float32))
    assert pcm.dtype == np.int16
    assert pcm.tolist() == [0, 32767, -32767, 32767, -32767]


def test_to_mono_averages_channels() -> None:
    interleaved = np.array([100, 200, -100, -200], dtype=np.int16)  # 2 канала
    assert to_mono(interleaved, 2).tolist() == [150, -150]


@pytest.mark.parametrize("resample", [resample_pcm, resample_linear])
def test_resample_16k_to_24k(resample: object) -> None:
    """Ресемпл 16 kHz -> 24 kHz: длина x1.5, частота тона на месте."""
    source = tone(1.0, sample_rate=16_000, freq_hz=440.0)
    resampled = resample(source, 16_000, OUTPUT_SAMPLE_RATE)  # type: ignore[operator]

    assert resampled.dtype == np.int16
    assert resampled.size == pytest.approx(source.size * 1.5, abs=1)
    assert dominant_hz(resampled, OUTPUT_SAMPLE_RATE) == pytest.approx(440.0, abs=5.0)


def test_resample_same_rate_is_noop() -> None:
    source = tone(0.1, sample_rate=OUTPUT_SAMPLE_RATE)
    assert np.array_equal(resample_pcm(source, OUTPUT_SAMPLE_RATE, OUTPUT_SAMPLE_RATE), source)


def test_iter_chunks_covers_signal_without_loss() -> None:
    source = tone(0.25)
    chunks = list(iter_chunks(source, OUTPUT_SAMPLE_RATE, CHUNK_MS_MIN))
    assert all(chunk.size <= 480 for chunk in chunks)
    assert np.array_equal(np.concatenate(chunks), source)


def test_rechunker_emits_fixed_size_chunks() -> None:
    rechunker = Rechunker(480)
    produced: list[Pcm16] = []
    for piece in (tone(0.05), tone(0.011), tone(0.004)):
        produced.extend(rechunker.push(piece))
    assert all(chunk.size == 480 for chunk in produced)
    tail = rechunker.flush()
    assert len(tail) == 1
    total = sum(chunk.size for chunk in produced) + tail[0].size
    assert total == round(OUTPUT_SAMPLE_RATE * 0.065)


def test_wav_roundtrip(tmp_path: Path) -> None:
    source = tone(0.5)
    path = tmp_path / "roundtrip.wav"
    write_wav(path, source, OUTPUT_SAMPLE_RATE)
    restored, sample_rate = read_wav(path)
    assert sample_rate == OUTPUT_SAMPLE_RATE
    assert np.array_equal(restored, source)


# --- VoiceStore ------------------------------------------------------------


def test_create_voice_writes_profile(store: VoiceStore, sample_10s: Path) -> None:
    profile = store.create_voice(sample_10s, name="me", lang=Lang.RU)

    assert profile.id.startswith("v_")
    assert len(profile.id) == len("v_") + 8
    assert profile.name == "me"
    assert profile.lang is Lang.RU
    assert profile.created_at_ms > 0
    assert profile.sample_path.name == SAMPLE_FILENAME
    assert profile.sample_path.is_file()

    meta = json.loads((profile.dir / META_FILENAME).read_text(encoding="utf-8"))
    assert meta == {
        "id": profile.id,
        "name": "me",
        "lang": "ru",
        "sample_path": SAMPLE_FILENAME,
        "created_at_ms": profile.created_at_ms,
    }
    assert set(profile.to_row()) == {"id", "name", "lang", "sample_path", "created_at"}


def test_create_voice_resamples_to_24k(store: VoiceStore, tmp_path: Path) -> None:
    source = write_tone(tmp_path / "16k.wav", 6.0, sample_rate=16_000)
    profile = store.create_voice(source, name="from16k", lang=Lang.EN)

    samples, sample_rate = read_wav(profile.sample_path)
    assert sample_rate == OUTPUT_SAMPLE_RATE
    assert samples.size == pytest.approx(6.0 * OUTPUT_SAMPLE_RATE, rel=0.01)


def test_create_voice_warns_on_short_sample(
    store: VoiceStore,
    sample_10s: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="engine.tts.voices"):
        store.create_voice(sample_10s, name="short", lang=Lang.RU)
    assert any("короткий" in record.getMessage() for record in caplog.records)


def test_create_voice_no_warning_on_long_sample(
    store: VoiceStore,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sample = write_tone(tmp_path / "long.wav", 20.0)
    with caplog.at_level(logging.WARNING, logger="engine.tts.voices"):
        store.create_voice(sample, name="long", lang=Lang.RU)
    assert not caplog.records


@pytest.mark.parametrize("duration", [MIN_SAMPLE_SECONDS - 1.0, MAX_SAMPLE_SECONDS + 1.0])
def test_create_voice_rejects_bad_duration(
    store: VoiceStore, tmp_path: Path, duration: float
) -> None:
    sample = write_tone(tmp_path / f"bad_{int(duration)}.wav", duration)
    with pytest.raises(VoiceError):
        store.create_voice(sample, name="bad", lang=Lang.RU)
    assert store.list() == []


def test_create_voice_rejects_non_wav(store: VoiceStore, tmp_path: Path) -> None:
    fake = tmp_path / "not_audio.wav"
    fake.write_bytes(b"definitely not a wav file")
    with pytest.raises(VoiceError):
        store.create_voice(fake, name="bad", lang=Lang.RU)


def test_create_voice_rejects_duplicate_name(store: VoiceStore, sample_10s: Path) -> None:
    store.create_voice(sample_10s, name="me", lang=Lang.RU)
    with pytest.raises(VoiceError):
        store.create_voice(sample_10s, name="me", lang=Lang.RU)


def test_get_list_delete(store: VoiceStore, sample_10s: Path, tmp_path: Path) -> None:
    first = store.create_voice(sample_10s, name="first", lang=Lang.RU)
    second = store.create_voice(write_tone(tmp_path / "s2.wav", 8.0), name="second", lang=Lang.EN)

    assert store.get(first.id) == first
    assert [profile.id for profile in store.list()] == [first.id, second.id]
    assert store.exists(first.id)

    store.delete(first.id)
    assert not store.exists(first.id)
    assert [profile.id for profile in store.list()] == [second.id]

    with pytest.raises(VoiceNotFoundError):
        store.get(first.id)
    with pytest.raises(VoiceNotFoundError):
        store.delete(first.id)


def test_list_empty_store(tmp_path: Path) -> None:
    assert VoiceStore(tmp_path / "missing").list() == []


def test_create_voice_calls_latents_fn(store: VoiceStore, sample_10s: Path) -> None:
    seen: list[tuple[Path, Path]] = []

    def latents_fn(sample_path: Path, voice_dir: Path) -> None:
        seen.append((sample_path, voice_dir))
        (voice_dir / "latents.pt").write_bytes(b"fake-latents")

    profile = store.create_voice(sample_10s, name="me", lang=Lang.RU, latents_fn=latents_fn)
    assert seen == [(profile.sample_path, profile.dir)]
    assert profile.latents_path.is_file()


def test_create_voice_cleans_up_when_latents_fail(store: VoiceStore, sample_10s: Path) -> None:
    def broken(sample_path: Path, voice_dir: Path) -> None:
        raise RuntimeError("модель не загрузилась")

    with pytest.raises(RuntimeError):
        store.create_voice(sample_10s, name="me", lang=Lang.RU, latents_fn=broken)
    assert store.list() == []


def test_invalid_voice_id_rejected(store: VoiceStore) -> None:
    for bad in ("", "../etc", "a/b"):
        with pytest.raises(VoiceError):
            store.path_for(bad)


# --- FakeTts ---------------------------------------------------------------


async def test_fake_tts_chunks_are_int16_24k() -> None:
    fake = FakeTts(chunk_ms=CHUNK_MS_MIN, ms_per_word=60)
    text = "один два три четыре пять"
    chunks = [chunk async for chunk in fake.synthesize(text, Lang.RU)]

    assert chunks, "фейк обязан отдать хотя бы один чанк"
    assert all(chunk.dtype == np.int16 for chunk in chunks)
    assert all(chunk.ndim == 1 for chunk in chunks)
    assert all(chunk.size <= 480 for chunk in chunks)

    total = sum(chunk.size for chunk in chunks)
    expected = round(OUTPUT_SAMPLE_RATE * 5 * 60 / 1000)  # 5 слов * 60 мс
    assert total == expected
    assert 1000 * total / OUTPUT_SAMPLE_RATE == pytest.approx(300.0, abs=1.0)


async def test_fake_tts_duration_scales_with_text() -> None:
    fake = FakeTts()
    short = sum([chunk.size async for chunk in fake.synthesize("одно слово", Lang.EN)])
    long = sum([chunk.size async for chunk in fake.synthesize("одно слово " * 5, Lang.EN)])
    assert long == 5 * short


async def test_fake_tts_rejects_unsupported_lang() -> None:
    fake = FakeTts(langs=(Lang.RU,))
    with pytest.raises(TtsError):
        [chunk async for chunk in fake.synthesize("hello", Lang.EN)]


def test_fake_tts_is_tts_provider() -> None:
    assert isinstance(FakeTts(), TtsProvider)


# --- события tts.chunk -----------------------------------------------------


def test_chunk_to_event_matches_contract() -> None:
    pcm = tone(0.02)
    envelope = chunk_to_event(Stream.OUT, 7, pcm)

    assert envelope.type == EVENT_TTS_CHUNK
    assert isinstance(envelope.payload, TtsChunk)
    assert envelope.payload.stream is Stream.OUT
    assert envelope.payload.seq == 7

    parsed = parse_event(to_json(envelope))  # валидация JSON-схемой контракта
    assert isinstance(parsed.payload, TtsChunk)
    restored = pcm_from_event(parsed.payload)
    assert np.array_equal(restored, pcm)
    assert base64.b64decode(parsed.payload.pcm_base64) == pcm.astype("<i2").tobytes()


async def test_router_events_have_sequential_seq(fake_config: TtsConfig) -> None:
    router = create_tts(fake_config)
    events = [
        envelope
        async for envelope in router.synthesize_events(
            "раз два три", Lang.RU, None, stream=Stream.IN
        )
    ]

    assert len(events) > 1
    seqs = [envelope.payload.seq for envelope in events if isinstance(envelope.payload, TtsChunk)]
    assert seqs == list(range(len(events)))
    for envelope in events:
        parse_event(to_json(envelope))  # каждое событие валидно по контракту


# --- TtsRouter -------------------------------------------------------------


@pytest.fixture
def recording_router(fake_config: TtsConfig) -> tuple[TtsRouter, RecordingTts, RecordingTts]:
    """Роутер с подменёнными провайдерами: XTTS-подобный и kk без клона."""
    cloning = RecordingTts(frozenset({Lang.RU, Lang.EN}), cloning=True)
    kazakh = RecordingTts(frozenset({Lang.KK}), cloning=False)
    router = TtsRouter(
        fake_config,
        providers={Lang.RU: cloning, Lang.EN: cloning, Lang.KK: kazakh},
    )
    return router, cloning, kazakh


@pytest.mark.parametrize("lang", [Lang.RU, Lang.EN])
async def test_router_uses_cloning_provider_for_ru_en(
    recording_router: tuple[TtsRouter, RecordingTts, RecordingTts],
    lang: Lang,
) -> None:
    router, cloning, kazakh = recording_router
    chunks = [chunk async for chunk in router.synthesize("текст", lang, "v_12345678")]

    assert chunks
    assert cloning.calls == [("текст", lang, "v_12345678")]
    assert kazakh.calls == []


async def test_router_kk_ignores_voice_id(
    recording_router: tuple[TtsRouter, RecordingTts, RecordingTts],
) -> None:
    router, cloning, kazakh = recording_router
    [chunk async for chunk in router.synthesize("мәтін", Lang.KK, "v_12345678")]

    assert kazakh.calls == [("мәтін", Lang.KK, None)]
    assert cloning.calls == []
    assert router.effective_voice_id(Lang.KK, "v_12345678") is None


async def test_router_warmup_touches_every_provider(
    recording_router: tuple[TtsRouter, RecordingTts, RecordingTts],
) -> None:
    router, cloning, kazakh = recording_router
    await router.warmup()
    assert cloning.warmups == 2  # один и тот же объект для ru и en
    assert kazakh.warmups == 1


def test_router_rejects_provider_without_lang(fake_config: TtsConfig) -> None:
    router = TtsRouter(fake_config, providers={Lang.KK: FakeTts(langs=(Lang.RU,))})
    with pytest.raises(TtsError):
        router.provider_for(Lang.KK)


def test_router_builds_fake_provider_in_fake_backend(fake_config: TtsConfig) -> None:
    router = create_tts(fake_config)
    assert isinstance(router.provider_for(Lang.RU), FakeTts)
    assert isinstance(router.provider_for(Lang.KK), FakeTts)
    assert router.provider_for(Lang.RU) is router.provider_for(Lang.RU)  # создаётся один раз

    # фейк повторяет правила маршрутизации: клон у ru/en, у kk его нет
    assert router.provider_for(Lang.RU).supports_cloning
    assert not router.provider_for(Lang.KK).supports_cloning
    assert router.effective_voice_id(Lang.RU, "v_12345678") == "v_12345678"
    assert router.effective_voice_id(Lang.KK, "v_12345678") is None


def test_router_picks_providers_by_lang_in_real_backend(voices_dir: Path) -> None:
    """Реальный режим: классы провайдеров выбираются по языку, модели не грузятся."""
    from engine.tts.kk_mms import MmsKazakhTts
    from engine.tts.xtts import XttsProvider

    router = TtsRouter(TtsConfig(voices_dir=voices_dir))
    assert isinstance(router.provider_for(Lang.RU), XttsProvider)
    assert isinstance(router.provider_for(Lang.EN), XttsProvider)
    assert isinstance(router.provider_for(Lang.KK), MmsKazakhTts)
    assert router.provider_for(Lang.RU).supports_cloning
    assert not router.provider_for(Lang.KK).supports_cloning


def test_router_kazakhtts2_backend_selected(voices_dir: Path) -> None:
    router = TtsRouter(TtsConfig(voices_dir=voices_dir, kk_backend=KkBackend.KAZAKHTTS2))
    assert isinstance(router.provider_for(Lang.KK), KazakhTts2Provider)


async def test_kazakhtts2_stub_raises() -> None:
    provider = KazakhTts2Provider(TtsConfig())
    assert provider.supports(Lang.KK)
    assert not provider.supports_cloning
    with pytest.raises(NotImplementedError):
        await provider.warmup()
    with pytest.raises(NotImplementedError):
        [chunk async for chunk in provider.synthesize("мәтін", Lang.KK)]


async def test_router_synthesize_array(fake_config: TtsConfig) -> None:
    router = create_tts(fake_config)
    pcm = await router.synthesize_array("раз два три", Lang.EN)
    assert pcm.dtype == np.int16
    assert pcm.size == round(OUTPUT_SAMPLE_RATE * 3 * 60 / 1000)


# --- XTTS: обвязка без реальной модели --------------------------------------


class StubXttsModel:
    """Заглушка XTTS: блокирующий генератор float32-чанков, как inference_stream."""

    def __init__(self, n_chunks: int = 3, chunk_size: int = 1000, fail: bool = False) -> None:
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size
        self.fail = fail
        self.calls: list[tuple[str, str, object, object, int]] = []
        self.speaker_manager = SimpleNamespace(
            speakers={"Ana Florence": {"gpt_cond_latent": "gpt", "speaker_embedding": "spk"}}
        )

    def inference_stream(
        self,
        text: str,
        lang: str,
        gpt_cond_latent: object,
        speaker_embedding: object,
        stream_chunk_size: int = 20,
    ) -> Iterator[np.ndarray]:
        self.calls.append((text, lang, gpt_cond_latent, speaker_embedding, stream_chunk_size))
        for index in range(self.n_chunks):
            if self.fail:
                raise RuntimeError("CUDA out of memory")
            time.sleep(0.001)  # блокирующая работа: должна уйти в отдельный поток
            yield np.full(self.chunk_size, 0.1 * (index + 1), dtype=np.float32)


@pytest.fixture
def stub_xtts(
    monkeypatch: pytest.MonkeyPatch, voices_dir: Path
) -> tuple[XttsProvider, StubXttsModel, list[int]]:
    """Провайдер XTTS с подменённой загрузкой модели; список — счётчик загрузок."""
    model = StubXttsModel()
    loads: list[int] = []

    def fake_load(self: XttsProvider) -> StubXttsModel:
        loads.append(1)
        return model

    monkeypatch.setattr(XttsProvider, "_load_model", fake_load)
    config = TtsConfig(voices_dir=voices_dir, chunk_ms=CHUNK_MS_MIN, stream_chunk_size=20)
    return XttsProvider(config, VoiceStore(voices_dir)), model, loads


async def test_xtts_streams_rechunked_int16(
    stub_xtts: tuple[XttsProvider, StubXttsModel, list[int]],
) -> None:
    """Чанки модели (1000 сэмплов) режутся в ровные 20 мс = 480 сэмплов."""
    provider, model, _ = stub_xtts
    chunks = [chunk async for chunk in provider.synthesize("привет", Lang.RU)]

    assert chunks
    assert all(chunk.dtype == np.int16 for chunk in chunks)
    assert all(chunk.size == 480 for chunk in chunks[:-1])
    assert chunks[-1].size <= 480
    assert sum(chunk.size for chunk in chunks) == 3 * 1000

    text, lang, gpt, spk, stream_chunk_size = model.calls[0]
    assert (text, lang) == ("привет", "ru")
    assert (gpt, spk) == ("gpt", "spk")  # встроенный голос из speakers_xtts.pth
    assert stream_chunk_size == 20


async def test_xtts_loads_model_once(
    stub_xtts: tuple[XttsProvider, StubXttsModel, list[int]],
) -> None:
    provider, _, loads = stub_xtts
    await provider.warmup()
    [chunk async for chunk in provider.synthesize("раз", Lang.RU)]
    [chunk async for chunk in provider.synthesize("два", Lang.EN)]
    assert len(loads) == 1  # singleton: 2.5 GB VRAM второй раз не выделяем


async def test_xtts_inference_error_becomes_tts_error(
    monkeypatch: pytest.MonkeyPatch, voices_dir: Path
) -> None:
    monkeypatch.setattr(XttsProvider, "_load_model", lambda self: StubXttsModel(fail=True))
    provider = XttsProvider(TtsConfig(voices_dir=voices_dir), VoiceStore(voices_dir))
    with pytest.raises(TtsError):
        [chunk async for chunk in provider.synthesize("привет", Lang.RU)]


async def test_xtts_early_break_does_not_hang(
    stub_xtts: tuple[XttsProvider, StubXttsModel, list[int]],
) -> None:
    """Потребитель ушёл после первого чанка — генератор закрывается без зависаний."""
    provider, _, _ = stub_xtts
    stream = provider.synthesize("привет", Lang.RU)
    first = await anext(stream)
    assert first.size == 480
    await stream.aclose()


async def test_xtts_rejects_kk_and_empty_text(
    stub_xtts: tuple[XttsProvider, StubXttsModel, list[int]],
) -> None:
    provider, model, _ = stub_xtts
    with pytest.raises(TtsError):
        [chunk async for chunk in provider.synthesize("мәтін", Lang.KK)]
    assert [chunk async for chunk in provider.synthesize("   ", Lang.RU)] == []
    assert model.calls == []


async def test_xtts_unknown_voice_id_raises(
    stub_xtts: tuple[XttsProvider, StubXttsModel, list[int]],
) -> None:
    provider, _, _ = stub_xtts
    with pytest.raises(VoiceNotFoundError):
        [chunk async for chunk in provider.synthesize("привет", Lang.RU, "v_deadbeef")]


def test_xtts_checkpoint_missing_message(tmp_path: Path) -> None:
    from engine.tts.xtts import _find_checkpoint_dir

    with pytest.raises(TtsError, match="download_models"):
        _find_checkpoint_dir(tmp_path / "models", "coqui/XTTS-v2")


def test_xtts_finds_checkpoint_dir(tmp_path: Path) -> None:
    from engine.tts.xtts import _find_checkpoint_dir

    checkpoint = tmp_path / "xtts"
    checkpoint.mkdir()
    for name in ("config.json", "model.pth"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    assert _find_checkpoint_dir(tmp_path, "coqui/XTTS-v2") == checkpoint


# --- конфигурация ----------------------------------------------------------


def test_config_defaults() -> None:
    config = TtsConfig()
    assert config.device is Device.AUTO
    assert config.backend is Backend.REAL
    assert config.kk_backend is KkBackend.MMS
    assert config.xtts_model_id == "coqui/XTTS-v2"
    assert config.kk_model_id == "facebook/mms-tts-kaz"
    assert config.voices_dir.name == "voices"
    assert CHUNK_MS_MIN <= config.chunk_ms <= CHUNK_MS_MAX


def test_config_from_env(tmp_path: Path) -> None:
    env = {
        ENV_DEVICE: "cpu",
        ENV_BACKEND: "fake",
        ENV_KK_BACKEND: "kazakhtts2",
        ENV_MODELS_DIR: str(tmp_path / "models"),
        ENV_VOICES_DIR: str(tmp_path / "voices"),
    }
    config = TtsConfig.from_env(env)

    assert config.device is Device.CPU
    assert config.resolve_device() is Device.CPU
    assert config.backend is Backend.FAKE
    assert config.kk_backend is KkBackend.KAZAKHTTS2
    assert config.models_dir == tmp_path / "models"
    assert config.voices_dir == tmp_path / "voices"


def test_config_from_empty_env_is_default() -> None:
    assert TtsConfig.from_env({}) == TtsConfig()


def test_config_rejects_unknown_env_value() -> None:
    with pytest.raises(TtsError):
        TtsConfig.from_env({ENV_BACKEND: "turbo"})
    with pytest.raises(TtsError):
        TtsConfig.from_env({ENV_DEVICE: "tpu"})


def test_config_rejects_bad_chunk_ms() -> None:
    with pytest.raises(TtsError):
        TtsConfig(chunk_ms=5)
    with pytest.raises(TtsError):
        TtsConfig(chunk_ms=500)


def test_create_tts_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(ENV_BACKEND, "fake")
    monkeypatch.setenv(ENV_VOICES_DIR, str(tmp_path / "voices"))
    router = create_tts()
    assert router.config.backend is Backend.FAKE
    assert router.voices.root == tmp_path / "voices"


# --- тесты с реальными моделями (RT_REAL_MODELS=1) -------------------------


@pytest.fixture
def real_config() -> TtsConfig:
    return TtsConfig.from_env()


@pytest.mark.real_models
async def test_real_xtts_synthesizes_short_phrase(real_config: TtsConfig, tmp_path: Path) -> None:
    """XTTS-v2: загрузка чекпоинта + синтез короткой фразы встроенным голосом."""
    from engine.tts.xtts import XttsProvider

    provider = XttsProvider(real_config, VoiceStore(tmp_path / "voices"))
    await provider.warmup()

    chunks = [chunk async for chunk in provider.synthesize("Привет, это тест.", Lang.RU)]
    assert chunks
    assert all(chunk.dtype == np.int16 for chunk in chunks)
    assert all(chunk.size <= OUTPUT_SAMPLE_RATE * CHUNK_MS_MAX // 1000 for chunk in chunks)
    total_ms = 1000 * sum(chunk.size for chunk in chunks) / OUTPUT_SAMPLE_RATE
    assert total_ms > 300
    assert np.abs(np.concatenate(chunks)).max() > 0


@pytest.mark.real_models
async def test_real_mms_kk_synthesizes(real_config: TtsConfig) -> None:
    """MMS-TTS kk: синтез на CPU, 16 kHz -> 24 kHz."""
    from engine.tts.kk_mms import MmsKazakhTts

    provider = MmsKazakhTts(real_config)
    await provider.warmup()

    chunks = [chunk async for chunk in provider.synthesize("Сәлеметсіз бе!", Lang.KK)]
    assert chunks
    assert all(chunk.dtype == np.int16 for chunk in chunks)
    total_ms = 1000 * sum(chunk.size for chunk in chunks) / OUTPUT_SAMPLE_RATE
    assert total_ms > 200


@pytest.mark.real_models
async def test_real_xtts_voice_cloning(real_config: TtsConfig, tmp_path: Path) -> None:
    """Полный путь: профиль голоса -> латенты -> синтез клоном."""
    from engine.tts.xtts import XttsProvider

    store = VoiceStore(tmp_path / "voices")
    provider = XttsProvider(real_config, store)
    sample = write_tone(tmp_path / "sample.wav", 16.0)
    profile = store.create_voice(
        sample, name="test-clone", lang=Lang.RU, latents_fn=provider.compute_latents
    )
    assert profile.latents_path.is_file()

    chunks = [chunk async for chunk in provider.synthesize("Проверка клона.", Lang.RU, profile.id)]
    assert chunks


def test_shared_fixture_wav_is_resampled(tone_wav: Path) -> None:
    """Общая фикстура conftest (16 kHz) читается и ресемплится в 24 kHz."""
    samples, sample_rate = read_wav(tone_wav)
    assert sample_rate == 16_000
    resampled = resample_pcm(samples, sample_rate, OUTPUT_SAMPLE_RATE)
    assert resampled.size == pytest.approx(samples.size * 1.5, abs=1)
    assert dominant_hz(resampled, OUTPUT_SAMPLE_RATE) == pytest.approx(440.0, abs=10.0)
