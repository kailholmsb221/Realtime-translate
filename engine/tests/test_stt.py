"""Тесты модуля stt: сегментация, события, конфиг, фабрики (зона агента B).

Реальные модели (faster-whisper, Silero) не поднимаются: используются фейковые
реализации тех же протоколов (CLAUDE.md, технический стандарт). Тесты с
реальными моделями помечены ``@pytest.mark.real_models`` и пропускаются без
``RT_REAL_MODELS=1``.

Запуск::

    python -m pytest engine/tests/test_stt.py
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from engine.contracts.events import (
    EVENT_STT_FINAL,
    EVENT_STT_PARTIAL,
    Envelope,
    Lang,
    SttFinal,
    SttPartial,
    parse_event,
    to_json,
    validate_envelope,
)
from engine.stt import (
    EnergyVad,
    FakeTranscriber,
    SttConfig,
    SttEngine,
    create_engine,
    create_transcriber,
    create_vad,
)
from engine.stt.base import (
    SAMPLE_RATE,
    Int16Array,
    TranscriptResult,
    VadEventKind,
    chunk_samples,
    samples_to_ms,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
if str(FIXTURES_DIR) not in sys.path:  # генератор фикстур лежит рядом с ними
    sys.path.insert(0, str(FIXTURES_DIR))

from stt_fixtures import (  # noqa: E402  — после правки sys.path
    SPEECH_PATTERN_FILENAME,
    make_stt_fixtures,
    silence,
    tone,
)

WavReader = Callable[[Path], tuple[np.ndarray, int]]

CHUNK_MS = 30
"""Чанк 30 мс — как отдаёт audio_io."""


# --- вспомогательное -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Chunk:
    """Аудиочанк для движка: утиная типизация ``pcm`` + ``ts_ms``."""

    pcm: np.ndarray
    ts_ms: int


@dataclass(frozen=True, slots=True)
class BytesChunk:
    """Чанк в стиле audio_io: ``pcm`` — байты, сэмплы в свойстве."""

    pcm: bytes
    ts_ms: int

    @property
    def samples(self) -> Int16Array:
        return np.frombuffer(self.pcm, dtype="<i2").astype(np.int16)


def iter_chunks(samples: np.ndarray, chunk_ms: int = CHUNK_MS) -> list[Chunk]:
    """Нарезать массив на чанки по ``chunk_ms``."""
    step = chunk_ms * SAMPLE_RATE // 1000
    return [
        Chunk(pcm=samples[offset : offset + step], ts_ms=samples_to_ms(offset))
        for offset in range(0, samples.size, step)
    ]


async def async_source(samples: np.ndarray, chunk_ms: int = CHUNK_MS) -> AsyncIterator[Chunk]:
    """Асинхронный источник чанков (даёт event loop шанс переключиться)."""
    for chunk in iter_chunks(samples, chunk_ms):
        yield chunk
        await asyncio.sleep(0)


def fast_config(**overrides: object) -> SttConfig:
    """Конфиг для тестов: короткие интервалы, детерминированный порог энергии."""
    params: dict[str, object] = {
        "energy_threshold": 0.02,
        "min_speech_ms": 250,
        "min_silence_ms": 500,
        "partial_interval_ms": 200,
        "max_utterance_ms": 15_000,
        "language": None,
        "fallback_lang": "ru",
    }
    params.update(overrides)
    return SttConfig(**params)  # type: ignore[arg-type]


async def collect(engine: SttEngine, samples: np.ndarray, **kwargs: object) -> list[Envelope]:
    """Прогнать массив через движок и собрать все события."""
    return [
        envelope
        async for envelope in engine.run(async_source(samples), **kwargs)  # type: ignore[arg-type]
    ]


@pytest.fixture(scope="session")
def speech_pattern_wav() -> Path:
    """WAV «тон — тишина — тон — тишина»: две фразы с известными границами."""
    make_stt_fixtures(FIXTURES_DIR)
    path: Path = FIXTURES_DIR / SPEECH_PATTERN_FILENAME
    return path


# --- сегментация -----------------------------------------------------------


def test_speech_pattern_fixture_format(speech_pattern_wav: Path, read_wav: WavReader) -> None:
    """Фикстура создаётся и лежит в формате движка: 16 kHz mono int16, 4 с."""
    samples, sample_rate = read_wav(speech_pattern_wav)
    assert sample_rate == SAMPLE_RATE
    assert samples.dtype == np.int16
    assert samples_to_ms(samples.size) == 4000


def test_energy_vad_segments_two_utterances(speech_pattern_wav: Path, read_wav: WavReader) -> None:
    """«Тон — тишина — тон» даёт ровно 2 фразы с ожидаемыми таймкодами."""
    samples, _ = read_wav(speech_pattern_wav)
    vad = EnergyVad(threshold=0.02, min_speech_ms=250, min_silence_ms=500)

    events = []
    for chunk in iter_chunks(samples):
        events.extend(vad.process(chunk.pcm))
    events.extend(vad.flush())

    kinds = [event.kind for event in events]
    assert kinds == [
        VadEventKind.SPEECH_START,
        VadEventKind.SPEECH_END,
        VadEventKind.SPEECH_START,
        VadEventKind.SPEECH_END,
    ]

    starts = [event.ts_ms for event in events if event.is_start]
    ends = [event.ts_ms for event in events if event.is_end]
    assert starts[0] == pytest.approx(0, abs=60)
    assert ends[0] == pytest.approx(1000, abs=60)
    assert starts[1] == pytest.approx(2000, abs=60)
    assert ends[1] == pytest.approx(3000, abs=60)


def test_energy_vad_reset_restarts_timeline() -> None:
    """После reset() таймлайн и состояние начинаются заново."""
    vad = EnergyVad(threshold=0.02, min_speech_ms=250, min_silence_ms=500)
    vad.process(tone(1000))
    assert vad.flush()  # фраза была открыта

    vad.reset()
    events = vad.process(tone(1000))
    assert [event.ts_ms for event in events if event.is_start] == [0]


def test_energy_vad_ignores_short_blip() -> None:
    """Всплеск короче min_speech_ms фразой не считается."""
    vad = EnergyVad(threshold=0.02, min_speech_ms=250, min_silence_ms=500)
    events = vad.process(np.concatenate([silence(200), tone(90), silence(800)]))
    assert events == []


# --- движок ----------------------------------------------------------------


async def test_engine_emits_partial_then_final(
    speech_pattern_wav: Path, read_wav: WavReader
) -> None:
    """Движок выдаёт partial во время речи и final по её концу."""
    samples, _ = read_wav(speech_pattern_wav)
    config = fast_config()
    transcriber = FakeTranscriber("привет мир", lang="ru")
    engine = SttEngine(EnergyVad.from_config(config), transcriber, config)

    events = await collect(engine, samples, stream="in", lang="ru")
    types = [envelope.type for envelope in events]

    assert types.count(EVENT_STT_FINAL) == 2
    assert EVENT_STT_PARTIAL in types
    assert types.index(EVENT_STT_PARTIAL) < types.index(EVENT_STT_FINAL)
    assert engine.last_latency_ms is not None


async def test_engine_final_timecodes_match_segments(
    speech_pattern_wav: Path, read_wav: WavReader
) -> None:
    """Таймкоды stt.final соответствуют границам фраз в фикстуре."""
    samples, _ = read_wav(speech_pattern_wav)
    config = fast_config()
    engine = SttEngine(EnergyVad.from_config(config), FakeTranscriber("фраза", lang="ru"), config)

    events = await collect(engine, samples, stream="in", lang="ru")
    finals = [e.payload for e in events if isinstance(e.payload, SttFinal)]

    assert len(finals) == 2
    assert finals[0].t_start_ms == pytest.approx(0, abs=60)
    assert finals[0].t_end_ms == pytest.approx(1000, abs=60)
    assert finals[1].t_start_ms == pytest.approx(2000, abs=60)
    assert finals[1].t_end_ms == pytest.approx(3000, abs=60)
    assert all(f.t_start_ms < f.t_end_ms for f in finals)


async def test_engine_events_pass_contract_validation(
    speech_pattern_wav: Path, read_wav: WavReader
) -> None:
    """Каждое событие движка проходит валидацию схемами контрактов."""
    samples, _ = read_wav(speech_pattern_wav)
    config = fast_config()
    engine = SttEngine(EnergyVad.from_config(config), FakeTranscriber("привет", lang="ru"), config)

    events = await collect(engine, samples, stream="out", lang="en")
    assert events

    for envelope in events:
        validate_envelope(envelope.to_dict())
        restored = parse_event(to_json(envelope))
        assert restored.type == envelope.type
        assert restored.payload == envelope.payload
        payload = envelope.payload
        assert isinstance(payload, (SttPartial, SttFinal))
        assert payload.stream.value == "out"
        assert payload.lang is Lang.EN


async def test_max_utterance_forces_final() -> None:
    """Непрерывная речь режется по max_utterance_ms."""
    samples = np.concatenate([tone(3000), silence(700)])
    config = fast_config(max_utterance_ms=1000, partial_interval_ms=0)
    engine = SttEngine(
        EnergyVad.from_config(config), FakeTranscriber("длинная речь", lang="ru"), config
    )

    events = await collect(engine, samples, stream="in", lang="ru")
    finals = [e.payload for e in events if isinstance(e.payload, SttFinal)]

    assert len(finals) >= 3, "фраза 3 с при max_utterance_ms=1000 должна дать 3 final"
    for final in finals:
        assert final.t_end_ms - final.t_start_ms <= 1100
    assert finals[0].t_start_ms == pytest.approx(0, abs=60)
    # куски идут подряд, без дыр и перекрытий
    for previous, current in pairwise(finals):
        assert current.t_start_ms == pytest.approx(previous.t_end_ms, abs=60)


async def test_engine_runs_one_transcription_at_a_time(
    speech_pattern_wav: Path, read_wav: WavReader
) -> None:
    """Пока считается предыдущая транскрипция, тик partial пропускается."""

    class CountingTranscriber:
        """Медленный распознаватель, считающий одновременные вызовы."""

        def __init__(self) -> None:
            self.calls = 0
            self.max_concurrent = 0
            self._active = 0
            self._lock = threading.Lock()

        def transcribe(self, pcm_int16_16k: Int16Array, lang: str | None) -> TranscriptResult:
            with self._lock:
                self.calls += 1
                self._active += 1
                self.max_concurrent = max(self.max_concurrent, self._active)
            time.sleep(0.02)
            with self._lock:
                self._active -= 1
            return TranscriptResult(
                text="текст", lang=lang or "ru", duration_ms=samples_to_ms(pcm_int16_16k.size)
            )

    samples, _ = read_wav(speech_pattern_wav)
    config = fast_config(partial_interval_ms=30)
    transcriber = CountingTranscriber()
    engine = SttEngine(EnergyVad.from_config(config), transcriber, config)

    await collect(engine, samples, stream="in", lang="ru")

    assert transcriber.calls > 2
    assert transcriber.max_concurrent == 1


async def test_engine_skips_empty_transcription() -> None:
    """Пустой текст не превращается в событие."""
    samples = np.concatenate([tone(800), silence(800)])
    config = fast_config()
    engine = SttEngine(EnergyVad.from_config(config), FakeTranscriber("", lang="ru"), config)

    assert await collect(engine, samples, stream="in", lang="ru") == []


async def test_engine_falls_back_to_contract_lang() -> None:
    """Язык вне ru/en/kk заменяется на fallback_lang из конфига."""
    samples = np.concatenate([tone(800), silence(800)])
    config = fast_config(fallback_lang="ru")
    engine = SttEngine(EnergyVad.from_config(config), FakeTranscriber("hallo", lang="de"), config)

    events = await collect(engine, samples, stream="in")
    payloads = [e.payload for e in events if isinstance(e.payload, (SttPartial, SttFinal))]
    assert payloads
    assert all(payload.lang is Lang.RU for payload in payloads)


async def test_engine_accepts_audio_io_style_chunks() -> None:
    """Чанк с pcm в байтах (как в audio_io) тоже принимается."""
    samples = np.concatenate([tone(800), silence(800)])
    config = fast_config(partial_interval_ms=0)
    engine = SttEngine(EnergyVad.from_config(config), FakeTranscriber("байты", lang="ru"), config)

    async def source() -> AsyncIterator[BytesChunk]:
        for chunk in iter_chunks(samples):
            yield BytesChunk(pcm=chunk.pcm.astype("<i2").tobytes(), ts_ms=chunk.ts_ms)
            await asyncio.sleep(0)

    events = [envelope async for envelope in engine.run(source(), stream="in", lang="ru")]
    assert [e.type for e in events] == [EVENT_STT_FINAL]


async def test_engine_survives_transcriber_failure() -> None:
    """Падение распознавателя не роняет поток событий."""

    class BrokenTranscriber:
        def transcribe(self, pcm_int16_16k: Int16Array, lang: str | None) -> TranscriptResult:
            raise RuntimeError("модель умерла")

    samples = np.concatenate([tone(800), silence(800)])
    config = fast_config(partial_interval_ms=0)
    engine = SttEngine(EnergyVad.from_config(config), BrokenTranscriber(), config)

    assert await collect(engine, samples, stream="in", lang="ru") == []


def test_chunk_samples_duck_typing() -> None:
    """chunk_samples понимает и numpy-, и bytes-чанки, иначе TypeError."""
    pcm = tone(30)
    assert np.array_equal(chunk_samples(Chunk(pcm=pcm, ts_ms=0)), pcm)
    assert np.array_equal(chunk_samples(BytesChunk(pcm=pcm.astype("<i2").tobytes(), ts_ms=0)), pcm)
    with pytest.raises(TypeError):
        chunk_samples(object())


# --- конфигурация и фабрики ------------------------------------------------


def test_config_defaults_follow_architecture() -> None:
    """Значения по умолчанию — из ARCHITECTURE.md 4.2."""
    config = SttConfig()
    assert config.model_size == "small"
    assert config.compute_type == "int8_float16"
    assert config.device == "auto"
    assert (config.min_speech_ms, config.min_silence_ms) == (250, 500)
    assert (config.max_utterance_ms, config.partial_interval_ms) == (15_000, 1_000)


def test_config_from_env() -> None:
    """Конфиг читается из переменных окружения."""
    env = {
        "RT_STT_MODEL": "small",
        "RT_STT_DEVICE": "cpu",
        "RT_STT_COMPUTE": "int8",
        "RT_STT_KK_MODEL": "models/issai-whisper-kk",
        "RT_MODELS_DIR": "/opt/models",
        "RT_STT_LANG": "ru",
        "RT_STT_PARTIAL_INTERVAL_MS": "500",
    }
    config = SttConfig.from_env(env)

    assert config.model_size == "small"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.language == "ru"
    assert config.models_dir == Path("/opt/models")
    assert config.partial_interval_ms == 500
    assert config.model_overrides == {"kk": "models/issai-whisper-kk"}


def test_config_from_empty_env_uses_defaults() -> None:
    """Пустое окружение не ломает конфиг."""
    config = SttConfig.from_env({})
    assert (config.model_size, config.device, config.language) == ("small", "auto", None)
    assert config.models_dir is None
    assert config.model_overrides == {}


def test_config_from_env_overrides_win() -> None:
    """Явные аргументы перебивают окружение."""
    config = SttConfig.from_env({"RT_STT_DEVICE": "cpu"}, device="cuda")
    assert config.device == "cuda"


def test_config_rejects_oversized_model() -> None:
    """Модель крупнее small запрещена бюджетом VRAM (CLAUDE.md, закон 4)."""
    with pytest.raises(ValueError, match="VRAM"):
        SttConfig(model_size="large-v3")
    with pytest.raises(ValueError, match="VRAM"):
        SttConfig.from_env({"RT_STT_MODEL": "medium"})


def test_config_rejects_bad_values() -> None:
    """Неразбираемые значения окружения и плохое устройство отбрасываются."""
    with pytest.raises(ValueError):
        SttConfig.from_env({"RT_STT_DEVICE": "tpu"})
    with pytest.raises(ValueError):
        SttConfig.from_env({"RT_STT_MIN_SPEECH_MS": "около трёхсот"})


def test_kk_model_override_selects_path() -> None:
    """Для kk берётся файнтюн, для остальных языков — базовая модель."""
    config = SttConfig.from_env({"RT_STT_KK_MODEL": "models/issai-whisper-kk"})
    assert config.model_for("kk") == "models/issai-whisper-kk"
    assert config.model_for("KK") == "models/issai-whisper-kk"
    assert config.model_for("ru") == "small"
    assert config.model_for(None) == "small"


def test_transcriber_reports_kk_override_and_cpu_fallback() -> None:
    """Без CUDA распознаватель уходит на cpu/int8, kk-модель — из конфига."""
    config = SttConfig.from_env(
        {"RT_STT_KK_MODEL": "models/issai-whisper-kk", "RT_STT_DEVICE": "auto"}
    )
    transcriber = create_transcriber(config, "faster_whisper")

    assert transcriber.model_name("kk") == "models/issai-whisper-kk"  # type: ignore[attr-defined]
    assert transcriber.model_name("ru") == "small"  # type: ignore[attr-defined]
    if transcriber.device == "cpu":  # type: ignore[attr-defined]
        assert transcriber.compute_type == "int8"  # type: ignore[attr-defined]


def test_factories_build_expected_backends() -> None:
    """Фабрики отдают нужные реализации и ругаются на неизвестный бэкенд."""
    config = fast_config()

    assert isinstance(create_vad(config, "energy"), EnergyVad)
    assert isinstance(create_transcriber(config, "fake"), FakeTranscriber)

    engine = create_engine(config, backend="fake", vad_backend="energy")
    assert isinstance(engine, SttEngine)
    assert isinstance(engine.vad, EnergyVad)
    assert isinstance(engine.transcriber, FakeTranscriber)

    with pytest.raises(ValueError, match="backend VAD"):
        create_vad(config, "magic")
    with pytest.raises(ValueError, match="backend"):
        create_transcriber(config, "magic")


def test_fake_transcriber_by_duration() -> None:
    """FakeTranscriber умеет отдавать разный текст по длительности буфера."""
    transcriber = FakeTranscriber("коротко", by_duration={0: "коротко", 1000: "длинно"})

    assert transcriber.transcribe(tone(300), "ru").text == "коротко"
    assert transcriber.transcribe(tone(1500), "ru").text == "длинно"
    assert transcriber.calls == [(300, "ru"), (1500, "ru")]


def test_partial_and_final_payload_shapes() -> None:
    """Датаклассы контрактов собираются из результатов stt без сюрпризов."""
    partial = SttPartial.from_payload({"stream": "in", "lang": "kk", "text": "сәлем"})
    final = SttFinal.from_payload(
        {"stream": "out", "lang": "ru", "text": "привет", "t_start_ms": 0, "t_end_ms": 900}
    )
    assert partial.lang is Lang.KK
    assert final.t_end_ms == 900


# --- реальные модели (RT_REAL_MODELS=1) ------------------------------------


@pytest.mark.real_models
def test_real_faster_whisper_returns_result(speech_pattern_wav: Path, read_wav: WavReader) -> None:
    """Реальная модель грузится и отдаёт TranscriptResult.

    Речевого WAV в фикстурах нет (синтезировать речь мы не можем), поэтому
    проверяется только, что модель поднимается и возвращает контрактный
    результат — текст на тоне может быть любым, в том числе пустым.
    """
    config = SttConfig.from_env()
    transcriber = create_transcriber(config, "faster_whisper")
    samples, _ = read_wav(speech_pattern_wav)

    result = transcriber.transcribe(samples, "ru")

    assert isinstance(result, TranscriptResult)
    assert result.duration_ms == pytest.approx(4000, abs=50)
    assert isinstance(result.text, str)


@pytest.mark.real_models
def test_real_silero_vad_processes_stream(speech_pattern_wav: Path, read_wav: WavReader) -> None:
    """Silero VAD грузится и переваривает поток чанков по 30 мс."""
    config = SttConfig.from_env()
    vad = create_vad(config, "silero")
    samples, _ = read_wav(speech_pattern_wav)

    events = []
    for chunk in iter_chunks(samples):
        events.extend(vad.process(chunk.pcm))

    # На синтетическом тоне Silero имеет право не услышать речь — проверяем,
    # что события корректны по форме, а не их количество.
    assert all(event.ts_ms >= 0 for event in events)
    vad.reset()
