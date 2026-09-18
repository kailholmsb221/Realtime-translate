"""Тесты конвейера оркестратора на фейковых бэкендах.

Фикстура ``stt_speech_pattern.wav`` — «тон — тишина — тон — тишина», то есть
ровно две фразы. Отсюда ожидания: 2 × ``stt.final``, 2 × ``translation.ready``
в том же порядке и с ``ref_utterance_id`` вставленных строк, 2 ×
``metrics.latency``, ненулевой WAV на выходе и записи в БД.

Реальные модели не поднимаются: EnergyVad + FakeTranscriber, FakeProvider,
FakeTts, FakeSource/FakeSink.
"""

from __future__ import annotations

import asyncio
import sys
import wave
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import pytest

from engine.audio_io import FakeSink, FakeSource
from engine.audio_io.base import AudioChunk, AudioIOError, BaseAudioSource
from engine.contracts.events import (
    EVENT_METRICS_LATENCY,
    EVENT_STT_FINAL,
    EVENT_TRANSLATION_READY,
    Envelope,
    Lang,
    MetricsLatency,
    Stream,
    SttFinal,
    TranslationReady,
    validate_envelope,
)
from engine.orchestrator.autoclone import (
    AUTO_NAME_PREFIX,
    AutoCloneConfig,
    AutoCloner,
    supports_auto_clone,
)
from engine.orchestrator.bus import EventBus
from engine.orchestrator.db import Database
from engine.orchestrator.pipeline import Pipeline, WavRecorder
from engine.stt.base import SttConfig
from engine.stt.engine import SttEngine
from engine.stt.fake import EnergyVad, FakeTranscriber
from engine.translate.base import TranslationResult
from engine.translate.fake import FakeProvider
from engine.translate.service import TranslationService
from engine.tts.audio import Pcm16
from engine.tts.base import Backend, TtsConfig
from engine.tts.router import TtsRouter
from engine.tts.voices import VoiceStore

pytestmark = pytest.mark.asyncio

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPEECH_PATTERN = FIXTURES_DIR / "stt_speech_pattern.wav"
if str(FIXTURES_DIR) not in sys.path:  # генератор фикстур лежит рядом с ними
    sys.path.insert(0, str(FIXTURES_DIR))

from stt_fixtures import silence, tone, write_wav  # noqa: E402 — после правки sys.path

#: Тестовые пороги автоклона: VoiceStore принимает сэмпл не короче 5 секунд.
AUTO_MIN_SECONDS = 6.0
AUTO_MAX_SECONDS = 10.0


def stt_config() -> SttConfig:
    """Быстрая конфигурация STT для фикстуры «тон — тишина — тон»."""
    return SttConfig(
        energy_threshold=0.02,
        min_speech_ms=250,
        min_silence_ms=500,
        partial_interval_ms=400,
        max_utterance_ms=5_000,
    )


def make_engine(config: SttConfig | None = None, text: str = "тестовая фраза") -> SttEngine:
    """Движок STT на фейковых VAD и распознавателе."""
    cfg = config or stt_config()
    return SttEngine(EnergyVad.from_config(cfg), FakeTranscriber(text, lang="en"), cfg)


def make_router() -> TtsRouter:
    """Маршрутизатор TTS в фейковом режиме (тон вместо речи)."""
    return TtsRouter(TtsConfig(backend=Backend.FAKE, chunk_ms=20))


def read_sample(path: Path) -> tuple[np.ndarray, int]:
    """Прочитать WAV профиля голоса (VoiceStore пишет 24 kHz mono int16)."""
    with wave.open(str(path), "rb") as handle:
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    return np.frombuffer(frames, dtype="<i2"), rate


def long_speech_wav(path: Path, phrases: int = 6, speech_ms: int = 2_000) -> Path:
    """WAV «тон — тишина» ×N: набирается достаточно речи для автоклона."""
    parts = []
    for index in range(phrases):
        parts.append(tone(speech_ms, freq_hz=440.0 + 40.0 * index))
        parts.append(silence(1_000))
    write_wav(path, np.concatenate(parts))
    return path


class _RecordingRouter(TtsRouter):
    """Роутер TTS, запоминающий, каким голосом синтезировалась каждая фраза."""

    def __init__(self) -> None:
        super().__init__(TtsConfig(backend=Backend.FAKE, chunk_ms=20))
        self.voice_ids: list[str | None] = []

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        self.voice_ids.append(voice_id)
        async for chunk in super().synthesize(text, lang, voice_id):
            yield chunk


def make_cloner(
    tmp_path: Path,
    *,
    enabled: bool = True,
    keep: bool = False,
    lang: Lang = Lang.EN,
    session_id: int | None = 1,
    db: Database | None = None,
) -> AutoCloner:
    """Автоклон с тестовыми порогами и хранилищем голосов в каталоге теста."""
    config = AutoCloneConfig(
        enabled=enabled,
        min_seconds=AUTO_MIN_SECONDS,
        max_seconds=AUTO_MAX_SECONDS,
        keep=keep,
    )
    return AutoCloner(
        config=config,
        voices=VoiceStore(tmp_path / "voices"),
        lang=lang,
        session_id=session_id,
        db=db,
    )


class _BoomSource(BaseAudioSource):
    """Источник, который обрывается ошибкой устройства посреди потока."""

    async def _open(self) -> None:
        return None

    async def _close(self) -> None:
        return None

    def _iter_chunks(self) -> AsyncIterator[AudioChunk]:
        async def gen() -> AsyncIterator[AudioChunk]:
            for index in range(3):
                yield AudioChunk.from_array(np.zeros(480, dtype=np.int16), index * 30)
            raise AudioIOError("устройство пропало")

        return gen()


class _BoomProvider:
    """Провайдер перевода, который всегда падает (проверка устойчивости)."""

    def warmup(self) -> None:
        return None

    async def translate(
        self,
        text: str,
        src: Lang,
        dst: Lang,
        context: Sequence[str] = (),
    ) -> TranslationResult:
        raise RuntimeError("перевод недоступен")


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """База движка в каталоге теста."""
    database = Database(tmp_path / "translator.db")
    await database.init()
    try:
        yield database
    finally:
        await database.close()


def drain(queue: asyncio.Queue[Envelope]) -> list[Envelope]:
    """Забрать все накопленные события подписки."""
    out: list[Envelope] = []
    while True:
        try:
            out.append(queue.get_nowait())
        except asyncio.QueueEmpty:
            return out


def of_type(events: Sequence[Envelope], type_name: str) -> list[Envelope]:
    """Отфильтровать события по типу с сохранением порядка."""
    return [event for event in events if event.type == type_name]


async def run_pipeline(
    *,
    db: Database | None = None,
    session_id: int | None = None,
    recorder: WavRecorder | None = None,
    out_wav: Path | None = None,
    provider: object | None = None,
    stream: Stream = Stream.IN,
    source_wav: Path | None = None,
    auto_cloner: AutoCloner | None = None,
    tts_router: TtsRouter | None = None,
    lang_src: Lang = Lang.EN,
    lang_dst: Lang = Lang.RU,
    voice_id: str | None = None,
) -> tuple[Pipeline, list[Envelope], FakeSink]:
    """Прогнать фикстуру через конвейер и вернуть его, события и приёмник."""
    bus = EventBus()
    queue = bus.subscribe()
    source = FakeSource.from_wav(source_wav if source_wav is not None else SPEECH_PATTERN)
    sink = FakeSink(path=out_wav)
    service = TranslationService(provider if provider is not None else FakeProvider())  # type: ignore[arg-type]
    pipeline = Pipeline(
        stream=stream,
        lang_src=lang_src,
        lang_dst=lang_dst,
        voice_id=voice_id,
        source=source,
        sink=sink,
        stt_engine=make_engine(),
        translation_service=service,
        tts_router=tts_router if tts_router is not None else make_router(),
        bus=bus,
        db=db,
        session_id=session_id,
        recorder=recorder,
        auto_cloner=auto_cloner,
    )
    await pipeline.run()
    await sink.drain()
    await sink.close()
    events = drain(queue)
    bus.unsubscribe(queue)
    return pipeline, events, sink


async def test_pipeline_emits_two_utterances(tmp_path: Path) -> None:
    """Из фикстуры получаются 2 фразы, 2 перевода и 2 метрики — в этом порядке."""
    _, events, sink = await run_pipeline(out_wav=tmp_path / "out.wav")

    finals = of_type(events, EVENT_STT_FINAL)
    ready = of_type(events, EVENT_TRANSLATION_READY)
    metrics = of_type(events, EVENT_METRICS_LATENCY)
    assert len(finals) == 2
    assert len(ready) == 2
    assert len(metrics) == 2

    order = [event.type for event in events if event.type != "stt.partial"]
    assert order == [
        EVENT_STT_FINAL,
        EVENT_TRANSLATION_READY,
        EVENT_METRICS_LATENCY,
        EVENT_STT_FINAL,
        EVENT_TRANSLATION_READY,
        EVENT_METRICS_LATENCY,
    ]
    assert sink.n_frames > 0


async def test_pipeline_events_match_contracts(tmp_path: Path) -> None:
    """Каждое событие конвейера проходит валидацию схемами контрактов."""
    _, events, _ = await run_pipeline(out_wav=tmp_path / "out.wav")
    assert events
    for envelope in events:
        validate_envelope(envelope.to_dict())


async def test_translation_refers_to_inserted_rows(db: Database, tmp_path: Path) -> None:
    """``ref_utterance_id`` — id вставленных строк, порядок совпадает с фразами."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    _, events, _ = await run_pipeline(db=db, session_id=session_id, out_wav=tmp_path / "out.wav")

    rows = await db.list_utterances(session_id)
    assert len(rows) == 2

    ready = [event.payload for event in of_type(events, EVENT_TRANSLATION_READY)]
    refs = [payload.ref_utterance_id for payload in ready if isinstance(payload, TranslationReady)]
    assert refs == [row["id"] for row in rows]

    for row in rows:
        assert row["translation"] == f"[ru] {row['text']}"
        assert row["translation_lang"] == "ru"
        assert row["speaker"] == "in"
        assert row["lang"] == "en"


async def test_metrics_are_non_negative(tmp_path: Path) -> None:
    """``metrics.latency`` не отрицательные (в файловом режиме время «сжато»)."""
    _, events, _ = await run_pipeline(out_wav=tmp_path / "out.wav")
    payloads = [event.payload for event in of_type(events, EVENT_METRICS_LATENCY)]
    assert payloads
    for payload in payloads:
        assert isinstance(payload, MetricsLatency)
        assert payload.stream is Stream.IN
        assert min(payload.stt_ms, payload.mt_ms, payload.tts_ms, payload.total_ms) >= 0


async def test_sink_receives_tts_audio(tmp_path: Path) -> None:
    """Синтез уезжает в приёмник и сохраняется в WAV 16 kHz."""
    out = tmp_path / "out.wav"
    _, _, sink = await run_pipeline(out_wav=out)
    assert sink.n_frames > 0
    assert out.is_file()
    with wave.open(str(out), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getframerate() == 16_000
        assert handle.getnframes() == sink.n_frames


async def test_recorder_writes_source_wav(tmp_path: Path) -> None:
    """Рекордер пишет исходный поток 16 kHz в WAV сессии."""
    recorder = WavRecorder(tmp_path / "recordings" / "1_in.wav")
    await run_pipeline(recorder=recorder, out_wav=tmp_path / "out.wav")

    assert recorder.path.is_file()
    with wave.open(str(recorder.path), "rb") as handle:
        assert handle.getframerate() == 16_000
        assert handle.getnchannels() == 1
        # Фикстура — 4 секунды; допускаем добивку последнего чанка нулями.
        assert handle.getnframes() >= 16_000 * 4


async def test_outbound_stream_translates_back(tmp_path: Path) -> None:
    """Поток ``out`` помечает реплики как речь пользователя."""
    _, events, _ = await run_pipeline(stream=Stream.OUT, out_wav=tmp_path / "out.wav")
    finals = [event.payload for event in of_type(events, EVENT_STT_FINAL)]
    assert finals
    for payload in finals:
        assert isinstance(payload, SttFinal)
        assert payload.stream is Stream.OUT


async def test_translation_failure_does_not_break_pipeline(db: Database, tmp_path: Path) -> None:
    """Падение перевода теряет одну фразу, но конвейер продолжает работать."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    pipeline, events, _ = await run_pipeline(
        db=db,
        session_id=session_id,
        out_wav=tmp_path / "out.wav",
        provider=_BoomProvider(),
    )

    assert len(of_type(events, EVENT_STT_FINAL)) == 2
    assert of_type(events, EVENT_TRANSLATION_READY) == []
    assert pipeline.stats.errors == 2
    rows = await db.list_utterances(session_id)
    assert len(rows) == 2
    assert all(row["translation"] is None for row in rows)


async def test_source_failure_does_not_leak_exception(tmp_path: Path) -> None:
    """Регрессия: обрыв источника не роняет задачу конвейера с исключением.

    В live-режиме задачу конвейера никто не ждёт до ``session.stop``, поэтому
    исключение от устройства (``AudioIOError`` по таймауту WASAPI) уходило в
    «никуда»: поток тихо умирал, движок продолжал считать сессию живой, а
    ошибка всплывала лишь как «Task exception was never retrieved».
    """
    bus = EventBus()
    queue = bus.subscribe()
    sink = FakeSink(path=tmp_path / "out.wav")
    pipeline = Pipeline(
        stream=Stream.IN,
        lang_src=Lang.EN,
        lang_dst=Lang.RU,
        voice_id=None,
        source=_BoomSource(),
        sink=sink,
        stt_engine=make_engine(),
        translation_service=TranslationService(FakeProvider()),
        tts_router=make_router(),
        bus=bus,
    )

    task = asyncio.create_task(pipeline.run())
    await asyncio.wait_for(task, timeout=5.0)

    assert task.exception() is None
    assert pipeline.stats.errors == 1
    bus.unsubscribe(queue)
    await sink.close()


async def test_pipeline_without_db_keeps_ref_none(tmp_path: Path) -> None:
    """Без БД (файловый режим) ``ref_utterance_id`` остаётся ``null``."""
    _, events, _ = await run_pipeline(out_wav=tmp_path / "out.wav")
    ready = [event.payload for event in of_type(events, EVENT_TRANSLATION_READY)]
    assert ready
    for payload in ready:
        assert isinstance(payload, TranslationReady)
        assert payload.ref_utterance_id is None
        assert payload.src_lang is Lang.EN
        assert payload.dst_lang is Lang.RU


# --- автоклон голоса собеседника ------------------------------------------


async def test_auto_clone_creates_profile_once(db: Database, tmp_path: Path) -> None:
    """Набрав нужные секунды речи, конвейер создаёт ровно один автопрофиль."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    cloner = make_cloner(tmp_path, db=db, session_id=session_id)

    pipeline, events, _ = await run_pipeline(
        db=db,
        session_id=session_id,
        out_wav=tmp_path / "out.wav",
        source_wav=long_speech_wav(tmp_path / "speech.wav"),
        auto_cloner=cloner,
    )

    assert len(of_type(events, EVENT_STT_FINAL)) >= 3
    profile = cloner.profile
    assert profile is not None
    assert profile.id.startswith("v_")
    assert profile.name == f"{AUTO_NAME_PREFIX}session{session_id}_in"
    assert profile.lang is Lang.EN
    assert profile.sample_path.is_file()

    # Профиль ровно один — повторно клон не запускается.
    assert len(cloner.voices.list()) == 1
    assert pipeline.voice_id == profile.id

    # И он же лежит в таблице voices как индекс для истории и UI.
    rows = await db.list_voices()
    assert [row["id"] for row in rows] == [profile.id]
    assert rows[0]["name"].startswith(AUTO_NAME_PREFIX)


async def test_auto_clone_switches_voice_for_next_phrases(tmp_path: Path) -> None:
    """До клона синтез идёт голосом по умолчанию, после — автопрофилем."""
    cloner = make_cloner(tmp_path)
    router = _RecordingRouter()

    await run_pipeline(
        out_wav=tmp_path / "out.wav",
        source_wav=long_speech_wav(tmp_path / "speech.wav"),
        auto_cloner=cloner,
        tts_router=router,
    )

    profile = cloner.profile
    assert profile is not None
    assert router.voice_ids[0] is None, "первая фраза — встроенный голос XTTS"
    assert router.voice_ids[-1] == profile.id, "после клона звучит голос собеседника"
    assert None in router.voice_ids and profile.id in router.voice_ids


async def test_auto_clone_language_rules() -> None:
    """Клон возможен только когда и сэмпл, и синтез — на ru/en (kk без клона)."""
    assert supports_auto_clone(Lang.EN, Lang.RU)
    assert supports_auto_clone(Lang.RU, Lang.EN)
    assert not supports_auto_clone(Lang.KK, Lang.RU)  # сэмпл на kk — XTTS не умеет
    assert not supports_auto_clone(Lang.RU, Lang.KK)  # синтез на kk — клон не нужен


async def test_auto_clone_disabled_by_config(tmp_path: Path) -> None:
    """``RT_AUTO_CLONE=0`` — речь не копится и профиль не создаётся."""
    cloner = make_cloner(tmp_path, enabled=False)

    pipeline, _, _ = await run_pipeline(
        out_wav=tmp_path / "out.wav",
        source_wav=long_speech_wav(tmp_path / "speech.wav"),
        auto_cloner=cloner,
    )

    assert cloner.profile is None
    assert cloner.collected_seconds == 0.0
    assert not cloner.started
    assert pipeline.voice_id is None
    assert cloner.voices.list() == []


async def test_auto_clone_cleanup_removes_profile(db: Database, tmp_path: Path) -> None:
    """При остановке сессии автопрофиль удаляется (KEEP=0) — с диска и из БД."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    cloner = make_cloner(tmp_path, db=db, session_id=session_id)

    await run_pipeline(
        db=db,
        session_id=session_id,
        out_wav=tmp_path / "out.wav",
        source_wav=long_speech_wav(tmp_path / "speech.wav"),
        auto_cloner=cloner,
    )
    profile = cloner.profile
    assert profile is not None
    voice_dir = profile.dir

    await cloner.cleanup()  # это делает Session.stop() по session.stop

    assert cloner.profile is None
    assert not voice_dir.exists()
    assert await db.list_voices() == []


async def test_auto_clone_cleanup_keeps_profile_when_asked(db: Database, tmp_path: Path) -> None:
    """``RT_AUTO_CLONE_KEEP=1`` — профиль переживает сессию."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    cloner = make_cloner(tmp_path, db=db, session_id=session_id, keep=True)

    await run_pipeline(
        db=db,
        session_id=session_id,
        out_wav=tmp_path / "out.wav",
        source_wav=long_speech_wav(tmp_path / "speech.wav"),
        auto_cloner=cloner,
    )
    profile = cloner.profile
    assert profile is not None

    await cloner.cleanup()

    assert cloner.profile is not None
    assert profile.sample_path.is_file()
    assert [row["id"] for row in await db.list_voices()] == [profile.id]


async def test_auto_clone_uses_only_speech_segments(tmp_path: Path) -> None:
    """В сэмпл попадает речь по таймкодам ``stt.final``, а не весь поток."""
    cloner = make_cloner(tmp_path)
    source = long_speech_wav(tmp_path / "speech.wav", phrases=4, speech_ms=2_000)

    await run_pipeline(
        out_wav=tmp_path / "out.wav",
        source_wav=source,
        auto_cloner=cloner,
    )

    profile = cloner.profile
    assert profile is not None
    sample, rate = read_sample(profile.sample_path)
    duration = sample.size / rate
    # Речи в источнике 8 с, тишины 4 с; в сэмпл ушла только речь.
    assert AUTO_MIN_SECONDS <= duration <= AUTO_MAX_SECONDS
    assert duration < 12.0
