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
import wave
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from engine.audio_io import FakeSink, FakeSource
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
from engine.orchestrator.bus import EventBus
from engine.orchestrator.db import Database
from engine.orchestrator.pipeline import Pipeline, WavRecorder
from engine.stt.base import SttConfig
from engine.stt.engine import SttEngine
from engine.stt.fake import EnergyVad, FakeTranscriber
from engine.translate.base import TranslationResult
from engine.translate.fake import FakeProvider
from engine.translate.service import TranslationService
from engine.tts.base import Backend, TtsConfig
from engine.tts.router import TtsRouter

pytestmark = pytest.mark.asyncio

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPEECH_PATTERN = FIXTURES_DIR / "stt_speech_pattern.wav"


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
) -> tuple[Pipeline, list[Envelope], FakeSink]:
    """Прогнать фикстуру через конвейер и вернуть его, события и приёмник."""
    bus = EventBus()
    queue = bus.subscribe()
    source = FakeSource.from_wav(SPEECH_PATTERN)
    sink = FakeSink(path=out_wav)
    service = TranslationService(provider if provider is not None else FakeProvider())  # type: ignore[arg-type]
    pipeline = Pipeline(
        stream=stream,
        lang_src=Lang.EN,
        lang_dst=Lang.RU,
        voice_id=None,
        source=source,
        sink=sink,
        stt_engine=make_engine(),
        translation_service=service,
        tts_router=make_router(),
        bus=bus,
        db=db,
        session_id=session_id,
        recorder=recorder,
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
