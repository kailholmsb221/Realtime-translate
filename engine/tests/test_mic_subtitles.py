"""Тесты режима «только мой голос» — ``scripts/mic_subtitles.py``.

Реальный WebSocket на свободном порту, но без моделей и без микрофона: вместо
whisper — фейковый распознаватель, вместо NLLB — фейковый провайдер, вместо
устройства — WAV-фикстура. Проверяется то, ради чего режим и сделан: наружу
идёт цепочка ``stt.final`` → ``translation.ready`` → ``metrics.latency``,
работает только поток ``out`` (микрофон), перевод озвучивается в приёмник, а
микрофон на это время глушится полудуплексом; с ``--silent`` не звучит ничего.

``scripts/`` лежит на ``sys.path`` — это делает ``engine/tests/conftest.py``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import mic_subtitles
import pytest
import websockets
from aec import EchoCanceller
from websockets.asyncio.server import serve as ws_serve

from engine.audio_io import ENGINE_FORMAT, FakeSink, FakeSource
from engine.contracts.events import (
    EVENT_METRICS_LATENCY,
    EVENT_SESSION_STATE,
    EVENT_STT_FINAL,
    EVENT_TRANSLATION_READY,
    EVENT_TTS_CHUNK,
    Envelope,
    Lang,
    MetricsLatency,
    SessionStart,
    SessionState,
    SessionStatus,
    SessionStop,
    Stream,
    SttFinal,
    TranslationReady,
    parse_event,
    to_json,
)
from engine.stt import create_engine, create_transcriber
from engine.stt.base import SttConfig
from engine.translate.fake import FakeProvider
from engine.translate.service import TranslationService
from engine.tts import create_tts
from engine.tts.base import Backend as TtsBackend
from engine.tts.base import TtsConfig

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPEECH_PATTERN = FIXTURES_DIR / "stt_speech_pattern.wav"
WS_TIMEOUT = 20.0

SPOKEN = "привет как слышно"


def build_app(
    text: str = SPOKEN,
    provider: FakeProvider | None = None,
    tts: object | None = None,
    sink: FakeSink | None = None,
    gate: mic_subtitles.HalfDuplexGate | None = None,
    aec: EchoCanceller | None = None,
) -> mic_subtitles.MicSubtitles:
    """Режим на фейках: WAV вместо микрофона, фейковые whisper, NLLB и TTS."""
    config = SttConfig(min_silence_ms=200, max_utterance_ms=2_000, energy_threshold=0.01)
    engine = create_engine(
        config,
        vad_backend="energy",
        transcriber=create_transcriber(config, "fake", text=text, lang=Lang.RU.value),
    )
    return mic_subtitles.MicSubtitles(
        engine,
        TranslationService(provider or FakeProvider()),
        mic_subtitles.Broadcaster(),
        lambda: FakeSource.from_wav(SPEECH_PATTERN, fmt=ENGINE_FORMAT),
        Lang.RU,
        Lang.EN,
        tts=tts,  # type: ignore[arg-type]  # фейковый роутер того же интерфейса
        sink_factory=(None if sink is None else lambda: sink),
        gate=gate,
        aec=aec,
        printer=lambda *_: None,
    )


async def collect(
    client: websockets.ClientConnection,
    until: str,
    limit: int = 40,
) -> list[Envelope]:
    """Читать конверты, пока не придёт событие ``until`` (или не кончится лимит)."""
    received: list[Envelope] = []
    for _ in range(limit):
        raw = await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT)
        envelope = parse_event(raw)
        received.append(envelope)
        if envelope.type == until:
            return received
    raise AssertionError(f"событие {until} так и не пришло: {[e.type for e in received]}")


@pytest.fixture
async def connected() -> AsyncIterator[
    tuple[mic_subtitles.MicSubtitles, websockets.ClientConnection]
]:
    """Поднятый режим и подключённый к нему клиент UI."""
    app = build_app()
    async with ws_serve(app.handle_client, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            hello = parse_event(await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT))
            assert hello.type == EVENT_SESSION_STATE
            assert isinstance(hello.payload, SessionState)
            assert hello.payload.status is SessionStatus.IDLE
            try:
                yield app, client
            finally:
                await app.stop()


async def test_chain_reaches_ui_without_speaking(
    connected: tuple[mic_subtitles.MicSubtitles, websockets.ClientConnection],
) -> None:
    """Наружу идёт цепочка whisper → NLLB → метрики, и ничего не озвучивается."""
    app, client = connected
    await app.start()
    events = await collect(client, EVENT_METRICS_LATENCY)
    types = [e.type for e in events]

    assert EVENT_TTS_CHUNK not in types, "режим не должен ничего синтезировать"
    assert types.index(EVENT_STT_FINAL) < types.index(EVENT_TRANSLATION_READY)

    final = next(e.payload for e in events if e.type == EVENT_STT_FINAL)
    ready = next(e.payload for e in events if e.type == EVENT_TRANSLATION_READY)
    metrics = next(e.payload for e in events if e.type == EVENT_METRICS_LATENCY)
    assert isinstance(final, SttFinal)
    assert isinstance(ready, TranslationReady)
    assert isinstance(metrics, MetricsLatency)

    # Ровно то, что показывает панель: текст whisper = вход NLLB, рядом перевод.
    assert final.text == SPOKEN
    assert ready.src_text == SPOKEN
    assert ready.text == f"[{Lang.EN.value}] {SPOKEN}"
    assert (ready.src_lang, ready.dst_lang) == (Lang.RU, Lang.EN)

    # Работает только микрофон: поток `in` (loopback) не поднимается вовсе.
    assert {e.payload.stream for e in events if hasattr(e.payload, "stream")} == {Stream.OUT}
    assert metrics.tts_ms == 0


async def test_session_commands_switch_languages(
    connected: tuple[mic_subtitles.MicSubtitles, websockets.ClientConnection],
) -> None:
    """`session.start` из UI задаёт языки: `lang_out` — мой, `lang_in` — перевода."""
    app, client = connected
    await client.send(
        to_json(SessionStart(lang_in=Lang.KK, lang_out=Lang.RU, voice_id=None, record=False))
    )

    events = await collect(client, EVENT_SESSION_STATE)
    state = events[-1].payload
    assert isinstance(state, SessionState)
    assert state.status is SessionStatus.RUNNING
    assert (state.langs.out, state.langs.in_) == (Lang.RU, Lang.KK)
    assert app.running

    await client.send(to_json(SessionStop()))
    stopped = await collect(client, EVENT_SESSION_STATE)
    assert isinstance(stopped[-1].payload, SessionState)
    assert stopped[-1].payload.status is SessionStatus.IDLE
    assert not app.running


async def test_garbage_from_ui_is_ignored(
    connected: tuple[mic_subtitles.MicSubtitles, websockets.ClientConnection],
) -> None:
    """Кривая команда не роняет режим — он остаётся управляемым."""
    _app, client = connected
    await client.send("не json")
    await client.send('{"type": "session.start", "ts": 1, "payload": {}}')

    await client.send(
        to_json(SessionStart(lang_in=Lang.EN, lang_out=Lang.RU, voice_id=None, record=False))
    )
    events = await collect(client, EVENT_SESSION_STATE)
    assert isinstance(events[-1].payload, SessionState)
    assert events[-1].payload.status is SessionStatus.RUNNING


async def test_whisper_hallucinations_are_not_translated() -> None:
    """Заученные «титры» whisper не уходят ни в NLLB, ни в панель."""
    provider = FakeProvider()
    app = build_app(text="Субтитры сделал DimaTorzok", provider=provider)
    async with ws_serve(app.handle_client, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT)  # session.state idle
            await app.start()
            received: list[Envelope] = []
            with_timeout = asyncio.get_running_loop().time() + 10.0
            while asyncio.get_running_loop().time() < with_timeout:
                try:
                    raw = await asyncio.wait_for(client.recv(), timeout=2.0)
                except TimeoutError:
                    break
                received.append(parse_event(raw))
            await app.stop()

    # Артефакт отсекается до публикации: в панели его нет ни финалом, ни переводом.
    assert all(e.type not in (EVENT_STT_FINAL, EVENT_TRANSLATION_READY) for e in received), [
        e.type for e in received
    ]
    assert provider.calls == 0
    assert app.index == 0


async def test_translation_is_spoken() -> None:
    """С озвучкой перевод уходит в приёмник, и метрика несёт время синтеза."""
    sink = FakeSink()
    # Перевод должен быть латиницей, иначе сработает защита looks_untranslated.
    provider = FakeProvider(table={(Lang.RU, Lang.EN, SPOKEN): "hello, can you hear me"})
    app = build_app(
        provider=provider,
        tts=create_tts(TtsConfig(backend=TtsBackend.FAKE)),
        sink=sink,
        gate=mic_subtitles.HalfDuplexGate(tail_ms=50),
    )

    async with ws_serve(app.handle_client, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT)  # session.state idle
            await app.start()
            events = await collect(client, EVENT_METRICS_LATENCY)
            await app.stop()

    ready = next(e.payload for e in events if e.type == EVENT_TRANSLATION_READY)
    metrics = next(e.payload for e in events if e.type == EVENT_METRICS_LATENCY)
    assert isinstance(ready, TranslationReady)
    assert isinstance(metrics, MetricsLatency)
    assert ready.text == "hello, can you hear me"
    assert sink.n_frames > 0, "перевод должен был прозвучать"


async def test_russian_output_is_not_spoken_as_english() -> None:
    """Если NLLB вернул кириллицу вместо английского, в динамик она не уходит.

    Иначе программа озвучивает русский текст «английским» голосом, микрофон
    слышит его и переводит снова — пользователю кажется, что она отвечает.
    """
    sink = FakeSink()
    app = build_app(  # FakeProvider отдаёт "[en] привет как слышно" — кириллица
        tts=create_tts(TtsConfig(backend=TtsBackend.FAKE)),
        sink=sink,
        gate=mic_subtitles.HalfDuplexGate(tail_ms=50),
    )

    async with ws_serve(app.handle_client, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT)
            await app.start()
            events = await collect(client, EVENT_METRICS_LATENCY)
            await app.stop()

    metrics = next(e.payload for e in events if e.type == EVENT_METRICS_LATENCY)
    assert isinstance(metrics, MetricsLatency)
    assert sink.n_frames == 0, "непереведённый текст озвучивать нельзя"
    assert metrics.tts_ms == 0
    # В панель фраза всё равно попадает — видно, что NLLB вернул исходник.
    assert any(e.type == EVENT_TRANSLATION_READY for e in events)


async def test_silent_mode_keeps_everything_quiet() -> None:
    """Без TTS ничего не звучит, а цепочка до панели всё равно доходит."""
    app = build_app()
    assert not app.speaking

    async with ws_serve(app.handle_client, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT)
            await app.start()
            events = await collect(client, EVENT_METRICS_LATENCY)
            await app.stop()

    metrics = next(e.payload for e in events if e.type == EVENT_METRICS_LATENCY)
    assert isinstance(metrics, MetricsLatency)
    assert metrics.tts_ms == 0


def test_gate_mutes_by_stream_timecodes() -> None:
    """Окно полудуплекса задано в таймкодах потока, а не по часам.

    Whisper отстаёт от микрофона, и чанк, записанный во время озвучки, доходит
    до потребителя уже после неё. Ловить его по «сейчас» нельзя — только по
    ``ts_ms``, иначе программа услышит собственный голос и переведёт сам себя.
    """
    gate = mic_subtitles.HalfDuplexGate(tail_ms=200)
    assert not gate.blocks(1_000)

    gate.mute_from(2_000)
    assert not gate.blocks(1_999), "до начала озвучки микрофон открыт"
    assert gate.blocks(2_500), "во время озвучки — закрыт"
    assert gate.blocks(60_000), "конца ещё не было — окно не закрывается само"

    gate.release_at(3_000)
    assert gate.blocks(3_100), "хвост 200 мс ещё держит"
    assert not gate.blocks(3_300), "после хвоста микрофон снова открыт"


def test_untranslated_output_is_not_spoken() -> None:
    """Если NLLB вернул русский текст как «перевод на en», его не озвучивают."""
    assert mic_subtitles.looks_untranslated("А я сейчас поколю", Lang.EN)
    assert not mic_subtitles.looks_untranslated("I'm going to chop wood", Lang.EN)
    assert mic_subtitles.looks_untranslated("hello there", Lang.RU)
    assert not mic_subtitles.looks_untranslated("привет", Lang.RU)
    assert not mic_subtitles.looks_untranslated("  ", Lang.EN)


async def test_echo_cancellation_keeps_microphone_open() -> None:
    """С эхоподавлением микрофон не глушится: говорить можно поверх перевода.

    Полудуплекс тут не подключён вовсе — вместо него опорный сигнал озвучки
    уходит в фильтр, и тот вычитает эхо из входа.
    """
    canceller = EchoCanceller(16_000)
    sink = FakeSink()
    provider = FakeProvider(table={(Lang.RU, Lang.EN, SPOKEN): "hello, can you hear me"})
    app = build_app(
        provider=provider,
        tts=create_tts(TtsConfig(backend=TtsBackend.FAKE)),
        sink=sink,
        aec=canceller,
    )

    async with ws_serve(app.handle_client, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT)  # session.state idle
            await app.start()
            await collect(client, EVENT_METRICS_LATENCY)
            await app.stop()

    assert canceller.stats.blocks > 0, "микрофон должен идти через эхоподавитель"
    assert sink.n_frames > 0, "перевод должен был прозвучать"


def test_hallucination_filter() -> None:
    """Фильтр ловит артефакты и не трогает обычную речь."""
    assert mic_subtitles.is_hallucination("Субтитры сделал DimaTorzok")
    assert mic_subtitles.is_hallucination("  ПРОДОЛЖЕНИЕ СЛЕДУЕТ...  ")
    assert not mic_subtitles.is_hallucination("привет, как слышно?")
    assert not mic_subtitles.is_hallucination("Расскажи мне что-нибудь интересное")


def test_sound_tags_are_hallucinations() -> None:
    """Подписи звуков из ютуб-субтитров («СПОКОЙНАЯ МУЗЫКА») — не речь."""
    assert mic_subtitles.is_hallucination("СПОКОЙНАЯ МУЗЫКА")
    assert mic_subtitles.is_hallucination("спокойная мелодия")
    assert mic_subtitles.is_hallucination("СТОН!")
    assert mic_subtitles.is_hallucination("Редактор субтитров Н.Закомолдина")
    # Слово «музыка» внутри живой фразы — не тег.
    assert not mic_subtitles.is_hallucination("мне нравится эта музыка, включи громче")


def test_foreign_text_is_treated_as_echo() -> None:
    """Латиница при языке ru — не речь пользователя, а остаток озвучки."""
    assert mic_subtitles.looks_foreign("What's up?", Lang.RU)
    assert mic_subtitles.looks_foreign("Hey, what is up", Lang.RU)
    assert not mic_subtitles.looks_foreign("Привет, как дела", Lang.RU)
    assert not mic_subtitles.looks_foreign("Окей, го", Lang.RU)
    # И в обратную сторону: для английского источника кириллица — чужая.
    assert mic_subtitles.looks_foreign("Привет", Lang.EN)
    assert not mic_subtitles.looks_foreign("hello there", Lang.EN)
    assert not mic_subtitles.looks_foreign("...", Lang.RU)


async def test_echo_in_latin_never_reaches_panel() -> None:
    """Whisper записал английское эхо при языке ru — фраза отбрасывается целиком."""
    provider = FakeProvider()
    app = build_app(text="What's up? What's up?", provider=provider)
    async with ws_serve(app.handle_client, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        async with websockets.connect(f"ws://127.0.0.1:{port}") as client:
            await asyncio.wait_for(client.recv(), timeout=WS_TIMEOUT)
            await app.start()
            received: list[Envelope] = []
            deadline = asyncio.get_running_loop().time() + 8.0
            while asyncio.get_running_loop().time() < deadline:
                try:
                    raw = await asyncio.wait_for(client.recv(), timeout=2.0)
                except TimeoutError:
                    break
                received.append(parse_event(raw))
            await app.stop()

    assert all(e.type not in (EVENT_STT_FINAL, EVENT_TRANSLATION_READY) for e in received)
    assert provider.calls == 0
    assert app.echo_leaks > 0


def test_watchdog_flags_dead_microphone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Цифровой ноль на входе дольше порога — явное сообщение, а не тишина в логе.

    Ровно так и пропало полчаса: аппаратный mute микрофона выглядел как
    «пользователь молчит».
    """
    lines: list[str] = []
    watchdog = mic_subtitles.LoopWatchdog(printer=lines.append)
    clock = {"now": 100.0}
    monkeypatch.setattr(mic_subtitles.time, "perf_counter", lambda: clock["now"])

    watchdog.note_input(0.02)  # живой сигнал
    assert not watchdog.input_dead

    for _ in range(10):
        clock["now"] += 1.0
        watchdog.note_input(0.0)  # цифровой ноль
    assert watchdog.input_dead
    watchdog._report()
    assert any("цифровой ноль" in line.lower() for line in lines)

    watchdog.note_input(0.03)
    assert not watchdog.input_dead
    assert any("снова отдаёт сигнал" in line for line in lines)


def test_group_repetition_is_garbage() -> None:
    """Зацикливание на группе букв — мусор, даже если это одно «слово»."""
    assert mic_subtitles.is_hallucination(
        "үйтіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңіңі"
    )
    assert mic_subtitles.is_hallucination("lalalalalala")
    assert mic_subtitles.is_garbage("Құр ұр ұр ұр ұр ұр ұр ұр ұр ұр")
    assert not mic_subtitles.is_hallucination("Мама мыла раму")
    assert not mic_subtitles.is_hallucination("Сәлеметсіз бе, қалыңыз қалай")


def test_speaker_echo_is_recognised_by_text() -> None:
    """Собственный перевод, вернувшийся через динамик, узнаётся по тексту.

    Для пары kk ↔ ru проверка по алфавиту бессильна — оба кириллические, и
    без этой защиты режим уходил в петлю «и дом твой, и дом твой…».
    """
    app = build_app()
    app._spoken.append("и дом твой, и дом твой, и дом твой")
    echo = mic_subtitles.Envelope.wrap(
        mic_subtitles.SttFinal(
            stream=Stream.OUT,
            lang=Lang.KK,
            text="и дом твой и дом твой",
            t_start_ms=0,
            t_end_ms=900,
        )
    )
    speech = mic_subtitles.Envelope.wrap(
        mic_subtitles.SttFinal(
            stream=Stream.OUT, lang=Lang.KK, text="бүгін ауа райы жақсы", t_start_ms=0, t_end_ms=900
        )
    )
    assert app._is_noise(echo)
    assert not app._is_noise(speech)
    assert mic_subtitles.similarity("Hey, what's up?", "hey what's up") > 0.8
    assert mic_subtitles.similarity("совсем другая фраза", "hello there") < 0.3


def test_stale_audio_is_skipped_when_behind(monkeypatch: pytest.MonkeyPatch) -> None:
    """Чанк, отставший от реального времени на секунды, пропускается.

    Иначе при перегрузе распознавание получает речь, сказанную полминуты
    назад, и перевод приходит безнадёжно поздно.
    """
    app = build_app()
    clock = {"now": 1000.0}
    monkeypatch.setattr(mic_subtitles.time, "perf_counter", lambda: clock["now"])
    app._t0 = 1000.0

    clock["now"] = 1001.0  # прошла секунда реального времени
    assert not app._behind(ts_ms=900)
    clock["now"] = 1010.0
    assert app._behind(ts_ms=900)
    assert app.dropped_ms == 30
    assert not app._behind(ts_ms=9_900)
    assert not app._dropping


def test_overlong_source_is_not_translated() -> None:
    """Исходник длиннее живой фразы — мусор, NLLB на нём генерирует секунды."""
    app = build_app()
    long_text = "бұл жерде " * 60
    envelope = mic_subtitles.Envelope.wrap(
        mic_subtitles.SttFinal(
            stream=Stream.OUT, lang=Lang.KK, text=long_text, t_start_ms=0, t_end_ms=7000
        )
    )
    assert app._is_noise(envelope)


def test_repetition_spam_filter() -> None:
    """Залипание на одном слове отсекается, короткие повторы — нет."""
    assert mic_subtitles.is_repetition_spam("No, no, no, no, no, no, no, no")
    assert mic_subtitles.is_repetition_spam("Субтитры субтитры субтитры субтитры субтитры субтитры")
    assert not mic_subtitles.is_repetition_spam("да да")
    assert not mic_subtitles.is_repetition_spam("Привет, как меня слышно сейчас, всё хорошо")


def test_parser_defaults() -> None:
    """Значения по умолчанию: ru→en, порт движка, темп ~2 секунды."""
    args = mic_subtitles.build_parser().parse_args([])
    assert (args.lang_src, args.lang_dst) == ("ru", "en")
    assert args.port == mic_subtitles.DEFAULT_WS_PORT
    assert args.max_phrase_ms == mic_subtitles.DEFAULT_MAX_PHRASE_MS
    assert args.silence_ms == mic_subtitles.DEFAULT_SILENCE_MS
