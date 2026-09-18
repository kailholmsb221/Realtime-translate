"""Тесты серверов движка: WebSocket :8765 и REST :8766 (на свободных портах).

Поднимается настоящий :class:`~engine.orchestrator.server.EngineServer` на
фейковом бэкенде: вместо loopback и микрофона — WAV-фикстура, вместо моделей —
фейки. Проверяются жизненный цикл сессии по контрактам, устойчивость к мусору
от UI и REST-контракт истории, утверждённый владельцем (см. ui/README.md).
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiohttp
import numpy as np
import pytest
import websockets

from engine.contracts.events import (
    EVENT_SESSION_STATE,
    EVENT_STT_FINAL,
    EVENT_TRANSLATION_READY,
    Envelope,
    Lang,
    SessionStart,
    SessionState,
    SessionStop,
    Stream,
    parse_event,
    to_json,
)
from engine.orchestrator.config import OrchestratorConfig
from engine.orchestrator.db import Database
from engine.orchestrator.runtime import Runtime
from engine.orchestrator.server import EngineServer
from engine.tts.audio import write_wav
from engine.tts.base import OUTPUT_SAMPLE_RATE, Backend, TtsConfig

pytestmark = pytest.mark.asyncio

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
SPEECH_PATTERN = FIXTURES_DIR / "stt_speech_pattern.wav"
WS_TIMEOUT = 20.0


@pytest.fixture
async def server(tmp_path: Path) -> AsyncIterator[EngineServer]:
    """Движок на фейковом бэкенде и свободных портах."""
    config = OrchestratorConfig(
        ws_host="127.0.0.1",
        ws_port=0,
        http_port=0,
        backend="fake",
        db_path=tmp_path / "translator.db",
        recordings_dir=tmp_path / "recordings",
        fake_in=SPEECH_PATTERN,
        fake_out=SPEECH_PATTERN,
        # Профили голосов — в каталоге теста, а не в ./voices репозитория.
        tts=TtsConfig(backend=Backend.FAKE, voices_dir=tmp_path / "voices"),
    )
    runtime = Runtime.create(config)
    db = Database(config.db_path, config.schema_path)
    engine = EngineServer(runtime.config, runtime, db)
    await engine.start()
    try:
        yield engine
    finally:
        await engine.close()
        await db.close()


async def recv_event(socket: Any) -> Envelope:
    """Получить и разобрать одно событие (с таймаутом, чтобы тест не висел)."""
    raw = await asyncio.wait_for(socket.recv(), timeout=WS_TIMEOUT)
    return parse_event(raw)


async def recv_until(socket: Any, *types: str, limit: int = 60) -> list[Envelope]:
    """Читать события, пока не встретится один из типов ``types``."""
    events: list[Envelope] = []
    for _ in range(limit):
        envelope = await recv_event(socket)
        events.append(envelope)
        if envelope.type in types:
            return events
    raise AssertionError(f"не дождались событий {types}: получено {[e.type for e in events]}")


def ws_url(server: EngineServer) -> str:
    """Адрес WebSocket запущенного движка."""
    return f"ws://127.0.0.1:{server.ws_port}"


def http_url(server: EngineServer, path: str) -> str:
    """Адрес REST-эндпоинта запущенного движка."""
    return f"http://127.0.0.1:{server.http_port}{path}"


def start_command(record: bool = False) -> str:
    """JSON команды ``session.start`` (en → ru, без клона голоса)."""
    return to_json(SessionStart(lang_in=Lang.EN, lang_out=Lang.RU, voice_id=None, record=record))


async def test_client_gets_idle_state_on_connect(server: EngineServer) -> None:
    """Сразу после подключения UI получает ``session.state`` со статусом idle."""
    async with websockets.connect(ws_url(server)) as socket:
        envelope = await recv_event(socket)
        assert envelope.type == EVENT_SESSION_STATE
        payload = envelope.payload
        assert isinstance(payload, SessionState)
        assert payload.status.value == "idle"
        assert payload.session_id is None


async def test_session_start_stop_cycle(server: EngineServer) -> None:
    """start → running + события конвейера, stop → idle и закрытая сессия в БД."""
    async with websockets.connect(ws_url(server)) as socket:
        await recv_event(socket)  # idle

        await socket.send(start_command(record=True))
        running = await recv_until(socket, EVENT_SESSION_STATE)
        state = running[-1].payload
        assert isinstance(state, SessionState)
        assert state.status.value == "running"
        assert state.session_id == 1
        assert state.langs.in_ is Lang.EN
        assert state.langs.out is Lang.RU

        events = await recv_until(socket, EVENT_TRANSLATION_READY)
        assert any(event.type == EVENT_STT_FINAL for event in events)

        await socket.send(to_json(SessionStop()))
        idle = await recv_until(socket, EVENT_SESSION_STATE)
        stopped = idle[-1].payload
        assert isinstance(stopped, SessionState)
        assert stopped.status.value == "idle"
        assert stopped.session_id is None

    assert server.session is None
    async with (
        aiohttp.ClientSession() as http,
        http.get(http_url(server, "/api/sessions")) as response,
    ):
        body = await response.json()
    session = body["sessions"][0]
    assert session["ended_at"] is not None
    assert session["duration_ms"] is not None
    assert session["utterance_count"] >= 1
    assert session["audio_path"], "record=true — путь к WAV должен быть записан"


async def test_invalid_commands_are_ignored(server: EngineServer) -> None:
    """Мусор и события не-команды не роняют движок и не стартуют сессию."""
    async with websockets.connect(ws_url(server)) as socket:
        await recv_event(socket)  # idle

        await socket.send("не json")
        await socket.send(json.dumps({"type": "session.start", "ts": 1, "payload": {}}))
        await socket.send(json.dumps({"type": "нет такого", "ts": 1, "payload": {}}))
        await socket.send(json.dumps({"type": "stt.final", "ts": 1, "payload": {}}))

        # Движок жив: следующая валидная команда обрабатывается.
        await socket.send(start_command())
        envelope = await recv_until(socket, EVENT_SESSION_STATE)
        state = envelope[-1].payload
        assert isinstance(state, SessionState)
        assert state.status.value == "running"

        await socket.send(to_json(SessionStop()))
        await recv_until(socket, EVENT_SESSION_STATE)


async def test_rest_session_transcript(server: EngineServer) -> None:
    """``GET /api/sessions/{id}`` отдаёт сессию и её реплики."""
    async with websockets.connect(ws_url(server)) as socket:
        await recv_event(socket)
        await socket.send(start_command())
        await recv_until(socket, EVENT_TRANSLATION_READY)
        await socket.send(to_json(SessionStop()))
        await recv_until(socket, EVENT_SESSION_STATE)

    async with aiohttp.ClientSession() as http:
        async with http.get(http_url(server, "/api/sessions/1")) as response:
            assert response.status == 200
            body = await response.json()
        async with http.get(http_url(server, "/api/sessions/404")) as response:
            assert response.status == 404
            assert (await response.json())["error"] == "session not found"

    assert body["session"]["id"] == 1
    assert body["utterances"]
    first = body["utterances"][0]
    assert set(first) == {
        "id",
        "session_id",
        "t_start_ms",
        "t_end_ms",
        "speaker",
        "lang",
        "text",
        "translation",
        "translation_lang",
    }
    assert first["speaker"] in {"in", "out"}


async def test_rest_sessions_empty_and_cors(server: EngineServer) -> None:
    """Пустая история — пустой список; ответы несут CORS-заголовок для UI."""
    async with aiohttp.ClientSession() as http:
        async with http.get(http_url(server, "/api/sessions")) as response:
            assert response.status == 200
            assert response.headers["Access-Control-Allow-Origin"] == "http://localhost:3000"
            assert await response.json() == {"sessions": []}
        async with http.get(http_url(server, "/api/sessions/1/audio")) as response:
            assert response.status == 404


async def test_rest_voices_create_and_list(server: EngineServer, tmp_path: Path) -> None:
    """``POST /api/voices`` создаёт профиль на диске и строку в БД."""
    sample = tmp_path / "sample.wav"
    write_wav(sample, np.zeros(OUTPUT_SAMPLE_RATE * 16, dtype=np.int16), OUTPUT_SAMPLE_RATE)

    async with aiohttp.ClientSession() as http:
        async with http.get(http_url(server, "/api/voices")) as response:
            assert await response.json() == {"voices": []}

        form = aiohttp.FormData()
        form.add_field("name", "мой голос")
        form.add_field("lang", "ru")
        form.add_field("sample", sample.read_bytes(), filename="sample.wav")
        async with http.post(http_url(server, "/api/voices"), data=form) as response:
            assert response.status == 201, await response.text()
            created = (await response.json())["voice"]

        async with http.get(http_url(server, "/api/voices")) as response:
            voices = (await response.json())["voices"]

        bad = aiohttp.FormData()
        bad.add_field("name", "без языка")
        bad.add_field("lang", "de")
        bad.add_field("sample", sample.read_bytes(), filename="sample.wav")
        async with http.post(http_url(server, "/api/voices"), data=bad) as response:
            assert response.status == 400

    assert created["id"].startswith("v_")
    assert created["lang"] == "ru"
    assert voices == [created]
    assert Path(created["sample_path"]).is_file()


async def test_rest_health(server: EngineServer) -> None:
    """Служебный ``/api/health`` показывает режим движка."""
    async with (
        aiohttp.ClientSession() as http,
        http.get(http_url(server, "/api/health")) as response,
    ):
        assert await response.json() == {"status": "ok", "backend": "fake"}


async def test_auto_clone_attached_to_inbound_only(server: EngineServer) -> None:
    """Голос из session.start уходит в outbound, а inbound клонирует собеседника."""
    async with websockets.connect(ws_url(server)) as socket:
        await recv_event(socket)  # idle
        await socket.send(
            to_json(
                SessionStart(lang_in=Lang.EN, lang_out=Lang.RU, voice_id="v_user", record=False)
            )
        )
        await recv_until(socket, EVENT_SESSION_STATE)

        session = server.session
        assert session is not None
        inbound, outbound = session.pipelines
        assert inbound.stream is Stream.IN
        assert outbound.stream is Stream.OUT
        # voice_id пользователя — только там, где звучит его речь.
        assert outbound.voice_id == "v_user"
        assert inbound.voice_id is None, "пока клон не готов — встроенный голос XTTS"
        # Автоклон подключён к inbound и только к нему.
        assert len(session.auto_cloners) == 1
        assert inbound.auto_cloner is session.auto_cloners[0]
        assert outbound.auto_cloner is None

        await socket.send(to_json(SessionStop()))
        await recv_until(socket, EVENT_SESSION_STATE)


async def test_auto_clone_skipped_for_kazakh_session(server: EngineServer) -> None:
    """Для kk автоклон не создаётся: клона у казахского TTS нет."""
    async with websockets.connect(ws_url(server)) as socket:
        await recv_event(socket)  # idle
        await socket.send(
            to_json(SessionStart(lang_in=Lang.KK, lang_out=Lang.RU, voice_id=None, record=False))
        )
        await recv_until(socket, EVENT_SESSION_STATE)

        session = server.session
        assert session is not None
        assert session.auto_cloners == ()
        assert all(pipeline.auto_cloner is None for pipeline in session.pipelines)

        await socket.send(to_json(SessionStop()))
        await recv_until(socket, EVENT_SESSION_STATE)
