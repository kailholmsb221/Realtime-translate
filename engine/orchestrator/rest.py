"""REST-сервер истории и голосов на ``aiohttp`` (порт 8766).

Контракт утверждён владельцем по предложению из ``ui/README.md`` (раздел
«ПРЕДЛОЖЕНИЕ: REST-контракт истории для оркестратора»):

===================================  =====================================
``GET  /api/sessions``               список сессий, новые сверху
``GET  /api/sessions/{id}``          сессия + её реплики
``GET  /api/sessions/{id}/audio``    WAV записи сессии (поддержан ``Range``)
``GET  /api/voices``                 голосовые профили
``POST /api/voices``                 multipart: ``sample`` (WAV), ``name``, ``lang``
``GET  /api/health``                 служебная проверка живости
===================================  =====================================

Поля ответов названы ровно как колонки в ``db/schema.sql`` плюс производные
``utterance_count`` и ``duration_ms`` у сессии. CORS открыт для источника из
``OrchestratorConfig.cors_origin`` (по умолчанию UI на ``localhost:3000``).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Final

from aiohttp import BodyPartReader, web

from engine.contracts.events import Lang
from engine.orchestrator.config import REPO_ROOT, OrchestratorConfig
from engine.orchestrator.db import Database
from engine.tts.voices import VoiceError, VoiceStore

__all__ = ["create_app"]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.rest")

#: Ключи приложения aiohttp.
KEY_DB: Final[web.AppKey[Database]] = web.AppKey("db", Database)
KEY_CONFIG: Final[web.AppKey[OrchestratorConfig]] = web.AppKey("config", OrchestratorConfig)
KEY_VOICES: Final[web.AppKey[VoiceStore]] = web.AppKey("voices", VoiceStore)

Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]

#: Максимальный размер сэмпла голоса, байт (60 с × 24 kHz × 2 байта с запасом).
MAX_SAMPLE_BYTES: Final[int] = 16 * 1024 * 1024


def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": message}, status=status)


def _resolve_path(raw: str) -> Path:
    """Путь из БД: относительный считается от корня репозитория."""
    path = Path(raw).expanduser()
    return path if path.is_absolute() else (REPO_ROOT / path)


@web.middleware
async def _cors_middleware(request: web.Request, handler: Handler) -> web.StreamResponse:
    """Добавить CORS-заголовки и ответить на preflight."""
    origin = request.app[KEY_CONFIG].cors_origin
    if request.method == "OPTIONS":
        response: web.StreamResponse = web.Response(status=204)
    else:
        try:
            response = await handler(request)
        except web.HTTPException as exc:
            response = exc
    response.headers["Access-Control-Allow-Origin"] = origin
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Vary"] = "Origin"
    return response


async def _health(request: web.Request) -> web.StreamResponse:
    """Живость движка и его режим."""
    config = request.app[KEY_CONFIG]
    return web.json_response({"status": "ok", "backend": config.backend})


async def _list_sessions(request: web.Request) -> web.StreamResponse:
    """``GET /api/sessions`` — список сессий с производными полями."""
    rows = await request.app[KEY_DB].list_sessions()
    return web.json_response({"sessions": rows})


def _session_id(request: web.Request) -> int | None:
    raw = request.match_info.get("session_id", "")
    try:
        return int(raw)
    except ValueError:
        return None


async def _get_session(request: web.Request) -> web.StreamResponse:
    """``GET /api/sessions/{id}`` — сессия и её транскрипт."""
    session_id = _session_id(request)
    if session_id is None:
        return _error(404, "session not found")
    db = request.app[KEY_DB]
    session = await db.get_session(session_id)
    if session is None:
        return _error(404, "session not found")
    utterances = await db.list_utterances(session_id)
    return web.json_response({"session": session, "utterances": utterances})


async def _get_session_audio(request: web.Request) -> web.StreamResponse:
    """``GET /api/sessions/{id}/audio`` — WAV записи сессии."""
    session_id = _session_id(request)
    if session_id is None:
        return _error(404, "session not found")
    session = await request.app[KEY_DB].get_session(session_id)
    if session is None:
        return _error(404, "session not found")
    raw = session.get("audio_path")
    if not raw:
        return _error(404, "session has no audio")
    path = _resolve_path(str(raw))
    if not path.is_file():
        logger.warning("audio_path сессии %d указывает в никуда: %s", session_id, path)
        return _error(404, "audio file is missing")
    return web.FileResponse(path, headers={"Content-Type": "audio/wav"})


async def _list_voices(request: web.Request) -> web.StreamResponse:
    """``GET /api/voices`` — строки таблицы ``voices``."""
    rows = await request.app[KEY_DB].list_voices()
    return web.json_response({"voices": rows})


async def _read_voice_form(request: web.Request, target_dir: Path) -> tuple[dict[str, str], Path]:
    """Разобрать multipart: сохранить сэмпл на диск, вернуть поля и путь к нему."""
    reader = await request.multipart()
    fields: dict[str, str] = {}
    sample_path: Path | None = None

    async for part in reader:
        if not isinstance(part, BodyPartReader):  # pragma: no cover — вложенный multipart
            continue
        if part.name == "sample":
            filename = Path(part.filename or "sample.wav").name
            sample_path = target_dir / filename
            size = 0
            with sample_path.open("wb") as handle:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > MAX_SAMPLE_BYTES:
                        raise VoiceError(
                            f"сэмпл больше {MAX_SAMPLE_BYTES // (1024 * 1024)} МБ — "
                            "нужен WAV на 15-30 секунд"
                        )
                    handle.write(chunk)
        elif part.name:
            fields[part.name] = (await part.text()).strip()

    if sample_path is None:
        raise VoiceError("нет файла в поле 'sample' (WAV 15-30 секунд)")
    return fields, sample_path


async def _create_voice(request: web.Request) -> web.StreamResponse:
    """``POST /api/voices`` — создать профиль голоса и записать его в БД.

    Правовая заметка (ARCHITECTURE.md раздел 8): клонировать можно только
    собственный голос пользователя или голос с явного согласия владельца.
    """
    store = request.app[KEY_VOICES]
    db = request.app[KEY_DB]

    with tempfile.TemporaryDirectory(prefix="rt-voice-") as tmp:
        try:
            fields, sample_path = await _read_voice_form(request, Path(tmp))
        except VoiceError as exc:
            return _error(400, str(exc))

        name = fields.get("name", "")
        lang_raw = fields.get("lang", "")
        if not name:
            return _error(400, "нужно поле 'name'")
        try:
            lang = Lang(lang_raw)
        except ValueError:
            return _error(400, f"поле 'lang' должно быть ru|en|kk, получено {lang_raw!r}")

        try:
            profile = await asyncio.to_thread(store.create_voice, sample_path, name, lang)
        except VoiceError as exc:
            return _error(400, str(exc))

    try:
        row = await db.upsert_voice(
            profile.id, profile.name, profile.lang, profile.sample_path, profile.created_at_ms
        )
    except sqlite3.IntegrityError as exc:
        await asyncio.to_thread(store.delete, profile.id)
        return _error(409, f"имя голоса занято: {exc}")

    logger.info("создан голос %s (%s, %s)", profile.id, profile.name, profile.lang.value)
    return web.json_response({"voice": row}, status=201)


def create_app(
    db: Database,
    config: OrchestratorConfig,
    voices: VoiceStore,
) -> web.Application:
    """Собрать aiohttp-приложение REST.

    Args:
        db: открытая база движка.
        config: конфигурация (нужен ``cors_origin``).
        voices: хранилище голосовых профилей на диске.

    Returns:
        Приложение, готовое к ``AppRunner``/``TCPSite`` или к тестовому клиенту.
    """
    app = web.Application(middlewares=[_cors_middleware])
    app[KEY_DB] = db
    app[KEY_CONFIG] = config
    app[KEY_VOICES] = voices

    routes: list[Any] = [
        web.get("/api/health", _health),
        web.get("/api/sessions", _list_sessions),
        web.get("/api/sessions/{session_id}", _get_session),
        web.get("/api/sessions/{session_id}/audio", _get_session_audio),
        web.get("/api/voices", _list_voices),
        web.post("/api/voices", _create_voice),
        web.options("/{tail:.*}", _health),
    ]
    app.add_routes(routes)
    return app
