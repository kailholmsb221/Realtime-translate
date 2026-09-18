"""Сессия и серверы движка: WebSocket (8765) + REST (8766).

ARCHITECTURE.md 4.5: два конвейера на сессию и WebSocket-сервер, который
транслирует все события в UI.

Семантика ``session.start`` (контракты, раздел 5):

* ``lang_in`` — язык собеседника, он же язык потока ``in`` (WASAPI loopback);
* ``lang_out`` — язык пользователя, он же язык потока ``out`` (микрофон).

Отсюда направления перевода:

===========  =========================  ===========================  =================
поток        источник                   перевод                      приёмник
===========  =========================  ===========================  =================
``in``       loopback (Zoom)            ``lang_in`` → ``lang_out``   наушники
``out``      микрофон                   ``lang_out`` → ``lang_in``   VB-Cable (Zoom)
===========  =========================  ===========================  =================

Голоса: ``voice_id`` из ``session.start`` — это голос **пользователя**, он идёт
в outbound-конвейер. Голос собеседника движок клонирует сам в первые секунды
разговора (:class:`~engine.orchestrator.autoclone.AutoCloner`), пока клон не
готов — inbound звучит встроенным голосом XTTS.

Команды принимаются только валидные (``parse_event``); всё остальное пишется
в лог и игнорируется — кривое сообщение из UI не должно ронять движок.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, Final

import websockets
from aiohttp import web
from websockets.asyncio.server import Server as WsServer
from websockets.asyncio.server import ServerConnection

from engine.audio_io import AudioSink, AudioSource
from engine.contracts.events import (
    COMMAND_SESSION_START,
    COMMAND_SESSION_STOP,
    ContractError,
    Envelope,
    Lang,
    Langs,
    SessionStart,
    SessionState,
    SessionStatus,
    Stream,
    parse_event,
    to_json,
)
from engine.orchestrator.autoclone import AutoCloner, supports_auto_clone
from engine.orchestrator.bus import EventBus
from engine.orchestrator.config import OrchestratorConfig
from engine.orchestrator.db import Database
from engine.orchestrator.pipeline import Pipeline, WavRecorder
from engine.orchestrator.rest import create_app
from engine.orchestrator.runtime import Runtime

__all__ = ["EngineServer", "Session", "serve"]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.server")

#: Языки в ``session.state``, пока сессия не запускалась (как initialState в UI).
DEFAULT_LANGS: Final[Langs] = Langs(in_=Lang.EN, out=Lang.RU)


class Session:
    """Одна сессия перевода: два конвейера, запись и строка в БД.

    Args:
        config: конфигурация движка.
        runtime: общие компоненты (модели, хранилище голосов).
        bus: шина событий.
        db: база движка.
        command: команда ``session.start`` из UI.
    """

    __slots__ = (
        "_auto_cloners",
        "_bus",
        "_command",
        "_config",
        "_db",
        "_pipelines",
        "_recorders",
        "_runtime",
        "_sinks",
        "_sources",
        "_tasks",
        "session_id",
    )

    def __init__(
        self,
        config: OrchestratorConfig,
        runtime: Runtime,
        bus: EventBus,
        db: Database,
        command: SessionStart,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._bus = bus
        self._db = db
        self._command = command
        self._pipelines: list[Pipeline] = []
        self._tasks: list[asyncio.Task[None]] = []
        self._sources: list[AudioSource] = []
        self._sinks: list[AudioSink] = []
        self._recorders: list[WavRecorder] = []
        self._auto_cloners: list[AutoCloner] = []
        self.session_id: int | None = None

    @property
    def langs(self) -> Langs:
        """Языки сессии в форме контракта ``session.state``."""
        return Langs(in_=self._command.lang_in, out=self._command.lang_out)

    @property
    def voice_id(self) -> str | None:
        """Голосовой профиль сессии."""
        return self._command.voice_id

    @property
    def pipelines(self) -> tuple[Pipeline, ...]:
        """Запущенные конвейеры (для тестов и smoke-скрипта)."""
        return tuple(self._pipelines)

    def state(self, status: SessionStatus = SessionStatus.RUNNING) -> SessionState:
        """Событие ``session.state`` для этой сессии."""
        return SessionState(
            status=status,
            session_id=self.session_id if status is SessionStatus.RUNNING else None,
            langs=self.langs,
            voice_id=self.voice_id,
        )

    async def start(self) -> None:
        """Создать строку сессии, поднять оба конвейера и запустить их."""
        self.session_id = await self._db.create_session(
            self._command.lang_in, self._command.lang_out
        )
        if self._command.record:
            # audio_path сессии — WAV входящего потока (речь собеседника);
            # речь пользователя пишется рядом как <id>_out.wav.
            await self._db.set_audio_path(
                self.session_id, self._config.recording_path(self.session_id, Stream.IN.value)
            )

        for stream in (Stream.IN, Stream.OUT):
            self._pipelines.append(self._build_pipeline(stream))
        self._tasks = [
            asyncio.create_task(pipeline.run(), name=f"session-{self.session_id}-{stream.value}")
            for stream, pipeline in zip((Stream.IN, Stream.OUT), self._pipelines, strict=True)
        ]
        logger.info(
            "сессия %s запущена: in %s->%s, out %s->%s, voice_id=%s, record=%s",
            self.session_id,
            self._command.lang_in.value,
            self._command.lang_out.value,
            self._command.lang_out.value,
            self._command.lang_in.value,
            self._command.voice_id,
            self._command.record,
        )

    def _build_pipeline(self, stream: Stream) -> Pipeline:
        lang_in, lang_out = self._command.lang_in, self._command.lang_out
        lang_src, lang_dst = (lang_in, lang_out) if stream is Stream.IN else (lang_out, lang_in)
        # voice_id из session.start — голос пользователя: он нужен только там,
        # где озвучивается его речь, то есть в outbound (решение владельца).
        voice_id = self._command.voice_id if stream is Stream.OUT else None
        auto_cloner = self._build_auto_cloner(stream, lang_src, lang_dst)

        source = self._runtime.create_source(stream)
        sink = self._runtime.create_sink(stream, self.session_id)
        self._sources.append(source)
        self._sinks.append(sink)

        recorder: WavRecorder | None = None
        if self._command.record and self.session_id is not None:
            recorder = WavRecorder(self._config.recording_path(self.session_id, stream.value))
            self._recorders.append(recorder)

        return Pipeline(
            stream=stream,
            lang_src=lang_src,
            lang_dst=lang_dst,
            voice_id=voice_id,
            source=source,
            sink=sink,
            stt_engine=self._runtime.stt_engine(),
            translation_service=self._runtime.translation_service(),
            tts_router=self._runtime.tts,
            bus=self._bus,
            db=self._db,
            session_id=self.session_id,
            recorder=recorder,
            auto_cloner=auto_cloner,
            emit_tts_chunks=self._config.emit_tts_chunks,
            drain_timeout_s=self._config.drain_timeout_s,
        )

    def _build_auto_cloner(
        self, stream: Stream, lang_src: Lang, lang_dst: Lang
    ) -> AutoCloner | None:
        """Автоклон голоса собеседника — только для потока ``in`` и только ru/en."""
        config = self._config.auto_clone
        if stream is not Stream.IN or not config.enabled:
            return None
        if not supports_auto_clone(lang_src, lang_dst):
            logger.info(
                "автоклон голоса пропущен: пара %s->%s вне ru/en (kk синтезируется без клона)",
                lang_src.value,
                lang_dst.value,
            )
            return None
        cloner = AutoCloner(
            config=config,
            voices=self._runtime.voices,
            lang=lang_src,
            session_id=self.session_id,
            db=self._db,
            latents_fn=self._runtime.latents_fn(),
            stream=stream.value,
        )
        self._auto_cloners.append(cloner)
        return cloner

    @property
    def auto_cloners(self) -> tuple[AutoCloner, ...]:
        """Автоклоны сессии (для тестов и отладки)."""
        return tuple(self._auto_cloners)

    async def wait(self) -> None:
        """Дождаться, пока оба конвейера сами закончатся (файловые источники)."""
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)

    async def stop(self) -> None:
        """Остановить конвейеры, закрыть устройства и закрыть сессию в БД."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

        for source in self._sources:
            with contextlib.suppress(Exception):
                await source.close()
        for sink in self._sinks:
            with contextlib.suppress(Exception):
                await sink.drain()
                await sink.close()
        for recorder in self._recorders:
            recorder.close()
        for cloner in self._auto_cloners:
            with contextlib.suppress(Exception):
                await cloner.cleanup()
        self._sources.clear()
        self._sinks.clear()

        if self.session_id is not None:
            with contextlib.suppress(Exception):
                await self._db.end_session(self.session_id)
        logger.info("сессия %s остановлена", self.session_id)


class EngineServer:
    """WebSocket-сервер команд/событий и REST-сервер истории в одном процессе."""

    __slots__ = (
        "_broadcast",
        "_bus",
        "_clients",
        "_config",
        "_db",
        "_http_runner",
        "_http_site",
        "_lock",
        "_runtime",
        "_session",
        "_started",
        "_ws_server",
    )

    def __init__(
        self,
        config: OrchestratorConfig,
        runtime: Runtime,
        db: Database,
        bus: EventBus | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime
        self._db = db
        self._bus = bus if bus is not None else EventBus()
        self._clients: set[ServerConnection] = set()
        self._session: Session | None = None
        self._lock = asyncio.Lock()
        self._ws_server: WsServer | None = None
        self._http_runner: web.AppRunner | None = None
        self._http_site: web.TCPSite | None = None
        self._broadcast: asyncio.Task[None] | None = None
        self._started = False

    # --- свойства ----------------------------------------------------------

    @property
    def bus(self) -> EventBus:
        """Шина событий движка."""
        return self._bus

    @property
    def session(self) -> Session | None:
        """Текущая сессия или ``None``, если движок простаивает."""
        return self._session

    @property
    def ws_port(self) -> int:
        """Фактический порт WebSocket (важно при ``ws_port=0`` в тестах)."""
        if self._ws_server is None:
            return self._config.ws_port
        sockets = self._ws_server.sockets
        return int(sockets[0].getsockname()[1]) if sockets else self._config.ws_port

    @property
    def http_port(self) -> int:
        """Фактический порт REST."""
        if self._http_runner is None or not self._http_runner.addresses:
            return self._config.http_port
        return int(self._http_runner.addresses[0][1])

    def state(self) -> SessionState:
        """Текущее состояние движка для ``session.state``."""
        if self._session is None:
            return SessionState(
                status=SessionStatus.IDLE, session_id=None, langs=DEFAULT_LANGS, voice_id=None
            )
        return self._session.state(SessionStatus.RUNNING)

    # --- жизненный цикл ----------------------------------------------------

    async def start(self) -> None:
        """Поднять оба сервера и фоновую рассылку событий."""
        if self._started:
            return
        await self._db.init()
        await self._db.sync_voices_from_store(self._runtime.voices)

        self._ws_server = await websockets.serve(
            self._handle_client, self._config.ws_host, self._config.ws_port
        )
        app = create_app(self._db, self._config, self._runtime.voices)
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        self._http_site = web.TCPSite(
            self._http_runner, self._config.ws_host, self._config.http_port
        )
        await self._http_site.start()

        self._broadcast = asyncio.create_task(self._broadcast_loop(), name="ws-broadcast")
        self._started = True
        logger.info(
            "движок слушает: ws://%s:%d, http://%s:%d (backend=%s)",
            self._config.ws_host,
            self.ws_port,
            self._config.ws_host,
            self.http_port,
            self._config.backend,
        )

    async def close(self) -> None:
        """Остановить сессию, серверы и рассылку."""
        if not self._started:
            return
        self._started = False

        async with self._lock:
            if self._session is not None:
                await self._session.stop()
                self._session = None

        if self._broadcast is not None:
            self._broadcast.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._broadcast
            self._broadcast = None

        for client in list(self._clients):
            with contextlib.suppress(Exception):
                await client.close()
        self._clients.clear()

        if self._ws_server is not None:
            self._ws_server.close()
            with contextlib.suppress(Exception):
                await self._ws_server.wait_closed()
            self._ws_server = None

        if self._http_site is not None:
            await self._http_site.stop()
            self._http_site = None
        if self._http_runner is not None:
            await self._http_runner.cleanup()
            self._http_runner = None
        logger.info("движок остановлен")

    async def serve_forever(self) -> None:
        """Работать, пока задачу не отменят (``Ctrl+C`` в CLI)."""
        await self.start()
        try:
            await asyncio.Event().wait()
        finally:
            await self.close()

    # --- WebSocket ---------------------------------------------------------

    async def _handle_client(self, connection: ServerConnection) -> None:
        """Обслужить одного клиента UI: отдать состояние и читать команды."""
        self._clients.add(connection)
        logger.info("UI подключился (%d клиентов)", len(self._clients))
        try:
            await connection.send(to_json(self.state()))
            async for raw in connection:
                await self._on_message(raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(connection)
            logger.info("UI отключился (%d клиентов)", len(self._clients))

    async def _on_message(self, raw: str | bytes) -> None:
        """Разобрать команду UI; невалидное — в лог и мимо."""
        try:
            envelope = parse_event(raw)
        except ContractError as exc:
            logger.warning("невалидная команда от UI, игнорирую: %s", exc)
            return

        payload = envelope.payload
        if envelope.type == COMMAND_SESSION_START and isinstance(payload, SessionStart):
            await self.start_session(payload)
        elif envelope.type == COMMAND_SESSION_STOP:
            await self.stop_session()
        else:
            logger.warning("событие %r — не команда UI, игнорирую", envelope.type)

    async def _broadcast_loop(self) -> None:
        """Раздавать события шины всем подключённым клиентам."""
        with self._bus.subscription() as queue:
            while True:
                envelope = await queue.get()
                try:
                    await self._send_all(envelope)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    # Рассылка — единственный канал событий в UI: её падение
                    # оставило бы окно навсегда без субтитров. Теряем одно
                    # событие, но цикл продолжает работать.
                    logger.exception("не удалось разослать событие %r", envelope.type)

    async def _send_all(self, envelope: Envelope) -> None:
        clients = list(self._clients)
        if not clients:
            return
        try:
            raw = to_json(envelope)
        except ContractError:
            logger.exception("событие %r не прошло валидацию, не отправляю", envelope.type)
            return
        results: list[Any] = await asyncio.gather(
            *(client.send(raw) for client in clients), return_exceptions=True
        )
        for client, result in zip(clients, results, strict=True):
            if isinstance(result, BaseException):
                logger.debug("клиент отвалился при отправке: %s", result)
                self._clients.discard(client)

    # --- управление сессией -------------------------------------------------

    async def start_session(self, command: SessionStart) -> Session | None:
        """Запустить сессию (если уже идёт — сначала остановить предыдущую)."""
        async with self._lock:
            if self._session is not None:
                await self._session.stop()
                self._session = None
            session = Session(self._config, self._runtime, self._bus, self._db, command)
            try:
                await session.start()
            except Exception:
                logger.exception("не удалось запустить сессию")
                with contextlib.suppress(Exception):
                    await session.stop()
                await self._bus.publish(Envelope.wrap(self.state()))
                return None
            self._session = session
        await self._bus.publish(Envelope.wrap(session.state(SessionStatus.RUNNING)))
        return session

    async def stop_session(self) -> None:
        """Остановить текущую сессию и сообщить UI ``status="idle"``."""
        async with self._lock:
            session, self._session = self._session, None
            if session is not None:
                await session.stop()
            state = (
                session.state(SessionStatus.IDLE)
                if session is not None
                else SessionState(SessionStatus.IDLE, None, DEFAULT_LANGS, None)
            )
        await self._bus.publish(Envelope.wrap(state))


async def serve(config: OrchestratorConfig, *, warmup: bool = True) -> None:
    """Поднять движок в live-режиме и работать до отмены задачи.

    Прогревает модели (real-бэкенд), открывает базу, синхронизирует голоса с
    диска и запускает оба сервера.

    Args:
        config: конфигурация движка.
        warmup: грузить ли модели заранее; ``False`` — только для отладки,
            тогда первая фраза будет ждать загрузку whisper и XTTS.
    """
    runtime = Runtime.create(config)
    db = Database(config.db_path, config.schema_path)
    server = EngineServer(runtime.config, runtime, db)
    try:
        if warmup:
            await runtime.warmup()
        await server.serve_forever()
    finally:
        await server.close()
        await runtime.aclose()
        await db.close()
