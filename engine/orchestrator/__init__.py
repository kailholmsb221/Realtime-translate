"""Пакет orchestrator: склейка конвейера, WebSocket :8765, REST :8766, запись в БД.

Зона этапа 2, см. ARCHITECTURE.md раздел 4.5 и README этого пакета.

Быстрый старт::

    python -m engine.orchestrator                  # live: ws://127.0.0.1:8765 + REST :8766
    python -m engine.orchestrator --backend fake \\
        --file in.wav --from en --to ru --out out.wav   # файловый прогон без серверов

Из кода::

    from engine.orchestrator import OrchestratorConfig, serve

    await serve(OrchestratorConfig.from_env())

Голос пользователя приходит в ``session.start`` (``voice_id``) и озвучивает
outbound; голос собеседника движок клонирует сам в первые секунды разговора
(:class:`~engine.orchestrator.autoclone.AutoCloner`).

Тяжёлые модели создаются только фабриками чужих модулей и только в
:class:`~engine.orchestrator.runtime.Runtime`; импорт самого пакета их не тянет.
"""

from __future__ import annotations

from engine.orchestrator.autoclone import AutoCloneConfig, AutoCloner
from engine.orchestrator.bus import EventBus
from engine.orchestrator.config import OrchestratorConfig
from engine.orchestrator.db import Database
from engine.orchestrator.filemode import FileModeResult, run_file_mode
from engine.orchestrator.pipeline import Pipeline, PipelineStats, WavRecorder
from engine.orchestrator.rest import create_app
from engine.orchestrator.runtime import Runtime
from engine.orchestrator.server import EngineServer, Session, serve

__all__ = [
    "AutoCloneConfig",
    "AutoCloner",
    "Database",
    "EngineServer",
    "EventBus",
    "FileModeResult",
    "OrchestratorConfig",
    "Pipeline",
    "PipelineStats",
    "Runtime",
    "Session",
    "WavRecorder",
    "create_app",
    "run_file_mode",
    "serve",
]
