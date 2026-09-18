"""Конфигурация оркестратора (ARCHITECTURE.md 4.5).

Собирает в одном месте собственные настройки (порты, путь к БД, каталог
записей, выбор бэкенда) и конфигурации остальных модулей движка — каждая
читается своей фабрикой ``from_env()``, чужие переменные окружения здесь не
дублируются (CLAUDE.md, закон 1).

Пример::

    from engine.orchestrator.config import OrchestratorConfig

    config = OrchestratorConfig.from_env()
    print(config.ws_url, config.http_url, config.backend)
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Final, Literal

from engine.audio_io.config import AudioConfig
from engine.stt.base import SttConfig
from engine.translate.base import TranslateConfig
from engine.tts.base import Backend as TtsBackend
from engine.tts.base import TtsConfig

__all__ = [
    "BACKENDS",
    "DEFAULT_CORS_ORIGIN",
    "DEFAULT_DB_PATH",
    "DEFAULT_HTTP_PORT",
    "DEFAULT_RECORDINGS_DIR",
    "DEFAULT_WS_HOST",
    "DEFAULT_WS_PORT",
    "ENV_BACKEND",
    "ENV_CORS_ORIGIN",
    "ENV_DB_PATH",
    "ENV_HTTP_PORT",
    "ENV_RECORDINGS_DIR",
    "ENV_TTS_EVENTS",
    "ENV_WS_HOST",
    "ENV_WS_PORT",
    "REPO_ROOT",
    "Backend",
    "OrchestratorConfig",
]

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

#: ``real`` — живые модели и устройства, ``fake`` — фейки (Linux, CI, проверка UI).
Backend = Literal["real", "fake"]
BACKENDS: Final[tuple[str, ...]] = ("real", "fake")

DEFAULT_WS_HOST: Final[str] = "127.0.0.1"
DEFAULT_WS_PORT: Final[int] = 8765
DEFAULT_HTTP_PORT: Final[int] = 8766
DEFAULT_DB_PATH: Final[Path] = REPO_ROOT / "db" / "translator.db"
DEFAULT_RECORDINGS_DIR: Final[Path] = REPO_ROOT / "recordings"
DEFAULT_SCHEMA_PATH: Final[Path] = REPO_ROOT / "db" / "schema.sql"
DEFAULT_CORS_ORIGIN: Final[str] = "http://localhost:3000"

ENV_WS_HOST: Final[str] = "RT_WS_HOST"
ENV_WS_PORT: Final[str] = "RT_WS_PORT"
ENV_HTTP_PORT: Final[str] = "RT_HTTP_PORT"
ENV_DB_PATH: Final[str] = "RT_DB_PATH"
ENV_RECORDINGS_DIR: Final[str] = "RT_RECORDINGS_DIR"
ENV_BACKEND: Final[str] = "RT_BACKEND"
ENV_CORS_ORIGIN: Final[str] = "RT_CORS_ORIGIN"
ENV_TTS_EVENTS: Final[str] = "RT_WS_TTS_CHUNKS"

_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _clean(env.get(key))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key}={raw!r}: ожидалось целое число") from exc


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = _clean(env.get(key))
    if raw is None:
        return default
    value = raw.lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{key}={raw!r}: ожидалось булево значение")


def _env_path(env: Mapping[str, str], key: str, default: Path) -> Path:
    raw = _clean(env.get(key))
    return Path(raw).expanduser() if raw else default


def _env_backend(env: Mapping[str, str], key: str, default: Backend) -> Backend:
    raw = _clean(env.get(key))
    if raw is None:
        return default
    value = raw.lower()
    if value not in BACKENDS:
        raise ValueError(f"{key}={raw!r}: допустимо {' | '.join(BACKENDS)}")
    return "real" if value == "real" else "fake"


@dataclass(frozen=True, slots=True)
class OrchestratorConfig:
    """Настройки движка целиком.

    Attributes:
        ws_host: адрес, на котором слушают оба сервера (по умолчанию только
            localhost — движок не выставляется наружу).
        ws_port: порт WebSocket-сервера событий и команд (контракты, раздел 5).
        http_port: порт REST-сервера истории и голосов (см. README модуля).
        db_path: файл SQLite (``RT_DB_PATH``).
        recordings_dir: каталог WAV-записей сессий (``RT_RECORDINGS_DIR``).
        schema_path: ``db/schema.sql``, применяется при старте (идемпотентно).
        backend: ``real`` — живые модели и устройства Windows, ``fake`` —
            фейковые реализации тех же интерфейсов (Linux, CI, проверка UI).
        cors_origin: источник, которому REST разрешает запросы (UI на :3000).
        emit_tts_chunks: слать ли в WebSocket служебные события ``tts.chunk``
            (UI их игнорирует, трафик заметный — по умолчанию выключено).
        drain_timeout_s: сколько ждать до-обработки очереди фраз при остановке.
        fake_in: WAV для фейкового источника потока ``in`` (live-режим на Linux).
        fake_out: WAV для фейкового источника потока ``out``.
        audio: конфигурация :mod:`engine.audio_io`.
        stt: конфигурация :mod:`engine.stt`.
        translate: конфигурация :mod:`engine.translate`.
        tts: конфигурация :mod:`engine.tts`.
    """

    ws_host: str = DEFAULT_WS_HOST
    ws_port: int = DEFAULT_WS_PORT
    http_port: int = DEFAULT_HTTP_PORT
    db_path: Path = DEFAULT_DB_PATH
    recordings_dir: Path = DEFAULT_RECORDINGS_DIR
    schema_path: Path = DEFAULT_SCHEMA_PATH
    backend: Backend = "real"
    cors_origin: str = DEFAULT_CORS_ORIGIN
    emit_tts_chunks: bool = False
    drain_timeout_s: float = 10.0
    fake_in: Path | None = None
    fake_out: Path | None = None
    audio: AudioConfig = field(default_factory=AudioConfig)
    stt: SttConfig = field(default_factory=SttConfig)
    translate: TranslateConfig = field(default_factory=TranslateConfig)
    tts: TtsConfig = field(default_factory=TtsConfig)

    def __post_init__(self) -> None:
        for name, port in (("ws_port", self.ws_port), ("http_port", self.http_port)):
            if not 0 <= port <= 65535:
                raise ValueError(f"{name}={port}: порт вне диапазона 0..65535")
        if self.ws_port and self.ws_port == self.http_port:
            raise ValueError("ws_port и http_port должны различаться")
        if self.drain_timeout_s <= 0:
            raise ValueError(f"drain_timeout_s должен быть > 0, получено {self.drain_timeout_s}")

    @property
    def ws_url(self) -> str:
        """Адрес WebSocket для UI (``NEXT_PUBLIC_ENGINE_WS``)."""
        return f"ws://{self.ws_host}:{self.ws_port}"

    @property
    def http_url(self) -> str:
        """Адрес REST для UI (``ENGINE_HTTP``)."""
        return f"http://{self.ws_host}:{self.http_port}"

    @property
    def is_fake(self) -> bool:
        """Работаем ли на фейковых бэкендах."""
        return self.backend == "fake"

    def recording_path(self, session_id: int, stream: str) -> Path:
        """Куда пишется WAV одного потока сессии."""
        return self.recordings_dir / f"{session_id}_{stream}.wav"

    def for_backend(self) -> OrchestratorConfig:
        """Согласовать вложенные конфиги с :attr:`backend`.

        При ``backend="fake"`` модули tts и translate тоже переводятся в свои
        фейковые режимы, чтобы ни одна тяжёлая модель не загрузилась.
        """
        if not self.is_fake:
            return self
        return replace(
            self,
            translate=replace(self.translate, backend="fake"),
            tts=replace(self.tts, backend=TtsBackend.FAKE),
        )

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OrchestratorConfig:
        """Собрать конфигурацию из переменных окружения.

        Свои переменные: ``RT_WS_HOST``, ``RT_WS_PORT``, ``RT_HTTP_PORT``,
        ``RT_DB_PATH``, ``RT_RECORDINGS_DIR``, ``RT_BACKEND``,
        ``RT_CORS_ORIGIN``, ``RT_WS_TTS_CHUNKS``. Остальное читают
        ``AudioConfig.from_env`` / ``SttConfig.from_env`` /
        ``TranslateConfig.from_env`` / ``TtsConfig.from_env``.

        Args:
            env: словарь переменных; по умолчанию ``os.environ``.
                ``TranslateConfig`` читает только ``os.environ`` — так решено
                в чужом модуле, здесь это не переопределяется (закон 1).

        Raises:
            ValueError: переменная задана значением, которое нельзя разобрать.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            ws_host=_clean(source.get(ENV_WS_HOST)) or DEFAULT_WS_HOST,
            ws_port=_env_int(source, ENV_WS_PORT, DEFAULT_WS_PORT),
            http_port=_env_int(source, ENV_HTTP_PORT, DEFAULT_HTTP_PORT),
            db_path=_env_path(source, ENV_DB_PATH, DEFAULT_DB_PATH),
            recordings_dir=_env_path(source, ENV_RECORDINGS_DIR, DEFAULT_RECORDINGS_DIR),
            backend=_env_backend(source, ENV_BACKEND, "real"),
            cors_origin=_clean(source.get(ENV_CORS_ORIGIN)) or DEFAULT_CORS_ORIGIN,
            emit_tts_chunks=_env_bool(source, ENV_TTS_EVENTS, False),
            audio=AudioConfig.from_env(source),
            stt=SttConfig.from_env(source),
            translate=TranslateConfig.from_env(),
            tts=TtsConfig.from_env(source),
        )

    def describe(self) -> str:
        """Однострочная сводка для логов при старте."""
        return (
            f"backend={self.backend}, ws={self.ws_url}, http={self.http_url}, "
            f"db={self.db_path}, recordings={self.recordings_dir}"
        )
