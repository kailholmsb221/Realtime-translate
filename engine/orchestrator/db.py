"""Доступ к SQLite (``db/schema.sql``) поверх ``aiosqlite``.

Единственный модуль, который пишет в базу (db/README.md). Схему не меняет —
только применяет её из ``db/schema.sql`` (CLAUDE.md, закон 2): файл
идемпотентен (``CREATE TABLE IF NOT EXISTS``), поэтому ``init()`` безопасно
вызывать при каждом старте.

Соглашения (db/README.md):

* все временные метки — unix-время в миллисекундах;
* ``utterances.t_start_ms`` / ``t_end_ms`` — смещение от начала сессии;
* ``utterances.speaker`` совпадает со ``stream`` из контрактов (``in``/``out``);
* ``voices.id`` — это ``voice_id`` из контрактов; источник правды — профили на
  диске (:class:`~engine.tts.voices.VoiceStore`), таблица служит индексом для
  истории и UI и синхронизируется методом :meth:`Database.sync_voices_from_store`.

Пример::

    db = Database(Path("db/translator.db"))
    await db.init()
    session_id = await db.create_session(Lang.EN, Lang.RU)
    row_id = await db.add_utterance(session_id, 0, 940, Stream.IN, Lang.EN, "hi")
    await db.set_translation(row_id, "привет", Lang.RU)
    await db.close()
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Final

import aiosqlite

from engine.contracts.events import Lang, Stream
from engine.orchestrator.config import DEFAULT_SCHEMA_PATH
from engine.tts.voices import VoiceProfile, VoiceStore

__all__ = ["Database", "SessionRow", "UtteranceRow", "VoiceRow"]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.db")

#: Строка ``sessions`` плюс производные поля ``utterance_count`` / ``duration_ms``.
SessionRow = dict[str, Any]
#: Строка ``utterances`` как есть.
UtteranceRow = dict[str, Any]
#: Строка ``voices`` как есть.
VoiceRow = dict[str, Any]

_SESSION_COLUMNS: Final[str] = """
    s.id            AS id,
    s.started_at    AS started_at,
    s.ended_at      AS ended_at,
    s.lang_from     AS lang_from,
    s.lang_to       AS lang_to,
    s.audio_path    AS audio_path,
    (SELECT COUNT(*) FROM utterances u WHERE u.session_id = s.id) AS utterance_count,
    CASE WHEN s.ended_at IS NULL THEN NULL ELSE s.ended_at - s.started_at END AS duration_ms
"""


def now_ms() -> int:
    """Текущее unix-время в миллисекундах (как поле ``ts`` в контрактах)."""
    return int(time.time() * 1000)


class Database:
    """Асинхронный доступ к базе движка.

    Args:
        path: файл SQLite; каталог создаётся при :meth:`init`.
        schema_path: путь к ``db/schema.sql``.
    """

    __slots__ = ("_conn", "_lock", "_path", "_schema_path")

    def __init__(self, path: str | Path, schema_path: str | Path = DEFAULT_SCHEMA_PATH) -> None:
        self._path = Path(path).expanduser()
        self._schema_path = Path(schema_path).expanduser()
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> Path:
        """Файл базы."""
        return self._path

    @property
    def is_open(self) -> bool:
        """Открыто ли соединение."""
        return self._conn is not None

    # --- жизненный цикл ----------------------------------------------------

    async def init(self) -> None:
        """Открыть соединение и применить ``db/schema.sql`` (идемпотентно)."""
        if self._conn is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        conn = await aiosqlite.connect(self._path)
        conn.row_factory = aiosqlite.Row
        try:
            schema = self._schema_path.read_text(encoding="utf-8")
        except OSError as exc:
            await conn.close()
            raise RuntimeError(f"не читается схема {self._schema_path}: {exc}") from exc
        await conn.executescript(schema)
        # SQLite выключает внешние ключи в каждом новом соединении (db/README.md).
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.commit()
        self._conn = conn
        logger.info("база готова: %s", self._path)

    async def close(self) -> None:
        """Закрыть соединение (повторный вызов безопасен)."""
        conn, self._conn = self._conn, None
        if conn is not None:
            await conn.close()

    async def __aenter__(self) -> Database:
        await self.init()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.init() не вызван")
        return self._conn

    # --- сессии ------------------------------------------------------------

    async def create_session(
        self,
        lang_from: Lang | str,
        lang_to: Lang | str,
        started_at: int | None = None,
        audio_path: str | Path | None = None,
    ) -> int:
        """Создать сессию и вернуть её ``id`` (он же ``session_id`` в событиях)."""
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO sessions (started_at, ended_at, lang_from, lang_to, audio_path) "
                "VALUES (?, NULL, ?, ?, ?)",
                (
                    now_ms() if started_at is None else started_at,
                    str(Lang(lang_from).value),
                    str(Lang(lang_to).value),
                    None if audio_path is None else str(audio_path),
                ),
            )
            await self._db.commit()
        session_id = cursor.lastrowid
        if session_id is None:  # pragma: no cover — sqlite всегда отдаёт rowid
            raise RuntimeError("SQLite не вернул id новой сессии")
        logger.info("сессия %d начата (%s -> %s)", session_id, lang_from, lang_to)
        return int(session_id)

    async def set_audio_path(self, session_id: int, audio_path: str | Path | None) -> None:
        """Прописать путь к WAV сессии (запись включается после создания строки)."""
        async with self._lock:
            await self._db.execute(
                "UPDATE sessions SET audio_path = ? WHERE id = ?",
                (None if audio_path is None else str(audio_path), session_id),
            )
            await self._db.commit()

    async def end_session(self, session_id: int, ended_at: int | None = None) -> None:
        """Закрыть сессию (``ended_at``); повторный вызов просто перезапишет метку."""
        async with self._lock:
            await self._db.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ?",
                (now_ms() if ended_at is None else ended_at, session_id),
            )
            await self._db.commit()
        logger.info("сессия %d закрыта", session_id)

    async def list_sessions(self, limit: int | None = None) -> list[SessionRow]:
        """Список сессий, новые сверху, с ``utterance_count`` и ``duration_ms``."""
        sql = f"SELECT {_SESSION_COLUMNS} FROM sessions s ORDER BY s.started_at DESC, s.id DESC"
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        async with self._db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_session(self, session_id: int) -> SessionRow | None:
        """Одна сессия с производными полями или ``None``, если её нет."""
        sql = f"SELECT {_SESSION_COLUMNS} FROM sessions s WHERE s.id = ?"
        async with self._db.execute(sql, (session_id,)) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row is not None else None

    # --- реплики -----------------------------------------------------------

    async def add_utterance(
        self,
        session_id: int,
        t_start_ms: int,
        t_end_ms: int,
        speaker: Stream | str,
        lang: Lang | str,
        text: str,
        translation: str | None = None,
        translation_lang: Lang | str | None = None,
    ) -> int:
        """Записать реплику (``stt.final``) и вернуть её ``id``.

        Этот ``id`` уезжает в ``translation.ready.ref_utterance_id`` — по нему
        UI связывает оригинал и перевод (решение владельца, db/README.md).
        """
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO utterances (session_id, t_start_ms, t_end_ms, speaker, lang, "
                "text, translation, translation_lang) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    max(0, int(t_start_ms)),
                    max(0, int(t_end_ms)),
                    str(Stream(speaker).value),
                    str(Lang(lang).value),
                    text,
                    translation,
                    None if translation_lang is None else str(Lang(translation_lang).value),
                ),
            )
            await self._db.commit()
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover — sqlite всегда отдаёт rowid
            raise RuntimeError("SQLite не вернул id новой реплики")
        return int(row_id)

    async def set_translation(
        self,
        utterance_id: int,
        text: str,
        lang: Lang | str,
    ) -> None:
        """Дописать перевод к реплике (``translation.ready``)."""
        async with self._lock:
            await self._db.execute(
                "UPDATE utterances SET translation = ?, translation_lang = ? WHERE id = ?",
                (text, str(Lang(lang).value), utterance_id),
            )
            await self._db.commit()

    async def list_utterances(self, session_id: int) -> list[UtteranceRow]:
        """Транскрипт сессии по возрастанию ``t_start_ms``."""
        async with self._db.execute(
            "SELECT id, session_id, t_start_ms, t_end_ms, speaker, lang, text, translation, "
            "translation_lang FROM utterances WHERE session_id = ? ORDER BY t_start_ms, id",
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # --- голоса ------------------------------------------------------------

    async def list_voices(self) -> list[VoiceRow]:
        """Голосовые профили, новые снизу (порядок создания)."""
        async with self._db.execute(
            "SELECT id, name, lang, sample_path, created_at FROM voices ORDER BY created_at, id"
        ) as cursor:
            rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_voice(self, voice_id: str) -> VoiceRow | None:
        """Один профиль или ``None``."""
        async with self._db.execute(
            "SELECT id, name, lang, sample_path, created_at FROM voices WHERE id = ?",
            (voice_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row is not None else None

    async def upsert_voice(
        self,
        voice_id: str,
        name: str,
        lang: Lang | str,
        sample_path: str | Path,
        created_at: int | None = None,
    ) -> VoiceRow:
        """Добавить или обновить строку профиля голоса.

        Raises:
            sqlite3.IntegrityError: имя профиля занято другим ``voice_id``
                (``name UNIQUE`` в ``db/schema.sql``).
        """
        row: VoiceRow = {
            "id": voice_id,
            "name": name,
            "lang": str(Lang(lang).value),
            "sample_path": str(sample_path),
            "created_at": now_ms() if created_at is None else created_at,
        }
        async with self._lock:
            await self._db.execute(
                "INSERT INTO voices (id, name, lang, sample_path, created_at) "
                "VALUES (:id, :name, :lang, :sample_path, :created_at) "
                "ON CONFLICT(id) DO UPDATE SET name = excluded.name, lang = excluded.lang, "
                "sample_path = excluded.sample_path, created_at = excluded.created_at",
                row,
            )
            await self._db.commit()
        return row

    async def delete_voice(self, voice_id: str) -> None:
        """Убрать профиль из индекса (файлы на диске удаляет ``VoiceStore``)."""
        async with self._lock:
            await self._db.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
            await self._db.commit()

    async def sync_voices_from_store(self, store: VoiceStore) -> list[VoiceRow]:
        """Подтянуть в таблицу профили с диска (источник правды — диск).

        Профили, которых нет в БД, добавляются; существующие обновляются.
        Строки, которых уже нет на диске, не удаляются: история сессий может
        ссылаться на удалённый голос.

        Returns:
            Строки, которые были добавлены или обновлены.
        """
        profiles: list[VoiceProfile] = await asyncio.to_thread(store.list)
        synced: list[VoiceRow] = []
        for profile in profiles:
            try:
                synced.append(
                    await self.upsert_voice(
                        profile.id,
                        profile.name,
                        profile.lang,
                        profile.sample_path,
                        profile.created_at_ms,
                    )
                )
            except sqlite3.IntegrityError as exc:
                logger.warning(
                    "профиль %s (%s) не попал в таблицу voices: %s", profile.id, profile.name, exc
                )
        if synced:
            logger.info("синхронизировано голосов с диска: %d", len(synced))
        return synced
