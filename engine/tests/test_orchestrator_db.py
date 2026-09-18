"""Тесты слоя БД оркестратора (``engine/orchestrator/db.py``).

Проверяют, что движок работает с ``db/schema.sql`` как есть: применяет схему,
пишет реплики и переводы, считает производные поля для REST и синхронизирует
голосовые профили с диска. Реальные модели не нужны.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import numpy as np
import pytest

from engine.contracts.events import Lang, Stream
from engine.orchestrator.db import Database
from engine.tts.audio import write_wav
from engine.tts.base import OUTPUT_SAMPLE_RATE
from engine.tts.voices import VoiceStore

pytestmark = pytest.mark.asyncio

SAMPLE_SECONDS = 16


@pytest.fixture
async def db(tmp_path: Path) -> AsyncIterator[Database]:
    """Свежая база в каталоге теста; закрывается после теста."""
    database = Database(tmp_path / "translator.db")
    await database.init()
    try:
        yield database
    finally:
        await database.close()


def _sample_wav(path: Path, seconds: int = SAMPLE_SECONDS) -> Path:
    """WAV нужной длительности для ``VoiceStore`` (5-60 секунд)."""
    write_wav(path, np.zeros(OUTPUT_SAMPLE_RATE * seconds, dtype=np.int16), OUTPUT_SAMPLE_RATE)
    return path


async def test_init_applies_schema(tmp_path: Path) -> None:
    """``init()`` создаёт файл и таблицы из ``db/schema.sql`` (идемпотентно)."""
    database = Database(tmp_path / "nested" / "translator.db")
    await database.init()
    await database.init()  # повторный вызов безопасен
    try:
        assert database.path.is_file()
        assert await database.list_sessions() == []
        assert await database.list_voices() == []
    finally:
        await database.close()


async def test_session_lifecycle_and_derived_fields(db: Database) -> None:
    """Идущая сессия — ``duration_ms is None``; закрытая считает длительность."""
    session_id = await db.create_session(Lang.EN, Lang.RU, started_at=1_000)

    running = await db.get_session(session_id)
    assert running is not None
    assert running["lang_from"] == "en"
    assert running["lang_to"] == "ru"
    assert running["ended_at"] is None
    assert running["duration_ms"] is None
    assert running["utterance_count"] == 0

    await db.add_utterance(session_id, 0, 940, Stream.IN, Lang.EN, "hi")
    await db.end_session(session_id, ended_at=6_000)

    ended = await db.get_session(session_id)
    assert ended is not None
    assert ended["duration_ms"] == 5_000
    assert ended["utterance_count"] == 1


async def test_get_session_unknown_returns_none(db: Database) -> None:
    """Неизвестный id — ``None`` (REST превратит это в 404)."""
    assert await db.get_session(404) is None


async def test_list_sessions_newest_first(db: Database) -> None:
    """Список истории отсортирован по ``started_at`` вниз."""
    old = await db.create_session(Lang.RU, Lang.EN, started_at=1_000)
    new = await db.create_session(Lang.EN, Lang.RU, started_at=2_000)
    rows = await db.list_sessions()
    assert [row["id"] for row in rows] == [new, old]
    assert [row["id"] for row in await db.list_sessions(limit=1)] == [new]


async def test_add_utterance_and_set_translation(db: Database) -> None:
    """Реплика пишется без перевода, затем перевод дописывается по её id."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    first = await db.add_utterance(session_id, 0, 940, Stream.IN, Lang.EN, "hi there")
    second = await db.add_utterance(session_id, 2_000, 3_000, Stream.OUT, Lang.RU, "привет")
    assert second > first

    rows = await db.list_utterances(session_id)
    assert [row["text"] for row in rows] == ["hi there", "привет"]
    assert rows[0]["translation"] is None
    assert rows[0]["speaker"] == "in"
    assert rows[1]["speaker"] == "out"

    await db.set_translation(first, "привет", Lang.RU)
    rows = await db.list_utterances(session_id)
    assert rows[0]["translation"] == "привет"
    assert rows[0]["translation_lang"] == "ru"


async def test_utterances_ordered_by_time(db: Database) -> None:
    """Транскрипт отдаётся по возрастанию ``t_start_ms``, а не по id."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    await db.add_utterance(session_id, 5_000, 5_500, Stream.IN, Lang.EN, "later")
    await db.add_utterance(session_id, 1_000, 1_500, Stream.IN, Lang.EN, "earlier")
    rows = await db.list_utterances(session_id)
    assert [row["text"] for row in rows] == ["earlier", "later"]


async def test_set_audio_path(db: Database) -> None:
    """``audio_path`` проставляется после создания строки (запись включена)."""
    session_id = await db.create_session(Lang.EN, Lang.RU)
    await db.set_audio_path(session_id, "recordings/1_in.wav")
    session = await db.get_session(session_id)
    assert session is not None
    assert session["audio_path"] == "recordings/1_in.wav"


async def test_upsert_voice_is_idempotent(db: Database) -> None:
    """Повторный upsert того же ``voice_id`` обновляет строку, а не дублирует."""
    await db.upsert_voice("v_1a2b3c4d", "мой голос", Lang.RU, "voices/v_1a2b3c4d/sample.wav", 10)
    await db.upsert_voice("v_1a2b3c4d", "мой голос 2", Lang.EN, "voices/v_1a2b3c4d/sample.wav", 20)

    voices = await db.list_voices()
    assert len(voices) == 1
    assert voices[0]["name"] == "мой голос 2"
    assert voices[0]["lang"] == "en"
    assert await db.get_voice("v_1a2b3c4d") == voices[0]


async def test_delete_voice(db: Database) -> None:
    """Строку профиля можно убрать из индекса."""
    await db.upsert_voice("v_1a2b3c4d", "голос", Lang.RU, "voices/v/sample.wav", 1)
    await db.delete_voice("v_1a2b3c4d")
    assert await db.list_voices() == []


async def test_sync_voices_from_store(db: Database, tmp_path: Path) -> None:
    """Профили с диска попадают в таблицу; диск — источник правды."""
    store = VoiceStore(tmp_path / "voices")
    profile = store.create_voice(_sample_wav(tmp_path / "sample.wav"), name="мой", lang=Lang.RU)

    synced = await db.sync_voices_from_store(store)
    assert [row["id"] for row in synced] == [profile.id]

    rows = await db.list_voices()
    assert len(rows) == 1
    assert rows[0]["id"] == profile.id
    assert rows[0]["name"] == "мой"
    assert rows[0]["lang"] == "ru"
    assert rows[0]["sample_path"] == str(profile.sample_path)
    assert rows[0]["created_at"] == profile.created_at_ms

    # Повторная синхронизация ничего не ломает и не плодит строк.
    await db.sync_voices_from_store(store)
    assert len(await db.list_voices()) == 1


async def test_sync_voices_keeps_rows_missing_on_disk(db: Database, tmp_path: Path) -> None:
    """Строки удалённых с диска голосов остаются: на них ссылается история."""
    await db.upsert_voice("v_deleted", "старый", Lang.RU, "voices/v_deleted/sample.wav", 1)
    store = VoiceStore(tmp_path / "voices")
    await db.sync_voices_from_store(store)
    assert [row["id"] for row in await db.list_voices()] == ["v_deleted"]
