-- Realtime Translator — схема SQLite (ARCHITECTURE.md раздел 4.7).
-- МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ ВЛАДЕЛЬЦА ПРОЕКТА (CLAUDE.md, закон 2).
--
-- Применение:  sqlite3 db/translator.db < db/schema.sql
-- Все временные метки — INTEGER, unix-время в миллисекундах (как поле `ts`
-- в контрактах событий). Языки — те же три, что в ARCHITECTURE.md разделе 2.

PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- Сессии перевода: один звонок = одна строка.
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  INTEGER NOT NULL,                 -- unix ms
    ended_at    INTEGER,                          -- NULL пока сессия идёт
    lang_from   TEXT    NOT NULL CHECK (lang_from IN ('ru', 'en', 'kk')),
    lang_to     TEXT    NOT NULL CHECK (lang_to   IN ('ru', 'en', 'kk')),
    audio_path  TEXT,                             -- WAV сессии на диске, NULL если record=false
    CHECK (ended_at IS NULL OR ended_at >= started_at)
);

-- Реплики: результат stt.final + соответствующий translation.ready.
-- speaker — сторона разговора, совпадает со `stream` из контрактов:
-- 'in' — собеседник (loopback из Zoom), 'out' — пользователь (микрофон).
CREATE TABLE IF NOT EXISTS utterances (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id       INTEGER NOT NULL,
    t_start_ms       INTEGER NOT NULL CHECK (t_start_ms >= 0),  -- от начала сессии
    t_end_ms         INTEGER NOT NULL CHECK (t_end_ms >= 0),
    speaker          TEXT    NOT NULL CHECK (speaker IN ('in', 'out')),  -- = stream
    lang             TEXT    NOT NULL CHECK (lang IN ('ru', 'en', 'kk')),
    text             TEXT    NOT NULL,            -- оригинал (stt.final.text)
    translation      TEXT,                        -- NULL пока перевод не готов
    translation_lang TEXT    CHECK (translation_lang IS NULL
                                    OR translation_lang IN ('ru', 'en', 'kk')),
    CHECK (t_end_ms >= t_start_ms),
    FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
);

-- Голосовые профили для клонирования (XTTS-v2). Для kk клона нет.
-- Правовая заметка (ARCHITECTURE.md раздел 8): клонировать можно только
-- собственный голос пользователя или голос с явного согласия владельца.
CREATE TABLE IF NOT EXISTS voices (
    id          TEXT    PRIMARY KEY,               -- voice_id из контрактов (строка, напр. 'v_ab12cd')
    name        TEXT    NOT NULL UNIQUE,
    lang        TEXT    NOT NULL CHECK (lang IN ('ru', 'en', 'kk')),
    sample_path TEXT    NOT NULL,                 -- WAV-сэмпл 15-30 сек
    created_at  INTEGER NOT NULL                  -- unix ms
);

-- Индексы: история сессий и выборка транскрипта.
CREATE INDEX IF NOT EXISTS idx_sessions_started_at
    ON sessions (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_utterances_session_id
    ON utterances (session_id);
CREATE INDEX IF NOT EXISTS idx_utterances_session_t_start
    ON utterances (session_id, t_start_ms);
CREATE INDEX IF NOT EXISTS idx_voices_lang
    ON voices (lang);
