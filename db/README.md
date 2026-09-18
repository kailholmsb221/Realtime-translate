# db — хранилище (SQLite)

> **МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ ВЛАДЕЛЬЦА ПРОЕКТА** (CLAUDE.md, закон 2).

SQLite достаточно для MVP: один пользователь, один звонок одновременно
(ARCHITECTURE.md раздел 4.7). Файл БД — `db/translator.db`, в git не попадает
(`.gitignore`: `*.db`). Пишет в БД только `engine/orchestrator` (этап 2),
читает ещё страница истории в UI — через движок, а не напрямую.

## Создание базы

```bash
sqlite3 db/translator.db < db/schema.sql
```

Схема идемпотентна (`CREATE TABLE IF NOT EXISTS`), повторный запуск безопасен.
Соединение должно включать `PRAGMA foreign_keys = ON` — SQLite выключает
внешние ключи по умолчанию в каждом новом соединении.

## Соглашения

- все временные метки — `INTEGER`, unix-время в **миллисекундах** (как поле `ts`
  в контрактах событий);
- `t_start_ms` / `t_end_ms` в `utterances` — смещение **от начала сессии**, мс;
- языки — `'ru' | 'en' | 'kk'`, закреплены `CHECK` (ARCHITECTURE.md раздел 2);
- `speaker` — `'in' | 'out'`, совпадает с полем `stream` из контрактов
  (`in` — собеседник из Zoom, `out` — пользователь).

## Таблицы

### `sessions` — один звонок

| поле | тип | описание |
|---|---|---|
| `id` | INTEGER PK | автоинкремент, он же `session_id` в событии `session.state` |
| `started_at` | INTEGER NOT NULL | начало сессии, unix ms |
| `ended_at` | INTEGER NULL | конец сессии; `NULL` — сессия идёт |
| `lang_from` | TEXT NOT NULL | язык источника (`lang_in` из `session.start`) |
| `lang_to` | TEXT NOT NULL | язык перевода (`lang_out` из `session.start`) |
| `audio_path` | TEXT NULL | путь к WAV сессии; `NULL`, если `record = false` |

Ограничение: `ended_at >= started_at`.

### `utterances` — реплики и их переводы

| поле | тип | описание |
|---|---|---|
| `id` | INTEGER PK | на него ссылается `ref_utterance_id` в `translation.ready` |
| `session_id` | INTEGER NOT NULL | FK → `sessions.id`, `ON DELETE CASCADE` |
| `t_start_ms` | INTEGER NOT NULL | начало фразы от старта сессии |
| `t_end_ms` | INTEGER NOT NULL | конец фразы, `>= t_start_ms` |
| `speaker` | TEXT NOT NULL | `'in'` или `'out'` |
| `lang` | TEXT NOT NULL | язык оригинала |
| `text` | TEXT NOT NULL | оригинал из `stt.final` |
| `translation` | TEXT NULL | перевод из `translation.ready`; `NULL` пока не готов |
| `translation_lang` | TEXT NULL | язык перевода |

Порядок записи: строка создаётся по `stt.final` (перевода ещё нет), затем
`UPDATE` по `translation.ready` — поэтому `translation` допускает `NULL`.

### `voices` — голосовые профили (XTTS-v2)

| поле | тип | описание |
|---|---|---|
| `id` | TEXT PK | = `voice_id` из контрактов (строка) |
| `name` | TEXT NOT NULL UNIQUE | человекочитаемое имя профиля |
| `lang` | TEXT NOT NULL | язык сэмпла |
| `sample_path` | TEXT NOT NULL | WAV-сэмпл 15–30 сек для клонирования |
| `created_at` | INTEGER NOT NULL | unix ms |

Для `kk` клонирования нет — используется стандартный голос KazakhTTS2.
Правовая заметка (ARCHITECTURE.md раздел 8): клонировать можно только
собственный голос пользователя или голос с явного согласия его владельца.

## Индексы

| индекс | назначение |
|---|---|
| `idx_sessions_started_at` | список сессий в истории, новые сверху |
| `idx_utterances_session_id` | все реплики сессии |
| `idx_utterances_session_t_start` | транскрипт сессии по порядку времени |
| `idx_voices_lang` | выбор голосовых профилей по языку |

## Миграции

Пока миграций нет — MVP пересоздаёт базу из `schema.sql`. Когда появятся,
класть их сюда как `db/migrations/NNN_<описание>.sql` (по согласованию
с владельцем).

## Решение владельца

`voices.id` — TEXT и равен `voice_id` из контрактов (строка вида `v_<hex>`), приведения типов не требуется. `utterances.speaker` = `stream` (`'in'|'out'`), закреплено CHECK-ограничением.
