# ARCHITECTURE.md — Realtime Translator (надстройка над Zoom/Meet)

## 1. Цель проекта

Локальное приложение, работающее поверх Zoom/Google Meet:

- переводит речь собеседника (en/kk → ru) в реальном времени: субтитры + озвучка в наушники;
- переводит речь пользователя (ru → en/kk) и отправляет переведённый голос в Zoom через виртуальный микрофон;
- для ru/en использует клон голоса говорящего (XTTS-v2), для kk — стандартный TTS-голос;
- записывает транскрипт и переводы разговора в базу данных;
- показывает маленькое окно-ассистент с живым транскриптом (LLM-подсказки — в следующих версиях).

## 2. Ограничения (не нарушать)

| Ограничение | Значение |
|---|---|
| Задержка "конец фразы → озвучка перевода" | ≤ 2.5 сек |
| GPU | RTX 4050 Laptop, 6 GB VRAM |
| CPU | Ryzen 5 5600H |
| Бюджет | 0 — только бесплатные open-source модели, никаких платных API |
| ОС | Windows (WASAPI loopback, VB-Audio Virtual Cable) |
| Языки | ru, en, kk |
| Масштаб | 1 пользователь, 1 звонок одновременно |

## 3. Структура репозитория

```
/engine                  # Python 3.11, asyncio
  /audio_io              # захват/вывод звука (Агент A)
  /stt                   # распознавание речи (Агент B)
  /translate             # перевод (Агент C)
  /tts                   # синтез речи (Агент D)
  /orchestrator          # склейка пайплайна, WebSocket-сервер, запись в БД (Этап 2)
  /contracts             # JSON-схемы событий и API — МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ
  /tests
/ui                      # Next.js (React, TypeScript) — субтитры, ассистент, история (Агент E)
/db                      # схема БД, миграции
/scripts                 # установка моделей, smoke-тесты
ARCHITECTURE.md
CLAUDE.md
```

## 4. Компоненты

### 4.1 audio_io (Python)

- Захват WASAPI loopback (звук собеседника из Zoom) — 16 kHz mono chunks.
- Захват физического микрофона пользователя.
- Вывод: (а) наушники — перевод собеседника; (б) VB-Audio Virtual Cable — перевод пользователя (в Zoom микрофоном выбран кабель).
- Библиотеки: sounddevice / pyaudiowpatch (WASAPI loopback).
- Интерфейс: async-генераторы аудиочанков + async-функции воспроизведения. Абстракция устройства — чтобы позже добавить Linux.

### 4.2 stt (Python)

- Silero VAD режет поток на фразы.
- faster-whisper, модель `small`, `compute_type=int8_float16` (≈1 GB VRAM).
- Для kk предусмотреть подмену модели на файнтюн (например ISSAI whisper-kk) через конфиг.
- Вход: аудиочанки 16 kHz. Выход: события `stt.partial` / `stt.final` (см. контракты).

### 4.3 translate (Python)

- NLLB-200-distilled-600M, направления ru↔en, ru↔kk, en↔kk.
- Запуск на CPU (GPU оставляем STT и TTS). Абстракция Provider — позже можно подключить LLM API.
- Вход: `stt.final`. Выход: `translation.ready`.

### 4.4 tts (Python)

- XTTS-v2 для ru/en — клонирование по сэмплу 15–30 сек (≈2–3 GB VRAM).
- Для kk — без клона: KazakhTTS2 (ISSAI) если доступен загружаемый чекпоинт, иначе `facebook/mms-tts-kaz` (VITS, CPU, ~150 MB) — оба open-source, выбор через конфиг.
- Единый интерфейс: `synthesize(text, lang, voice_id) -> async аудиочанки 24 kHz`.
- Эндпоинт/функция `create_voice(sample_wav) -> voice_id`, профили голосов хранятся на диске.

### 4.5 orchestrator (Python)

- Два конвейера: inbound (loopback → STT → MT → TTS → наушники + субтитры) и outbound (микрофон → STT → MT → TTS → вирт. кабель + субтитры).
- WebSocket-сервер (localhost:8765) — транслирует все события в UI.
- Пишет `stt.final` + `translation.ready` в БД с таймкодами; сырое аудио сессии — WAV-файлы на диск.
- Управление сессией: start/stop записи, выбор направления языков, выбор голосового профиля.

### 4.6 ui (Next.js)

- Маленькое окно (localhost:3000): две ленты субтитров (оригинал/перевод), индикаторы задержки, кнопки start/stop.
- Панель ассистента: живой транскрипт последних N минут (LLM-подсказки — заглушка с TODO).
- Страница истории: список сессий из БД, просмотр транскрипта, проигрывание аудио.

### 4.7 Хранилище

SQLite (файл в `/db`) для MVP — достаточно для 1 пользователя. Схема:

```
sessions(id, started_at, ended_at, lang_from, lang_to, audio_path)
utterances(id, session_id, t_start_ms, t_end_ms, speaker, lang, text, translation, translation_lang)  -- speaker = stream ('in'|'out')
voices(id TEXT, name, lang, sample_path, created_at)  -- id = voice_id из контрактов
```

## 5. Контракты событий (WebSocket, JSON)

Все события: `{ "type": string, "ts": unix_ms, "payload": {...} }`

| type | payload |
|---|---|
| `stt.partial` | `{stream: "in"\|"out", lang, text}` |
| `stt.final` | `{stream, lang, text, t_start_ms, t_end_ms}` |
| `translation.ready` | `{stream, src_lang, dst_lang, src_text, text, ref_utterance_id}` |
| `tts.chunk` | `{stream, seq, pcm_base64}` — служебное, UI не обязателен |
| `session.state` | `{status: "idle"\|"running", session_id, langs, voice_id}` |
| `metrics.latency` | `{stream, stt_ms, mt_ms, tts_ms, total_ms}` |

Команды от UI к движку (WebSocket, тот же конверт `{type, ts, payload}`):

| type | payload |
|---|---|
| `session.start` | `{lang_in, lang_out, voice_id, record: bool}` |
| `session.stop` | `{}` |

Полные JSON-схемы — в `/engine/contracts/`. Изменение схем = отдельный PR с явным одобрением владельца.

## 6. Бюджет VRAM (6 GB)

| Компонент | VRAM |
|---|---|
| faster-whisper small int8 | ~1.0 GB |
| XTTS-v2 | ~2.5 GB |
| NLLB-600M | 0 (CPU) |
| kk TTS (MMS-TTS kaz / KazakhTTS2) | 0 (CPU) |
| Резерв/фрагментация | ~1.5 GB |

Модели загружаются один раз при старте движка и живут в памяти. Одновременная загрузка whisper `medium` и XTTS запрещена.

## 7. Этапы реализации

1. **Файловый режим:** WAV на входе → переведённый WAV на выходе (без realtime). Смоук-тест всей цепочки.
2. **Inbound live:** loopback → субтитры + голос в наушники (одно направление).
3. **Outbound live:** микрофон → вирт. кабель (Zoom слышит перевод).
4. **Запись:** сессии и транскрипты в SQLite, история в UI.
5. **Ассистент:** LLM-подсказки по транскрипту (когда появится бюджет/модель).

## 8. Правовые заметки

- Клонировать можно только собственный голос пользователя (или с явного согласия владельца голоса).
- При записи разговора собеседник должен быть уведомлён — UI показывает статус записи, ответственность за уведомление на пользователе.
