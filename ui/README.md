# ui — окно перевода, ассистент и история (Агент E)

Next.js 15 (App Router) + TypeScript strict + Tailwind v4. Состояние — только
React (`useState`/`useReducer`-логика в `src/lib/state.ts`), сторонних
стейт-менеджеров и UI-китов нет (CLAUDE.md, «Технический стандарт»).

Реализует ARCHITECTURE.md раздел 4.6:

- маленькое окно (~480×640, тёмная тема) с двумя лентами субтитров
  «Собеседник» (`stream: "in"`) и «Вы» (`stream: "out"`): оригинал + перевод;
- индикаторы задержки `stt/mt/tts/total` по каждому потоку, `total` подсвечивается
  красным при превышении бюджета 2500 мс (ARCHITECTURE.md раздел 2);
- кнопки Start/Stop, выбор `lang_in`/`lang_out` (ru/en/kk), `voice_id` и флаг записи
  с подписью «Запись включена — уведомите собеседника» (ARCHITECTURE.md раздел 8);
- панель ассистента: живой транскрипт последних 5 минут + кнопка «Подсказать ответ»
  (пока отключена, TODO — LLM-подсказки, этап 5);
- `/history` — список сессий, `/history/[id]` — транскрипт с таймкодами, переводами
  и `<audio>`, если у сессии есть `audio_path`.

Вёрстка рассчитана на ширину от 360 px (окно можно сжимать).

## Запуск

```bash
cd ui
npm install          # если lock-файла нет и npm падает на peer-deps: npm install --legacy-peer-deps
npm run mock         # терминал 1: фейковый движок на ws://localhost:8765
npm run dev          # терминал 2: http://localhost:3000
```

Без движка и без мока UI тоже открывается: статус покажет «нет связи», Start будет
недоступен, история придёт из фикстур `mocks/data.json`.

Прочие команды:

| Команда | Что делает |
|---|---|
| `npm test` | vitest (редьюсер, парсер, контракты, сценарий, рендер окна) |
| `npm run test:watch` | то же в watch-режиме |
| `npm run test:coverage` | отчёт покрытия по `src/lib` и `src/components` |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run lint` | ESLint (`next/core-web-vitals` + `next/typescript`) |
| `npm run build` | production-сборка |
| `npm run mock` | mock-движок (`MOCK_WS_PORT`, `MOCK_LOOP=1` — крутить сценарий по кругу) |

## Переменные окружения

Пример — в `.env.example`, локальные значения кладите в `.env.local`.

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `NEXT_PUBLIC_ENGINE_WS` | `ws://localhost:8765` | WebSocket движка (события + команды) |
| `ENGINE_HTTP` | `http://localhost:8766` | REST движка; читается на сервере Next.js в route handlers |
| `NEXT_PUBLIC_ENGINE_HTTP` | не задана | если задать — браузер ходит в REST движка напрямую, минуя заглушки (нужен CORS) |
| `MOCK_WS_PORT` | `8765` | порт mock-движка |
| `MOCK_LOOP` | не задана | `1` — повторять сценарий бесконечно |

## Структура

```
ui/
  src/lib/contracts.ts    типы событий 1:1 с engine/contracts/*.json, isEvent, parseEnvelope
  src/lib/state.ts        редьюсер: ленты субтитров, привязка переводов, метрики, транскрипт
  src/lib/ws.ts           useEngineSocket: реконнект, статус, буфер событий, send
  src/lib/api.ts          REST-клиент истории и голосов + типы строк БД
  src/lib/format.ts       форматтеры времени/языков
  src/lib/server/history.ts  серверная логика заглушек: прокси в движок или mocks/data.json
  src/components/         StatusBar, LatencyBar, Controls, SubtitleLane, AssistantPanel, TranslatorWindow
  src/app/page.tsx        главное окно (client) — связывает сокет и TranslatorWindow
  src/app/history/…       список сессий и транскрипт сессии
  src/app/api/…           route handlers-заглушки REST (работают без движка)
  mocks/server.mjs        WebSocket mock-движок (пакет ws)
  mocks/scenario.json     сценарий событий (валидируется по JSON-схемам в тестах)
  mocks/data.json         фикстуры истории (строки sessions/utterances/voices)
  tests/                  vitest + @testing-library/react (jsdom)
```

### Как UI работает с контрактами

`src/lib/contracts.ts` — зеркало `engine/contracts/*.json` на TypeScript:
`Envelope<T>`, `SttPartial`, `SttFinal`, `TranslationReady`, `TtsChunk`,
`SessionState`, `MetricsLatency`, `SessionStart`, `SessionStop`, `Lang`, `Stream`.
Схемы не копируются и не меняются — они неприкосновенны (CLAUDE.md, закон 2).
Расхождение ловит `tests/contracts.test.ts`: он читает JSON-схемы с диска и
сверяет набор `type`-констант и required-поля payload с ключами TS-типов.

`parseEnvelope(raw)` возвращает `Envelope | null`: проверяет `type` (известный),
`ts` (integer), `payload` (все поля на месте, лишних нет, enum-ы соблюдены).
Невалидные сообщения не роняют UI — они считаются в `invalidCount`.
`tts.chunk` разбирается, но редьюсером игнорируется (аудио играет движок).

Команды собираются хелперами `sessionStart(payload)` / `sessionStop()`;
хук отдаёт удобные `engine.start(payload)` и `engine.stop()`.

## ПРЕДЛОЖЕНИЕ: REST-контракт истории для оркестратора

> Это **предложение** Агента E, а не утверждённый контракт. В `ARCHITECTURE.md`
> и `engine/contracts/` REST не описан — там только WebSocket. Прошу владельца
> проекта утвердить (или поправить) формы ниже; тогда их можно будет вынести в
> `engine/contracts/` отдельным PR. Пока контракт не утверждён, UI ходит в
> собственные заглушки `src/app/api/**` поверх `mocks/data.json`.

База: `http://localhost:8766` (порт рядом с WebSocket 8765). Кодировка UTF-8,
все временные метки — `integer`, unix-время в миллисекундах, как `ts` в событиях.
Поля названы ровно как колонки в `db/schema.sql`, чтобы строки отдавались без
переименований.

### `GET /api/sessions`

Список сессий, новые сверху.

```json
{
  "sessions": [
    {
      "id": 3,
      "started_at": 1758240000000,
      "ended_at": 1758240315000,
      "lang_from": "en",
      "lang_to": "ru",
      "audio_path": "recordings/session-3.wav",
      "utterance_count": 5,
      "duration_ms": 315000
    }
  ]
}
```

- `id`, `started_at`, `ended_at`, `lang_from`, `lang_to`, `audio_path` — строка
  таблицы `sessions` как есть (`ended_at` и `audio_path` могут быть `null`);
- `utterance_count` — **производное**, `COUNT(*)` по `utterances` этой сессии;
- `duration_ms` — **производное**, `ended_at - started_at`, `null` для идущей сессии.

Оба производных поля нужны списку истории (ARCHITECTURE.md 4.6: «дата, языки,
длительность, кол-во реплик»). Если движку их считать неудобно — скажите, UI
посчитает `duration_ms` сам, но `utterance_count` без запроса всех реплик не получить.

### `GET /api/sessions/:id`

Сессия и её транскрипт (реплики по возрастанию `t_start_ms`).

```json
{
  "session": { "...": "как в /api/sessions" },
  "utterances": [
    {
      "id": 101,
      "session_id": 3,
      "t_start_ms": 1200,
      "t_end_ms": 4100,
      "speaker": "in",
      "lang": "en",
      "text": "Hi, can you hear me well?",
      "translation": "Привет, хорошо меня слышно?",
      "translation_lang": "ru"
    }
  ]
}
```

Строки `utterances` — 1:1 с таблицей: `speaker` совпадает со `stream`
(`"in" | "out"`), `translation` и `translation_lang` могут быть `null`, пока
перевод не готов. Неизвестный `id` → `404` с телом `{"error": "session not found"}`.

### `GET /api/sessions/:id/audio`

Сырое аудио сессии (`sessions.audio_path`), `Content-Type: audio/wav`.
Желательна поддержка `Range` — плеер перематывает. Если записи нет
(`audio_path IS NULL`) → `404`, UI тогда не рисует `<audio>`.

### `GET /api/voices`

```json
{
  "voices": [
    {
      "id": "v_ab12cd",
      "name": "Мой голос (ru)",
      "lang": "ru",
      "sample_path": "voices/v_ab12cd.wav",
      "created_at": 1757980800000
    }
  ]
}
```

Строки таблицы `voices`; `id` — это `voice_id` из контрактов, он же уходит в
`session.start`. «Голос по умолчанию» отдельной записью не нужен — в UI это
пункт «без клона», который шлёт `voice_id: null` (для kk клон не предусмотрен).

### Заглушки на стороне UI

`src/app/api/**` сначала пробуют сходить в `ENGINE_HTTP` (таймаут 1.5 с) и
проксируют ответ; если движок недоступен — отдают `mocks/data.json` с заголовком
`x-rt-source: mock`. Аудио в мок-режиме — синтезированный тон (реальных WAV в
репозитории нет, `*.wav` в `.gitignore`). Когда оркестратор поднимет REST,
менять UI не потребуется.

## Тесты

`npm test` — 67 тестов, 5 файлов:

| Файл | Что покрывает |
|---|---|
| `tests/contracts.test.ts` | сверка TS-типов с реальными `engine/contracts/*.json` (набор `type`, required-поля payload, `additionalProperties: false`), `parseEnvelope` на валидных и 13 невалидных входах, сборка команд |
| `tests/state.test.ts` | редьюсер: partial заменяет живую строку, final фиксирует, привязка `translation.ready` (по `ref_utterance_id`, по `src_text`, к последней), независимость потоков, ограничение длины ленты, очистка при новой сессии, метрики, окно транскрипта 5 минут |
| `tests/scenario.test.ts` | `mocks/scenario.json` валиден по JSON-схемам через `ajv` (draft 2020-12), покрывает оба потока и все три языка, проигрывается редьюсером; `mocks/data.json` соответствует ограничениям `db/schema.sql` |
| `tests/window.test.tsx` | рендер главного окна с фейковым состоянием: субтитры и переводы видны, подсветка `total > 2500`, Start заблокирован без соединения, Start шлёт выбранные языки/голос/`record`, кнопка подсказок — заглушка |
| `tests/ws.test.ts` | экспоненциальная задержка реконнекта с потолком 10 с |

## TODO

- **Ассистент-LLM** (ARCHITECTURE.md этап 5): кнопка «Подсказать ответ» отключена;
  нужен эндпоинт движка вида `POST /api/assist { session_id, transcript }` и
  стриминг подсказки в панель.
- **Electron always-on-top**: окно поверх Zoom, запоминание позиции/размера,
  системный трей. Сейчас это обычная вкладка браузера; вся вёрстка уже рассчитана
  на 480×640 и сжимается до 360 px, так что обёртка не потребует переделки UI.
- **История**: пагинация `/api/sessions` (сейчас отдаётся весь список), поиск по
  тексту реплик, экспорт транскрипта.
- **Аудио**: подсветка текущей реплики при воспроизведении (нужен `Range` на
  эндпоинте аудио и таймкоды — они уже есть).
- **`ref_utterance_id`**: пока `stt.final` не несёт id реплики, привязка перевода
  делается эвристикой (см. вопрос ниже).

## Вопросы к владельцу проекта

1. **REST-контракт истории** (раздел выше) — утвердить формы ответов, порт 8766 и
   производные поля `utterance_count` / `duration_ms`.
2. **`ref_utterance_id` не с чем сопоставлять.** `translation.ready` несёт
   `ref_utterance_id` (id строки в `utterances`), но `stt.final` id реплики не
   содержит — у UI нет ключа, по которому он мог бы её найти. Сейчас UI
   привязывает перевод по `src_text`, а при неудаче — к последней реплике без
   перевода. Это ломается при двух одинаковых фразах подряд. Предлагаю добавить
   `utterance_id: integer | null` в payload `stt.final` — но это изменение
   контракта, поэтому нужен ваш явный ОК (CLAUDE.md, закон 2), кодом я его не делал.
3. **Кто уведомляет о записи.** UI показывает статус записи и предупреждение у
   чекбокса (ARCHITECTURE.md раздел 8). Нужен ли дополнительно звуковой/визуальный
   сигнал собеседнику со стороны движка?
