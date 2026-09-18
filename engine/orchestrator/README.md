# engine/orchestrator — движок: конвейеры, WebSocket, REST, запись в БД

Зона этапа 2, см. [ARCHITECTURE.md](../../ARCHITECTURE.md) 4.5.

Оркестратор склеивает четыре модуля движка в два конвейера, раздаёт события в
UI по WebSocket, отдаёт историю по REST и пишет транскрипт в SQLite. Своих
моделей у него нет: всё берётся через публичные фабрики чужих модулей
(`create_source`/`create_sink`, `create_engine`, `create_provider`, `create_tts`),
а общение — только конвертами из `engine/contracts/` (CLAUDE.md, закон 2).

## Что внутри

| Файл | Роль |
|---|---|
| `config.py` | `OrchestratorConfig`: порты, БД, записи, бэкенд + вложенные конфиги модулей |
| `bus.py` | `EventBus`: широковещательная шина конвертов (WS-клиенты и внутренние подписчики) |
| `db.py` | `Database` на `aiosqlite`: применение `db/schema.sql` и весь CRUD |
| `pipeline.py` | `Pipeline` одного потока + `WavRecorder` записи исходного аудио |
| `runtime.py` | `Runtime`: сборка компонентов и прогрев моделей (whisper → XTTS → NLLB) |
| `server.py` | `Session` (два конвейера) и `EngineServer` (WebSocket + REST) |
| `rest.py` | aiohttp-приложение REST истории и голосов |
| `filemode.py` | `run_file_mode`: WAV → конвейер → WAV, без серверов |
| `__main__.py` | CLI `python -m engine.orchestrator` |

## Архитектура сессии

```
                         session.start {lang_in, lang_out, voice_id, record}
                                            │
          ┌─────────────────────────────────┴─────────────────────────────────┐
          ▼                                                                   ▼
  inbound (stream "in")                                          outbound (stream "out")
  loopback (речь собеседника)                                    микрофон (речь пользователя)
          │ 16 kHz                                                           │ 16 kHz
          ▼                                                                   ▼
  SttEngine (VAD + whisper) ──► stt.partial / stt.final ──► EventBus ──► WebSocket ──► UI
          │                                                                   │
          ▼ очередь фраз (по одной, по порядку)                               ▼
  INSERT utterances ──► id                                            то же самое
          ▼
  TranslationService  lang_in → lang_out          (для outbound: lang_out → lang_in)
          ▼
  translation.ready {ref_utterance_id = id}  +  UPDATE utterances.translation
          ▼
  TtsRouter.synthesize → чанки PCM 24 kHz
          ▼                                                                   ▼
  наушники пользователя                                     CABLE Input (Zoom слышит перевод)
          ▼
  metrics.latency {stt_ms, mt_ms, tts_ms, total_ms}
```

Ключевые решения:

* **`lang_in` — язык собеседника** (поток `in`), **`lang_out` — язык
  пользователя** (поток `out`). Inbound переводит `lang_in → lang_out` и играет
  в наушники, outbound — `lang_out → lang_in` и играет в виртуальный кабель.
* **`ref_utterance_id`** — это `id` строки `utterances`, вставленной по
  `stt.final`. Переводы одного потока публикуются строго в порядке своих фраз:
  их обрабатывает один фоновый воркер на конвейер, очередь FIFO.
* **STT не ждёт перевод и синтез.** Распознавание крутится в основном цикле,
  а «БД → перевод → TTS → метрики» — в фоновой задаче. Ошибка любого этапа
  логируется и теряет одну фразу, а не сессию.
* **Модели живут в одном экземпляре на процесс** (`Runtime`), состояние потока
  (VAD, таймлайн, окно контекста) — своё на каждый конвейер.
* **Запись** (`record: true`): исходный PCM 16 kHz каждого потока пишется в
  `recordings/<session_id>_<stream>.wav`. В `sessions.audio_path` попадает WAV
  потока `in` (речь собеседника) — его проигрывает страница истории; речь
  пользователя лежит рядом как `<session_id>_out.wav`.
* **`tts.chunk` по умолчанию не уходит в UI** (служебное событие, трафик
  заметный). Включается `RT_WS_TTS_CHUNKS=1`.

## Порты

| Порт | Протокол | Что отдаёт |
|---|---|---|
| 8765 | WebSocket (`websockets`) | все события движка + приём `session.start` / `session.stop` |
| 8766 | HTTP (`aiohttp`) | REST истории и голосов, CORS для `http://localhost:3000` |

Оба сервера слушают `127.0.0.1` — движок наружу не выставляется.

## REST API (порт 8766)

Контракт утверждён владельцем по предложению из [ui/README.md](../../ui/README.md).
Все временные метки — unix-время в миллисекундах, поля названы как колонки в
`db/schema.sql`.

### `GET /api/sessions`

```json
{
  "sessions": [
    {
      "id": 3,
      "started_at": 1758240000000,
      "ended_at": 1758240315000,
      "lang_from": "en",
      "lang_to": "ru",
      "audio_path": "/path/recordings/3_in.wav",
      "utterance_count": 5,
      "duration_ms": 315000
    }
  ]
}
```

Новые сверху. `utterance_count` — `COUNT(*)` реплик сессии, `duration_ms` —
`ended_at - started_at` (`null`, пока сессия идёт).

### `GET /api/sessions/{id}`

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

Реплики по возрастанию `t_start_ms`. Неизвестный `id` → `404`
`{"error": "session not found"}`.

### `GET /api/sessions/{id}/audio`

WAV записи сессии, `Content-Type: audio/wav`, `Range` поддержан (aiohttp
`FileResponse`). Нет записи или файл пропал → `404`.

### `GET /api/voices`

```json
{ "voices": [{ "id": "v_1a2b3c4d", "name": "Мой голос", "lang": "ru",
               "sample_path": "/path/voices/v_1a2b3c4d/sample.wav",
               "created_at": 1757980800000 }] }
```

### `POST /api/voices`

`multipart/form-data`: `sample` — WAV 15–30 секунд, `name` — имя профиля,
`lang` — `ru|en|kk`. Создаёт профиль в `VoiceStore` (диск) и строку в таблице
`voices`, отвечает `201` и телом `{"voice": {...}}`.

```bash
curl -F "sample=@sample.wav" -F "name=Мой голос" -F "lang=ru" \
     http://127.0.0.1:8766/api/voices
```

Ошибки: `400` — нет файла/имени, неизвестный язык, сэмпл не WAV или не той
длительности; `409` — имя уже занято (`name UNIQUE` в схеме).

Правовая заметка (ARCHITECTURE.md раздел 8): клонировать можно только
собственный голос или голос с явного согласия владельца.

### `GET /api/health`

`{"status": "ok", "backend": "real"}` — живость и режим движка.

## Источник правды по голосам

Профили на диске (`RT_VOICES_DIR`, по умолчанию `./voices`) — источник правды,
таблица `voices` — индекс для истории и UI. При старте движок синхронизирует
таблицу с диском: профили, которых нет в БД, добавляются, существующие
обновляются. Строки голосов, удалённых с диска, не стираются — на них может
ссылаться история.

## CLI

```bash
# live-режим (по умолчанию): WebSocket :8765 + REST :8766
python -m engine.orchestrator
python -m engine.orchestrator serve --backend real

# live на Linux/без железа: вместо loopback и микрофона — WAV-файлы
python -m engine.orchestrator --backend fake --fake-in in.wav --fake-out mic.wav

# файловый режим: WAV → конвейер → WAV, серверы не поднимаются,
# события печатаются JSON-строками в stdout
python -m engine.orchestrator --backend fake \
    --file engine/tests/fixtures/stt_speech_pattern.wav \
    --from en --to ru --out out.wav
python -m engine.orchestrator --file sample.wav --from ru --to en \
    --voice v_1a2b3c4d --out out.wav --stream out --realtime
```

| Флаг | Значение |
|---|---|
| `--backend real\|fake` | реальные модели и устройства или фейки (по умолчанию `RT_BACKEND`) |
| `--file` | входной WAV — включает файловый режим |
| `--out` | выходной WAV (по умолчанию `<вход>_out.wav`) |
| `--from` / `--to` | языки оригинала и перевода (`ru\|en\|kk`) |
| `--voice` | `voice_id` профиля голоса |
| `--stream in\|out` | каким потоком считать запись в файловом режиме |
| `--realtime` | читать файл в темпе живого звука (честные `total_ms`) |
| `--ws-port` / `--http-port` | порты серверов |
| `--db` | файл SQLite |
| `--fake-in` / `--fake-out` | WAV вместо loopback и микрофона при `--backend fake` |
| `--no-warmup` | не грузить модели при старте (отладка) |
| `--log-level` | уровень логирования (по умолчанию `LOG_LEVEL`) |

Коды возврата: `0` — успех, `1` — ошибка аргументов, `2` — движок не может
работать (нет аудиобэкенда, не грузятся модели).

## Переменные окружения

Свои:

| Переменная | По умолчанию | Смысл |
|---|---|---|
| `RT_BACKEND` | `real` | `real` — модели и устройства, `fake` — заглушки |
| `RT_DB_PATH` | `./db/translator.db` | файл SQLite |
| `RT_RECORDINGS_DIR` | `./recordings` | куда писать WAV сессий |
| `RT_WS_HOST` | `127.0.0.1` | адрес обоих серверов |
| `RT_WS_PORT` | `8765` | порт WebSocket |
| `RT_HTTP_PORT` | `8766` | порт REST |
| `RT_CORS_ORIGIN` | `http://localhost:3000` | кому REST разрешает запросы |
| `RT_WS_TTS_CHUNKS` | `0` | `1` — слать в UI служебные `tts.chunk` |
| `LOG_LEVEL` | `INFO` | уровень логирования |

Чужие переменные движок не дублирует: аудиоустройства (`RT_AUDIO_*`),
распознавание (`RT_STT_*`), перевод (`RT_MT_*`), синтез (`RT_TTS_*`,
`RT_VOICES_DIR`) и кэш моделей (`RT_MODELS_DIR`) читают сами модули — см. их
README.

## Прогрев и VRAM

При старте live-режима с `--backend real` модели грузятся заранее и в порядке
из ARCHITECTURE.md раздела 6:

1. faster-whisper `small` int8_float16 — GPU, ~1.0 GB;
2. XTTS-v2 — GPU, ~2.5 GB;
3. NLLB-200-distilled-600M — CPU, в отдельном потоке.

После каждого шага в лог пишется время и `torch.cuda.memory_allocated()`, если
torch доступен. Порядок важен: XTTS должен вставать на уже занятую whisper
память, а не наоборот.

## Порядок запуска на Windows 11 (целевая машина)

```powershell
# 0. VB-Audio Virtual Cable — ставится вручную и требует перезагрузки
#    https://vb-audio.com/Cable/ , подробности в engine/audio_io/README.md
#    В Zoom: Микрофон = CABLE Output, Динамик = ваши наушники.

# 1. окружение и зависимости
python -m venv venv
venv\Scripts\activate
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts\check_env.py

# 2. модели (кэш ./models, переопределяется RT_MODELS_DIR)
python scripts\download_models.py

# 3. база
sqlite3 db\translator.db < db\schema.sql     # необязательно: движок применит схему сам

# 4. профиль голоса (15-30 секунд своей речи)
python scripts\tts_smoke.py --create-voice sample.wav --name me --lang ru
#    или через REST уже запущенного движка:
#    curl -F "sample=@sample.wav" -F "name=me" -F "lang=ru" http://127.0.0.1:8766/api/voices

# 5. проверка всей цепочки на одном файле
python scripts\e2e_smoke.py --real --file sample.wav --from ru --to en --realtime

# 6. движок
python -m engine.orchestrator

# 7. UI (второй терминал)
cd ui
npm install
npm run dev        # http://localhost:3000
```

В UI выберите языки, голос и нажмите Start: движок поднимет обе цепочки,
субтитры и задержки поедут в окно, история появится на `/history`.

## Проверка без Windows и без моделей

```bash
pip install -r engine/orchestrator/requirements-orchestrator.txt
python scripts/e2e_smoke.py                    # фейки + фикстура, полный конвейер
python -m engine.orchestrator --backend fake \
    --fake-in engine/tests/fixtures/stt_speech_pattern.wav \
    --fake-out engine/tests/fixtures/stt_speech_pattern.wav
```

Дальше `cd ui && npm run dev` — UI подключится к настоящему движку на
`ws://localhost:8765`, история придёт из REST на 8766. Так проверяются
контракты, ленты субтитров и история без единой модели и без звуковой карты.

## Troubleshooting

| Симптом | Причина и что делать |
|---|---|
| `live-режим ... работает только на Windows` (код 2) | запустили `serve` с `--backend real` не на Windows — используйте `--backend fake` |
| `AudioBackendUnavailable` на Windows | не установлены `sounddevice`/`pyaudiowpatch` — `pip install -r engine/audio_io/requirements-audio_io.txt` |
| В наушниках эхо, движок переводит сам себя | перевод играет в колонки, и loopback ловит его обратно — нужны наушники (engine/audio_io/README.md) |
| Zoom не слышит перевод | в Zoom микрофон не `CABLE Output`; проверка: `python scripts/audio_smoke.py --tone "CABLE Input"` |
| UI показывает «нет связи» | движок не запущен или занят порт 8765: `python -m engine.orchestrator --ws-port 8865` и `NEXT_PUBLIC_ENGINE_WS` в `ui/.env.local` |
| История пустая, хотя сессия была | UI ходит в свои заглушки — задайте `ENGINE_HTTP=http://localhost:8766` в `ui/.env.local` |
| `404` на `/api/sessions/{id}/audio` | сессия записана без `record: true` или WAV удалён из `recordings/` |
| `409` при создании голоса | имя профиля занято (`name UNIQUE`), возьмите другое |
| `total_ms` больше 2500 мс | бюджет ARCHITECTURE.md раздела 2 превышен: проверьте, что whisper на CUDA (`RT_STT_DEVICE=cuda`), уменьшите `RT_STT_MAX_UTTERANCE_MS`, держите `RT_MT_BEAMS=1` |
| Нехватка VRAM при старте XTTS | закройте другие GPU-приложения; крайняя мера — `RT_TTS_DEVICE=cpu` (в реальное время не уложится) |
| В логе «очередь подписчика переполнена» | UI не успевает читать события; старые события отбрасываются, конвейер при этом не тормозит |

## Тесты

```bash
python -m pytest engine/tests/test_orchestrator_db.py \
                 engine/tests/test_orchestrator_pipeline.py \
                 engine/tests/test_orchestrator_server.py \
                 engine/tests/test_e2e_file_mode.py -q
ruff check engine/orchestrator scripts/e2e_smoke.py
mypy engine/orchestrator
```

Замечание: `mypy engine/orchestrator` проверяет и импортируемые модули, поэтому
показывает 3 ошибки `no-any-return` в `engine/tts/audio.py` (чужая зона, те же
ошибки видны при `mypy engine/tts` — CLAUDE.md, закон 8: не чиним сами).
В файлах самого оркестратора ошибок нет.

Покрыто (всё на фейках, без моделей и без звуковой карты):

* `test_orchestrator_db.py` — применение `db/schema.sql`, жизненный цикл
  сессии, производные `utterance_count` / `duration_ms`, порядок реплик,
  запись перевода, upsert и синхронизация голосов с диска;
* `test_orchestrator_pipeline.py` — из фикстуры «тон — тишина — тон» получаются
  ровно 2 `stt.final`, 2 `translation.ready` с корректными `ref_utterance_id` и
  порядком, 2 `metrics.latency`; непустой WAV в приёмнике, строки в БД, WAV
  рекордера, устойчивость к падению перевода;
* `test_orchestrator_server.py` — настоящие серверы на свободных портах:
  `session.state` при подключении, `start → running`, события конвейера,
  `stop → idle`, игнорирование невалидных команд, REST `/api/sessions`,
  `/api/sessions/{id}`, `/api/voices` (включая `POST`), CORS и `404`;
* `test_e2e_file_mode.py` — CLI отдельным процессом: валидный выходной WAV и
  валидные по контрактам события в stdout.

## TODO

* **Ассистент-LLM** (ARCHITECTURE.md этап 5): сейчас движок отдаёт только живой
  транскрипт, подсказок нет. Понадобится эндпоинт вида
  `POST /api/assist {session_id, transcript}` и стриминг ответа — по решению
  владельца отложено.
* **Пагинация и поиск** в `GET /api/sessions` (сейчас отдаётся весь список).
* **Пересоздание голосов через REST**: есть `POST /api/voices`, удаления
  (`DELETE /api/voices/{id}`) пока нет — профиль удаляется руками из
  `RT_VOICES_DIR`, строка в БД остаётся как ссылка для истории.
* **Резюме сессии**: `sessions.audio_path` указывает на WAV потока `in`;
  объединённая дорожка «собеседник + пользователь» не пишется.
