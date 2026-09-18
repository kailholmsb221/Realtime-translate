# Realtime Translator

Локальный синхронный переводчик речи — надстройка над Zoom / Google Meet.

- речь собеседника (en/kk → ru): субтитры в окне-ассистенте + озвучка в наушники;
- речь пользователя (ru → en/kk): переведённый голос уходит в Zoom через виртуальный микрофон;
- для ru/en голос клонируется по сэмплу говорящего (XTTS-v2), для kk — стандартный TTS;
- транскрипт и переводы пишутся в SQLite, история доступна в UI.

Всё работает локально и бесплатно: только open-source модели, никаких платных API.
Цель по задержке «конец фразы → озвучка перевода» — **≤ 2.5 с** на RTX 4050 (6 GB VRAM).

Полное описание — в [ARCHITECTURE.md](ARCHITECTURE.md).
Правила работы над проектом (в том числе для агентов) — в [CLAUDE.md](CLAUDE.md).

## Требования

| Что | Версия / примечание |
|---|---|
| ОС | Windows 10/11 (WASAPI loopback) |
| Python | 3.11 |
| Node.js | 20+ (для UI) |
| GPU | NVIDIA с CUDA, ≥ 6 GB VRAM (эталон — RTX 4050 Laptop) |
| VB-Audio Virtual Cable | ставится вручную: <https://vb-audio.com/Cable/> |

**VB-Audio Virtual Cable** нужен, чтобы отдавать переведённый голос в Zoom:
после установки в Zoom выберите «CABLE Input» микрофоном, а движок будет писать
в него звук. Установку системных программ и драйверов автоматика не делает —
поставьте сами и перезагрузитесь.

## Быстрый старт на Windows 11

Полная последовательность от чистой машины до работающего перевода в Zoom.

**0. VB-Audio Virtual Cable** — ставится вручную и требует перезагрузки:
скачайте с <https://vb-audio.com/Cable/>, запустите `VBCABLE_Setup_x64.exe` от
администратора, перезагрузитесь. В Zoom: *Настройки → Звук → Микрофон* =
**CABLE Output**, *Динамик* = ваши наушники (наушники обязательны, иначе
loopback поймает собственный перевод). Подробности — в
[engine/audio_io/README.md](engine/audio_io/README.md).

**1. Окружение и зависимости.** PyTorch с CUDA ставится первым и отдельно:

```powershell
git clone <repo> realtime-translator
cd realtime-translator

python -m venv venv
venv\Scripts\activate

pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python scripts\check_env.py          # Python, CUDA, VRAM, аудиоустройства, VB-Cable
```

**2. Модели** (кэш — `./models`, переопределяется `RT_MODELS_DIR`):

```powershell
python scripts\download_models.py             # всё, что качается автоматически
python scripts\download_models.py --list      # что именно будет скачано
python scripts\download_models.py --only whisper nllb xtts kk_tts
```

KazakhTTS2 (ISSAI) — ручной шаг: загружаемого репозитория на Hugging Face нет,
по умолчанию для kk используется `facebook/mms-tts-kaz` (ключ `kk_tts`).

**3. База.** Движок применяет `db/schema.sql` сам при старте; вручную — так:

```powershell
sqlite3 db\translator.db < db\schema.sql
```

**4. Профиль голоса** (сэмпл 15–30 секунд своей речи, WAV):

```powershell
python scripts\tts_smoke.py --create-voice sample.wav --name me --lang ru
```

Тот же профиль можно создать у уже запущенного движка:
`curl -F "sample=@sample.wav" -F "name=me" -F "lang=ru" http://127.0.0.1:8766/api/voices`.
Клонировать разрешено только свой голос или голос с явного согласия владельца.

**5. Проверка всей цепочки на одном файле** (WAV → перевод → WAV + задержки):

```powershell
python scripts\e2e_smoke.py --real --file sample.wav --from ru --to en --realtime
```

**6. Движок и UI** (два терминала):

```powershell
python -m engine.orchestrator                 # WebSocket :8765, REST :8766
cd ui && npm install && npm run dev           # http://localhost:3000
```

В окне UI выберите языки (`lang_in` — язык собеседника, `lang_out` — ваш),
голосовой профиль, при необходимости включите запись и нажмите Start.
История сессий — на `/history`.

## Проверка без железа (Linux/CI, без моделей)

Весь движок работает на фейковых бэкендах: вместо WASAPI и микрофона — WAV,
вместо whisper/NLLB/XTTS — заглушки тех же интерфейсов.

```bash
pip install -r engine/orchestrator/requirements-orchestrator.txt

# полный конвейер на фикстуре: события и задержки в консоль
python scripts/e2e_smoke.py

# то же через CLI движка: на выходе WAV, в stdout — JSON-события контрактов
python -m engine.orchestrator --backend fake \
    --file engine/tests/fixtures/stt_speech_pattern.wav \
    --from en --to ru --out out.wav

# live-режим без звуковой карты: серверы поднимаются, аудио берётся из WAV
python -m engine.orchestrator --backend fake \
    --fake-in engine/tests/fixtures/stt_speech_pattern.wav \
    --fake-out engine/tests/fixtures/stt_speech_pattern.wav
```

UI при этом можно запускать и против настоящего движка (`npm run dev`), и
против собственного мока (`npm run mock`) — см. [ui/README.md](ui/README.md).

Подробности по портам, REST API, переменным окружения и troubleshooting — в
[engine/orchestrator/README.md](engine/orchestrator/README.md).

## Тесты

```bash
pytest engine/tests            # тесты движка (все на фейках, без моделей и звука)
ruff check engine scripts      # линт
ruff format --check engine scripts
mypy engine                    # типы
cd ui && npm test              # тесты UI
```

Тесты, поднимающие реальные модели, помечены `@pytest.mark.real_models` и по
умолчанию пропускаются. Чтобы прогнать их:

```bash
RT_REAL_MODELS=1 pytest engine/tests      # Windows: set RT_REAL_MODELS=1
```

WAV-фикстуры (тон 440 Гц и тишина, 16 kHz mono int16) лежат в
`engine/tests/fixtures/` и пересоздаются командой:

```bash
python scripts/make_fixtures.py [--force]
```

## Структура

```
engine/               Python 3.11, asyncio
  audio_io/           захват/вывод звука: WASAPI loopback, микрофон, наушники, VB-Cable
  stt/                распознавание речи: Silero VAD + faster-whisper small
  translate/          перевод: NLLB-200-distilled-600M на CPU
  tts/                синтез: XTTS-v2 (ru/en, клон голоса), KazakhTTS2 (kk)
  orchestrator/       конвейеры, WebSocket :8765, REST :8766, запись в БД
  contracts/          JSON-схемы событий и events.py — МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ
  tests/              pytest, фикстуры в tests/fixtures/
ui/                   Next.js + TypeScript: субтитры, ассистент, история
db/                   schema.sql (SQLite) — МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ
scripts/              check_env.py, download_models.py, make_fixtures.py, e2e_smoke.py
```

Зависимости разнесены по модулям: у каждого свой `requirements-<модуль>.txt`,
корневой `requirements.txt` подключает их через `-r` (плюс `requirements-core.txt`
и `requirements-dev.txt`).

## Переменные окружения

| Переменная | Значение |
|---|---|
| `RT_MODELS_DIR` | каталог кэша моделей (по умолчанию `./models`) |
| `RT_REAL_MODELS` | `1` — не пропускать тесты с реальными моделями |
| `RT_KAZAKHTTS_REPO` | id репозитория KazakhTTS2, если владелец его зафиксирует (по умолчанию пусто — ручная установка) |
| `RT_BACKEND` | `real` (модели и устройства) или `fake` (заглушки) |
| `RT_DB_PATH` | файл SQLite (по умолчанию `./db/translator.db`) |
| `RT_RECORDINGS_DIR` | каталог WAV-записей сессий (по умолчанию `./recordings`) |
| `RT_WS_PORT` / `RT_HTTP_PORT` | порты движка (8765 / 8766) |
| `LOG_LEVEL` | уровень логирования движка |

Переменные конкретных модулей (`RT_AUDIO_*`, `RT_STT_*`, `RT_MT_*`, `RT_TTS_*`,
`RT_VOICES_DIR`) описаны в README этих модулей.

## Контракты

Все модули общаются только событиями из `engine/contracts/`
(конверт `{type, ts, payload}`, JSON Schema draft 2020-12):

```python
from engine.contracts.events import Lang, SttFinal, Stream, parse_event, to_json

raw = to_json(SttFinal(Stream.IN, Lang.EN, "hello there", 0, 940))
envelope = parse_event(raw)
```

Схемы и схема БД неприкосновенны: изменение — только с явного разрешения
владельца проекта, отдельным PR (CLAUDE.md, закон 2).
См. [engine/contracts/README.md](engine/contracts/README.md) и [db/README.md](db/README.md).

## Правовые заметки

- Клонировать можно только собственный голос пользователя либо голос с явного
  согласия его владельца.
- При записи разговора собеседник должен быть уведомлён; UI показывает статус
  записи, ответственность за уведомление — на пользователе.
