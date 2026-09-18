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

**1. Окружение и зависимости.** PyTorch с CUDA ставится **первым и отдельно**:
с PyPI на Windows приезжает CPU-сборка, и тогда XTTS и whisper работают в разы
медленнее реального времени.

```powershell
git clone <repo> realtime-translator
cd realtime-translator

python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip

# RTX 4050 Laptop (Ada, sm_89) — проверенная связка: torch 2.5.1 + CUDA 12.1
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# проверка: должно напечатать True и имя видеокарты
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

pip install -r requirements.txt   # torch уже стоит — pip его не тронет
python scripts\check_env.py       # Python, CUDA, VRAM, аудиоустройства, VB-Cable
```

Почему именно так:

* **CUDA 12.x обязательна.** `faster-whisper` 1.2 тянет `ctranslate2 >= 4.0`,
  а свежие `ctranslate2` собраны только под **CUDA 12 + cuDNN 9**. Индекс
  `cu121` (или `cu124`) подходит, `cu118` — нет.
* **cuBLAS и cuDNN 9** для `ctranslate2` приезжают вместе с CUDA-сборкой
  torch. Если whisper падает с ошибкой про `cudnn64_9.dll`, доставьте их явно:
  `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12` (подробности — в
  [engine/stt/README.md](engine/stt/README.md)).
* **Драйвер NVIDIA** — из ветки с поддержкой CUDA 12 (Windows: 527.41 и
  новее). Отдельно ставить CUDA Toolkit не нужно: всё необходимое лежит
  внутри колёс torch.
* **`coqui-tts` (XTTS-v2) тестируется с PyTorch 2.2+** и с версии 0.27.4 не
  тянет torch за собой — поэтому torch и ставится руками. Если возьмёте torch
  **2.6 и новее** (индексы `cu124` / `cu126`), обязательно возьмите и свежий
  `coqui-tts` (>= 0.26): в torch 2.6 у `torch.load` по умолчанию включился
  `weights_only=True`, и старые версии не открывают чекпойнт XTTS.

```powershell
# альтернатива посвежее (torch 2.6 + CUDA 12.4), тогда и coqui-tts >= 0.26
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

**2. Модели** (кэш — `./models`, переопределяется `RT_MODELS_DIR`):

```powershell
python scripts\download_models.py             # всё, что качается автоматически
python scripts\download_models.py --list      # что именно будет скачано
python scripts\download_models.py --only whisper nllb xtts kk_tts
```

KazakhTTS2 (ISSAI) — ручной шаг: загружаемого репозитория на Hugging Face нет,
по умолчанию для kk используется `facebook/mms-tts-kaz` (ключ `kk_tts`).

**3. База.** Отдельный шаг не нужен: движок применяет `db/schema.sql` сам при
старте (файл идемпотентный). Если хочется создать базу заранее, понадобится
консольный `sqlite3` — в Windows он не входит в систему, скачайте
*sqlite-tools* с <https://sqlite.org/download.html>:

```powershell
sqlite3 db\translator.db ".read db/schema.sql"
```

**4. Профиль своего голоса** (сэмпл 15–30 секунд своей речи, WAV):

```powershell
python scripts\tts_smoke.py --create-voice sample.wav --name me --lang ru
```

Тот же профиль можно создать у уже запущенного движка:
`curl -F "sample=@sample.wav" -F "name=me" -F "lang=ru" http://127.0.0.1:8766/api/voices`.
Полученный `voice_id` выбирается в UI — им озвучивается **ваша** речь для
собеседника.

**Голос собеседника заводить не нужно**: движок клонирует его сам в первые
~12 секунд разговора (накапливает только речевые участки, создаёт профиль в
фоне и дальше озвучивает перевод этим голосом). До готовности клона звучит
встроенный голос XTTS. Автоклон настраивается переменными `RT_AUTO_CLONE`,
`RT_AUTO_CLONE_MIN_S`, `RT_AUTO_CLONE_MAX_S`, `RT_AUTO_CLONE_KEEP` и по
умолчанию удаляет профиль собеседника вместе с сессией.

Правовая заметка: клонировать можно только собственный голос или голос с
явного согласия его владельца — предупредите собеседника (ARCHITECTURE.md
раздел 8).

**5. Проверка всей цепочки на одном файле** (WAV → перевод → WAV + задержки):

```powershell
python scripts\e2e_smoke.py --real --file sample.wav --from ru --to en --realtime
```

**6. Движок и UI** (два терминала).

Терминал 1 — движок (не закрывайте, он держит модели в памяти):

```powershell
venv\Scripts\activate
python -m engine.orchestrator          # WebSocket :8765, REST :8766
```

В логе должно появиться: прогрев `STT (faster-whisper small)` → `TTS (XTTS-v2)`
→ `MT (NLLB-200, CPU)` с временем и занятой VRAM, затем строка
`движок слушает: ws://127.0.0.1:8765, http://127.0.0.1:8766 (backend=real)`.

Терминал 2 — UI:

```powershell
cd ui
npm install
npm run dev                            # http://localhost:3000
```

**Что видно в окне** (`http://localhost:3000`, маленькое окно на 480 px —
держите его поверх Zoom):

* верхняя строка — «движок на связи», «сессия идёт / не запущена», направление
  языков, красная метка «● запись» и ссылка «История»;
* полоса задержек: `stt / mt / tts / total` по каждому потоку; `total` больше
  2500 мс подсвечивается — бюджет ARCHITECTURE.md раздела 2 превышен;
* селекторы «Вход» (`lang_in` — язык собеседника) и «Выход» (`lang_out` — ваш
  язык), список голосов (автоклоны `auto_*` в нём не показываются), галочка
  записи и кнопки Start / Stop;
* две ленты субтитров: **Собеседник** (перевод звучит в наушниках) и **Вы**
  (перевод уходит в Zoom через CABLE Input). В каждой реплике сверху оригинал,
  снизу перевод (пока его нет — курсивное «перевод…»); последняя строка курсивом
  с мигающим курсором — ещё не законченная фраза (`stt.partial`);
* внизу панель ассистента — транскрипт последних 5 минут (кнопка подсказок
  пока заглушка).

История сессий — на `http://localhost:3000/history`, транскрипт с
таймкодами и проигрывателем записи — на `/history/<id>`.

## Что проверить первым делом

Если что-то не работает, идите по этому списку сверху вниз: каждый шаг
проверяет ровно один слой и не требует предыдущего звонка в Zoom.

| # | Команда | Что должно получиться |
|---|---|---|
| 1 | `python scripts\check_env.py` | `Python 3.11+ : да`, `CUDA : да`, `VB-Cable : да` |
| 2 | `python scripts\download_models.py --list` | список моделей и отметки, что уже скачано |
| 3 | `python scripts\audio_smoke.py --list` | в списке есть ваш микрофон, наушники, `CABLE Input`, loopback-устройство |
| 4 | `python scripts\audio_smoke.py --tone "CABLE Input"` | Zoom (микрофон = CABLE Output) показывает уровень сигнала |
| 5 | `python scripts\audio_smoke.py --record-loopback 5 check.wav` | в `check.wav` слышно то, что играло в наушниках |
| 6 | `python scripts\translate_smoke.py --bench` | таблица шести направлений ru↔en↔kk с временем на CPU |
| 7 | `python scripts\stt_smoke.py check.wav --lang ru` | события `stt.partial` / `stt.final` с распознанным текстом |
| 8 | `python scripts\tts_smoke.py --say "проверка связи" --lang ru` | WAV в `out_tts\`, голос звучит внятно |
| 9 | `python scripts\e2e_smoke.py --real --file sample.wav --from ru --to en --realtime` | события всей цепочки и `total_ms` в пределах 2500 мс |
| 10 | `python -m engine.orchestrator` + `cd ui && npm run dev` | окно UI пишет «движок на связи», Start активен |

Шаги 1–5 — железо и драйверы, 6–8 — модели по отдельности, 9 — вся цепочка,
10 — сборка целиком. Без GPU те же скрипты гоняются на заглушках:
`--fake` у `stt_smoke` / `tts_smoke` / `translate_smoke` и `e2e_smoke.py` без
`--real`.

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
cd ui && npm test              # тесты UI (включая прогон записи событий движка)
cd ui && npx tsc --noEmit && npm run lint && npm run build
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
| `RT_AUTO_CLONE` | `0` — не клонировать голос собеседника (по умолчанию `1`) |
| `RT_AUTO_CLONE_MIN_S` / `RT_AUTO_CLONE_MAX_S` | сколько речи копить для клона (12 / 30 с) |
| `RT_AUTO_CLONE_KEEP` | `1` — оставлять автопрофиль собеседника после сессии |
| `LOG_LEVEL` | уровень логирования движка |

Переменные конкретных модулей (`RT_AUDIO_*`, `RT_STT_*`, `RT_MT_*`, `RT_TTS_*`,
`RT_VOICES_DIR`) описаны в README этих модулей.

## Известные ограничения

Про что стоит знать до первого звонка — это не баги, а сознательные рамки MVP.

**Задержка.** Бюджет «конец фразы → озвучка перевода» — 2.5 с, и он считается
**от конца фразы**: движок ждёт паузу (`min_silence_ms`, по умолчанию 500 мс),
только потом распознаёт целиком. Разговор идёт «по рации»: вы слышите перевод
реплики, когда собеседник её уже договорил. Длинная фраза закрывается
принудительно через `RT_STT_MAX_UTTERANCE_MS` (15 с). Если `total_ms` в окне
регулярно больше 2500 — см. таблицу troubleshooting в
[engine/orchestrator/README.md](engine/orchestrator/README.md).

**Казахский — без клона голоса.** XTTS-v2 клонирует только ru и en. Для kk
берётся `facebook/mms-tts-kaz` (VITS, CPU): голос стандартный, `voice_id`
игнорируется, автоклон собеседника не создаётся. KazakhTTS2 (ISSAI) можно
подключить через `RT_TTS_KK_BACKEND=kazakhtts2` + `RT_KAZAKHTTS_REPO`, но
готового загружаемого чекпойнта нет — это ручная установка.

**Лицензии моделей — некоммерческое использование.** Веса XTTS-v2 идут под
*Coqui Public Model License* (CPML): личное и исследовательское использование,
коммерческое — нет. `facebook/mms-tts-kaz` — CC-BY-NC 4.0, тоже
non-commercial. faster-whisper (MIT) и NLLB-200 (CC-BY-NC 4.0) — код
свободный, веса NLLB тоже некоммерческие. Итог: **сборка годится для личных
звонков и исследований, не для продукта**. Платных API в проекте нет и быть не
должно (CLAUDE.md, закон 3).

**Голос — биометрия.** Автоклон голоса собеседника допустим только с его
согласия; по умолчанию профиль удаляется вместе с сессией
(`RT_AUTO_CLONE_KEEP=0`), выключается целиком через `RT_AUTO_CLONE=0`.

**Только Windows и только один звонок.** Живой звук — WASAPI loopback и
VB-Cable, то есть Windows 10/11; на Linux/macOS работает лишь файловый и
фейковый режимы. Один пользователь, одна сессия на процесс: новый
`session.start` останавливает предыдущую сессию.

**Нужны наушники.** Loopback снимает всё, что играет в устройстве вывода.
Если перевод идёт в колонки, движок услышит сам себя и начнёт переводить
собственную речь.

**Качество.** whisper `small` — компромисс под 6 GB VRAM: имена, термины и
акцент он путает. NLLB-200-distilled-600M переводит фразу без контекста
диалога (окно контекста в коде есть, но NLLB его не использует). Первая фраза
после старта звучит встроенным голосом XTTS — клон собеседника готов примерно
через 12 секунд его речи.

**Чего ещё нет.** LLM-подсказок ассистента (кнопка — заглушка), пагинации и
поиска в истории, удаления голосов через REST, объединённой дорожки
«собеседник + пользователь» в записи: `sessions.audio_path` указывает на WAV
входящего потока, речь пользователя лежит рядом как `<id>_out.wav`.

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
  согласия его владельца. Движок клонирует голос собеседника автоматически,
  чтобы перевод звучал его голосом: предупредите собеседника, а если согласия
  нет — выключите автоклон (`RT_AUTO_CLONE=0`). По умолчанию автопрофиль
  удаляется вместе с сессией.
- При записи разговора собеседник должен быть уведомлён; UI показывает статус
  записи, ответственность за уведомление — на пользователе.
