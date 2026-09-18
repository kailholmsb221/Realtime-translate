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

## Установка

```bash
git clone <repo> realtime-translator
cd realtime-translator

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

PyTorch с CUDA ставится отдельно под вашу версию CUDA, например:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Проверьте окружение (Python, CUDA, VRAM, аудиоустройства, VB-Cable):

```bash
python scripts/check_env.py
```

Скачайте модели (кэш — `./models`, переопределяется `RT_MODELS_DIR`):

```bash
python scripts/download_models.py             # всё сразу
python scripts/download_models.py --list      # что именно будет скачано
python scripts/download_models.py --only whisper nllb
```

Создайте базу:

```bash
sqlite3 db/translator.db < db/schema.sql
```

## Запуск

```bash
python -m engine.orchestrator      # движок (появится на этапе 2)
cd ui && npm install && npm run dev  # UI на http://localhost:3000
```

Пока оркестратора нет — используйте smoke-скрипты в `scripts/`.
WebSocket движка: `ws://localhost:8765`.

## Тесты

```bash
pytest engine/tests            # тесты движка
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
  orchestrator/       пайплайн, WebSocket-сервер :8765, запись в БД
  contracts/          JSON-схемы событий и events.py — МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ
  tests/              pytest, фикстуры в tests/fixtures/
ui/                   Next.js + TypeScript: субтитры, ассистент, история
db/                   schema.sql (SQLite) — МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ
scripts/              check_env.py, download_models.py, make_fixtures.py
```

Зависимости разнесены по модулям: у каждого свой `requirements-<модуль>.txt`,
корневой `requirements.txt` подключает их через `-r` (плюс `requirements-core.txt`
и `requirements-dev.txt`).

## Переменные окружения

| Переменная | Значение |
|---|---|
| `RT_MODELS_DIR` | каталог кэша моделей (по умолчанию `./models`) |
| `RT_REAL_MODELS` | `1` — не пропускать тесты с реальными моделями |
| `RT_KAZAKHTTS_REPO` | репозиторий KazakhTTS2 на Hugging Face |
| `LOG_LEVEL` | уровень логирования движка |

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
