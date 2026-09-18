# engine/tts — синтез речи

Зона агента D, ARCHITECTURE.md 4.4. Превращает переведённый текст в аудио
**PCM 24 kHz mono int16** — формат, зафиксированный контрактом
[`tts.chunk.json`](../contracts/tts.chunk.json). Ресемплинг под устройство
вывода делает orchestrator/audio_io.

| Язык | Провайдер | Устройство | Клон голоса |
|---|---|---|---|
| ru, en | XTTS-v2 (`coqui-tts`) | CUDA (~2.5 GB VRAM) | да, по сэмплу 15-30 с |
| kk | `facebook/mms-tts-kaz` (VITS) | CPU (~150 MB) | **нет**, один встроенный голос |
| любой | `FakeTts` (тон) | — | — |

## Быстрый старт

```bash
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r engine/tts/requirements-tts.txt
python scripts/download_models.py --only xtts        # чекпоинт в ./models/xtts

# проверка обвязки без моделей: по WAV на ru/en/kk в ./out_tts/
python scripts/tts_smoke.py --fake

# реальный синтез встроенным голосом XTTS
python scripts/tts_smoke.py --say "Привет, это тест" --lang ru --out out.wav

# профиль своего голоса и синтез клоном
python scripts/tts_smoke.py --create-voice sample.wav --name me --lang ru
python scripts/tts_smoke.py --say "Привет" --lang ru --voice v_1a2b3c4d --out out.wav
```

Из кода:

```python
from engine.contracts.events import Lang, Stream, to_json
from engine.tts import TtsConfig, create_tts

tts = create_tts(TtsConfig.from_env())
await tts.warmup([Lang.RU])                       # загрузить модель заранее

# сырые чанки PCM 24 kHz mono int16 — в наушники / виртуальный кабель
async for chunk in tts.synthesize("привет", Lang.RU, "v_1a2b3c4d"):
    player.write(chunk)

# те же чанки готовыми событиями контракта — в WebSocket
async for envelope in tts.synthesize_events("привет", Lang.RU, "v_1a2b3c4d", stream=Stream.IN):
    await websocket.send(to_json(envelope))       # {"type": "tts.chunk", ...}
```

## Как устроено

```
base.py             TtsProvider (протокол), TtsConfig, OUTPUT_SAMPLE_RATE = 24000
router.py           TtsRouter: язык -> провайдер, chunk_to_event -> Envelope(tts.chunk)
xtts.py             XTTS-v2: клон голоса, потоковый inference_stream
kk_mms.py           facebook/mms-tts-kaz: kk на CPU, 16 kHz -> 24 kHz
kk_kazakhtts2.py    адаптер-заготовка KazakhTTS2 (ISSAI/ESPnet), NotImplementedError
voices.py           VoiceStore: профили голосов на диске (1:1 с таблицей voices)
fake.py             FakeTts: тон вместо речи (тесты, файловый режим)
audio.py            WAV, ресемплинг, нарезка на чанки
```

* **Ленивая загрузка.** `torch`, `TTS`, `transformers` импортируются только
  внутри провайдеров — пакет `engine.tts` и юнит-тесты работают без них.
* **Singleton модели.** XTTS грузится один раз на процесс (ключ — каталог
  чекпоинта + устройство): второй раз 2.5 GB VRAM не выделяется.
* **Стриминг.** `Xtts.inference_stream` — блокирующий генератор, он крутится в
  отдельном потоке (`asyncio.to_thread`), чанки едут в корутину через
  `asyncio.Queue`; на выходе — ровные чанки `chunk_ms` (20-100 мс).
* **Голос по умолчанию.** Если `voice_id is None`, ru/en синтезируются
  встроенным голосом XTTS из `speakers_xtts.pth` (`RT_TTS_XTTS_SPEAKER`,
  по умолчанию «Ana Florence»).
* **kk игнорирует `voice_id`.** Контракт разрешает `voice_id` в `session.start`,
  но клонирования для казахского нет — роутер обнуляет его сам.

## Переменные окружения

| Переменная | Значения | По умолчанию |
|---|---|---|
| `RT_TTS_BACKEND` | `real`, `fake` | `real` |
| `RT_TTS_DEVICE` | `auto`, `cuda`, `cpu` | `auto` (cuda, если есть GPU) |
| `RT_TTS_KK_BACKEND` | `mms`, `kazakhtts2` | `mms` |
| `RT_MODELS_DIR` | путь | `./models` |
| `RT_VOICES_DIR` | путь | `./voices` |
| `RT_TTS_XTTS_MODEL_ID` | id репозитория | `coqui/XTTS-v2` |
| `RT_TTS_XTTS_SPEAKER` | имя встроенного голоса | `Ana Florence` |
| `RT_TTS_KK_MODEL_ID` | id репозитория | `facebook/mms-tts-kaz` |
| `LOG_LEVEL` | уровень logging | `INFO` |

## Бюджет VRAM (ARCHITECTURE.md раздел 6)

XTTS-v2 занимает ~2.5 GB рядом с faster-whisper `small` int8 (~1.0 GB) — на
RTX 4050 6 GB остаётся ~1.5 GB запаса. Поэтому:

* казахский TTS и NLLB держим на CPU;
* модель загружается один раз и живёт в памяти движка;
* одновременная загрузка whisper `medium` и XTTS запрещена;
* `RT_TTS_DEVICE=cpu` спасает при нехватке памяти, но XTTS на CPU медленнее
  реального времени — в задержку 2.5 с не уложится.

## Как записать сэмпл голоса

1. Длительность **15-30 секунд** непрерывной речи (минимум 5 с, максимум 60 с;
   короче 15 с — будет предупреждение и качество клона хуже).
2. Тихое помещение, без музыки, фоновых голосов и эха; микрофон на расстоянии
   ладони, без клиппинга.
3. Читайте обычным темпом связный текст (не отдельные слова), на том языке,
   который указываете в `--lang`.
4. Формат: несжатый WAV (PCM). Хранилище само сведёт в моно и ресемплит в
   24 kHz. Перекодировать можно так:
   `ffmpeg -i запись.m4a -ac 1 -ar 24000 -c:a pcm_s16le sample.wav`.
5. `python scripts/tts_smoke.py --create-voice sample.wav --name me --lang ru`
   напечатает `voice_id` — его же кладёт в БД orchestrator и отдаёт UI в
   `session.start`.

Профиль на диске: `<RT_VOICES_DIR>/<voice_id>/{meta.json, sample.wav, latents.pt}`.
Поля `meta.json` совпадают со строкой таблицы `voices` из `db/schema.sql`
(`created_at_ms` -> `created_at`); писать в SQLite — работа orchestrator.

## Правовая заметка

Клонировать разрешено **только собственный голос** пользователя или голос, на
использование которого владелец дал явное согласие (ARCHITECTURE.md раздел 8).
Голос — биометрический признак: чужой сэмпл без согласия — это и нарушение
закона, и лицензии модели.

## Лицензии моделей

* **XTTS-v2 — Coqui Public Model License (CPML), только некоммерческое
  использование.** Коммерческое применение весов требует отдельной лицензии у
  правообладателя. Код `coqui-tts` (форк idiap) — MPL-2.0.
* `facebook/mms-tts-kaz` — CC-BY-NC 4.0 (MMS, Meta): тоже некоммерческое.
* KazakhTTS2 (ISSAI) — код рецептов Apache-2.0, условия по чекпоинтам смотрите
  в репозитории https://github.com/IS2AI/Kazakh_TTS.

Проект личный и некоммерческий (ARCHITECTURE.md раздел 2: бюджет 0), под эти
условия он подходит; для коммерческого продукта модели придётся менять.

## Известные ограничения

* **Казахский без клона.** MMS-TTS даёт один фиксированный голос; интонация
  беднее XTTS. Улучшение — KazakhTTS2, см. `kk_kazakhtts2.py` (нужен ESPnet).
* MMS-TTS синтезирует фразу целиком, а не потоком: первый чанк приходит после
  полного синтеза куска (длинный текст режется по предложениям).
* XTTS не нормализует числа и аббревиатуры под русскую морфологию — «2024»
  может прозвучать не так, как ожидается.
* Клон по сэмплу 15-30 с не воспроизводит эмоции и сильный акцент.
* На CPU XTTS в реальное время не укладывается — нужен CUDA-torch.

## Тесты

```bash
python -m pytest engine/tests/test_tts.py          # без моделей, быстрые
RT_REAL_MODELS=1 python -m pytest engine/tests/test_tts.py -m real_models   # с XTTS и MMS
ruff check engine/tts scripts/tts_smoke.py && mypy engine/tts
```

Покрыто: валидация и жизненный цикл профилей голоса, ресемплинг 16k -> 24k,
нарезка на чанки, `FakeTts`, выбор провайдера по языку, игнорирование `voice_id`
для kk, соответствие событий `tts.chunk` контракту (включая обратный разбор
base64), чтение конфигурации из окружения. Тесты с реальными моделями помечены
`@pytest.mark.real_models` и пропускаются без `RT_REAL_MODELS=1`.
