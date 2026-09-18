# engine/stt — распознавание речи

Зона агента B, см. ARCHITECTURE.md 4.2.

Поток аудиочанков 16 kHz mono int16 → события `stt.partial` / `stt.final`
(контракты в `engine/contracts/`). VAD режет поток на фразы, faster-whisper их
распознаёт.

## Как работает

```
чанки 30 мс ──► VAD (Silero) ──► сегментатор SttEngine ──► faster-whisper ──► stt.partial
   (audio_io)                     preroll + буфер фразы     (asyncio.to_thread)  stt.final
```

1. **VAD.** `SileroVad` копит чанки и режет их на окна ровно по 512 сэмплов
   (требование Silero при 16 kHz), отдаёт `speech_start(ts_ms)` /
   `speech_end(ts_ms)` с таймкодами от начала потока.
2. **Сегментатор.** `SttEngine` держит «хвост» потока длиной `preroll_ms`,
   поэтому фраза начинается не с момента срабатывания VAD, а с реального начала
   речи (первый слог не съедается). Между `speech_start` и `speech_end` аудио
   копится в буфер фразы; хвостовая тишина обрезается по таймкоду `speech_end`.
3. **Partial.** Каждые `partial_interval_ms` во время речи накопленный буфер
   уходит на распознавание в отдельный поток (`asyncio.to_thread`) → событие
   `stt.partial`. Одновременно работает **не более одной** транскрипции: если
   предыдущая ещё считается, тик пропускается (иначе GPU захлебнётся).
4. **Final.** По `speech_end` (или по достижении `max_utterance_ms` —
   принудительная нарезка длинного монолога) считается финальный текст →
   `stt.final` с `t_start_ms` / `t_end_ms` от начала потока.
5. Каждое событие собирается датаклассом из `engine.contracts.events` и
   валидируется JSON-схемой до выдачи наружу. Время транскрипции пишется в
   логгер `engine.stt` и доступно в `SttEngine.last_latency_ms`.

Вход движка — любой async-итератор объектов с полями `pcm` и `ts_ms` (утиная
типизация): подходят и чанки `engine.audio_io.AudioChunk` (`pcm` в байтах), и
самодельные с numpy-массивом. Прямого импорта `engine.audio_io` здесь нет.

## Состав

| Файл | Что внутри |
|---|---|
| `base.py` | протоколы `VoiceActivityDetector` / `Transcriber`, `VadEvent`, `TranscriptResult`, `SttConfig` |
| `vad_silero.py` | `SileroVad` — обёртка над `silero_vad.VADIterator` |
| `whisper_fw.py` | `FasterWhisperTranscriber` — faster-whisper, выбор модели по языку, fallback на CPU |
| `engine.py` | `SttEngine` — сегментация, partial/final, метрики |
| `fake.py` | `EnergyVad` (RMS-порог, чистый numpy) и `FakeTranscriber` — тесты и файловый режим |
| `__init__.py` | фабрики `create_vad`, `create_transcriber`, `create_engine` |

Тяжёлые модели импортируются **лениво**, внутри фабрик и методов: импорт пакета
не требует `torch`/`faster-whisper`, юнит-тесты зелёные без них.

## Установка

```bash
# CUDA-сборка torch для RTX 4050 — ставится первой
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r engine/stt/requirements-stt.txt
```

### Установка на Windows

* faster-whisper на GPU использует CTranslate2, которому нужны библиотеки
  cuBLAS и cuDNN 9. Проще всего поставить их через pip:
  `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`, либо положить DLL из
  NVIDIA cuDNN рядом и добавить каталог в `PATH`.
* Если CUDA не поднялась, движок **не падает**: `resolve_runtime()` пишет
  warning в лог и переключается на `device=cpu`, `compute_type=int8`. Скорость
  на CPU (Ryzen 5 5600H) ниже реального времени для фраз длиннее ~3 с — это
  режим «чтобы работало», а не рабочий.
* Кэш моделей: `RT_MODELS_DIR` (каталог `models/` в `.gitignore`).

## Конфигурация

`SttConfig.from_env()` читает окружение; любые поля можно перебить аргументами.

| Переменная | Поле | По умолчанию | Смысл |
|---|---|---|---|
| `RT_STT_MODEL` | `model_size` | `small` | размер модели whisper или путь к файнтюну |
| `RT_STT_DEVICE` | `device` | `auto` | `cuda` / `cpu` / `auto` |
| `RT_STT_COMPUTE` | `compute_type` | `int8_float16` | тип вычислений CTranslate2 |
| `RT_STT_KK_MODEL` | `model_overrides["kk"]` | — | модель для казахского |
| `RT_MODELS_DIR` | `models_dir` | кэш HF | куда качать и откуда брать модели |
| `RT_STT_LANG` | `language` | `None` | фиксированный язык; пусто — автоопределение |
| `RT_STT_FALLBACK_LANG` | `fallback_lang` | `ru` | чем заменить язык вне `ru/en/kk` |
| `RT_STT_VAD_THRESHOLD` | `vad_threshold` | `0.5` | порог вероятности речи Silero |
| `RT_STT_ENERGY_THRESHOLD` | `energy_threshold` | `0.02` | порог RMS для `EnergyVad` |
| `RT_STT_MIN_SPEECH_MS` | `min_speech_ms` | `250` | короче — не речь |
| `RT_STT_MIN_SILENCE_MS` | `min_silence_ms` | `500` | столько тишины закрывает фразу |
| `RT_STT_MAX_UTTERANCE_MS` | `max_utterance_ms` | `15000` | предел длины фразы |
| `RT_STT_PARTIAL_INTERVAL_MS` | `partial_interval_ms` | `1000` | период `stt.partial` |
| `RT_STT_BEAM_SIZE` | `beam_size` | `1` | beam search whisper |
| `LOG_LEVEL` | — | `INFO` | уровень логирования |

`model_size` крупнее `small` (`medium`, `large*`, `turbo`) отвергается с
`ValueError`: бюджет VRAM 6 GB, ARCHITECTURE.md раздел 6 и CLAUDE.md закон 4.

## Использование

```python
from engine.stt import create_engine
from engine.stt.base import SttConfig

engine = create_engine(SttConfig.from_env())          # silero + faster-whisper
async for envelope in engine.run(source, stream="in", lang="ru"):
    print(envelope.type, envelope.payload)            # stt.partial / stt.final

# без моделей — для тестов и файлового режима
engine = create_engine(SttConfig(), backend="fake", vad_backend="energy")
```

`lang=None` — автоопределение языка моделью; если whisper вернёт язык вне
контракта (`ru/en/kk`), берётся `fallback_lang`.

## Казахский: подмена модели

Базовый whisper `small` понимает казахский плохо, поэтому язык `kk` можно
увести на отдельный файнтюн:

```bash
set RT_STT_KK_MODEL=models/whisper-kk-ct2     # Windows
export RT_STT_KK_MODEL=models/whisper-kk-ct2  # bash
```

`SttConfig.model_for("kk")` вернёт этот путь, `model_for("ru")` — `small`;
`FasterWhisperTranscriber` держит обе модели в кэше по имени, так что
переключение языка не перезагружает основную модель. Следите за VRAM: две
модели `small` int8 — это ~2 GB вместо 1 GB.

**Где брать модель (проверено в сентябре 2026).** Официальная организация ISSAI
на Hugging Face (<https://huggingface.co/issai>) публикует датасеты
(Kazakh Speech Corpus 2), TTS (`issai/KazGenericTTS`), переводчик
(`issai/tilmash`), но **готового whisper-файнтюна для kk у неё нет**. Ближайший
вариант «от ISSAI» — <https://huggingface.co/akuzdeuov/whisper-base.kk>: whisper
`base`, обучен на Kazakh Speech Corpus 2 (ISSAI, >1000 ч), WER 15.36 % на
тестовой части. Есть и сторонние файнтюны (`abilmansplus/whisper-turbo-kaz-rus-v1`,
`alibiserikbay/kazakh-russian-mixed-stt`), но модели размера `medium`/`large`/`turbo`
брать нельзя — бюджет VRAM.

Если нужной модели нет или появилась новая — ищите так:

1. <https://huggingface.co/models?language=kk&pipeline_tag=automatic-speech-recognition&sort=downloads>
   — фильтр «kk + ASR», сортировка по загрузкам;
2. в поиске HF: `whisper kazakh`, `whisper-kk`, `issai whisper`;
3. берите размер `small` или `base` (не больше — CLAUDE.md закон 4).

Модель в формате transformers нужно сконвертировать в CTranslate2:

```bash
pip install ctranslate2 transformers
ct2-transformers-converter --model akuzdeuov/whisper-base.kk \
    --output_dir models/whisper-kk-ct2 --copy_files tokenizer.json preprocessor_config.json \
    --quantization int8_float16
```

## Smoke-тест

```bash
# без моделей: EnergyVad + FakeTranscriber, проверяет сегментацию и события
python scripts/stt_smoke.py engine/tests/fixtures/stt_speech_pattern.wav --fake

# реальные модели: Silero VAD + faster-whisper small
python scripts/stt_smoke.py sample.wav --lang ru
python scripts/stt_smoke.py sample.wav --lang en --realtime   # в темпе реального времени
python scripts/stt_smoke.py sample.wav --device cpu --partial-interval-ms 500
```

Вход — WAV 16 kHz mono int16. Другой формат скрипт не конвертирует, а
подсказывает команду `ffmpeg`. Вывод — строка на событие со временем от старта и
итоговая сводка: число `final` / `partial`, RTF и задержка транскрипции.

## Замеры

Скрипт печатает медиану и максимум `SttEngine.last_latency_ms` по финальным
фразам — это чистое время распознавания (без VAD и ожидания конца фразы).
Ориентиры для бюджета «конец фразы → озвучка ≤ 2.5 с» (ARCHITECTURE.md раздел 2):

| Конфигурация | Ожидание |
|---|---|
| `small` int8_float16, RTX 4050 | ~0.15–0.4 с на фразу 3–5 с (RTF ≈ 0.1) |
| `small` int8, CPU Ryzen 5 5600H | ~2–5 с на ту же фразу — в realtime не укладывается |

Числа на целевой машине снимает владелец проекта: `--realtime` даёт честную
картину, `--log-level DEBUG` печатает время каждой транскрипции, включая
partial. Если задержка растёт — уменьшайте `max_utterance_ms` и увеличивайте
`partial_interval_ms` (partial-ы едят то же GPU-время, что и final).

## Тесты

```bash
python -m pytest engine/tests/test_stt.py         # без моделей, быстро
RT_REAL_MODELS=1 python -m pytest engine/tests/test_stt.py -m real_models
```

Покрыто: сегментация `EnergyVad` на фикстуре «тон — тишина — тон» (ровно 2
фразы с таймкодами), последовательность `partial → final`, валидация всех
событий схемами контрактов, принудительное закрытие по `max_utterance_ms`,
единственность одновременной транскрипции, чтение конфига из окружения, запрет
крупных моделей, подмена модели для `kk`, fallback на CPU, приём чанков
`audio_io` (pcm в байтах), устойчивость к падению распознавателя.

Фикстура `stt_speech_pattern.wav` генерируется скриптом
`engine/tests/fixtures/stt_fixtures.py` (создаётся автоматически при запуске
тестов, руками — `python engine/tests/fixtures/stt_fixtures.py --force`).
Тесты с реальными моделями помечены `@pytest.mark.real_models` и пропускаются
без `RT_REAL_MODELS=1`.
