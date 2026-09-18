# engine/translate

Зона агента C, см. ARCHITECTURE.md 4.3.

**Назначение:** перевод фраз между ru / en / kk. Вход — событие `stt.final`,
выход — `translation.ready` (контракты в `engine/contracts/`).
Модель: `facebook/nllb-200-distilled-600M` на **CPU** — GPU (6 GB) целиком
отдана STT и TTS (ARCHITECTURE.md раздел 6, перевод = 0 GB VRAM).

## Что внутри

| файл | роль |
|---|---|
| `base.py` | `TranslationProvider` (Protocol), `TranslationResult`, `TranslateConfig` |
| `nllb.py` | реальный провайдер: `transformers` + NLLB-200-distilled-600M, CPU |
| `fake.py` | `FakeProvider` — детерминированная заглушка для тестов и смоука |
| `cache.py` | LRU-кэш последних 256 переводов |
| `service.py` | `TranslationService`: `stt.final` -> `translation.ready` + окно контекста |
| `__init__.py` | фабрика `create_provider(config)` |

`torch` и `transformers` импортируются **лениво**, внутри `NllbProvider._load()`,
поэтому `import engine.translate` и все юнит-тесты работают без установленных
весов и без torch (CLAUDE.md, техстандарт).

## Как работает

1. Оркестратор отдаёт конверт `stt.final` в `TranslationService.handle(event, dst, ref_utterance_id)`.
2. Сервис берёт `payload.lang` как язык оригинала, `dst` — как целевой:
   - `src == dst` — короткое замыкание: текст возвращается как есть, `latency_ms = 0`;
   - пустая/пробельная фраза — пустой перевод без обращения к модели;
   - иначе зовётся провайдер с окном контекста этого потока.
3. Провайдер `NllbProvider`:
   - коды языков NLLB (FLORES-200): `ru -> rus_Cyrl`, `en -> eng_Latn`, `kk -> kaz_Cyrl`;
   - `tokenizer.src_lang = <код источника>`,
     `forced_bos_token_id = tokenizer.convert_tokens_to_ids(<код цели>)`;
   - генерация под `torch.inference_mode()` в `asyncio.to_thread` — event loop не блокируется;
   - перед генерацией смотрит LRU-кэш (256 последних фраз на все направления):
     повторная реплика («да», «ok, thanks») отдаётся мгновенно.
4. Сервис собирает `TranslationReady(stream, src_lang, dst_lang, src_text, text,
   ref_utterance_id)`, оборачивает в конверт и **валидирует JSON-схемой** —
   некорректное событие в WebSocket не уедет.

Окно контекста — последние **3** фразы по каждому потоку (`in` / `out`),
доступно через `service.context(stream)`. NLLB его игнорирует (модель переводит
пофразно); окно сделано под будущего LLM-провайдера.

## Запуск

```bash
# зависимости модуля
pip install -r engine/translate/requirements-translate.txt
# (CPU-колесо torch экономит ~2 GB: pip install torch --index-url https://download.pytorch.org/whl/cpu)

# веса заранее (иначе скачаются при первом переводе)
python scripts/download_models.py --only nllb

# один перевод
python scripts/translate_smoke.py --from ru --to en "Привет, как дела?"

# без моделей вообще — проверить обвязку
python scripts/translate_smoke.py --fake --from ru --to kk "Привет"

# бенчмарк: 6 направлений на фразе из 15 слов, таблица латентностей
python scripts/translate_smoke.py --bench
```

Из кода:

```python
from engine.contracts.events import Lang
from engine.translate import TranslateConfig, TranslationService, create_provider

provider = create_provider(TranslateConfig.from_env())
provider.warmup()                      # синхронно, на старте движка
service = TranslationService(provider)
ready = await service.handle(stt_final_envelope, Lang.RU, ref_utterance_id=None)
```

## Конфигурация

`TranslateConfig.from_env()` читает:

| переменная | по умолчанию | смысл |
|---|---|---|
| `RT_MT_BACKEND` | `nllb` | `nllb` \| `fake` |
| `RT_MT_MODEL` | `facebook/nllb-200-distilled-600M` | id модели или путь |
| `RT_MT_DEVICE` | `cpu` | устройство инференса (GPU занята — не меняйте без нужды) |
| `RT_MT_THREADS` | `min(4, cpu_count)` | `torch.set_num_threads` |
| `RT_MT_INT8` | `1` | динамическая квантизация Linear-слоёв |
| `RT_MT_BEAMS` | `1` | `1` = greedy (быстро), `2` = качественнее и ~в 1.6-2 раза дольше |
| `RT_MT_MAX_NEW_TOKENS` | `128` | потолок длины перевода |
| `RT_MT_CACHE` | `256` | размер LRU-кэша, `0` выключает |
| `RT_MODELS_DIR` | `./models` | кэш весов; если есть `<dir>/nllb/config.json`, грузим оттуда |

Квантизация — `torch.quantization.quantize_dynamic(model, {torch.nn.Linear},
dtype=torch.qint8)`, применяется только на `device="cpu"`.

## Ожидаемая латентность (Ryzen 5 5600H, CPU, 4 потока)

Оценка для фразы ~15 слов, `num_beams=1`, int8-квантизация:

| сценарий | ожидание |
|---|---|
| первая загрузка модели (`warmup`) | 5-15 с (с диска; первое скачивание — дольше) |
| фраза 5-8 слов | ~200-400 мс |
| фраза ~15 слов | ~400-800 мс |
| `num_beams=2` | +60-100 % к времени генерации |
| попадание в кэш | < 1 мс |

Бюджет всей цепочки — 2.5 с (ARCHITECTURE.md раздел 2), на перевод разумно
закладывать не больше ~700 мс. **Цифры — оценка, а не замер:** реальные числа
на своей машине снимайте бенчмарком

```bash
python scripts/translate_smoke.py --bench          # 3 прогона на направление
python scripts/translate_smoke.py --bench --beams 2 --runs 5
```

Бенчмарк отключает кэш (`cache_size=0`), иначе повторы показали бы 0 мс.
Если не укладываетесь: уменьшите `RT_MT_MAX_NEW_TOKENS`, держите `RT_MT_BEAMS=1`,
проверьте, что int8 включён, и не отдавайте переводу больше 4 потоков —
остальные ядра нужны audio_io и STT.

## Как подключить LLM-провайдера позже

Интерфейс один — `engine/translate/base.py`:

```python
class MyLlmProvider:
    def warmup(self) -> None: ...
    async def translate(self, text, src, dst, context=()) -> TranslationResult: ...
```

Достаточно реализовать эти два метода (`TranslationProvider` — `Protocol`,
наследоваться не нужно) и вернуть его из `create_provider` по новому значению
`RT_MT_BACKEND`. `context` уже приходит: последние 3 фразы потока — то, чего
не хватает NLLB для согласования местоимений и терминов.
Важно: CLAUDE.md, закон 3 — **только бесплатные локальные модели**, платные API
подключать нельзя.

## Заметка про качество kk

NLLB-200-**distilled-600M** — компромисс по размеру. Для ru↔en качество
разговорных фраз хорошее; для казахского (`kaz_Cyrl`) заметно слабее:
встречаются пропуски частиц, кальки и неточный порядок слов, особенно в
направлении en↔kk (обучающих данных для пары меньше). Что с этим можно делать,
не ломая ограничений проекта:

- держать фразы короткими — STT уже режет по VAD, это работает в нашу пользу;
- если качество kk критично, обсудить с владельцем переход на `nllb-200-1.3B`
  (CPU, но заметно медленнее — бюджет 2.5 с под вопросом) — это изменение
  ARCHITECTURE.md, самовольно делать нельзя (CLAUDE.md, законы 4 и 10).

## Тесты

```bash
python -m pytest engine/tests/test_translate.py          # без моделей
RT_REAL_MODELS=1 python -m pytest engine/tests/test_translate.py -m real_models
```

Покрыто: коды языков NLLB, сборка `translation.ready` для всех 6 направлений,
короткое замыкание `src == dst`, пустой текст, LRU-кэш (счётчик обращений к
модели), окно контекста и его границы, чтение конфига из окружения, фабрика
провайдеров. Тесты с настоящей NLLB (6 пар фраз, проверка ключевых слов)
помечены `@pytest.mark.real_models` и пропускаются без `RT_REAL_MODELS=1`.
