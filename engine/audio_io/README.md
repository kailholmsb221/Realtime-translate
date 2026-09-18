# engine/audio_io — захват и вывод звука

Зона агента A, см. `ARCHITECTURE.md` 4.1.

Модуль отвечает за весь звук движка: снимает речь собеседника из Zoom/Meet
(WASAPI loopback), снимает микрофон пользователя, играет перевод в наушники и
отправляет переведённый голос в Zoom через VB-Audio Virtual Cable.

Формат внутри движка: **PCM 16 kHz mono int16, чанки по 30 мс = 480 фреймов**
(CLAUDE.md, технический стандарт). TTS отдаёт 24 kHz — ресемплинг под
устройство делает приёмник этого модуля.

## Схема маршрутизации

```
       ┌────────────── Zoom / Google Meet ──────────────┐
       │  голос собеседника            микрофон Zoom    │
       └───────┬────────────────────────────▲───────────┘
               │ играет в наушники          │ «CABLE Output»
               ▼                            │
   WASAPI loopback (pyaudiowpatch)   CABLE Input (VB-Audio) ◄── create_sink("cable")
               │ 48 kHz stereo               ▲
               │ ↓ даунмикс + ресемплинг     │ 24 kHz TTS → ресемплинг
               ▼ 16 kHz mono int16           │
        create_source("loopback")     ┌──────┴───────┐
               │                      │  TTS ru→en   │  outbound
               ▼                      └──────▲───────┘
        STT → MT → TTS (ru)                  │
               │                      create_source("mic")  ◄── микрофон (sounddevice)
               ▼
     create_sink("headphones") ──► наушники пользователя      inbound
```

Итого две цепочки (ARCHITECTURE.md 4.5):

* **inbound:** `loopback → STT → MT → TTS → наушники`;
* **outbound:** `микрофон → STT → MT → TTS → CABLE Input`, а в Zoom
  микрофоном выбран **CABLE Output** — так собеседник слышит перевод.

Важно: наушники обязательны. Если перевод играет в колонки, loopback снова
поймает его и получится петля обратной связи.

## Публичный API

```python
from engine.audio_io import AudioConfig, create_sink, create_source

cfg = AudioConfig.from_env()

src = create_source("loopback", cfg)      # "loopback" | "mic" | "file"
sink = create_sink("headphones", cfg)     # "headphones" | "cable" | "file"

async with src, sink:
    async for chunk in src:               # AudioChunk: pcm bytes, ts_ms, n_frames
        await sink.write(chunk)           # приёмник сам ресемплит под устройство
    await sink.drain()
```

Ключевые сущности:

| Объект | Назначение |
|---|---|
| `AudioFormat` | `sample_rate=16000, channels=1, dtype="int16"`; `ENGINE_FORMAT`, `TTS_FORMAT` (24 kHz) |
| `AudioChunk` | `pcm` (bytes или int16-массив на входе), `ts_ms`, `n_frames`, `fmt`; `samples`, `duration_ms` |
| `AudioSource` | протокол: `open/close`, `async for chunk in src`, async context manager |
| `AudioSink` | протокол: `write(chunk)`, `drain()`, `close()`, async context manager |
| `AudioDeviceInfo` | `id, name, kind (input/output/loopback), default_sample_rate, channels`, `is_virtual_cable` (подстрока `CABLE`) |
| `ChunkAssembler` | режет блоки драйвера на ровные чанки 30 мс с корректными `ts_ms` |
| `FakeSource` / `FakeSink` | файловый режим и тесты (WAV / numpy) |

Модули: `base.py` (абстракции), `pcm.py` (ресемплинг, даунмикс, WAV),
`devices.py` (реестр устройств), `config.py` (env), `windows.py` (WASAPI +
sounddevice), `fake.py` (файл/память), `__init__.py` (фабрики).

## Переменные окружения

| Переменная | Значение по умолчанию | Что делает |
|---|---|---|
| `RT_AUDIO_LOOPBACK` | loopback устройства вывода по умолчанию | подстрока имени устройства, с которого снимаем звук собеседника |
| `RT_AUDIO_MIC` | микрофон по умолчанию | подстрока имени микрофона |
| `RT_AUDIO_HEADPHONES` | вывод по умолчанию | подстрока имени наушников |
| `RT_AUDIO_CABLE` | `CABLE Input` | подстрока имени входа виртуального кабеля |
| `RT_AUDIO_CHUNK_MS` | `30` | длительность чанка захвата, мс |
| `RT_AUDIO_SAMPLE_RATE` | `16000` | частота движка (менять не нужно) |
| `RT_AUDIO_DEVICE_TIMEOUT` | `5` | сколько секунд ждать данных с устройства |
| `LOG_LEVEL` | `INFO` | уровень логирования |

Имена сравниваются без учёта регистра по подстроке, поэтому достаточно
`RT_AUDIO_HEADPHONES=Наушники` или `RT_AUDIO_CABLE=CABLE Input`.

## Установка

```powershell
pip install -r engine/audio_io/requirements-audio_io.txt
```

`pyaudiowpatch` ставится только на Windows, `scipy` опционален (с ним
ресемплинг идёт polyphase-фильтром, без него — линейной интерполяцией).

### VB-Audio Virtual Cable (ставит пользователь вручную)

1. Скачать VB-CABLE Driver с https://vb-audio.com/Cable/ (donationware, бесплатно).
2. Распаковать архив, запустить `VBCABLE_Setup_x64.exe` **от имени администратора**,
   нажать *Install Driver*, перезагрузить Windows.
3. В «Параметры → Система → Звук» появятся два устройства:
   `CABLE Input (VB-Audio Virtual Cable)` — выход, `CABLE Output (VB-Audio Virtual Cable)` — вход.
4. В Zoom: *Настройки → Звук → Микрофон* = **CABLE Output**,
   *Динамик* = ваши наушники.
5. Проверка: `python scripts/audio_smoke.py --tone "CABLE Input"` — тон уходит
   в кабель; в Zoom на вкладке звука должен дрожать индикатор микрофона.

Движок ничего не устанавливает сам (CLAUDE.md, закон 5).

## Smoke-скрипт

```powershell
python scripts/audio_smoke.py --list                        # устройства, метки loopback/CABLE
python scripts/audio_smoke.py --record-loopback 5 out.wav   # 5 с речи собеседника в WAV
python scripts/audio_smoke.py --record-mic 5 mic.wav        # 5 с микрофона в WAV
python scripts/audio_smoke.py --tone "CABLE Input"          # тон 440 Гц 1 с в устройство
python scripts/audio_smoke.py --loop-file sample.wav "Наушники"
```

Коды возврата: `0` — успех, `1` — ошибка аргументов, `2` — аудио недоступно
(не Windows, нет пакетов или нет устройства). На Linux скрипт печатает, какого
бэкенда не хватает, и выходит с кодом 2.

## Тесты

```bash
python -m pytest engine/tests/test_audio_io.py engine/tests/test_audio_io_windows.py -q
```

Тесты не трогают железо и работают на Linux без
`sounddevice`/`pyaudiowpatch`/`scipy`:

* `test_audio_io.py` — ресемплинг (длина и сохранение пика 440 Гц по FFT),
  даунмикс, WAV round-trip, нарезка ровно по 480 фреймов с таймкодами,
  `FakeSink` → WAV, поведение фабрики вне Windows, разбор env-переменных,
  определение виртуального кабеля по имени;
* `test_audio_io_windows.py` — `windows.py` на фейковых драйверах: 48 kHz
  stereo → 16 kHz mono, fallback int16 → float32, передача из потока драйвера
  в asyncio-очередь, таймаут молчащего устройства, буфер воспроизведения
  (ресемплинг 24 kHz → 48 kHz, обратное давление вместо обрыва фразы,
  отбрасывание при мёртвом потоке, тишина при underrun).

## Ограничения и известные особенности

* Живой звук — только Windows (WASAPI loopback). На других ОС любые
  `loopback/mic/headphones/cable` бросают `AudioBackendUnavailable`; работает
  файловый режим (`create_source("file", path=...)`).
* Нужны наушники, иначе loopback поймает собственный перевод (петля).
* Захват идёт в нативном формате устройства (обычно 48 kHz stereo float32);
  даунмикс — простое усреднение каналов, без учёта фазы.
* Ресемплинг без scipy — линейная интерполяция, то есть без анти-алиасного
  фильтра. Для речи в 16 kHz слышимой разницы нет, но scipy предпочтителен.
* Очередь захвата ограничена (по умолчанию 200 чанков ≈ 6 с): при залипании
  потребителя старые чанки отбрасываются, счётчик — в `source.stats`.
* Буфер воспроизведения тоже ограничен (`buffer_ms`, по умолчанию 4 с), но
  переполнение не режет фразу: `SoundDeviceSink.write` ждёт, пока драйвер
  вычерпает место (обратное давление). TTS отдаёт фразу быстрее реального
  времени, и без этого длинный перевод обрывался на середине. Если место так
  и не освободилось за `overflow_wait_s` (поток не играет), чанк отбрасывается
  с предупреждением и счётчиком `sink.stats.dropped_chunks`.
* Windows-часть (`windows.py`) на Linux не тестируется автоматически —
  проверяется вручную smoke-скриптом на целевой машине.
* Эксклюзивный режим WASAPI и выбор конкретного `hostapi` не реализованы;
  устройства берутся в общем (shared) режиме.
