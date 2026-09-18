# engine/contracts — контракты событий

> **МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ ВЛАДЕЛЬЦА ПРОЕКТА.**
> CLAUDE.md, закон 2: файлы в `engine/contracts/`, схема БД в `db/` и раздел 5
> `ARCHITECTURE.md` неприкосновенны. Если контракт мешает или в нём ошибка —
> остановись и напиши владельцу текстом предлагаемое изменение, не коммить код.
> Изменение схем = отдельный PR с явным одобрением владельца.

Это единственный разрешённый способ взаимодействия между модулями движка
(`audio_io`, `stt`, `translate`, `tts`, `orchestrator`) и UI. Прямые импорты
внутренностей чужого модуля запрещены.

## Конверт

Все сообщения WebSocket имеют вид:

```json
{ "type": "stt.final", "ts": 1758240000000, "payload": { "...": "..." } }
```

- `type` — строка из таблицы ниже;
- `ts` — unix-время в миллисекундах (`integer`);
- `payload` — объект, состав зависит от `type`.

Все поля обязательны, `additionalProperties: false` — лишнее поле = ошибка
валидации. Схемы — JSON Schema draft 2020-12.

## События движок → UI

| `type` | payload | схема |
|---|---|---|
| `stt.partial` | `stream`, `lang`, `text` | `stt.partial.json` |
| `stt.final` | `stream`, `lang`, `text`, `t_start_ms`, `t_end_ms` | `stt.final.json` |
| `translation.ready` | `stream`, `src_lang`, `dst_lang`, `src_text`, `text`, `ref_utterance_id` | `translation.ready.json` |
| `tts.chunk` | `stream`, `seq`, `pcm_base64` | `tts.chunk.json` |
| `session.state` | `status`, `session_id`, `langs`, `voice_id` | `session.state.json` |
| `metrics.latency` | `stream`, `stt_ms`, `mt_ms`, `tts_ms`, `total_ms` | `metrics.latency.json` |

## Команды UI → движок

| `type` | payload | схема |
|---|---|---|
| `session.start` | `lang_in`, `lang_out`, `voice_id`, `record` | `session.start.json` |
| `session.stop` | `{}` | `session.stop.json` |

## Типы значений

- `lang`, `src_lang`, `dst_lang`, `lang_in`, `lang_out` — enum `"ru" | "en" | "kk"`;
- `stream` — enum `"in" | "out"` (`in` — речь собеседника из Zoom, `out` — речь пользователя);
- `status` — enum `"idle" | "running"`;
- `ref_utterance_id` — `integer | null` (id строки в таблице `utterances`);
- `session_id` — `integer | null` (`null` при `status = "idle"`);
- `voice_id` — `string | null` (`null` = голос по умолчанию / без клона, напр. для kk);
- `langs` — объект `{"in": lang, "out": lang}`;
- `pcm_base64` — base64 от PCM 24 kHz mono int16 (TTS отдаёт 24 kHz, ресемплинг — за orchestrator/audio_io);
- `t_start_ms`, `t_end_ms`, `*_ms` — неотрицательные `integer`, миллисекунды.

`envelope.json` описывает только общий конверт (`payload` — любой объект) и
используется как первая ступень валидации; вторая ступень — схема конкретного типа.

## Использование из Python

```python
from engine.contracts.events import (
    ContractError,
    Lang,
    SttFinal,
    Stream,
    parse_event,
    to_json,
)

raw = to_json(SttFinal(Stream.IN, Lang.EN, "hello there", 0, 940))
envelope = parse_event(raw)  # -> Envelope
assert envelope.type == "stt.final"
assert envelope.payload.text == "hello there"

try:
    parse_event('{"type": "stt.partial", "ts": 1, "payload": {"stream": "in"}}')
except ContractError as exc:
    print(exc)  # не прошло валидацию схемой
```

- `to_json(event, ts=None)` — оборачивает payload-датакласс в конверт (`ts` по
  умолчанию — текущее время), валидирует и сериализует;
- `parse_event(raw)` — принимает `str | bytes | dict`, валидирует по схемам и
  возвращает `Envelope` с типизированным payload-ом;
- `validate_envelope(dict)` — валидация без разбора в датаклассы;
- датаклассы `frozen=True, slots=True` — иммутабельные, дешёвые;
- константы типов: `EVENT_STT_PARTIAL`, `EVENT_STT_FINAL`, `EVENT_TRANSLATION_READY`,
  `EVENT_TTS_CHUNK`, `EVENT_SESSION_STATE`, `EVENT_METRICS_LATENCY`,
  `EVENT_SESSION_START` (= `COMMAND_SESSION_START`), `EVENT_SESSION_STOP`
  (= `COMMAND_SESSION_STOP`); реестр `EVENT_CLASSES`, кортеж `ALL_TYPES`.

Ошибки контракта — всегда `ContractError` (наследник `ValueError`).

Тесты контрактов: `engine/tests/test_contracts.py`, запуск `pytest engine/tests`.
