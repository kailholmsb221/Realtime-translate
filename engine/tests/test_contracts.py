"""Тесты контрактов событий (ARCHITECTURE.md раздел 5).

Проверяют, что: валидные примеры каждого типа проходят валидацию, невалидные
(лишнее поле, неверный lang, пропущенное поле, не тот тип) — падают с
``ContractError``, и что ``to_json``/``parse_event`` дают round-trip.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from engine.contracts.events import (
    ALL_TYPES,
    EVENT_CLASSES,
    EVENT_METRICS_LATENCY,
    EVENT_SESSION_START,
    EVENT_SESSION_STATE,
    EVENT_SESSION_STOP,
    EVENT_STT_FINAL,
    EVENT_STT_PARTIAL,
    EVENT_TRANSLATION_READY,
    EVENT_TTS_CHUNK,
    ContractError,
    Envelope,
    Event,
    Lang,
    Langs,
    MetricsLatency,
    SessionStart,
    SessionState,
    SessionStatus,
    SessionStop,
    Stream,
    SttFinal,
    SttPartial,
    TranslationReady,
    TtsChunk,
    now_ms,
    parse_event,
    to_json,
    validate_envelope,
)

TS = 1_758_240_000_000

VALID_EVENTS: dict[str, Event] = {
    EVENT_STT_PARTIAL: SttPartial(stream=Stream.IN, lang=Lang.EN, text="hello the"),
    EVENT_STT_FINAL: SttFinal(
        stream=Stream.IN, lang=Lang.EN, text="hello there", t_start_ms=0, t_end_ms=940
    ),
    EVENT_TRANSLATION_READY: TranslationReady(
        stream=Stream.IN,
        src_lang=Lang.EN,
        dst_lang=Lang.RU,
        src_text="hello there",
        text="привет",
        ref_utterance_id=42,
    ),
    EVENT_TTS_CHUNK: TtsChunk(stream=Stream.OUT, seq=0, pcm_base64="AAAA"),
    EVENT_SESSION_STATE: SessionState(
        status=SessionStatus.RUNNING,
        session_id=7,
        langs=Langs(in_=Lang.EN, out=Lang.KK),
        voice_id="voice-1",
    ),
    EVENT_METRICS_LATENCY: MetricsLatency(
        stream=Stream.IN, stt_ms=400, mt_ms=180, tts_ms=700, total_ms=1280
    ),
    EVENT_SESSION_START: SessionStart(
        lang_in=Lang.EN, lang_out=Lang.RU, voice_id=None, record=True
    ),
    EVENT_SESSION_STOP: SessionStop(),
}


def envelope_dict(type_name: str) -> dict[str, Any]:
    """Валидный конверт указанного типа как обычный словарь."""
    return Envelope(type=type_name, ts=TS, payload=VALID_EVENTS[type_name]).to_dict()


def test_registry_covers_all_contract_types() -> None:
    """Реестр покрывает все 8 типов раздела 5 и совпадает с ALL_TYPES."""
    assert set(EVENT_CLASSES) == set(VALID_EVENTS)
    assert set(ALL_TYPES) == set(EVENT_CLASSES)
    assert len(ALL_TYPES) == 8


@pytest.mark.parametrize("type_name", sorted(VALID_EVENTS))
def test_valid_event_passes_validation(type_name: str) -> None:
    """Валидный пример каждого типа проходит обе ступени валидации."""
    assert validate_envelope(envelope_dict(type_name)) == type_name


@pytest.mark.parametrize("type_name", sorted(VALID_EVENTS))
def test_round_trip_to_json_parse_event(type_name: str) -> None:
    """to_json -> parse_event возвращает эквивалентный объект."""
    event = VALID_EVENTS[type_name]
    raw = to_json(event, ts=TS)

    restored = parse_event(raw)
    assert isinstance(restored, Envelope)
    assert restored.type == type_name
    assert restored.ts == TS
    assert restored.payload == event
    assert type(restored.payload) is type(event)

    # и второй круг даёт байт-в-байт тот же JSON
    assert to_json(restored) == raw


@pytest.mark.parametrize("type_name", sorted(VALID_EVENTS))
def test_parse_event_accepts_dict(type_name: str) -> None:
    """parse_event принимает не только строку, но и готовый словарь и байты."""
    data = envelope_dict(type_name)
    assert parse_event(data).payload == VALID_EVENTS[type_name]
    assert parse_event(json.dumps(data).encode()).payload == VALID_EVENTS[type_name]


def test_to_json_shape_and_default_ts() -> None:
    """Конверт — ровно {type, ts, payload}, ts по умолчанию близок к текущему."""
    before = now_ms()
    data = json.loads(to_json(VALID_EVENTS[EVENT_STT_PARTIAL]))
    after = now_ms()

    assert set(data) == {"type", "ts", "payload"}
    assert data["type"] == EVENT_STT_PARTIAL
    assert before <= data["ts"] <= after
    assert data["payload"] == {"stream": "in", "lang": "en", "text": "hello the"}


def test_session_stop_payload_is_empty() -> None:
    """У session.stop payload — пустой объект."""
    assert json.loads(to_json(SessionStop(), ts=TS))["payload"] == {}


def test_langs_uses_in_key_not_in_() -> None:
    """В JSON поле называется `in`, в Python — `in_` (ключевое слово)."""
    payload = json.loads(to_json(VALID_EVENTS[EVENT_SESSION_STATE], ts=TS))["payload"]
    assert payload["langs"] == {"in": "en", "out": "kk"}


def test_nullable_fields_accept_null() -> None:
    """ref_utterance_id / session_id / voice_id допускают null."""
    translation = TranslationReady(
        stream=Stream.OUT,
        src_lang=Lang.RU,
        dst_lang=Lang.KK,
        src_text="привет",
        text="сәлем",
        ref_utterance_id=None,
    )
    assert parse_event(to_json(translation, ts=TS)).payload == translation

    idle = SessionState(
        status=SessionStatus.IDLE,
        session_id=None,
        langs=Langs(in_=Lang.EN, out=Lang.RU),
        voice_id=None,
    )
    assert parse_event(to_json(idle, ts=TS)).payload == idle


# --- негативные случаи -----------------------------------------------------


@pytest.mark.parametrize("type_name", sorted(VALID_EVENTS))
def test_extra_payload_field_is_rejected(type_name: str) -> None:
    """Лишнее поле в payload запрещено (additionalProperties: false)."""
    data = envelope_dict(type_name)
    data["payload"]["unexpected"] = "nope"
    with pytest.raises(ContractError):
        parse_event(data)


@pytest.mark.parametrize("type_name", sorted(VALID_EVENTS))
def test_extra_envelope_field_is_rejected(type_name: str) -> None:
    """Лишнее поле в конверте тоже запрещено."""
    data = envelope_dict(type_name)
    data["extra"] = 1
    with pytest.raises(ContractError):
        parse_event(data)


@pytest.mark.parametrize(
    ("type_name", "field"),
    [
        (EVENT_STT_PARTIAL, "lang"),
        (EVENT_STT_FINAL, "lang"),
        (EVENT_TRANSLATION_READY, "src_lang"),
        (EVENT_TRANSLATION_READY, "dst_lang"),
        (EVENT_SESSION_START, "lang_in"),
        (EVENT_SESSION_START, "lang_out"),
    ],
)
def test_invalid_lang_is_rejected(type_name: str, field: str) -> None:
    """lang вне enum ru/en/kk не проходит."""
    data = envelope_dict(type_name)
    data["payload"][field] = "de"
    with pytest.raises(ContractError):
        parse_event(data)


def test_invalid_lang_in_nested_langs_is_rejected() -> None:
    """Вложенный langs тоже ограничен enum-ом."""
    data = envelope_dict(EVENT_SESSION_STATE)
    data["payload"]["langs"]["in"] = "fr"
    with pytest.raises(ContractError):
        parse_event(data)


@pytest.mark.parametrize("type_name", sorted(VALID_EVENTS))
def test_missing_envelope_field_is_rejected(type_name: str) -> None:
    """Все поля конверта обязательны."""
    for field in ("type", "ts", "payload"):
        data = envelope_dict(type_name)
        del data[field]
        with pytest.raises(ContractError):
            parse_event(data)


def test_missing_payload_field_is_rejected() -> None:
    """Все поля payload обязательны."""
    data = envelope_dict(EVENT_STT_FINAL)
    del data["payload"]["t_end_ms"]
    with pytest.raises(ContractError):
        parse_event(data)


def test_invalid_stream_is_rejected() -> None:
    """stream — только in/out."""
    data = envelope_dict(EVENT_STT_PARTIAL)
    data["payload"]["stream"] = "both"
    with pytest.raises(ContractError):
        parse_event(data)


def test_invalid_status_is_rejected() -> None:
    """status — только idle/running."""
    data = envelope_dict(EVENT_SESSION_STATE)
    data["payload"]["status"] = "paused"
    with pytest.raises(ContractError):
        parse_event(data)


def test_wrong_value_type_is_rejected() -> None:
    """ts — целое, таймкоды — целые, record — bool."""
    data = envelope_dict(EVENT_STT_FINAL)
    data["ts"] = "1758240000000"
    with pytest.raises(ContractError):
        parse_event(data)

    data = envelope_dict(EVENT_STT_FINAL)
    data["payload"]["t_start_ms"] = "0"
    with pytest.raises(ContractError):
        parse_event(data)

    data = envelope_dict(EVENT_SESSION_START)
    data["payload"]["record"] = "true"
    with pytest.raises(ContractError):
        parse_event(data)


def test_negative_timecode_is_rejected() -> None:
    """Таймкоды неотрицательны."""
    data = envelope_dict(EVENT_STT_FINAL)
    data["payload"]["t_start_ms"] = -1
    with pytest.raises(ContractError):
        parse_event(data)


def test_ref_utterance_id_must_be_int_or_null() -> None:
    """ref_utterance_id — integer или null, строка не проходит."""
    data = envelope_dict(EVENT_TRANSLATION_READY)
    data["payload"]["ref_utterance_id"] = "42"
    with pytest.raises(ContractError):
        parse_event(data)


def test_unknown_type_is_rejected() -> None:
    """Неизвестный тип события отвергается."""
    with pytest.raises(ContractError, match="Неизвестный тип"):
        parse_event({"type": "stt.magic", "ts": TS, "payload": {}})


def test_mismatched_payload_for_type_is_rejected() -> None:
    """payload одного типа под шапкой другого не проходит."""
    data = envelope_dict(EVENT_STT_PARTIAL)
    data["type"] = EVENT_STT_FINAL
    with pytest.raises(ContractError):
        parse_event(data)


def test_malformed_json_is_rejected() -> None:
    """Битый JSON и не-объект отвергаются с ContractError."""
    with pytest.raises(ContractError, match="Невалидный JSON"):
        parse_event("{not json")
    with pytest.raises(ContractError):
        parse_event("[1, 2, 3]")


def test_to_json_validates_before_sending() -> None:
    """to_json не выпускает наружу конверт, нарушающий схему."""
    broken = Envelope(type=EVENT_STT_PARTIAL, ts=TS, payload=VALID_EVENTS[EVENT_STT_FINAL])
    with pytest.raises(ContractError):
        to_json(broken)


def test_events_are_frozen() -> None:
    """Датаклассы событий иммутабельны (frozen=True)."""
    event = VALID_EVENTS[EVENT_STT_PARTIAL]
    with pytest.raises(AttributeError):
        event.text = "changed"  # type: ignore[attr-defined]
