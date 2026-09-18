"""Типизированные контракты событий Realtime Translator.

Единственный источник правды по формату сообщений между модулями движка и UI
(ARCHITECTURE.md раздел 5). JSON-схемы лежат рядом в этом же пакете и
подгружаются лениво; каждый конверт валидируется дважды — общей схемой
``envelope.json`` и схемой конкретного типа.

МЕНЯТЬ ТОЛЬКО С РАЗРЕШЕНИЯ ВЛАДЕЛЬЦА ПРОЕКТА (CLAUDE.md, закон 2).

Пример::

    from engine.contracts.events import Lang, SttFinal, Stream, parse_event, to_json

    raw = to_json(SttFinal(Stream.IN, Lang.EN, "hello", 0, 900))
    envelope = parse_event(raw)
    assert envelope.payload.text == "hello"
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from functools import cache
from pathlib import Path
from typing import Any, ClassVar, Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

__all__ = [
    "ALL_TYPES",
    "COMMAND_SESSION_START",
    "COMMAND_SESSION_STOP",
    "EVENT_CLASSES",
    "EVENT_METRICS_LATENCY",
    "EVENT_SESSION_START",
    "EVENT_SESSION_STATE",
    "EVENT_SESSION_STOP",
    "EVENT_STT_FINAL",
    "EVENT_STT_PARTIAL",
    "EVENT_TRANSLATION_READY",
    "EVENT_TTS_CHUNK",
    "ContractError",
    "Envelope",
    "Event",
    "Lang",
    "Langs",
    "MetricsLatency",
    "SessionStart",
    "SessionState",
    "SessionStatus",
    "SessionStop",
    "Stream",
    "SttFinal",
    "SttPartial",
    "TranslationReady",
    "TtsChunk",
    "now_ms",
    "parse_event",
    "to_json",
    "validate_envelope",
]

SCHEMA_DIR: Final[Path] = Path(__file__).resolve().parent
ENVELOPE_SCHEMA: Final[str] = "envelope"

# --- константы типов (значение поля "type" в конверте) ---------------------

EVENT_STT_PARTIAL: Final[str] = "stt.partial"
EVENT_STT_FINAL: Final[str] = "stt.final"
EVENT_TRANSLATION_READY: Final[str] = "translation.ready"
EVENT_TTS_CHUNK: Final[str] = "tts.chunk"
EVENT_SESSION_STATE: Final[str] = "session.state"
EVENT_METRICS_LATENCY: Final[str] = "metrics.latency"
EVENT_SESSION_START: Final[str] = "session.start"
EVENT_SESSION_STOP: Final[str] = "session.stop"

# Команды UI -> движок: те же строки, отдельные имена для читаемости.
COMMAND_SESSION_START: Final[str] = EVENT_SESSION_START
COMMAND_SESSION_STOP: Final[str] = EVENT_SESSION_STOP


class ContractError(ValueError):
    """Сообщение не соответствует контракту из ``engine/contracts``."""


class Lang(StrEnum):
    """Языки проекта (ARCHITECTURE.md раздел 2)."""

    RU = "ru"
    EN = "en"
    KK = "kk"


class Stream(StrEnum):
    """Направление потока: ``in`` — речь собеседника, ``out`` — речь пользователя."""

    IN = "in"
    OUT = "out"


class SessionStatus(StrEnum):
    """Состояние сессии движка."""

    IDLE = "idle"
    RUNNING = "running"


def now_ms() -> int:
    """Текущее unix-время в миллисекундах (значение поля ``ts``)."""
    return int(time.time() * 1000)


# --- события ---------------------------------------------------------------


class Event:
    """База для всех payload-ов. Без полей, чтобы не ломать ``slots``."""

    __slots__ = ()

    TYPE: ClassVar[str] = ""

    def to_payload(self) -> dict[str, Any]:
        """Сериализовать payload в JSON-совместимый словарь."""
        raise NotImplementedError

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Event:
        """Собрать payload из уже провалидированного словаря."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class SttPartial(Event):
    """``stt.partial`` — промежуточная гипотеза распознавания."""

    TYPE: ClassVar[str] = EVENT_STT_PARTIAL

    stream: Stream
    lang: Lang
    text: str

    def to_payload(self) -> dict[str, Any]:
        return {"stream": self.stream.value, "lang": self.lang.value, "text": self.text}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SttPartial:
        return cls(
            stream=Stream(payload["stream"]),
            lang=Lang(payload["lang"]),
            text=payload["text"],
        )


@dataclass(frozen=True, slots=True)
class SttFinal(Event):
    """``stt.final`` — окончательный текст фразы с таймкодами."""

    TYPE: ClassVar[str] = EVENT_STT_FINAL

    stream: Stream
    lang: Lang
    text: str
    t_start_ms: int
    t_end_ms: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "stream": self.stream.value,
            "lang": self.lang.value,
            "text": self.text,
            "t_start_ms": self.t_start_ms,
            "t_end_ms": self.t_end_ms,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SttFinal:
        return cls(
            stream=Stream(payload["stream"]),
            lang=Lang(payload["lang"]),
            text=payload["text"],
            t_start_ms=payload["t_start_ms"],
            t_end_ms=payload["t_end_ms"],
        )


@dataclass(frozen=True, slots=True)
class TranslationReady(Event):
    """``translation.ready`` — готовый перевод фразы."""

    TYPE: ClassVar[str] = EVENT_TRANSLATION_READY

    stream: Stream
    src_lang: Lang
    dst_lang: Lang
    src_text: str
    text: str
    ref_utterance_id: int | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "stream": self.stream.value,
            "src_lang": self.src_lang.value,
            "dst_lang": self.dst_lang.value,
            "src_text": self.src_text,
            "text": self.text,
            "ref_utterance_id": self.ref_utterance_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TranslationReady:
        return cls(
            stream=Stream(payload["stream"]),
            src_lang=Lang(payload["src_lang"]),
            dst_lang=Lang(payload["dst_lang"]),
            src_text=payload["src_text"],
            text=payload["text"],
            ref_utterance_id=payload["ref_utterance_id"],
        )


@dataclass(frozen=True, slots=True)
class TtsChunk(Event):
    """``tts.chunk`` — служебный чанк PCM 24 kHz mono int16 в base64."""

    TYPE: ClassVar[str] = EVENT_TTS_CHUNK

    stream: Stream
    seq: int
    pcm_base64: str

    def to_payload(self) -> dict[str, Any]:
        return {"stream": self.stream.value, "seq": self.seq, "pcm_base64": self.pcm_base64}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TtsChunk:
        return cls(
            stream=Stream(payload["stream"]),
            seq=payload["seq"],
            pcm_base64=payload["pcm_base64"],
        )


@dataclass(frozen=True, slots=True)
class Langs:
    """Пара языков сессии по направлениям потоков."""

    in_: Lang
    out: Lang

    def to_payload(self) -> dict[str, Any]:
        return {"in": self.in_.value, "out": self.out.value}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> Langs:
        return cls(in_=Lang(payload["in"]), out=Lang(payload["out"]))


@dataclass(frozen=True, slots=True)
class SessionState(Event):
    """``session.state`` — текущее состояние сессии движка."""

    TYPE: ClassVar[str] = EVENT_SESSION_STATE

    status: SessionStatus
    session_id: int | None
    langs: Langs
    voice_id: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "session_id": self.session_id,
            "langs": self.langs.to_payload(),
            "voice_id": self.voice_id,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SessionState:
        return cls(
            status=SessionStatus(payload["status"]),
            session_id=payload["session_id"],
            langs=Langs.from_payload(payload["langs"]),
            voice_id=payload["voice_id"],
        )


@dataclass(frozen=True, slots=True)
class MetricsLatency(Event):
    """``metrics.latency`` — задержка по стадиям пайплайна, мс."""

    TYPE: ClassVar[str] = EVENT_METRICS_LATENCY

    stream: Stream
    stt_ms: int
    mt_ms: int
    tts_ms: int
    total_ms: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "stream": self.stream.value,
            "stt_ms": self.stt_ms,
            "mt_ms": self.mt_ms,
            "tts_ms": self.tts_ms,
            "total_ms": self.total_ms,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> MetricsLatency:
        return cls(
            stream=Stream(payload["stream"]),
            stt_ms=payload["stt_ms"],
            mt_ms=payload["mt_ms"],
            tts_ms=payload["tts_ms"],
            total_ms=payload["total_ms"],
        )


@dataclass(frozen=True, slots=True)
class SessionStart(Event):
    """``session.start`` — команда UI начать сессию."""

    TYPE: ClassVar[str] = COMMAND_SESSION_START

    lang_in: Lang
    lang_out: Lang
    voice_id: str | None = None
    record: bool = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "lang_in": self.lang_in.value,
            "lang_out": self.lang_out.value,
            "voice_id": self.voice_id,
            "record": self.record,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SessionStart:
        return cls(
            lang_in=Lang(payload["lang_in"]),
            lang_out=Lang(payload["lang_out"]),
            voice_id=payload["voice_id"],
            record=payload["record"],
        )


@dataclass(frozen=True, slots=True)
class SessionStop(Event):
    """``session.stop`` — команда UI остановить сессию (пустой payload)."""

    TYPE: ClassVar[str] = COMMAND_SESSION_STOP

    def to_payload(self) -> dict[str, Any]:
        return {}

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SessionStop:
        return cls()


EVENT_CLASSES: Final[dict[str, type[Event]]] = {
    EVENT_STT_PARTIAL: SttPartial,
    EVENT_STT_FINAL: SttFinal,
    EVENT_TRANSLATION_READY: TranslationReady,
    EVENT_TTS_CHUNK: TtsChunk,
    EVENT_SESSION_STATE: SessionState,
    EVENT_METRICS_LATENCY: MetricsLatency,
    COMMAND_SESSION_START: SessionStart,
    COMMAND_SESSION_STOP: SessionStop,
}

ALL_TYPES: Final[tuple[str, ...]] = tuple(EVENT_CLASSES)


@dataclass(frozen=True, slots=True)
class Envelope:
    """Общий конверт ``{type, ts, payload}`` (ARCHITECTURE.md раздел 5)."""

    type: str
    ts: int
    payload: Event

    @classmethod
    def wrap(cls, event: Event, ts: int | None = None) -> Envelope:
        """Обернуть событие в конверт; ``ts`` по умолчанию — текущее время."""
        return cls(type=event.TYPE, ts=now_ms() if ts is None else ts, payload=event)

    def to_dict(self) -> dict[str, Any]:
        """JSON-совместимый словарь конверта."""
        return {"type": self.type, "ts": self.ts, "payload": self.payload.to_payload()}


# --- валидация -------------------------------------------------------------


@cache
def _validator(name: str) -> Draft202012Validator:
    path = SCHEMA_DIR / f"{name}.json"
    if not path.is_file():
        raise ContractError(f"Схема не найдена: {path}")
    schema = json.loads(path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def validate_envelope(data: Mapping[str, Any]) -> str:
    """Проверить конверт общей схемой и схемой его типа. Вернуть ``type``.

    Raises:
        ContractError: если данные не соответствуют схемам или тип неизвестен.
    """
    try:
        _validator(ENVELOPE_SCHEMA).validate(data)
    except ValidationError as exc:
        raise ContractError(f"envelope: {exc.message}") from exc

    type_name = str(data["type"])
    if type_name not in EVENT_CLASSES:
        raise ContractError(f"Неизвестный тип события: {type_name!r}")

    try:
        _validator(type_name).validate(data)
    except ValidationError as exc:
        raise ContractError(f"{type_name}: {exc.message}") from exc
    return type_name


def to_json(event: Event | Envelope, ts: int | None = None) -> str:
    """Сериализовать событие (или готовый конверт) в JSON-строку.

    Перед сериализацией результат валидируется по JSON-схемам, поэтому
    некорректное событие не уедет в WebSocket.

    Raises:
        ContractError: если результат не проходит валидацию.
    """
    envelope = event if isinstance(event, Envelope) else Envelope.wrap(event, ts)
    data = envelope.to_dict()
    validate_envelope(data)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def parse_event(raw: str | bytes | Mapping[str, Any]) -> Envelope:
    """Разобрать и провалидировать входящее сообщение.

    Args:
        raw: JSON-строка/байты или уже разобранный словарь.

    Returns:
        Конверт с типизированным payload-ом.

    Raises:
        ContractError: если это не JSON-объект или он нарушает контракт.
    """
    if isinstance(raw, (str, bytes)):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ContractError(f"Невалидный JSON: {exc}") from exc
    else:
        data = dict(raw)

    if not isinstance(data, dict):
        raise ContractError(f"Ожидался JSON-объект, получено {type(data).__name__}")

    type_name = validate_envelope(data)
    payload = EVENT_CLASSES[type_name].from_payload(data["payload"])
    return Envelope(type=type_name, ts=data["ts"], payload=payload)
