"""Обработчик событий перевода: `stt.final` -> `translation.ready`.

ARCHITECTURE.md 4.3: вход — `stt.final`, выход — `translation.ready`.
Сервис ничего не знает о провайдере, кроме интерфейса `TranslationProvider`,
поэтому NLLB заменяется на LLM-провайдера без правок оркестратора.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Final

from engine.contracts.events import (
    EVENT_STT_FINAL,
    Envelope,
    Lang,
    Stream,
    SttFinal,
    TranslationReady,
    validate_envelope,
)
from engine.translate.base import TranslationError, TranslationProvider, TranslationResult

__all__ = ["DEFAULT_CONTEXT_SIZE", "TranslationService"]

logger = logging.getLogger(__name__)

#: Сколько последних фраз потока держим как контекст диалога.
DEFAULT_CONTEXT_SIZE: Final[int] = 3


class TranslationService:
    """Переводит `stt.final` и собирает валидный `translation.ready`.

    Хранит скользящее окно последних `context_size` фраз по каждому потоку
    (`in` — собеседник, `out` — пользователь). NLLB контекст игнорирует, но
    будущему LLM-провайдеру он нужен, и окно уже есть на месте.
    """

    __slots__ = ("_context", "_context_size", "_provider", "last_latency_ms")

    def __init__(
        self,
        provider: TranslationProvider,
        context_size: int = DEFAULT_CONTEXT_SIZE,
    ) -> None:
        if context_size < 0:
            raise TranslationError(f"context_size должен быть >= 0, получено {context_size}")
        self._provider = provider
        self._context_size = context_size
        self._context: dict[Stream, deque[str]] = {
            stream: deque(maxlen=context_size) for stream in Stream
        }
        #: Латентность последнего перевода, мс (для `metrics.latency`).
        self.last_latency_ms = 0

    @property
    def provider(self) -> TranslationProvider:
        """Провайдер перевода."""
        return self._provider

    @property
    def context_size(self) -> int:
        """Размер окна контекста."""
        return self._context_size

    def context(self, stream: Stream) -> tuple[str, ...]:
        """Последние фразы потока (от старой к новой)."""
        return tuple(self._context[stream])

    def reset(self, stream: Stream | None = None) -> None:
        """Забыть контекст одного потока или всех (например, на `session.stop`)."""
        streams = (stream,) if stream is not None else tuple(Stream)
        for item in streams:
            self._context[item].clear()

    async def translate(
        self,
        text: str,
        src: Lang,
        dst: Lang,
        stream: Stream,
    ) -> TranslationResult:
        """Перевести фразу с учётом окна контекста потока и обновить окно.

        `src == dst` и пустой текст обрабатываются без вызова провайдера.
        """
        stripped = text.strip()
        if not stripped:
            result = TranslationResult(text="", src=src, dst=dst, latency_ms=0)
        elif src == dst:
            result = TranslationResult(text=text, src=src, dst=dst, latency_ms=0)
        else:
            result = await self._provider.translate(text, src, dst, self.context(stream))

        if stripped:
            self._context[stream].append(stripped)
        self.last_latency_ms = result.latency_ms
        return result

    async def handle(
        self,
        event: Envelope,
        dst: Lang,
        ref_utterance_id: int | None = None,
    ) -> Envelope:
        """Обработать конверт `stt.final` и вернуть конверт `translation.ready`.

        Args:
            event: конверт с payload-ом `SttFinal` (`engine/contracts`).
            dst: целевой язык.
            ref_utterance_id: id строки `utterances` в БД или `None`.

        Returns:
            Провалидированный конверт `translation.ready`.

        Raises:
            TranslationError: если пришёл не `stt.final`.
            ContractError: если собранное событие не проходит валидацию схемой.
        """
        payload = event.payload
        if event.type != EVENT_STT_FINAL or not isinstance(payload, SttFinal):
            raise TranslationError(f"Ожидался {EVENT_STT_FINAL}, получено {event.type!r}")

        result = await self.translate(payload.text, payload.lang, dst, payload.stream)
        logger.debug(
            "MT %s->%s (%s): %d мс",
            payload.lang.value,
            dst.value,
            payload.stream.value,
            result.latency_ms,
        )

        ready = TranslationReady(
            stream=payload.stream,
            src_lang=payload.lang,
            dst_lang=dst,
            src_text=payload.text,
            text=result.text,
            ref_utterance_id=ref_utterance_id,
        )
        envelope = Envelope.wrap(ready)
        validate_envelope(envelope.to_dict())
        return envelope
