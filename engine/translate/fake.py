"""Фейковый провайдер перевода — для юнит-тестов и смоук-прогонов без весов.

CLAUDE.md, техстандарт: «в юнит-тестах используются фейковые реализации того же
интерфейса». Перевод детерминированный: либо из словаря пар, либо префикс
`[dst] исходный текст`.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence

from engine.contracts.events import Lang
from engine.translate.base import TranslationResult

__all__ = ["FakeProvider", "FakeTable"]

#: Словарь подстановок: (src, dst, исходный текст) -> перевод.
FakeTable = Mapping[tuple[Lang, Lang, str], str]


class FakeProvider:
    """Детерминированный провайдер без моделей.

    Args:
        table: точные переводы для конкретных фраз; всё остальное получает
            префикс `[dst]`.
        delay: искусственная задержка перевода в секундах (тесты таймингов).
        prefix: шаблон ответа по умолчанию, поля `dst` и `text`.

    Атрибуты `calls`, `last_context` и `warmups` открыты специально —
    тесты проверяют по ним, дёргали ли провайдера.
    """

    __slots__ = ("_delay", "_prefix", "_table", "calls", "last_context", "warmups")

    def __init__(
        self,
        table: FakeTable | None = None,
        delay: float = 0.0,
        prefix: str = "[{dst}] {text}",
    ) -> None:
        self._table: dict[tuple[Lang, Lang, str], str] = dict(table or {})
        self._delay = delay
        self._prefix = prefix
        #: сколько раз реально вызывали перевод (без коротких замыканий)
        self.calls = 0
        #: сколько раз вызывали `warmup()`
        self.warmups = 0
        #: контекст последнего вызова (сервис передаёт окно последних фраз)
        self.last_context: tuple[str, ...] = ()

    def warmup(self) -> None:
        """Ничего не грузит, только считает вызовы."""
        self.warmups += 1

    async def translate(
        self,
        text: str,
        src: Lang,
        dst: Lang,
        context: Sequence[str] = (),
    ) -> TranslationResult:
        """Вернуть детерминированный «перевод» с теми же правилами, что и NLLB."""
        self.last_context = tuple(context)
        stripped = text.strip()
        if not stripped:
            return TranslationResult(text="", src=src, dst=dst, latency_ms=0)
        if src == dst:
            return TranslationResult(text=text, src=src, dst=dst, latency_ms=0)

        started = time.perf_counter()
        self.calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        translated = self._table.get(
            (src, dst, stripped),
            self._prefix.format(dst=dst.value, text=stripped),
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return TranslationResult(text=translated, src=src, dst=dst, latency_ms=latency_ms)
