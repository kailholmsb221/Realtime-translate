"""Шина событий движка: один издатель — много подписчиков.

Конвейеры публикуют сюда конверты :class:`~engine.contracts.events.Envelope`,
а сервер раздаёт их всем WebSocket-клиентам; внутренние потребители
(файловый режим, smoke-скрипт, тесты) подписываются тем же способом.

Шина нарочно не блокирует издателя: если подписчик не успевает разбирать
очередь, самые старые события выбрасываются, а счётчик потерь растёт —
конвейер важнее, чем полнота ленты субтитров.

Пример::

    bus = EventBus()
    with bus.subscription() as queue:
        await bus.publish(envelope)
        received = await queue.get()
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Iterator
from typing import Final

from engine.contracts.events import Envelope

__all__ = ["DEFAULT_QUEUE_SIZE", "EventBus"]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.bus")

#: Сколько событий держим на подписчика, прежде чем ронять старые.
DEFAULT_QUEUE_SIZE: Final[int] = 512


class EventBus:
    """Широковещательная шина конвертов событий.

    Attributes:
        dropped: сколько событий выброшено из-за переполнения очередей.
        published: сколько событий опубликовано с момента создания.
    """

    __slots__ = ("_queue_size", "_subscribers", "dropped", "published")

    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        if queue_size <= 0:
            raise ValueError(f"queue_size должен быть > 0, получено {queue_size}")
        self._queue_size = queue_size
        self._subscribers: list[asyncio.Queue[Envelope]] = []
        self.dropped = 0
        self.published = 0

    @property
    def subscribers(self) -> int:
        """Сколько сейчас подписчиков."""
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[Envelope]:
        """Подписаться: вернуть личную очередь событий."""
        queue: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Envelope]) -> None:
        """Отписаться (повторный вызов безопасен)."""
        with contextlib.suppress(ValueError):
            self._subscribers.remove(queue)

    @contextlib.contextmanager
    def subscription(self) -> Iterator[asyncio.Queue[Envelope]]:
        """Контекст-менеджер подписки: гарантированно отписывает."""
        queue = self.subscribe()
        try:
            yield queue
        finally:
            self.unsubscribe(queue)

    async def publish(self, envelope: Envelope) -> None:
        """Разослать конверт всем подписчикам.

        Не ждёт медленных подписчиков: при переполнении очереди самое старое
        событие выбрасывается, а потеря пишется в лог уровня DEBUG.
        """
        self.published += 1
        for queue in list(self._subscribers):
            self._offer(queue, envelope)

    def _offer(self, queue: asyncio.Queue[Envelope], envelope: Envelope) -> None:
        try:
            queue.put_nowait(envelope)
            return
        except asyncio.QueueFull:
            pass
        with contextlib.suppress(asyncio.QueueEmpty):
            queue.get_nowait()
            self.dropped += 1
            logger.debug("очередь подписчика переполнена, выброшено старое событие")
        with contextlib.suppress(asyncio.QueueFull):
            queue.put_nowait(envelope)
