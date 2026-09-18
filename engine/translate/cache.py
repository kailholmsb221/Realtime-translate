"""LRU-кэш переводов.

В разговоре одни и те же короткие фразы («да», «ага», «ok, thanks») повторяются
постоянно, а NLLB на CPU стоит сотни миллисекунд — кэш убирает этот хвост
из бюджета задержки (ARCHITECTURE.md раздел 2).
"""

from __future__ import annotations

from collections import OrderedDict

from engine.contracts.events import Lang

__all__ = ["CacheKey", "TranslationCache"]

#: Ключ кэша: направление перевода + исходный текст.
CacheKey = tuple[Lang, Lang, str]


class TranslationCache:
    """Потокобезопасности не требует: значения кладутся из одного event loop.

    Вытесняется наименее недавно использованный элемент. `maxsize <= 0`
    выключает кэш полностью.
    """

    __slots__ = ("_entries", "_hits", "_maxsize", "_misses")

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._entries: OrderedDict[CacheKey, str] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def maxsize(self) -> int:
        """Максимальное число хранимых переводов."""
        return self._maxsize

    @property
    def hits(self) -> int:
        """Сколько раз перевод нашёлся в кэше."""
        return self._hits

    @property
    def misses(self) -> int:
        """Сколько раз пришлось идти в модель."""
        return self._misses

    def get(self, src: Lang, dst: Lang, text: str) -> str | None:
        """Вернуть перевод из кэша или `None`."""
        if self._maxsize <= 0:
            return None
        key: CacheKey = (src, dst, text)
        value = self._entries.get(key)
        if value is None:
            self._misses += 1
            return None
        self._entries.move_to_end(key)
        self._hits += 1
        return value

    def put(self, src: Lang, dst: Lang, text: str, translation: str) -> None:
        """Положить перевод в кэш, вытеснив самый старый при переполнении."""
        if self._maxsize <= 0:
            return
        key: CacheKey = (src, dst, text)
        self._entries[key] = translation
        self._entries.move_to_end(key)
        while len(self._entries) > self._maxsize:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        """Очистить кэш и счётчики."""
        self._entries.clear()
        self._hits = 0
        self._misses = 0
