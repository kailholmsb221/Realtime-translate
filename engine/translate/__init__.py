"""Пакет translate: перевод (NLLB-200-distilled-600M на CPU, абстракция Provider).

Зона агента C. См. ARCHITECTURE.md раздел 4.3 и `engine/translate/README.md`.

Быстрый старт::

    from engine.contracts.events import Lang
    from engine.translate import TranslateConfig, TranslationService, create_provider

    provider = create_provider(TranslateConfig.from_env())
    service = TranslationService(provider)
    ready = await service.handle(stt_final_envelope, dst=Lang.RU, ref_utterance_id=None)

`transformers`/`torch` подтягиваются только в фабрике NLLB, поэтому импорт пакета
дёшев и не требует установленных весов (CLAUDE.md, техстандарт).
"""

from __future__ import annotations

from engine.translate.base import (
    DEFAULT_CACHE_SIZE,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_NUM_BEAMS,
    Backend,
    TranslateConfig,
    TranslationError,
    TranslationProvider,
    TranslationResult,
    default_num_threads,
)
from engine.translate.cache import TranslationCache
from engine.translate.fake import FakeProvider
from engine.translate.service import DEFAULT_CONTEXT_SIZE, TranslationService

__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_CONTEXT_SIZE",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NUM_BEAMS",
    "Backend",
    "FakeProvider",
    "TranslateConfig",
    "TranslationCache",
    "TranslationError",
    "TranslationProvider",
    "TranslationResult",
    "TranslationService",
    "create_provider",
    "default_num_threads",
]


def create_provider(config: TranslateConfig | None = None) -> TranslationProvider:
    """Создать провайдера перевода по конфигу (фабрика тяжёлых моделей).

    `backend="fake"` — `FakeProvider` без весов (тесты, смоук);
    `backend="nllb"` — `NllbProvider`, `transformers` импортируется здесь,
    веса грузятся лениво при первом переводе или `warmup()`.

    Raises:
        TranslationError: если бэкенд неизвестен.
    """
    cfg = config or TranslateConfig.from_env()
    if cfg.backend == "fake":
        return FakeProvider()
    if cfg.backend == "nllb":
        from engine.translate.nllb import NllbProvider

        return NllbProvider(cfg)
    raise TranslationError(f"Неизвестный бэкенд перевода: {cfg.backend!r}")
