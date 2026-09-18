"""Провайдер перевода на NLLB-200-distilled-600M (CPU).

ARCHITECTURE.md 4.3: направления ru↔en, ru↔kk, en↔kk; GPU оставлена STT и TTS,
перевод считается на CPU (раздел 6 — 0 GB VRAM).

`torch`/`transformers` импортируются лениво, внутри `_load()`, поэтому модуль
безопасно импортировать в юнит-тестах без установленных весов и без torch
(CLAUDE.md, техстандарт).

TODO: опциональный бэкенд ctranslate2 (конвертация NLLB в int8 даёт ещё ~2x
на CPU) — отдельная задача, интерфейс `TranslationProvider` менять не нужно.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Final

from engine.contracts.events import Lang
from engine.translate.base import TranslateConfig, TranslationError, TranslationResult
from engine.translate.cache import TranslationCache

if TYPE_CHECKING:  # pragma: no cover — только для аннотаций, не грузит torch
    from pathlib import Path

__all__ = ["NLLB_LANG_CODES", "NllbProvider", "nllb_code"]

logger = logging.getLogger(__name__)

#: Коды языков NLLB-200 (FLORES-200) для языков проекта.
NLLB_LANG_CODES: Final[dict[Lang, str]] = {
    Lang.RU: "rus_Cyrl",
    Lang.EN: "eng_Latn",
    Lang.KK: "kaz_Cyrl",
}

#: Потолок длины входа: реплика разговора, не документ.
MAX_INPUT_TOKENS: Final[int] = 512
#: Фраза для прогрева (`warmup`) — короткая, чтобы не тратить время на старте.
WARMUP_TEXT: Final[str] = "Привет."


def nllb_code(lang: Lang) -> str:
    """Код языка NLLB (`ru` -> `rus_Cyrl`).

    Raises:
        TranslationError: если язык не поддержан проектом.
    """
    try:
        return NLLB_LANG_CODES[lang]
    except KeyError as exc:
        raise TranslationError(f"Нет кода NLLB для языка {lang!r}") from exc


class NllbProvider:
    """Перевод через `transformers` + NLLB-200-distilled-600M.

    Модель грузится лениво — при первом `translate()` или явном `warmup()`.
    Инференс уходит в `asyncio.to_thread`, чтобы не блокировать event loop;
    одновременный доступ к токенизатору сериализуется мьютексом (у NLLB
    `tokenizer.src_lang` — изменяемое состояние).
    """

    __slots__ = ("_cache", "_config", "_lock", "_model", "_tokenizer")

    def __init__(self, config: TranslateConfig | None = None) -> None:
        self._config = config or TranslateConfig()
        self._cache = TranslationCache(self._config.cache_size)
        # Реентерабельный: `_generate` держит мьютекс и внутри вызывает `_load`.
        self._lock = threading.RLock()
        self._tokenizer: Any | None = None
        self._model: Any | None = None

    @property
    def config(self) -> TranslateConfig:
        """Конфигурация провайдера."""
        return self._config

    @property
    def cache(self) -> TranslationCache:
        """LRU-кэш переводов (для метрик и тестов)."""
        return self._cache

    @property
    def loaded(self) -> bool:
        """Загружены ли веса."""
        return self._model is not None

    # --- публичный интерфейс -------------------------------------------------

    def warmup(self) -> None:
        """Загрузить веса и прогнать одну короткую фразу (синхронно, на старте движка)."""
        self._load()
        try:
            self._generate(WARMUP_TEXT, nllb_code(Lang.RU), nllb_code(Lang.EN))
        except Exception:  # pragma: no cover — прогрев не должен ронять старт
            logger.warning("NLLB: прогрев не удался, модель загружена", exc_info=True)

    async def translate(
        self,
        text: str,
        src: Lang,
        dst: Lang,
        context: Sequence[str] = (),
    ) -> TranslationResult:
        """Перевести фразу. `context` NLLB не использует (нужен LLM-провайдеру).

        Пустая строка и `src == dst` возвращаются без обращения к модели.
        """
        del context  # NLLB переводит пофразно, контекст диалога не принимает
        stripped = text.strip()
        if not stripped:
            return TranslationResult(text="", src=src, dst=dst, latency_ms=0)
        if src == dst:
            return TranslationResult(text=text, src=src, dst=dst, latency_ms=0)

        src_code, dst_code = nllb_code(src), nllb_code(dst)
        started = time.perf_counter()

        cached = self._cache.get(src, dst, stripped)
        if cached is not None:
            return TranslationResult(text=cached, src=src, dst=dst, latency_ms=_elapsed_ms(started))

        translated = await asyncio.to_thread(self._generate, stripped, src_code, dst_code)
        self._cache.put(src, dst, stripped, translated)
        return TranslationResult(text=translated, src=src, dst=dst, latency_ms=_elapsed_ms(started))

    # --- внутреннее ----------------------------------------------------------

    def _load(self) -> None:
        """Лениво загрузить токенизатор и модель (идемпотентно)."""
        with self._lock:
            if self._model is not None:
                return
            cfg = self._config
            source = cfg.model_source()
            logger.info(
                "NLLB: загрузка %s (device=%s, threads=%d, int8=%s)",
                source,
                cfg.device,
                cfg.threads,
                cfg.quantize_int8,
            )
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover — зависит от окружения
                raise TranslationError(
                    "Нет transformers/torch: pip install -r engine/translate/"
                    "requirements-translate.txt"
                ) from exc

            torch.set_num_threads(cfg.threads)
            try:
                tokenizer = AutoTokenizer.from_pretrained(source)
                model = AutoModelForSeq2SeqLM.from_pretrained(source)
            except Exception as exc:  # pragma: no cover — сеть/диск
                raise TranslationError(f"Не удалось загрузить NLLB из {source!r}: {exc}") from exc

            model.eval()
            model.to(cfg.device)
            if cfg.quantize_int8 and cfg.device == "cpu":
                # Динамическая квантизация Linear-слоёв: ~1.5-2x к скорости на CPU.
                model = torch.quantization.quantize_dynamic(
                    model, {torch.nn.Linear}, dtype=torch.qint8
                )
            self._tokenizer = tokenizer
            self._model = model
            logger.info("NLLB: модель готова")

    def _generate(self, text: str, src_code: str, dst_code: str) -> str:
        """Синхронный инференс одной фразы. Вызывается из рабочего потока."""
        self._load()
        import torch

        with self._lock:
            tokenizer, model = self._tokenizer, self._model
            if tokenizer is None or model is None:  # pragma: no cover — защита от гонки
                raise TranslationError("NLLB: модель не загружена")
            tokenizer.src_lang = src_code
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            ).to(self._config.device)
            forced_bos_token_id = tokenizer.convert_tokens_to_ids(dst_code)
            with torch.inference_mode():
                tokens = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_new_tokens=self._config.max_new_tokens,
                    num_beams=self._config.num_beams,
                )
            decoded: str = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
        return decoded.strip()

    def local_model_dir(self) -> Path:
        """Каталог локальных весов (`scripts/download_models.py --only nllb`)."""
        return self._config.local_model_dir


def _elapsed_ms(started: float) -> int:
    """Прошедшее время в миллисекундах."""
    return int((time.perf_counter() - started) * 1000)
