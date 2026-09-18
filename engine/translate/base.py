"""Интерфейс модуля перевода (ARCHITECTURE.md 4.3).

Здесь только абстракции, никаких тяжёлых импортов: `torch`/`transformers`
подтягиваются лениво внутри конкретного провайдера (CLAUDE.md, техстандарт),
поэтому этот модуль импортируется в юнит-тестах без установленных моделей.

Пример::

    from engine.contracts.events import Lang
    from engine.translate import TranslateConfig, create_provider

    provider = create_provider(TranslateConfig.from_env())
    result = await provider.translate("Привет, как дела?", Lang.RU, Lang.EN)
    print(result.text, result.latency_ms)
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Protocol, runtime_checkable

from engine.contracts.events import Lang

__all__ = [
    "DEFAULT_CACHE_SIZE",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MODEL_ID",
    "DEFAULT_NUM_BEAMS",
    "Backend",
    "TranslateConfig",
    "TranslationError",
    "TranslationProvider",
    "TranslationResult",
    "default_num_threads",
]

#: Единственная разрешённая MT-модель проекта (ARCHITECTURE.md 4.3, раздел 6).
DEFAULT_MODEL_ID: Final[str] = "facebook/nllb-200-distilled-600M"
#: Потолок длины ответа: фраза разговора, а не документ.
DEFAULT_MAX_NEW_TOKENS: Final[int] = 128
#: 1 = greedy. Бюджет задержки — 2.5 с на всю цепочку (ARCHITECTURE.md 2),
#: поэтому по умолчанию без beam search. `num_beams=2` даёт заметно лучшее
#: качество ценой ~1.6-2x времени генерации — включайте через RT_MT_BEAMS=2,
#: если укладываетесь в бюджет (проверяйте `scripts/translate_smoke.py --bench`).
DEFAULT_NUM_BEAMS: Final[int] = 1
#: Размер LRU-кэша переводов (последние N уникальных фраз).
DEFAULT_CACHE_SIZE: Final[int] = 256
#: Верхняя граница потоков по умолчанию: 4 ядра Ryzen 5 5600H под MT,
#: остальные оставляем audio_io/STT/TTS.
DEFAULT_MAX_THREADS: Final[int] = 4

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR: Final[Path] = REPO_ROOT / "models"

#: Доступные бэкенды: реальная NLLB и фейк для тестов/смоука.
Backend = Literal["nllb", "fake"]

ENV_MODEL: Final[str] = "RT_MT_MODEL"
ENV_DEVICE: Final[str] = "RT_MT_DEVICE"
ENV_THREADS: Final[str] = "RT_MT_THREADS"
ENV_INT8: Final[str] = "RT_MT_INT8"
ENV_BACKEND: Final[str] = "RT_MT_BACKEND"
ENV_MAX_NEW_TOKENS: Final[str] = "RT_MT_MAX_NEW_TOKENS"
ENV_BEAMS: Final[str] = "RT_MT_BEAMS"
ENV_CACHE: Final[str] = "RT_MT_CACHE"
ENV_MODELS_DIR: Final[str] = "RT_MODELS_DIR"

_TRUE_VALUES: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


class TranslationError(RuntimeError):
    """Ошибка перевода: неизвестный язык, недоступная модель, битая конфигурация."""


@dataclass(frozen=True, slots=True)
class TranslationResult:
    """Результат перевода одной фразы."""

    text: str
    src: Lang
    dst: Lang
    #: Время перевода в миллисекундах (0 — без вызова модели: пусто или src == dst).
    latency_ms: int


@runtime_checkable
class TranslationProvider(Protocol):
    """Провайдер перевода. Реализации: NLLB на CPU, фейк, позже — LLM API.

    Контракт:

    * `translate` не блокирует event loop (тяжёлый инференс — в `asyncio.to_thread`);
    * пустой/пробельный текст возвращается как пустой без вызова модели;
    * `src == dst` — короткое замыкание, `latency_ms == 0`;
    * `context` — последние фразы диалога, провайдер вправе их игнорировать
      (NLLB игнорирует, LLM-провайдер сможет использовать).
    """

    async def translate(
        self,
        text: str,
        src: Lang,
        dst: Lang,
        context: Sequence[str] = (),
    ) -> TranslationResult:
        """Перевести фразу с `src` на `dst`."""
        ...

    def warmup(self) -> None:
        """Синхронно прогреть провайдера (загрузка весов, первый прогон)."""
        ...


def default_num_threads() -> int:
    """Число потоков инференса по умолчанию: `min(4, cpu_count)`."""
    return max(1, min(DEFAULT_MAX_THREADS, os.cpu_count() or 1))


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    return raw.strip() if raw and raw.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise TranslationError(f"{name}: ожидалось целое число, получено {raw!r}") from exc


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    value = raw.strip().lower()
    if value in _TRUE_VALUES:
        return True
    if value in _FALSE_VALUES:
        return False
    raise TranslationError(f"{name}: ожидалось булево значение, получено {raw!r}")


def _env_backend(name: str, default: Backend) -> Backend:
    value = _env_str(name, default).lower()
    if value not in ("nllb", "fake"):
        raise TranslationError(f"{name}: неизвестный бэкенд {value!r}, ожидалось nllb|fake")
    return "nllb" if value == "nllb" else "fake"


def _env_models_dir() -> Path:
    raw = os.environ.get(ENV_MODELS_DIR)
    if raw and raw.strip():
        return Path(raw.strip()).expanduser()
    return DEFAULT_MODELS_DIR


@dataclass(frozen=True, slots=True)
class TranslateConfig:
    """Конфигурация перевода.

    Перевод живёт на CPU: GPU (6 GB) занята STT и TTS (ARCHITECTURE.md раздел 6).
    """

    model_id: str = DEFAULT_MODEL_ID
    device: str = "cpu"
    num_threads: int | None = None
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    num_beams: int = DEFAULT_NUM_BEAMS
    models_dir: Path = DEFAULT_MODELS_DIR
    #: Динамическая квантизация Linear-слоёв в int8 (`torch.quantization.quantize_dynamic`):
    #: заметно ускоряет NLLB на CPU ценой небольшой потери качества.
    quantize_int8: bool = True
    backend: Backend = "nllb"
    cache_size: int = DEFAULT_CACHE_SIZE

    @property
    def threads(self) -> int:
        """Фактическое число потоков (`num_threads` или `min(4, cpu_count)`)."""
        return self.num_threads if self.num_threads else default_num_threads()

    @property
    def local_model_dir(self) -> Path:
        """Каталог локально скачанной модели (`scripts/download_models.py --only nllb`)."""
        return self.models_dir / "nllb"

    def model_source(self) -> str:
        """Откуда грузить веса: локальный каталог, если он есть, иначе id на Hugging Face."""
        local = self.local_model_dir
        if (local / "config.json").is_file():
            return str(local)
        return self.model_id

    @classmethod
    def from_env(cls) -> TranslateConfig:
        """Собрать конфиг из переменных окружения.

        `RT_MT_MODEL`, `RT_MT_DEVICE`, `RT_MT_THREADS`, `RT_MT_INT8`, `RT_MT_BACKEND`,
        `RT_MT_MAX_NEW_TOKENS`, `RT_MT_BEAMS`, `RT_MT_CACHE`, `RT_MODELS_DIR`.

        Raises:
            TranslationError: если значение переменной нельзя разобрать.
        """
        threads = _env_int(ENV_THREADS, 0)
        return cls(
            model_id=_env_str(ENV_MODEL, DEFAULT_MODEL_ID),
            device=_env_str(ENV_DEVICE, "cpu"),
            num_threads=threads if threads > 0 else None,
            max_new_tokens=_env_int(ENV_MAX_NEW_TOKENS, DEFAULT_MAX_NEW_TOKENS),
            num_beams=_env_int(ENV_BEAMS, DEFAULT_NUM_BEAMS),
            models_dir=_env_models_dir(),
            quantize_int8=_env_bool(ENV_INT8, True),
            backend=_env_backend(ENV_BACKEND, "nllb"),
            cache_size=_env_int(ENV_CACHE, DEFAULT_CACHE_SIZE),
        )
