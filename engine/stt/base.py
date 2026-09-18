"""Интерфейсы и конфигурация модуля stt (ARCHITECTURE.md 4.2, зона агента B).

Здесь нет тяжёлых зависимостей: только ``numpy`` и стандартная библиотека.
Модели (``faster-whisper``, ``silero-vad``, ``torch``) импортируются лениво
внутри фабрик и классов-реализаций — юнит-тесты работают без них
(CLAUDE.md, технический стандарт).

Основные сущности:

* :class:`VoiceActivityDetector` — протокол VAD: поток чанков -> события
  :class:`VadEvent` (``speech_start`` / ``speech_end``);
* :class:`Transcriber` — протокол распознавателя: PCM -> :class:`TranscriptResult`;
* :class:`SttConfig` — настройки модуля, читаются из переменных окружения.

Формат аудио везде — PCM 16 kHz mono int16 (CLAUDE.md, технический стандарт).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Final, Literal, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "BYTES_PER_SAMPLE",
    "DEFAULT_MODEL_SIZE",
    "ENV_COMPUTE",
    "ENV_DEVICE",
    "ENV_KK_MODEL",
    "ENV_MODEL",
    "ENV_MODELS_DIR",
    "FORBIDDEN_MODEL_SIZES",
    "SAMPLE_RATE",
    "Device",
    "Int16Array",
    "SttConfig",
    "SttError",
    "Transcriber",
    "TranscriptResult",
    "TranscriptSegment",
    "VadEvent",
    "VadEventKind",
    "VoiceActivityDetector",
    "chunk_samples",
    "ms_to_samples",
    "samples_to_ms",
    "to_float32",
]

SAMPLE_RATE: Final[int] = 16_000
"""Частота дискретизации аудио внутри движка, Гц."""

BYTES_PER_SAMPLE: Final[int] = 2
"""Размер одного сэмпла int16 в байтах."""

DEFAULT_MODEL_SIZE: Final[str] = "small"
"""Размер модели whisper по умолчанию (ARCHITECTURE.md 4.2, CLAUDE.md закон 4)."""

FORBIDDEN_MODEL_SIZES: Final[frozenset[str]] = frozenset(
    {
        "medium",
        "medium.en",
        "large",
        "large-v1",
        "large-v2",
        "large-v3",
        "large-v3-turbo",
        "turbo",
    }
)
"""Размеры whisper, запрещённые бюджетом VRAM 6 GB (CLAUDE.md, закон 4)."""

ENV_MODEL: Final[str] = "RT_STT_MODEL"
ENV_DEVICE: Final[str] = "RT_STT_DEVICE"
ENV_COMPUTE: Final[str] = "RT_STT_COMPUTE"
ENV_KK_MODEL: Final[str] = "RT_STT_KK_MODEL"
ENV_MODELS_DIR: Final[str] = "RT_MODELS_DIR"
ENV_LANGUAGE: Final[str] = "RT_STT_LANG"
ENV_FALLBACK_LANG: Final[str] = "RT_STT_FALLBACK_LANG"
ENV_VAD_THRESHOLD: Final[str] = "RT_STT_VAD_THRESHOLD"
ENV_ENERGY_THRESHOLD: Final[str] = "RT_STT_ENERGY_THRESHOLD"
ENV_MIN_SPEECH_MS: Final[str] = "RT_STT_MIN_SPEECH_MS"
ENV_MIN_SILENCE_MS: Final[str] = "RT_STT_MIN_SILENCE_MS"
ENV_MAX_UTTERANCE_MS: Final[str] = "RT_STT_MAX_UTTERANCE_MS"
ENV_PARTIAL_INTERVAL_MS: Final[str] = "RT_STT_PARTIAL_INTERVAL_MS"
ENV_BEAM_SIZE: Final[str] = "RT_STT_BEAM_SIZE"

Int16Array = npt.NDArray[np.int16]
"""PCM 16 kHz mono int16 — рабочий формат аудио внутри движка."""

Float32Array = npt.NDArray[np.float32]
"""PCM как float32 в диапазоне [-1, 1] — формат входа whisper/silero."""

Device = Literal["cuda", "cpu", "auto"]
"""Устройство вычислений; ``auto`` — cuda при наличии, иначе cpu."""


class SttError(RuntimeError):
    """Ошибка модуля stt (нет зависимости, не грузится модель и т. п.)."""


# --- VAD -------------------------------------------------------------------


class VadEventKind(StrEnum):
    """Тип события детектора речи."""

    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass(frozen=True, slots=True)
class VadEvent:
    """Граница речи на таймлайне потока.

    Attributes:
        kind: ``speech_start`` — речь началась, ``speech_end`` — закончилась.
        ts_ms: таймкод границы от начала потока, мс (неотрицательный).
    """

    kind: VadEventKind
    ts_ms: int

    def __post_init__(self) -> None:
        if self.ts_ms < 0:
            raise ValueError(f"ts_ms должен быть >= 0, получено {self.ts_ms}")

    @classmethod
    def speech_start(cls, ts_ms: int) -> VadEvent:
        """Событие начала речи."""
        return cls(VadEventKind.SPEECH_START, ts_ms)

    @classmethod
    def speech_end(cls, ts_ms: int) -> VadEvent:
        """Событие конца речи."""
        return cls(VadEventKind.SPEECH_END, ts_ms)

    @property
    def is_start(self) -> bool:
        """``True`` для ``speech_start``."""
        return self.kind is VadEventKind.SPEECH_START

    @property
    def is_end(self) -> bool:
        """``True`` для ``speech_end``."""
        return self.kind is VadEventKind.SPEECH_END


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """Детектор речевой активности: режет непрерывный поток на фразы."""

    def process(self, chunk_int16_16k: Int16Array) -> list[VadEvent]:
        """Скормить очередной чанк PCM 16 kHz mono int16.

        Реализация сама копит остатки, если её окно анализа не совпадает с
        размером чанка.

        Args:
            chunk_int16_16k: сэмплы чанка.

        Returns:
            События, найденные в этом чанке (обычно пустой список).
        """
        ...

    def reset(self) -> None:
        """Сбросить внутреннее состояние и таймлайн (новый поток)."""
        ...


# --- распознавание ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TranscriptSegment:
    """Сегмент распознанного текста с таймкодами внутри переданного PCM.

    Attributes:
        text: текст сегмента.
        t_start_ms: начало сегмента от начала переданного буфера, мс.
        t_end_ms: конец сегмента от начала переданного буфера, мс.
    """

    text: str
    t_start_ms: int
    t_end_ms: int


@dataclass(frozen=True, slots=True)
class TranscriptResult:
    """Результат распознавания одного буфера.

    Attributes:
        text: склеенный текст без ведущих/хвостовых пробелов.
        lang: язык (заданный или определённый моделью), код ISO 639-1.
        segments: сегменты с таймкодами относительно начала буфера.
        duration_ms: длительность распознанного аудио, мс.
    """

    text: str
    lang: str
    segments: tuple[TranscriptSegment, ...] = ()
    duration_ms: int = 0


@runtime_checkable
class Transcriber(Protocol):
    """Распознаватель речи: буфер PCM -> текст.

    Вызывается из отдельного потока (``asyncio.to_thread``), поэтому метод
    синхронный и обязан быть потокобезопасным относительно самого себя
    (движок гарантирует не более одного одновременного вызова).
    """

    def transcribe(self, pcm_int16_16k: Int16Array, lang: str | None) -> TranscriptResult:
        """Распознать буфер PCM 16 kHz mono int16.

        Args:
            pcm_int16_16k: аудио целиком (одна фраза или её начало).
            lang: код языка (``ru`` / ``en`` / ``kk``) или ``None`` —
                определить автоматически.

        Returns:
            Результат распознавания.
        """
        ...


# --- конфигурация ----------------------------------------------------------


def _env_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: ожидалось целое число, получено {raw!r}") from exc


def _env_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}: ожидалось число, получено {raw!r}") from exc


def _env_str(env: Mapping[str, str], name: str, default: str) -> str:
    raw = env.get(name)
    return raw.strip() if raw and raw.strip() else default


@dataclass(frozen=True, slots=True)
class SttConfig:
    """Настройки модуля stt (ARCHITECTURE.md 4.2).

    Attributes:
        model_size: размер или путь модели whisper. По умолчанию ``small``
            (CLAUDE.md, закон 4 — крупнее брать нельзя).
        compute_type: тип вычислений CTranslate2 (``int8_float16`` на GPU,
            ``int8`` на CPU).
        device: ``cuda`` / ``cpu`` / ``auto``.
        language: фиксированный язык распознавания или ``None`` —
            автоопределение моделью.
        fallback_lang: чем заменить язык, если модель определила язык вне
            ``ru/en/kk`` (контракты допускают только эти три).
        vad_threshold: порог вероятности речи для Silero VAD (0..1).
        energy_threshold: порог RMS (доля полной шкалы) для ``EnergyVad``.
        min_speech_ms: короче этого фрагмент речью не считается.
        min_silence_ms: столько тишины закрывает фразу.
        max_utterance_ms: предельная длина фразы, после неё принудительный
            ``stt.final``.
        partial_interval_ms: как часто во время речи считать ``stt.partial``.
        preroll_ms: сколько аудио до момента ``speech_start`` добавить в буфер
            фразы (страховка от срезанного первого слога).
        beam_size: beam search whisper; 1 — самый быстрый режим.
        models_dir: каталог кэша моделей (``RT_MODELS_DIR``); ``None`` —
            кэш huggingface по умолчанию.
        model_overrides: подмена модели для конкретных языков, например
            ``{"kk": "models/issai-whisper-kk"}``.
    """

    model_size: str = DEFAULT_MODEL_SIZE
    compute_type: str = "int8_float16"
    device: Device = "auto"
    language: str | None = None
    fallback_lang: str = "ru"
    vad_threshold: float = 0.5
    energy_threshold: float = 0.02
    min_speech_ms: int = 250
    min_silence_ms: int = 500
    max_utterance_ms: int = 15_000
    partial_interval_ms: int = 1_000
    preroll_ms: int = 300
    beam_size: int = 1
    models_dir: Path | None = None
    model_overrides: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.device not in ("cuda", "cpu", "auto"):
            raise ValueError(f"device: ожидалось cuda/cpu/auto, получено {self.device!r}")
        if self.model_size.lower() in FORBIDDEN_MODEL_SIZES:
            raise ValueError(
                f"модель whisper {self.model_size!r} не влезает в бюджет VRAM 6 GB "
                f"(ARCHITECTURE.md раздел 6, CLAUDE.md закон 4); разрешён {DEFAULT_MODEL_SIZE!r} "
                f"или путь к файнтюну сопоставимого размера"
            )
        for name, value in (
            ("min_speech_ms", self.min_speech_ms),
            ("min_silence_ms", self.min_silence_ms),
            ("max_utterance_ms", self.max_utterance_ms),
            ("partial_interval_ms", self.partial_interval_ms),
            ("preroll_ms", self.preroll_ms),
        ):
            if value < 0:
                raise ValueError(f"{name} должен быть >= 0, получено {value}")
        if self.max_utterance_ms and self.max_utterance_ms < self.min_speech_ms:
            raise ValueError("max_utterance_ms не может быть меньше min_speech_ms")
        if self.beam_size < 1:
            raise ValueError(f"beam_size должен быть >= 1, получено {self.beam_size}")
        # frozen + slots: нормализуем поле через object.__setattr__
        if isinstance(self.models_dir, str):
            object.__setattr__(self, "models_dir", Path(self.models_dir))
        object.__setattr__(self, "model_overrides", dict(self.model_overrides))

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None, /, **overrides: Any) -> SttConfig:
        """Собрать конфиг из переменных окружения.

        Читаются: ``RT_STT_MODEL``, ``RT_STT_DEVICE``, ``RT_STT_COMPUTE``,
        ``RT_STT_KK_MODEL``, ``RT_MODELS_DIR``, ``RT_STT_LANG``,
        ``RT_STT_FALLBACK_LANG``, ``RT_STT_VAD_THRESHOLD``,
        ``RT_STT_ENERGY_THRESHOLD``, ``RT_STT_MIN_SPEECH_MS``,
        ``RT_STT_MIN_SILENCE_MS``, ``RT_STT_MAX_UTTERANCE_MS``,
        ``RT_STT_PARTIAL_INTERVAL_MS``, ``RT_STT_BEAM_SIZE``.

        Args:
            env: словарь переменных; по умолчанию ``os.environ``.
            **overrides: значения полей, перебивающие окружение.

        Returns:
            Готовый конфиг.

        Raises:
            ValueError: переменная окружения не разбирается или нарушает
                ограничения (например запрещённый размер модели).
        """
        src: Mapping[str, str] = os.environ if env is None else env

        defaults = cls()
        model_overrides: dict[str, str] = {}
        kk_model = src.get(ENV_KK_MODEL, "").strip()
        if kk_model:
            model_overrides["kk"] = kk_model

        models_dir_raw = src.get(ENV_MODELS_DIR, "").strip()
        language = src.get(ENV_LANGUAGE, "").strip().lower()

        device_raw = _env_str(src, ENV_DEVICE, defaults.device).lower()
        if device_raw not in ("cuda", "cpu", "auto"):
            raise ValueError(f"{ENV_DEVICE}: ожидалось cuda/cpu/auto, получено {device_raw!r}")
        device: Device = device_raw  # type: ignore[assignment]

        config = cls(
            model_size=_env_str(src, ENV_MODEL, defaults.model_size),
            compute_type=_env_str(src, ENV_COMPUTE, defaults.compute_type),
            device=device,
            language=language or None,
            fallback_lang=_env_str(src, ENV_FALLBACK_LANG, defaults.fallback_lang).lower(),
            vad_threshold=_env_float(src, ENV_VAD_THRESHOLD, defaults.vad_threshold),
            energy_threshold=_env_float(src, ENV_ENERGY_THRESHOLD, defaults.energy_threshold),
            min_speech_ms=_env_int(src, ENV_MIN_SPEECH_MS, defaults.min_speech_ms),
            min_silence_ms=_env_int(src, ENV_MIN_SILENCE_MS, defaults.min_silence_ms),
            max_utterance_ms=_env_int(src, ENV_MAX_UTTERANCE_MS, defaults.max_utterance_ms),
            partial_interval_ms=_env_int(
                src, ENV_PARTIAL_INTERVAL_MS, defaults.partial_interval_ms
            ),
            beam_size=_env_int(src, ENV_BEAM_SIZE, defaults.beam_size),
            models_dir=Path(models_dir_raw) if models_dir_raw else None,
            model_overrides=model_overrides,
        )
        return replace(config, **overrides) if overrides else config

    def model_for(self, lang: str | None) -> str:
        """Какую модель брать для языка.

        Args:
            lang: код языка или ``None`` (автоопределение).

        Returns:
            ``model_overrides[lang]``, если задан, иначе :attr:`model_size`.
        """
        if lang:
            override = self.model_overrides.get(lang.lower())
            if override:
                return override
        return self.model_size


# --- утилиты аудио ---------------------------------------------------------


def ms_to_samples(ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    """Перевести миллисекунды в число сэмплов (округление вниз)."""
    return int(ms * sample_rate // 1000)


def samples_to_ms(n_samples: int, sample_rate: int = SAMPLE_RATE) -> int:
    """Перевести число сэмплов в миллисекунды (округление вниз)."""
    return int(n_samples * 1000 // sample_rate)


def to_float32(pcm_int16: Int16Array) -> Float32Array:
    """PCM int16 -> float32 в диапазоне [-1, 1] (вход whisper и silero)."""
    return np.asarray(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0


def chunk_samples(chunk: Any) -> Int16Array:
    """Достать int16-сэмплы из объекта-чанка по утиной типизации.

    Поддерживаются и чанк с полем ``pcm`` как ``numpy``-массив (описание в
    ARCHITECTURE.md 4.2), и чанк ``engine.audio_io`` с ``pcm`` как ``bytes``
    и свойством ``samples``. Прямого импорта чужого модуля нет
    (CLAUDE.md, технический стандарт).

    Args:
        chunk: объект с полем ``pcm`` (``bytes`` или int16-массив) либо со
            свойством ``samples``.

    Returns:
        Сэмплы как int16-массив.

    Raises:
        TypeError: у объекта нет ни ``samples``, ни ``pcm`` понятного типа.
    """
    samples = getattr(chunk, "samples", None)
    if isinstance(samples, np.ndarray):
        return np.asarray(samples, dtype=np.int16)

    pcm = getattr(chunk, "pcm", None)
    if isinstance(pcm, np.ndarray):
        return np.asarray(pcm, dtype=np.int16)
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        return np.frombuffer(bytes(pcm), dtype="<i2").astype(np.int16)

    raise TypeError(
        f"чанк {type(chunk).__name__} не похож на аудиочанк: "
        f"нужен pcm (bytes или numpy int16) либо свойство samples"
    )
