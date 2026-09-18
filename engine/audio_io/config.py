"""Конфигурация audio_io: имена устройств и env-переменные.

Устройства выбираются по подстроке имени (регистр не важен), потому что
индексы устройств в Windows меняются при переподключении наушников.

Env-переменные:

===========================  =======================================
``RT_AUDIO_LOOPBACK``        подстрока имени WASAPI loopback-устройства
                             (обычно = имя наушников/колонок, куда играет Zoom)
``RT_AUDIO_MIC``             подстрока имени физического микрофона
``RT_AUDIO_HEADPHONES``      подстрока имени наушников (куда играет перевод
                             речи собеседника)
``RT_AUDIO_CABLE``           подстрока имени входа виртуального кабеля
                             (по умолчанию ``CABLE Input``)
``RT_AUDIO_CHUNK_MS``        длительность чанка захвата, мс (по умолчанию 30)
``RT_AUDIO_SAMPLE_RATE``     частота движка, Гц (по умолчанию 16000)
``RT_AUDIO_DEVICE_TIMEOUT``  таймаут ожидания данных с устройства, с
===========================  =======================================

Пример::

    from engine.audio_io.config import AudioConfig

    cfg = AudioConfig.from_env()        # os.environ
    cfg = AudioConfig.from_env({"RT_AUDIO_MIC": "Microphone (Realtek"})
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from engine.audio_io.base import CHUNK_MS, SAMPLE_RATE_ENGINE, AudioFormat

__all__ = [
    "DEFAULT_CABLE_NAME",
    "ENV_CABLE",
    "ENV_CHUNK_MS",
    "ENV_DEVICE_TIMEOUT",
    "ENV_HEADPHONES",
    "ENV_LOOPBACK",
    "ENV_MIC",
    "ENV_SAMPLE_RATE",
    "AudioConfig",
]

ENV_LOOPBACK: Final[str] = "RT_AUDIO_LOOPBACK"
ENV_MIC: Final[str] = "RT_AUDIO_MIC"
ENV_HEADPHONES: Final[str] = "RT_AUDIO_HEADPHONES"
ENV_CABLE: Final[str] = "RT_AUDIO_CABLE"
ENV_CHUNK_MS: Final[str] = "RT_AUDIO_CHUNK_MS"
ENV_SAMPLE_RATE: Final[str] = "RT_AUDIO_SAMPLE_RATE"
ENV_DEVICE_TIMEOUT: Final[str] = "RT_AUDIO_DEVICE_TIMEOUT"

#: Имя устройства-входа VB-Audio Virtual Cable (то, куда пишем перевод).
DEFAULT_CABLE_NAME: Final[str] = "CABLE Input"


def _clean(value: str | None) -> str | None:
    """Пустая строка/пробелы -> ``None``."""
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _env_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = _clean(env.get(key))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{key}={raw!r}: ожидалось целое число") from exc


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = _clean(env.get(key))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key}={raw!r}: ожидалось число") from exc


@dataclass(frozen=True, slots=True)
class AudioConfig:
    """Настройки маршрутизации звука.

    Attributes:
        loopback_name: подстрока имени устройства вывода, с которого снимаем
            loopback (речь собеседника из Zoom). ``None`` — устройство вывода
            по умолчанию.
        mic_name: подстрока имени микрофона. ``None`` — микрофон по умолчанию.
        headphones_name: подстрока имени наушников. ``None`` — вывод по умолчанию.
        cable_name: подстрока имени входа виртуального кабеля.
        sample_rate: частота движка (16 kHz, менять не нужно).
        chunk_ms: длительность чанка захвата, мс.
        device_timeout_s: сколько ждать данных с устройства, прежде чем
            считать поток оборванным.
    """

    loopback_name: str | None = None
    mic_name: str | None = None
    headphones_name: str | None = None
    cable_name: str = DEFAULT_CABLE_NAME
    sample_rate: int = SAMPLE_RATE_ENGINE
    chunk_ms: int = CHUNK_MS
    device_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate должен быть > 0, получено {self.sample_rate}")
        if self.chunk_ms <= 0:
            raise ValueError(f"chunk_ms должен быть > 0, получено {self.chunk_ms}")
        if self.device_timeout_s <= 0:
            raise ValueError(f"device_timeout_s должен быть > 0, получено {self.device_timeout_s}")

    @property
    def format(self) -> AudioFormat:
        """Формат чанков движка по этой конфигурации."""
        return AudioFormat(sample_rate=self.sample_rate, channels=1)

    @property
    def chunk_frames(self) -> int:
        """Число фреймов в чанке при текущей частоте."""
        return self.format.frames_for_ms(self.chunk_ms)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AudioConfig:
        """Собрать конфиг из переменных окружения (по умолчанию ``os.environ``).

        Raises:
            ValueError: числовая переменная задана нечисловым значением.
        """
        source: Mapping[str, str] = os.environ if env is None else env
        return cls(
            loopback_name=_clean(source.get(ENV_LOOPBACK)),
            mic_name=_clean(source.get(ENV_MIC)),
            headphones_name=_clean(source.get(ENV_HEADPHONES)),
            cable_name=_clean(source.get(ENV_CABLE)) or DEFAULT_CABLE_NAME,
            sample_rate=_env_int(source, ENV_SAMPLE_RATE, SAMPLE_RATE_ENGINE),
            chunk_ms=_env_int(source, ENV_CHUNK_MS, CHUNK_MS),
            device_timeout_s=_env_float(source, ENV_DEVICE_TIMEOUT, 5.0),
        )

    def describe(self) -> str:
        """Человекочитаемая сводка (для логов и smoke-скрипта)."""
        return (
            f"loopback={self.loopback_name or '<default output>'}, "
            f"mic={self.mic_name or '<default input>'}, "
            f"headphones={self.headphones_name or '<default output>'}, "
            f"cable={self.cable_name}, "
            f"{self.sample_rate} Hz / {self.chunk_ms} ms"
        )
