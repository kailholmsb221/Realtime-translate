"""Общие фикстуры тестов движка.

Даёт всем агентам:

* ``fixtures_dir`` — каталог с WAV-фикстурами (генерируются автоматически);
* ``tone_wav`` / ``silence_wav`` — пути к конкретным сэмплам;
* ``read_wav`` — чтение WAV в ``numpy`` int16;
* авто-скип тестов с маркером ``real_models`` без ``RT_REAL_MODELS=1``.

Аудио фикстур — PCM 16 kHz mono int16, формат аудио внутри движка
(CLAUDE.md, технический стандарт).
"""

from __future__ import annotations

import os
import sys
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # запуск pytest без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

sys.path.insert(0, str(REPO_ROOT / "scripts"))

from make_fixtures import (  # noqa: E402  — после правки sys.path
    SILENCE_FILENAME,
    TONE_FILENAME,
    make_fixtures,
)

REAL_MODELS_ENV = "RT_REAL_MODELS"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Пропускать тесты с реальными моделями, если не задан RT_REAL_MODELS=1."""
    if os.environ.get(REAL_MODELS_ENV) == "1":
        return
    skip = pytest.mark.skip(reason=f"нужны реальные модели: задайте {REAL_MODELS_ENV}=1")
    for item in items:
        if "real_models" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    """Каталог WAV-фикстур; недостающие файлы генерируются на лету."""
    path = Path(__file__).resolve().parent / "fixtures"
    make_fixtures(path)
    return path


@pytest.fixture(scope="session")
def tone_wav(fixtures_dir: Path) -> Path:
    """WAV: синус 440 Гц, 1 с, 16 kHz mono int16."""
    return fixtures_dir / TONE_FILENAME


@pytest.fixture(scope="session")
def silence_wav(fixtures_dir: Path) -> Path:
    """WAV: тишина, 1 с, 16 kHz mono int16."""
    return fixtures_dir / SILENCE_FILENAME


@pytest.fixture(scope="session")
def read_wav() -> Callable[[Path], tuple[np.ndarray, int]]:
    """Функция чтения WAV: путь -> (int16-массив сэмплов, частота дискретизации)."""

    def _read(path: Path) -> tuple[np.ndarray, int]:
        with wave.open(str(path), "rb") as fh:
            sample_rate = fh.getframerate()
            frames = fh.readframes(fh.getnframes())
        return np.frombuffer(frames, dtype="<i2"), sample_rate

    return _read
