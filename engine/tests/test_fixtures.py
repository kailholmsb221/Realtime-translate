"""Проверка фундамента тестов: WAV-фикстуры и маркер real_models."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from engine.tests.conftest import REAL_MODELS_ENV

WavReader = Callable[[Path], tuple[np.ndarray, int]]


def test_fixtures_dir_exists(fixtures_dir: Path) -> None:
    """Каталог фикстур создаётся и содержит оба сэмпла."""
    assert fixtures_dir.is_dir()
    assert {p.name for p in fixtures_dir.glob("*.wav")} >= {
        "tone_440hz_1s.wav",
        "silence_1s.wav",
    }


def test_tone_fixture_format(tone_wav: Path, read_wav: WavReader) -> None:
    """Тон: 16 kHz mono int16, 1 секунда, ненулевая амплитуда."""
    samples, sample_rate = read_wav(tone_wav)
    assert sample_rate == 16_000
    assert samples.dtype == np.int16
    assert len(samples) == 16_000
    assert np.abs(samples).max() > 1000


def test_silence_fixture_is_silent(silence_wav: Path, read_wav: WavReader) -> None:
    """Тишина: та же длина и частота, все сэмплы нулевые."""
    samples, sample_rate = read_wav(silence_wav)
    assert sample_rate == 16_000
    assert len(samples) == 16_000
    assert not samples.any()


@pytest.mark.real_models
def test_real_models_marker_is_skipped_by_default() -> None:
    """Пример теста с реальными моделями: без RT_REAL_MODELS=1 он пропускается."""
    assert os.environ.get(REAL_MODELS_ENV) == "1"
