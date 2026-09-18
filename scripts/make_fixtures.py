#!/usr/bin/env python3
"""Генератор WAV-фикстур для тестов (CLAUDE.md: аудио-тесты — на фикстурах).

Создаёт в ``engine/tests/fixtures/`` короткие сэмплы PCM 16 kHz mono int16 —
формат аудио внутри движка:

* ``tone_440hz_1s.wav``  — синус 440 Гц, 1 секунда (с мягким fade in/out,
  чтобы не было щелчков на границах);
* ``silence_1s.wav``     — тишина, 1 секунда.

Запуск::

    python scripts/make_fixtures.py            # создать недостающие
    python scripts/make_fixtures.py --force    # перезаписать

Скрипт идемпотентен: без ``--force`` существующие файлы не трогает.
Фикстуры коммитятся в репозиторий (`.gitignore` их не игнорирует).
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # int16
TONE_HZ = 440.0
DURATION_S = 1.0
AMPLITUDE = 0.5  # доля от полной шкалы int16
FADE_MS = 10

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES_DIR = REPO_ROOT / "engine" / "tests" / "fixtures"

TONE_FILENAME = "tone_440hz_1s.wav"
SILENCE_FILENAME = "silence_1s.wav"


def tone(
    freq_hz: float = TONE_HZ,
    duration_s: float = DURATION_S,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    fade_ms: int = FADE_MS,
) -> np.ndarray:
    """Синус ``freq_hz`` как PCM int16 mono с коротким fade in/out."""
    n = round(duration_s * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    wave_f = amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    fade = min(round(fade_ms * sample_rate / 1000), n // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        wave_f[:fade] *= ramp
        wave_f[-fade:] *= ramp[::-1]

    return np.round(wave_f * np.iinfo(np.int16).max).astype(np.int16)


def silence(duration_s: float = DURATION_S, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Тишина как PCM int16 mono."""
    return np.zeros(round(duration_s * sample_rate), dtype=np.int16)


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Записать int16-массив в WAV (mono, 16 бит)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(CHANNELS)
        fh.setsampwidth(SAMPLE_WIDTH)
        fh.setframerate(sample_rate)
        fh.writeframes(samples.astype("<i2").tobytes())


def make_fixtures(fixtures_dir: Path = FIXTURES_DIR, *, force: bool = False) -> list[Path]:
    """Создать все фикстуры. Возвращает список фактически записанных файлов."""
    written: list[Path] = []
    for name, samples in ((TONE_FILENAME, tone()), (SILENCE_FILENAME, silence())):
        path = fixtures_dir / name
        if path.exists() and not force:
            continue
        write_wav(path, samples)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Сгенерировать WAV-фикстуры для тестов")
    parser.add_argument("--force", action="store_true", help="перезаписать существующие файлы")
    parser.add_argument(
        "--out",
        type=Path,
        default=FIXTURES_DIR,
        help=f"каталог фикстур (по умолчанию {FIXTURES_DIR})",
    )
    args = parser.parse_args()

    written = make_fixtures(args.out, force=args.force)
    if written:
        for path in written:
            print(f"создано: {path}")
    else:
        print(f"все фикстуры на месте: {args.out} (--force чтобы перезаписать)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
