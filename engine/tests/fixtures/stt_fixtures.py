"""Генератор WAV-фикстур модуля stt (зона агента B).

Лежит рядом с фикстурами, а не в ``scripts/make_fixtures.py``, чтобы не лезть
в чужую зону (CLAUDE.md, закон 1). Формат — PCM 16 kHz mono int16.

Создаёт ``stt_speech_pattern.wav``: «тон — тишина — тон — тишина», то есть две
отчётливые «фразы» с известными таймкодами. На нём проверяется сегментация VAD
и выдача событий ``stt.partial`` / ``stt.final``.

Запуск::

    python engine/tests/fixtures/stt_fixtures.py          # создать недостающее
    python engine/tests/fixtures/stt_fixtures.py --force  # перезаписать
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path
from typing import Final

import numpy as np

SAMPLE_RATE: Final[int] = 16_000
CHANNELS: Final[int] = 1
SAMPLE_WIDTH: Final[int] = 2  # int16
AMPLITUDE: Final[float] = 0.5
FADE_MS: Final[int] = 10

FIXTURES_DIR: Final[Path] = Path(__file__).resolve().parent
SPEECH_PATTERN_FILENAME: Final[str] = "stt_speech_pattern.wav"

SPEECH_PATTERN_SEGMENTS: Final[tuple[tuple[int, int], ...]] = ((0, 1000), (2000, 3000))
"""Ожидаемые границы «речи» в ``stt_speech_pattern.wav``, мс."""

SPEECH_PATTERN_TOTAL_MS: Final[int] = 4000
"""Полная длительность ``stt_speech_pattern.wav``, мс."""


def tone(
    duration_ms: int,
    freq_hz: float = 440.0,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    fade_ms: int = FADE_MS,
) -> np.ndarray:
    """Синус длиной ``duration_ms`` как PCM int16 mono с fade in/out."""
    n = round(duration_ms * sample_rate / 1000)
    t = np.arange(n, dtype=np.float64) / sample_rate
    values = amplitude * np.sin(2.0 * np.pi * freq_hz * t)

    fade = min(round(fade_ms * sample_rate / 1000), n // 2)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade, endpoint=False)
        values[:fade] *= ramp
        values[-fade:] *= ramp[::-1]
    return np.round(values * np.iinfo(np.int16).max).astype(np.int16)


def silence(duration_ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Тишина длиной ``duration_ms`` как PCM int16 mono."""
    return np.zeros(round(duration_ms * sample_rate / 1000), dtype=np.int16)


def speech_pattern(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """«Тон — тишина — тон — тишина»: две фразы по 1 с с паузами по 1 с.

    Границы «речи» — :data:`SPEECH_PATTERN_SEGMENTS`. Хвостовая тишина нужна,
    чтобы VAD успел закрыть вторую фразу до конца потока.
    """
    return np.concatenate(
        [
            tone(1000, sample_rate=sample_rate),
            silence(1000, sample_rate=sample_rate),
            tone(1000, freq_hz=660.0, sample_rate=sample_rate),
            silence(1000, sample_rate=sample_rate),
        ]
    )


def write_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Записать int16-массив в WAV (mono, 16 бит)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(CHANNELS)
        fh.setsampwidth(SAMPLE_WIDTH)
        fh.setframerate(sample_rate)
        fh.writeframes(samples.astype("<i2").tobytes())


def make_stt_fixtures(fixtures_dir: Path = FIXTURES_DIR, *, force: bool = False) -> list[Path]:
    """Создать фикстуры модуля stt. Возвращает список записанных файлов."""
    written: list[Path] = []
    for name, samples in ((SPEECH_PATTERN_FILENAME, speech_pattern()),):
        path = fixtures_dir / name
        if path.exists() and not force:
            continue
        write_wav(path, samples)
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Сгенерировать WAV-фикстуры модуля stt")
    parser.add_argument("--force", action="store_true", help="перезаписать существующие файлы")
    parser.add_argument(
        "--out",
        type=Path,
        default=FIXTURES_DIR,
        help=f"каталог фикстур (по умолчанию {FIXTURES_DIR})",
    )
    args = parser.parse_args()

    written = make_stt_fixtures(args.out, force=args.force)
    for path in written:
        print(f"создано: {path}")
    if not written:
        print(f"все фикстуры stt на месте: {args.out} (--force чтобы перезаписать)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
