#!/usr/bin/env python3
"""Smoke-тест модуля stt: WAV -> события ``stt.partial`` / ``stt.final``.

Примеры::

    # реальные модели: Silero VAD + faster-whisper small int8_float16
    python scripts/stt_smoke.py sample.wav --lang ru

    # без моделей (EnergyVad + FakeTranscriber) — проверить сегментацию
    python scripts/stt_smoke.py engine/tests/fixtures/stt_speech_pattern.wav --fake

    # имитировать живой поток (пауза между чанками как в реальном времени)
    python scripts/stt_smoke.py sample.wav --lang en --realtime

Вход — WAV 16 kHz mono int16 (формат аудио движка). Файл другого формата
скрипт не конвертирует, а сообщает об этом: конвертация — задача audio_io.

Вывод: строка на каждое событие с временем от старта прогона, а в конце —
сводка по задержке распознавания.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

from engine.contracts.events import Envelope, SttFinal, SttPartial  # noqa: E402
from engine.stt import create_engine  # noqa: E402
from engine.stt.base import SAMPLE_RATE, SttConfig, SttError, samples_to_ms  # noqa: E402

CHUNK_MS = 30
"""Размер чанка, как у audio_io (30 мс = 480 сэмплов при 16 kHz)."""


@dataclass(frozen=True, slots=True)
class WavChunk:
    """Минимальный аудиочанк для движка (утиная типизация: ``pcm`` + ``ts_ms``)."""

    pcm: np.ndarray
    ts_ms: int


def read_wav(path: Path) -> np.ndarray:
    """Прочитать WAV 16 kHz mono int16 в numpy-массив.

    Raises:
        SystemExit: файл не в формате движка.
    """
    with wave.open(str(path), "rb") as fh:
        channels = fh.getnchannels()
        width = fh.getsampwidth()
        rate = fh.getframerate()
        frames = fh.readframes(fh.getnframes())

    if (channels, width, rate) != (1, 2, SAMPLE_RATE):
        raise SystemExit(
            f"{path}: нужен WAV 16 kHz mono int16, а тут {rate} Гц, {channels} кан., "
            f"{width * 8} бит. Сконвертируйте, например: "
            f'ffmpeg -i "{path}" -ac 1 -ar 16000 -sample_fmt s16 out.wav'
        )
    return np.frombuffer(frames, dtype="<i2").astype(np.int16)


async def wav_source(
    samples: np.ndarray, chunk_ms: int = CHUNK_MS, realtime: bool = False
) -> AsyncIterator[WavChunk]:
    """Отдать массив чанками по ``chunk_ms`` как живой аудиопоток."""
    step = chunk_ms * SAMPLE_RATE // 1000
    for offset in range(0, samples.size, step):
        chunk = samples[offset : offset + step]
        yield WavChunk(pcm=chunk, ts_ms=samples_to_ms(offset))
        await asyncio.sleep(chunk_ms / 1000 if realtime else 0)


def describe(envelope: Envelope) -> str:
    """Человекочитаемая строка события."""
    payload = envelope.payload
    if isinstance(payload, SttFinal):
        return (
            f"stt.final   [{payload.stream.value}/{payload.lang.value}] "
            f"{payload.t_start_ms:>6}-{payload.t_end_ms:<6} мс  {payload.text!r}"
        )
    if isinstance(payload, SttPartial):
        return f"stt.partial [{payload.stream.value}/{payload.lang.value}] {payload.text!r}"
    return f"{envelope.type} {payload}"  # pragma: no cover — других событий stt не выдаёт


async def run_smoke(args: argparse.Namespace) -> int:
    """Прогнать WAV через движок и напечатать события."""
    samples = read_wav(args.wav)
    audio_ms = samples_to_ms(samples.size)

    overrides: dict[str, object] = {}
    if args.model:
        overrides["model_size"] = args.model
    if args.device:
        overrides["device"] = args.device
    if args.partial_interval_ms is not None:
        overrides["partial_interval_ms"] = args.partial_interval_ms
    if args.lang:
        overrides["language"] = args.lang

    config = SttConfig.from_env(**overrides)
    engine = create_engine(
        config,
        backend="fake" if args.fake else "faster_whisper",
        vad_backend="energy" if args.fake else "silero",
    )

    print(f"файл:   {args.wav} ({audio_ms} мс аудио)")
    print(
        f"бэкенд: vad={'energy' if args.fake else 'silero'}, "
        f"stt={'fake' if args.fake else 'faster_whisper'}, модель={config.model_for(args.lang)}"
    )
    print(
        f"конфиг: device={config.device}, compute={config.compute_type}, "
        f"partial={config.partial_interval_ms} мс, max={config.max_utterance_ms} мс"
    )
    print("-" * 78)

    started = time.perf_counter()
    finals = 0
    partials = 0
    latencies: list[int] = []

    source = wav_source(samples, realtime=args.realtime)
    async for envelope in engine.run(source, stream=args.stream, lang=args.lang):
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        print(f"{elapsed_ms:>7} мс  {describe(envelope)}")
        if isinstance(envelope.payload, SttFinal):
            finals += 1
            if engine.last_latency_ms is not None:
                latencies.append(engine.last_latency_ms)
        elif isinstance(envelope.payload, SttPartial):
            partials += 1

    total_ms = int((time.perf_counter() - started) * 1000)
    print("-" * 78)
    print(
        f"итого: {finals} final, {partials} partial за {total_ms} мс "
        f"(аудио {audio_ms} мс, RTF {total_ms / max(1, audio_ms):.2f})"
    )
    if latencies:
        print(
            f"задержка транскрипции final: медиана {int(np.median(latencies))} мс, "
            f"максимум {max(latencies)} мс"
        )
    if not finals:
        print("ВНИМАНИЕ: ни одного stt.final — проверьте уровень сигнала и пороги VAD")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-тест распознавания: WAV -> события stt.partial / stt.final"
    )
    parser.add_argument("wav", type=Path, help="WAV 16 kHz mono int16")
    parser.add_argument("--lang", choices=("ru", "en", "kk"), help="язык (иначе автоопределение)")
    parser.add_argument("--stream", choices=("in", "out"), default="in", help="направление потока")
    parser.add_argument(
        "--fake", action="store_true", help="без моделей: EnergyVad + FakeTranscriber"
    )
    parser.add_argument("--model", help="модель whisper или путь к файнтюну (по умолчанию small)")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), help="устройство вычислений")
    parser.add_argument(
        "--partial-interval-ms", type=int, dest="partial_interval_ms", help="период stt.partial, мс"
    )
    parser.add_argument(
        "--realtime", action="store_true", help="подавать чанки в темпе реального времени"
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    if not args.wav.is_file():
        raise SystemExit(f"файл не найден: {args.wav}")

    try:
        return asyncio.run(run_smoke(args))
    except SttError as exc:
        raise SystemExit(f"stt: {exc}") from exc
    except KeyboardInterrupt:  # pragma: no cover — ручная остановка
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
