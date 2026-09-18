#!/usr/bin/env python3
"""Smoke-тест audio_io: устройства, запись loopback/микрофона, воспроизведение.

Скрипт нужен, чтобы вручную проверить связку «Zoom -> loopback -> движок» и
«движок -> наушники / VB-Cable» на целевой машине (Windows). На Linux и без
установленных аудио-библиотек печатает, чего не хватает, и выходит с кодом 2.

Примеры::

    python scripts/audio_smoke.py --list
    python scripts/audio_smoke.py --record-loopback 5 out.wav
    python scripts/audio_smoke.py --record-mic 5 mic.wav
    python scripts/audio_smoke.py --tone "CABLE Input"
    python scripts/audio_smoke.py --loop-file sample.wav "Наушники"

Коды возврата: 0 — успех, 1 — ошибка аргументов, 2 — аудио недоступно
(не та ОС, нет пакетов, нет устройства).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

from engine.audio_io import (  # noqa: E402  — после правки sys.path
    AudioConfig,
    AudioIOError,
    AudioSource,
    DeviceKind,
    FakeSink,
    FakeSource,
    backend_report,
    list_devices,
)
from engine.audio_io.devices import (  # noqa: E402
    find_default_mic,
    find_loopback_device,
    match_device,
)
from engine.audio_io.pcm import float32_to_int16  # noqa: E402

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NO_AUDIO = 2

TONE_HZ = 440.0
TONE_SECONDS = 1.0

log = logging.getLogger("audio_smoke")


def _print_backends() -> None:
    print("Бэкенды:")
    for name, status in backend_report().items():
        print(f"  {name:<14} {status}")


def cmd_list() -> int:
    """Напечатать устройства с пометками loopback / CABLE."""
    devices = list_devices()
    _print_backends()
    if not devices:
        print(
            "\nУстройств не найдено. Живой звук работает только на Windows "
            "с установленными sounddevice и PyAudioWPatch:\n"
            "  pip install -r engine/audio_io/requirements-audio_io.txt",
            file=sys.stderr,
        )
        return EXIT_NO_AUDIO

    print(f"\nУстройства ({len(devices)}):")
    for dev in devices:
        print("  " + dev.describe())

    cables = [d for d in devices if d.is_virtual_cable]
    loopbacks = [d for d in devices if d.is_loopback]
    print(f"\nloopback-устройств: {len(loopbacks)}, устройств VB-Cable: {len(cables)}")
    if not cables:
        print("VB-Audio Virtual Cable не найден — см. engine/audio_io/README.md")
    print(f"Конфиг: {AudioConfig.from_env().describe()}")
    return EXIT_OK


async def _record(source: AudioSource, seconds: float, out: Path) -> int:
    """Записать ``seconds`` секунд из источника в WAV."""
    sink = FakeSink(path=out)
    frames_needed = round(seconds * 16_000)
    async with source, sink:
        async for chunk in source:
            await sink.write(chunk)
            if sink.n_frames >= frames_needed:
                break
    print(f"записано {sink.duration_ms / 1000:.2f} с -> {out}")
    return EXIT_OK


def cmd_record_loopback(seconds: float, out: Path) -> int:
    """Записать звук, который ОС играет в наушники (речь собеседника)."""
    cfg = AudioConfig.from_env()
    from engine.audio_io.windows import WasapiLoopbackSource

    device = find_loopback_device(name=cfg.loopback_name)
    print(f"loopback: {device.describe()}")
    source = WasapiLoopbackSource(device, fmt=cfg.format, chunk_ms=cfg.chunk_ms)
    return asyncio.run(_record(source, seconds, out))


def cmd_record_mic(seconds: float, out: Path) -> int:
    """Записать физический микрофон пользователя."""
    cfg = AudioConfig.from_env()
    from engine.audio_io.windows import MicSource

    device = find_default_mic(name=cfg.mic_name)
    print(f"микрофон: {device.describe()}")
    source = MicSource(device, fmt=cfg.format, chunk_ms=cfg.chunk_ms)
    return asyncio.run(_record(source, seconds, out))


async def _play(source: AudioSource, device_name: str) -> int:
    """Проиграть весь источник в устройство вывода по подстроке имени."""
    from engine.audio_io.windows import SoundDeviceSink

    device = match_device(list_devices(), DeviceKind.OUTPUT, device_name, what="устройство вывода")
    print(f"вывод: {device.describe()}")
    sink = SoundDeviceSink(device)
    async with source, sink:
        async for chunk in source:
            await sink.write(chunk)
        await sink.drain()
    return EXIT_OK


def cmd_tone(device_name: str) -> int:
    """Проиграть тестовый тон 440 Гц длительностью 1 с в указанное устройство."""
    cfg = AudioConfig.from_env()
    n = int(cfg.sample_rate * TONE_SECONDS)
    t = np.arange(n, dtype=np.float64) / cfg.sample_rate
    tone = float32_to_int16(0.3 * np.sin(2.0 * np.pi * TONE_HZ * t))
    source = FakeSource(tone, fmt=cfg.format, chunk_ms=cfg.chunk_ms, realtime=False)
    return asyncio.run(_play(source, device_name))


def cmd_loop_file(path: Path, device_name: str) -> int:
    """Проиграть WAV-файл в указанное устройство (проверка маршрутизации)."""
    cfg = AudioConfig.from_env()
    source = FakeSource.from_wav(path, fmt=cfg.format, chunk_ms=cfg.chunk_ms, realtime=True)
    return asyncio.run(_play(source, device_name))


def build_parser() -> argparse.ArgumentParser:
    """CLI smoke-скрипта."""
    parser = argparse.ArgumentParser(
        prog="audio_smoke.py",
        description="Smoke-тест audio_io: устройства, запись и воспроизведение",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="перечислить устройства")
    group.add_argument(
        "--record-loopback",
        nargs=2,
        metavar=("SECONDS", "OUT.WAV"),
        help="записать звук собеседника (WASAPI loopback) в WAV",
    )
    group.add_argument(
        "--record-mic",
        nargs=2,
        metavar=("SECONDS", "OUT.WAV"),
        help="записать микрофон в WAV",
    )
    group.add_argument(
        "--tone",
        metavar="DEVICE",
        help="проиграть тон 440 Гц 1 с в устройство (подстрока имени)",
    )
    group.add_argument(
        "--loop-file",
        nargs=2,
        metavar=("IN.WAV", "DEVICE"),
        help="проиграть WAV в устройство (подстрока имени)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        help="уровень логирования (по умолчанию LOG_LEVEL или INFO)",
    )
    return parser


def _seconds(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise ValueError("длительность должна быть > 0")
    return value


def main(argv: list[str] | None = None) -> int:
    """Точка входа CLI."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.list:
            return cmd_list()
        if args.record_loopback:
            seconds, out = args.record_loopback
            return cmd_record_loopback(_seconds(seconds), Path(out))
        if args.record_mic:
            return cmd_record_mic(_seconds(args.record_mic[0]), Path(args.record_mic[1]))
        if args.tone:
            return cmd_tone(args.tone)
        if args.loop_file:
            return cmd_loop_file(Path(args.loop_file[0]), args.loop_file[1])
    except ValueError as exc:
        print(f"ошибка аргументов: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except AudioIOError as exc:
        print(f"аудио недоступно: {exc}", file=sys.stderr)
        _print_backends()
        return EXIT_NO_AUDIO
    except FileNotFoundError as exc:
        print(f"файл не найден: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:  # pragma: no cover - ручной прогон
        print("прервано пользователем", file=sys.stderr)
        return EXIT_OK

    return EXIT_USAGE  # pragma: no cover - argparse не допустит


if __name__ == "__main__":
    raise SystemExit(main())
