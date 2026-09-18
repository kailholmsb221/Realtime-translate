#!/usr/bin/env python3
"""Сквозной smoke-тест движка: WAV → STT → MT → TTS → WAV.

Гоняет полный конвейер оркестратора в файловом режиме и печатает события с
задержками по стадиям. По умолчанию — на фейковых бэкендах и на фикстуре
``engine/tests/fixtures/stt_speech_pattern.wav``: работает на Linux без звука,
GPU и моделей. С ``--real`` берутся настоящие модели (faster-whisper small,
NLLB-200, XTTS-v2) — это прогон для целевой машины Windows + RTX 4050.

Запуск::

    python scripts/e2e_smoke.py                                  # фейки, фикстура
    python scripts/e2e_smoke.py --out out.wav --realtime         # в темпе живого звука
    python scripts/e2e_smoke.py --real --file sample.wav --from en --to ru
    python scripts/e2e_smoke.py --real --file me.wav --from ru --to en --voice v_1a2b3c4d

Коды возврата: ``0`` — успех, ``1`` — ошибка аргументов, ``2`` — движок не
смог отработать (нет моделей, не читается файл).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

from engine.contracts.events import (  # noqa: E402 — после правки sys.path
    EVENT_METRICS_LATENCY,
    EVENT_STT_FINAL,
    EVENT_TRANSLATION_READY,
    Lang,
    MetricsLatency,
    Stream,
    SttFinal,
    TranslationReady,
)
from engine.orchestrator.config import OrchestratorConfig  # noqa: E402
from engine.orchestrator.filemode import FileModeResult, run_file_mode  # noqa: E402

DEFAULT_FIXTURE: Final[Path] = (
    REPO_ROOT / "engine" / "tests" / "fixtures" / "stt_speech_pattern.wav"
)
# Каталог recordings/ уже в .gitignore — выход смоука не засоряет репозиторий.
DEFAULT_OUT: Final[Path] = REPO_ROOT / "recordings" / "e2e_smoke_out.wav"

#: Бюджет «конец фразы → озвучка перевода» (ARCHITECTURE.md раздел 2).
LATENCY_BUDGET_MS: Final[int] = 2_500

EXIT_OK: Final[int] = 0
EXIT_USAGE: Final[int] = 1
EXIT_UNAVAILABLE: Final[int] = 2


def build_parser() -> argparse.ArgumentParser:
    """Аргументы smoke-скрипта."""
    parser = argparse.ArgumentParser(
        description="Сквозной прогон конвейера движка на одном WAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--file", type=Path, default=DEFAULT_FIXTURE, help="входной WAV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="выходной WAV с озвучкой")
    parser.add_argument("--from", dest="lang_src", default="en", choices=[x.value for x in Lang])
    parser.add_argument("--to", dest="lang_dst", default="ru", choices=[x.value for x in Lang])
    parser.add_argument("--voice", dest="voice_id", default=None, help="voice_id профиля голоса")
    parser.add_argument(
        "--stream",
        default=Stream.IN.value,
        choices=[x.value for x in Stream],
        help="каким потоком считать запись",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="реальные модели вместо фейков (целевая машина, GPU)",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="читать файл в темпе реального времени — честные total_ms",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


def print_events(result: FileModeResult) -> None:
    """Печать распознанного, переведённого и задержек по фразам."""
    print("\nСобытия конвейера")
    print("-" * 72)
    for envelope in result.events:
        payload = envelope.payload
        if envelope.type == EVENT_STT_FINAL and isinstance(payload, SttFinal):
            print(
                f"  stt.final    [{payload.stream.value}] "
                f"{payload.t_start_ms:>6}-{payload.t_end_ms:<6} мс  "
                f"{payload.lang.value}: {payload.text}"
            )
        elif envelope.type == EVENT_TRANSLATION_READY and isinstance(payload, TranslationReady):
            print(
                f"  translation  [{payload.stream.value}] "
                f"{payload.src_lang.value}->{payload.dst_lang.value} "
                f"(ref={payload.ref_utterance_id}): {payload.text}"
            )
        elif envelope.type == EVENT_METRICS_LATENCY and isinstance(payload, MetricsLatency):
            print(
                f"  latency      [{payload.stream.value}] "
                f"stt={payload.stt_ms} мс, mt={payload.mt_ms} мс, "
                f"tts={payload.tts_ms} мс, total={payload.total_ms} мс"
            )


def print_summary(result: FileModeResult, realtime: bool) -> None:
    """Сводка: сколько фраз, какие задержки, укладываемся ли в бюджет."""
    metrics = [
        envelope.payload
        for envelope in result.metrics
        if isinstance(envelope.payload, MetricsLatency)
    ]
    print("\nИтог")
    print("-" * 72)
    print(f"  фраз (stt.final):     {len(result.finals)}")
    print(f"  переводов:            {len(result.translations)}")
    print(f"  выходной WAV:         {result.out_path} ({result.frames} фреймов)")
    if not metrics:
        print("  задержек нет — конвейер не дошёл до озвучки")
        return

    totals = [item.total_ms for item in metrics]
    print(f"  total_ms: медиана {statistics.median(totals):.0f}, максимум {max(totals)}")
    print(
        f"  по стадиям (медианы): stt {statistics.median([m.stt_ms for m in metrics]):.0f} мс, "
        f"mt {statistics.median([m.mt_ms for m in metrics]):.0f} мс, "
        f"tts {statistics.median([m.tts_ms for m in metrics]):.0f} мс"
    )
    if not realtime:
        print(
            "  (без --realtime файл читается быстрее живого звука — total_ms занижен, это не замер)"
        )
    elif max(totals) > LATENCY_BUDGET_MS:
        print(f"  ВНИМАНИЕ: бюджет {LATENCY_BUDGET_MS} мс превышен (ARCHITECTURE.md раздел 2)")


async def run(args: argparse.Namespace) -> int:
    """Прогнать конвейер и напечатать результат."""
    source: Path = args.file
    if not source.is_file():
        print(f"файл не найден: {source}", file=sys.stderr)
        return EXIT_USAGE

    config = replace(OrchestratorConfig.from_env(), backend="real" if args.real else "fake")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Бэкенд: {config.backend}, вход: {source}")
    print(f"Направление: {args.lang_src} -> {args.lang_dst}, voice_id={args.voice_id}")
    if args.real:
        print("Загружаю модели (первый запуск может занять минуты)…")

    try:
        result = await run_file_mode(
            config,
            source_path=source,
            out_path=args.out,
            lang_src=Lang(args.lang_src),
            lang_dst=Lang(args.lang_dst),
            voice_id=args.voice_id,
            stream=Stream(args.stream),
            realtime=args.realtime,
        )
    except Exception as exc:  # smoke-скрипт: показать причину, а не трейс
        print(f"прогон не удался: {exc}", file=sys.stderr)
        logging.getLogger("e2e_smoke").debug("подробности", exc_info=True)
        return EXIT_UNAVAILABLE

    print_events(result)
    print_summary(result, args.realtime)
    return EXIT_OK if result.finals else EXIT_UNAVAILABLE


def main(argv: list[str] | None = None) -> int:
    """Точка входа скрипта."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
