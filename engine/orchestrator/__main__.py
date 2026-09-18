"""Точка входа движка: ``python -m engine.orchestrator``.

Режимы:

* ``serve`` (по умолчанию) — live: WebSocket :8765 + REST :8766, два конвейера
  на сессию. На Windows работает с живыми устройствами, на других ОС требует
  ``--backend fake`` и WAV-файлов вместо микрофона и loopback;
* файловый — ``--file in.wav --from en --to ru --out out.wav``: один прогон
  WAV через конвейер без серверов, события печатаются JSON-строками.

Примеры::

    python -m engine.orchestrator                      # live, реальные модели
    python -m engine.orchestrator --backend fake \\
        --fake-in in.wav --fake-out mic.wav            # live на Linux, для UI
    python -m engine.orchestrator --backend fake \\
        --file engine/tests/fixtures/stt_speech_pattern.wav \\
        --from en --to ru --out out.wav
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
from dataclasses import replace
from pathlib import Path

from engine.audio_io.base import AudioBackendUnavailable
from engine.audio_io.devices import is_windows
from engine.contracts.events import Lang, Stream
from engine.orchestrator.config import BACKENDS, OrchestratorConfig
from engine.orchestrator.filemode import run_file_mode
from engine.orchestrator.server import serve

__all__ = ["build_parser", "main"]

logger = logging.getLogger("engine.orchestrator")

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_UNAVAILABLE = 2


def build_parser() -> argparse.ArgumentParser:
    """Аргументы командной строки движка."""
    parser = argparse.ArgumentParser(
        prog="python -m engine.orchestrator",
        description="Движок Realtime Translator: live-режим или прогон WAV через конвейер",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        nargs="?",
        default="serve",
        choices=("serve",),
        help="режим работы (пока только serve; файловый включается флагом --file)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=None,
        help="real — модели и устройства, fake — заглушки (по умолчанию из RT_BACKEND)",
    )
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "INFO"))

    group = parser.add_argument_group("файловый режим")
    group.add_argument("--file", type=Path, help="входной WAV (включает файловый режим)")
    group.add_argument("--out", type=Path, help="куда записать озвученный перевод (WAV)")
    group.add_argument("--from", dest="lang_src", choices=[lang.value for lang in Lang])
    group.add_argument("--to", dest="lang_dst", choices=[lang.value for lang in Lang])
    group.add_argument("--voice", dest="voice_id", default=None, help="voice_id профиля голоса")
    group.add_argument(
        "--stream",
        choices=[stream.value for stream in Stream],
        default=Stream.IN.value,
        help="каким потоком считать запись (in — собеседник, out — пользователь)",
    )
    group.add_argument(
        "--realtime",
        action="store_true",
        help="читать файл в темпе реального времени (честные latency)",
    )

    live = parser.add_argument_group("live-режим")
    live.add_argument(
        "--ws-port", type=int, default=None, help="порт WebSocket (по умолчанию 8765)"
    )
    live.add_argument("--http-port", type=int, default=None, help="порт REST (по умолчанию 8766)")
    live.add_argument("--db", type=Path, default=None, help="файл SQLite (по умолчанию RT_DB_PATH)")
    live.add_argument(
        "--fake-in", type=Path, default=None, help="WAV вместо loopback при --backend fake"
    )
    live.add_argument(
        "--fake-out", type=Path, default=None, help="WAV вместо микрофона при --backend fake"
    )
    live.add_argument(
        "--no-warmup", action="store_true", help="не прогревать модели при старте (отладка)"
    )
    return parser


def _config_from_args(args: argparse.Namespace) -> OrchestratorConfig:
    """Конфигурация из окружения, перебитая аргументами CLI."""
    config = OrchestratorConfig.from_env()
    changes: dict[str, object] = {}
    if args.backend is not None:
        changes["backend"] = args.backend
    if args.ws_port is not None:
        changes["ws_port"] = args.ws_port
    if args.http_port is not None:
        changes["http_port"] = args.http_port
    if args.db is not None:
        changes["db_path"] = args.db
    if args.fake_in is not None:
        changes["fake_in"] = args.fake_in
    if args.fake_out is not None:
        changes["fake_out"] = args.fake_out
    return replace(config, **changes) if changes else config  # type: ignore[arg-type]


def _check_live_backend(config: OrchestratorConfig) -> str | None:
    """Проверить, что live-режим вообще может работать на этой машине."""
    if config.is_fake or is_windows():
        return None
    return (
        "live-режим с реальными устройствами работает только на Windows "
        "(WASAPI loopback, VB-Audio Virtual Cable). На этой ОС запустите движок "
        "с --backend fake и WAV-файлами: python -m engine.orchestrator --backend fake "
        "--fake-in in.wav --fake-out mic.wav"
    )


async def _run_file(args: argparse.Namespace, config: OrchestratorConfig) -> int:
    source: Path = args.file
    if not source.is_file():
        print(f"файл не найден: {source}", file=sys.stderr)
        return EXIT_USAGE
    if args.lang_src is None or args.lang_dst is None:
        print("файловый режим требует --from и --to", file=sys.stderr)
        return EXIT_USAGE
    out: Path = args.out if args.out is not None else source.with_name(f"{source.stem}_out.wav")

    result = await run_file_mode(
        config,
        source_path=source,
        out_path=out,
        lang_src=Lang(args.lang_src),
        lang_dst=Lang(args.lang_dst),
        voice_id=args.voice_id,
        stream=Stream(args.stream),
        realtime=args.realtime,
        printer=lambda line: print(line, flush=True),
    )
    print(
        f"# готово: {out} ({result.frames} фреймов), "
        f"stt.final={len(result.finals)}, translation.ready={len(result.translations)}",
        flush=True,
    )
    return EXIT_OK


async def _run_serve(config: OrchestratorConfig, *, warmup: bool = True) -> int:
    problem = _check_live_backend(config)
    if problem is not None:
        print(problem, file=sys.stderr)
        return EXIT_UNAVAILABLE
    try:
        await serve(config, warmup=warmup)
    except AudioBackendUnavailable as exc:
        print(f"аудио недоступно: {exc}", file=sys.stderr)
        return EXIT_UNAVAILABLE
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    """Разобрать аргументы и запустить выбранный режим."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config = _config_from_args(args)
    if args.no_warmup:
        logger.info("прогрев отключён флагом --no-warmup")

    if args.file is not None:
        return asyncio.run(_run_file(args, config))

    logger.info("старт: %s", config.describe())
    with contextlib.suppress(KeyboardInterrupt):
        return asyncio.run(_run_serve(config, warmup=not args.no_warmup))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
