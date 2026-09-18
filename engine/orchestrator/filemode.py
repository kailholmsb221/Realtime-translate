"""Файловый режим: WAV на входе → переведённый WAV на выходе (ARCHITECTURE.md 7.1).

Тот же конвейер, что и в live-режиме, только источник — файл, приёмник — WAV,
а серверы не поднимаются. События печатаются JSON-строками (конверт контрактов
как есть), поэтому вывод можно скормить ``jq`` или разобрать в тесте.

Пример::

    python -m engine.orchestrator --backend fake \\
        --file engine/tests/fixtures/stt_speech_pattern.wav \\
        --from en --to ru --out out.wav
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

from engine.audio_io import FakeSink, FakeSource
from engine.contracts.events import (
    EVENT_METRICS_LATENCY,
    EVENT_STT_FINAL,
    EVENT_TRANSLATION_READY,
    Envelope,
    Lang,
    Stream,
    to_json,
)
from engine.orchestrator.bus import EventBus
from engine.orchestrator.config import OrchestratorConfig
from engine.orchestrator.db import Database
from engine.orchestrator.pipeline import Pipeline
from engine.orchestrator.runtime import Runtime

__all__ = ["FileModeResult", "run_file_mode"]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.filemode")


@dataclass(slots=True)
class FileModeResult:
    """Итог прогона файлового режима."""

    out_path: Path
    events: list[Envelope] = field(default_factory=list)
    frames: int = 0

    def of_type(self, type_name: str) -> list[Envelope]:
        """События одного типа в порядке появления."""
        return [event for event in self.events if event.type == type_name]

    @property
    def finals(self) -> list[Envelope]:
        """События ``stt.final``."""
        return self.of_type(EVENT_STT_FINAL)

    @property
    def translations(self) -> list[Envelope]:
        """События ``translation.ready``."""
        return self.of_type(EVENT_TRANSLATION_READY)

    @property
    def metrics(self) -> list[Envelope]:
        """События ``metrics.latency``."""
        return self.of_type(EVENT_METRICS_LATENCY)


async def run_file_mode(
    config: OrchestratorConfig,
    *,
    source_path: str | Path,
    out_path: str | Path,
    lang_src: Lang,
    lang_dst: Lang,
    voice_id: str | None = None,
    stream: Stream = Stream.IN,
    realtime: bool = False,
    printer: Callable[[str], None] | None = None,
    db: Database | None = None,
    session_id: int | None = None,
) -> FileModeResult:
    """Прогнать WAV через полный конвейер и записать озвученный перевод.

    Args:
        config: конфигурация движка (``backend`` решает, фейки или модели).
        source_path: входной WAV (любая частота — будет ресемплирован в 16 kHz).
        out_path: куда записать синтезированный перевод.
        lang_src: язык оригинала.
        lang_dst: язык перевода.
        voice_id: профиль голоса (для kk игнорируется).
        stream: каким потоком считать запись (``in`` по умолчанию).
        realtime: выдерживать темп реального времени при чтении файла.
        printer: куда печатать события; ``None`` — не печатать.
        db: база, если прогон нужно записать в историю.
        session_id: id сессии в БД (вместе с ``db``).

    Returns:
        Собранные события и путь к выходному WAV.
    """
    runtime = Runtime.create(config)
    bus = EventBus()
    result = FileModeResult(out_path=Path(out_path))

    source = FakeSource.from_wav(
        source_path,
        fmt=runtime.config.audio.format,
        chunk_ms=runtime.config.audio.chunk_ms,
        realtime=realtime,
    )
    sink = FakeSink(fmt=runtime.config.audio.format, path=Path(out_path))

    pipeline = Pipeline(
        stream=stream,
        lang_src=lang_src,
        lang_dst=lang_dst,
        voice_id=voice_id,
        source=source,
        sink=sink,
        stt_engine=runtime.stt_engine(),
        translation_service=runtime.translation_service(),
        tts_router=runtime.tts,
        bus=bus,
        db=db,
        session_id=session_id,
        emit_tts_chunks=runtime.config.emit_tts_chunks,
        drain_timeout_s=runtime.config.drain_timeout_s,
    )

    queue = bus.subscribe()
    collector = asyncio.create_task(_collect(queue, result, printer), name="filemode-events")
    try:
        await runtime.warmup((lang_dst,))
        await pipeline.run()
    finally:
        collector.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await collector
        bus.unsubscribe(queue)
        _drain(queue, result, printer)  # хвост, не успевший уйти в сборщик
        await sink.drain()
        await sink.close()
        await source.close()
        await runtime.aclose()

    result.frames = sink.n_frames
    logger.info(
        "файловый режим: фраз %d, переводов %d, выход %s (%d фреймов)",
        pipeline.stats.finals,
        pipeline.stats.translations,
        result.out_path,
        result.frames,
    )
    return result


async def _collect(
    queue: asyncio.Queue[Envelope],
    result: FileModeResult,
    printer: Callable[[str], None] | None,
) -> None:
    """Складывать события шины в результат и печатать их JSON-строками."""
    while True:
        envelope = await queue.get()
        _accept(envelope, result, printer)


def _drain(
    queue: asyncio.Queue[Envelope],
    result: FileModeResult,
    printer: Callable[[str], None] | None,
) -> None:
    """Забрать всё, что осталось в очереди после остановки конвейера."""
    while True:
        try:
            envelope = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        _accept(envelope, result, printer)


def _accept(
    envelope: Envelope,
    result: FileModeResult,
    printer: Callable[[str], None] | None,
) -> None:
    result.events.append(envelope)
    if printer is not None:
        printer(to_json(envelope))
