"""Один конвейер перевода: источник → STT → MT → TTS → приёмник (ARCHITECTURE.md 4.5).

Конвейер обслуживает **один** поток (``in`` — речь собеседника, ``out`` — речь
пользователя) и устроен так, чтобы распознавание никогда не ждало перевод и
синтез:

```
источник ─► (рекордер) ─► SttEngine.run ─► события stt.* ─► EventBus
                                   │
                                   └ stt.final ─► очередь ─► фоновый воркер:
                                        INSERT utterances
                                        ► translation.ready + UPDATE перевода
                                        ► TTS ─► чанки 24 kHz в приёмник
                                        ► metrics.latency
```

Воркер один на конвейер и разбирает очередь строго по порядку, поэтому
``translation.ready`` для потока приходят в том же порядке, что и
соответствующие ``stt.final``, а ``ref_utterance_id`` — это id уже вставленной
строки ``utterances`` (решение владельца).

Ошибка любого этапа логируется и не роняет конвейер: пропадает одна фраза,
а не вся сессия.

Inbound-конвейеру можно дать :class:`~engine.orchestrator.autoclone.AutoCloner`:
он копит речь собеседника по таймкодам ``stt.final`` и, набрав достаточно,
создаёт голосовой профиль в фоне. Как только профиль готов, следующие фразы
озвучиваются уже им (до этого — встроенным голосом XTTS).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
import wave
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from engine.audio_io import TTS_FORMAT, AudioChunk, AudioSink, AudioSource
from engine.contracts.events import (
    EVENT_STT_FINAL,
    Envelope,
    Lang,
    MetricsLatency,
    Stream,
    SttFinal,
    TranslationReady,
    validate_envelope,
)
from engine.orchestrator.autoclone import AutoCloner
from engine.orchestrator.bus import EventBus
from engine.orchestrator.db import Database
from engine.stt.engine import SttEngine
from engine.translate.service import TranslationService
from engine.tts.router import TtsRouter, chunk_to_event

__all__ = ["Pipeline", "PipelineStats", "WavRecorder"]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.pipeline")

#: Частота записи исходного потока на диск — формат движка (CLAUDE.md).
RECORD_SAMPLE_RATE: Final[int] = 16_000


class WavRecorder:
    """Пишет исходный PCM потока в WAV сессии (16 kHz mono int16).

    Файл открывается при первом чанке, поэтому для молчащего потока на диске
    ничего не появляется.

    Args:
        path: путь к WAV (каталог создаётся автоматически).
        sample_rate: частота записи; по умолчанию формат движка.
    """

    __slots__ = ("_path", "_sample_rate", "_wave", "frames")

    def __init__(self, path: str | Path, sample_rate: int = RECORD_SAMPLE_RATE) -> None:
        self._path = Path(path)
        self._sample_rate = sample_rate
        self._wave: wave.Wave_write | None = None
        #: сколько фреймов записано
        self.frames = 0

    @property
    def path(self) -> Path:
        """Путь к WAV."""
        return self._path

    def write(self, chunk: AudioChunk) -> None:
        """Дописать чанк (ошибки записи не роняют конвейер — только лог)."""
        try:
            if self._wave is None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                # Файл открыт на всё время потока и закрывается в close(),
                # поэтому context manager здесь неприменим.
                handle = wave.open(str(self._path), "wb")  # noqa: SIM115
                handle.setnchannels(1)
                handle.setsampwidth(2)
                handle.setframerate(self._sample_rate)
                self._wave = handle
            self._wave.writeframes(chunk.pcm)
            self.frames += chunk.n_frames
        except OSError:
            logger.exception("не удалось записать чанк в %s", self._path)

    def close(self) -> None:
        """Закрыть файл (повторный вызов безопасен)."""
        handle, self._wave = self._wave, None
        if handle is None:
            return
        with contextlib.suppress(OSError, wave.Error):
            handle.close()


@dataclass(slots=True)
class PipelineStats:
    """Счётчики конвейера — для тестов, smoke-скрипта и логов."""

    finals: int = 0
    translations: int = 0
    tts_chunks: int = 0
    errors: int = 0


class Pipeline:
    """Конвейер одного потока.

    Args:
        stream: ``in`` (собеседник) или ``out`` (пользователь).
        lang_src: язык оригинала этого потока.
        lang_dst: язык перевода этого потока.
        voice_id: профиль голоса для синтеза; для kk игнорируется роутером.
        source: источник аудио 16 kHz (``engine.audio_io``).
        sink: приёмник перевода (наушники, кабель или WAV).
        stt_engine: распознаватель; **свой на каждый конвейер** — движок
            держит состояние VAD и таймлайн потока.
        translation_service: сервис перевода (провайдер может быть общий).
        tts_router: маршрутизатор синтеза (модели общие на процесс).
        bus: шина событий.
        db: база; ``None`` — файловый режим без записи в SQLite.
        session_id: id сессии в БД; нужен вместе с ``db``.
        recorder: запись исходного аудио потока; ``None`` — не писать.
        auto_cloner: автоклон голоса говорящего (обычно только для потока
            ``in``); ``None`` — синтезировать голосом из ``voice_id``.
        emit_tts_chunks: публиковать ли служебные события ``tts.chunk``.
        drain_timeout_s: сколько ждать до-обработки очереди при остановке.
    """

    __slots__ = (
        "_auto_cloner",
        "_bus",
        "_db",
        "_drain_timeout_s",
        "_emit_tts_chunks",
        "_lang_dst",
        "_lang_src",
        "_queue",
        "_recorder",
        "_session_id",
        "_sink",
        "_source",
        "_started_at",
        "_stream",
        "_stt",
        "_translation",
        "_tts",
        "_tts_seq",
        "_voice_id",
        "stats",
    )

    def __init__(
        self,
        stream: Stream,
        lang_src: Lang,
        lang_dst: Lang,
        voice_id: str | None,
        source: AudioSource,
        sink: AudioSink,
        stt_engine: SttEngine,
        translation_service: TranslationService,
        tts_router: TtsRouter,
        bus: EventBus,
        db: Database | None = None,
        session_id: int | None = None,
        recorder: WavRecorder | None = None,
        auto_cloner: AutoCloner | None = None,
        *,
        emit_tts_chunks: bool = False,
        drain_timeout_s: float = 10.0,
    ) -> None:
        self._stream = stream
        self._lang_src = lang_src
        self._lang_dst = lang_dst
        self._voice_id = voice_id
        self._source = source
        self._sink = sink
        self._stt = stt_engine
        self._translation = translation_service
        self._tts = tts_router
        self._bus = bus
        self._db = db
        self._session_id = session_id
        self._recorder = recorder
        self._auto_cloner = auto_cloner
        self._emit_tts_chunks = emit_tts_chunks
        self._drain_timeout_s = drain_timeout_s
        # В очередь кладём фразу вместе с замером STT: пока она ждёт обработки,
        # движок уже может посчитать следующий partial и перезаписать latency.
        self._queue: asyncio.Queue[tuple[Envelope, int] | None] = asyncio.Queue()
        self._started_at = 0.0
        self._tts_seq = 0
        self.stats = PipelineStats()

    @property
    def stream(self) -> Stream:
        """Поток, который обслуживает конвейер."""
        return self._stream

    @property
    def recorder(self) -> WavRecorder | None:
        """Рекордер исходного аудио, если запись включена."""
        return self._recorder

    @property
    def auto_cloner(self) -> AutoCloner | None:
        """Автоклон голоса говорящего, если он подключён к конвейеру."""
        return self._auto_cloner

    @property
    def voice_id(self) -> str | None:
        """Каким голосом синтезируется перевод прямо сейчас.

        Готовый автоклон имеет приоритет над ``voice_id`` из ``session.start``:
        для потока ``in`` это и есть голос собеседника.
        """
        if self._auto_cloner is not None and self._auto_cloner.voice_id is not None:
            return self._auto_cloner.voice_id
        return self._voice_id

    # --- основной цикл -----------------------------------------------------

    async def run(self) -> None:
        """Крутить конвейер, пока источник не закончится или задача не отменится."""
        self._started_at = time.perf_counter()
        worker = asyncio.create_task(self._worker(), name=f"pipeline-{self._stream.value}")
        logger.info(
            "конвейер %s запущен: %s -> %s, voice_id=%s",
            self._stream.value,
            self._lang_src.value,
            self._lang_dst.value,
            self._voice_id,
        )
        try:
            async for envelope in self._stt.run(
                self._read_source(), self._stream, self._lang_src.value
            ):
                await self._bus.publish(envelope)
                if envelope.type == EVENT_STT_FINAL:
                    self.stats.finals += 1
                    self._queue.put_nowait((envelope, int(self._stt.last_latency_ms or 0)))
        finally:
            self._queue.put_nowait(None)
            await self._drain(worker)
            if self._auto_cloner is not None:
                await self._auto_cloner.wait()
            self._close_recorder()
            logger.info(
                "конвейер %s остановлен: фраз %d, переводов %d, чанков TTS %d, ошибок %d",
                self._stream.value,
                self.stats.finals,
                self.stats.translations,
                self.stats.tts_chunks,
                self.stats.errors,
            )

    async def _drain(self, worker: asyncio.Task[None]) -> None:
        """Дать воркеру доработать очередь; при затягивании — отменить."""
        try:
            await asyncio.wait_for(asyncio.shield(worker), timeout=self._drain_timeout_s)
        except TimeoutError:
            logger.warning("конвейер %s: очередь не разобрана вовремя", self._stream.value)
        except asyncio.CancelledError:
            pass
        if not worker.done():
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    def _close_recorder(self) -> None:
        if self._recorder is not None:
            self._recorder.close()

    async def _read_source(self) -> AsyncIterator[AudioChunk]:
        """Чанки источника; попутно пишет их в WAV сессии, если включена запись."""
        async for chunk in self._source:
            if self._recorder is not None:
                self._recorder.write(chunk)
            if self._auto_cloner is not None:
                self._auto_cloner.feed(chunk)
            yield chunk

    # --- обработка фразы ---------------------------------------------------

    async def _worker(self) -> None:
        """Последовательно обрабатывает ``stt.final``: БД → перевод → TTS → метрики."""
        while True:
            item = await self._queue.get()
            if item is None:
                return
            try:
                await self._handle_final(*item)
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats.errors += 1
                logger.exception("конвейер %s: фраза пропущена", self._stream.value)

    async def _handle_final(self, envelope: Envelope, stt_ms: int) -> None:
        payload = envelope.payload
        if not isinstance(payload, SttFinal):  # pragma: no cover — защита от чужого события
            return

        self._note_auto_clone(payload)
        row_id = await self._insert_utterance(payload)
        ready = await self._translate(envelope, row_id)
        if ready is None:
            return

        mt_ms = int(self._translation.last_latency_ms)
        translated = ready.payload
        text = translated.text if isinstance(translated, TranslationReady) else ""
        tts_ms, first_chunk_at = await self._speak(text)

        await self._publish_metrics(payload, stt_ms, mt_ms, tts_ms, first_chunk_at)

    def _note_auto_clone(self, payload: SttFinal) -> None:
        """Отдать автоклону речевой участок фразы и, если пора, запустить клон."""
        cloner = self._auto_cloner
        if cloner is None:
            return
        cloner.note_utterance(payload.t_start_ms, payload.t_end_ms)
        if cloner.maybe_start():
            logger.info(
                "конвейер %s: собрано %.1f с речи, клонирую голос говорящего",
                self._stream.value,
                cloner.collected_seconds,
            )

    async def _insert_utterance(self, payload: SttFinal) -> int | None:
        """Записать реплику в БД; вернуть её id (или ``None`` без БД/при ошибке)."""
        if self._db is None or self._session_id is None:
            return None
        try:
            return await self._db.add_utterance(
                self._session_id,
                payload.t_start_ms,
                payload.t_end_ms,
                payload.stream,
                payload.lang,
                payload.text,
            )
        except Exception:
            self.stats.errors += 1
            logger.exception("не удалось записать реплику в БД")
            return None

    async def _translate(self, envelope: Envelope, row_id: int | None) -> Envelope | None:
        """Перевести фразу, опубликовать ``translation.ready`` и дописать перевод в БД."""
        try:
            ready = await self._translation.handle(envelope, self._lang_dst, row_id)
        except Exception:
            self.stats.errors += 1
            logger.exception("перевод упал, фраза пропущена")
            return None

        await self._bus.publish(ready)
        self.stats.translations += 1

        payload = ready.payload
        if self._db is not None and row_id is not None and isinstance(payload, TranslationReady):
            try:
                await self._db.set_translation(row_id, payload.text, payload.dst_lang)
            except Exception:
                self.stats.errors += 1
                logger.exception("не удалось записать перевод в БД")
        return ready

    async def _speak(self, text: str) -> tuple[int, float | None]:
        """Синтезировать перевод в приёмник.

        Returns:
            ``(tts_ms, время первого чанка)`` — ``tts_ms`` это время до первого
            чанка (именно оно попадает в бюджет «конец фразы → озвучка»),
            второе значение — момент по ``perf_counter`` или ``None``, если
            синтезировать было нечего.
        """
        if not text.strip():
            return 0, None

        started = time.perf_counter()
        first_chunk_at: float | None = None
        try:
            voice_id = self.voice_id
            async for pcm in self._tts.synthesize(text, self._lang_dst, voice_id):
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                await self._write_chunk(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            self.stats.errors += 1
            logger.exception("синтез речи упал, фраза не озвучена")
        with contextlib.suppress(Exception):
            await self._sink.drain()
        if first_chunk_at is None:
            return 0, None
        return max(0, round((first_chunk_at - started) * 1000)), first_chunk_at

    async def _write_chunk(self, pcm: Any) -> None:
        """Отдать чанк TTS приёмнику (и, если включено, в WebSocket)."""
        chunk = AudioChunk.from_array(pcm, ts_ms=0, fmt=TTS_FORMAT)
        await self._sink.write(chunk)
        self.stats.tts_chunks += 1
        if self._emit_tts_chunks:
            await self._bus.publish(chunk_to_event(self._stream, self._tts_seq, pcm))
        self._tts_seq += 1

    async def _publish_metrics(
        self,
        payload: SttFinal,
        stt_ms: int,
        mt_ms: int,
        tts_ms: int,
        first_chunk_at: float | None,
    ) -> None:
        """Посчитать и опубликовать ``metrics.latency``.

        ``total_ms`` — от конца речи (``t_end_ms`` на таймлайне потока) до
        первого чанка озвучки. В файловом режиме аудио читается быстрее
        реального времени, поэтому значение зажимается снизу нулём.
        """
        if first_chunk_at is None:
            total_ms = stt_ms + mt_ms
        else:
            elapsed_ms = round((first_chunk_at - self._started_at) * 1000)
            total_ms = max(0, elapsed_ms - payload.t_end_ms)
        metrics = MetricsLatency(
            stream=self._stream,
            stt_ms=max(0, stt_ms),
            mt_ms=max(0, mt_ms),
            tts_ms=max(0, tts_ms),
            total_ms=max(0, total_ms),
        )
        envelope = Envelope.wrap(metrics)
        validate_envelope(envelope.to_dict())
        await self._bus.publish(envelope)
        logger.debug(
            "latency %s: stt=%d мс, mt=%d мс, tts=%d мс, total=%d мс",
            self._stream.value,
            metrics.stt_ms,
            metrics.mt_ms,
            metrics.tts_ms,
            metrics.total_ms,
        )
