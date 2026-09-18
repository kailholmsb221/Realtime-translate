"""Сегментатор и движок распознавания (ARCHITECTURE.md 4.2).

:class:`SttEngine` соединяет VAD и распознаватель: копит речь между
``speech_start`` и ``speech_end``, во время речи каждые
``partial_interval_ms`` выдаёт ``stt.partial``, а на конце фразы (или по
``max_utterance_ms``) — ``stt.final`` с таймкодами от начала потока.

Правила:

* распознавание идёт в отдельном потоке (``asyncio.to_thread``), event loop не
  блокируется;
* одновременно работает не более одной транскрипции — если предыдущая
  ``partial`` ещё считается, тик пропускается;
* каждое событие собирается датаклассом из :mod:`engine.contracts.events` и
  валидируется JSON-схемой перед выдачей;
* время транскрипции пишется в лог ``engine.stt`` и доступно в
  :attr:`SttEngine.last_latency_ms`.

Вход — любой async-итератор объектов с полями ``pcm`` и ``ts_ms``
(утиная типизация, см. :func:`engine.stt.base.chunk_samples`); прямого импорта
``engine.audio_io`` здесь нет (CLAUDE.md, технический стандарт).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from engine.contracts.events import (
    Envelope,
    Event,
    Lang,
    Stream,
    SttFinal,
    SttPartial,
    validate_envelope,
)
from engine.stt.base import (
    Int16Array,
    SttConfig,
    Transcriber,
    TranscriptResult,
    VoiceActivityDetector,
    chunk_samples,
    ms_to_samples,
    samples_to_ms,
)

__all__ = ["SttEngine"]

logger = logging.getLogger("engine.stt")

_EMPTY: Int16Array = np.zeros(0, dtype=np.int16)


@dataclass(slots=True)
class _RunState:
    """Состояние одного прогона :meth:`SttEngine.run`.

    ``tail`` — кольцевой хвост потока длиной ``preroll_samples``: из него
    берётся аудио, записанное до момента ``speech_start`` (VAD сообщает о
    начале речи с задержкой на ``min_speech_ms``).
    """

    preroll_samples: int
    pos_samples: int = 0
    tail: Int16Array = field(default_factory=lambda: _EMPTY)
    tail_start_samples: int = 0
    utterance: Int16Array | None = None
    utterance_start_samples: int = 0
    last_partial_ms: int = 0

    @property
    def active(self) -> bool:
        """Идёт ли сейчас фраза."""
        return self.utterance is not None

    @property
    def now_ms(self) -> int:
        """Текущая позиция на таймлайне потока, мс."""
        return samples_to_ms(self.pos_samples)

    @property
    def utterance_ms(self) -> int:
        """Длительность накопленной фразы, мс."""
        return 0 if self.utterance is None else samples_to_ms(self.utterance.size)

    def append(self, samples: Int16Array) -> None:
        """Добавить чанк в хвост потока и, если идёт фраза, в её буфер."""
        if samples.size == 0:
            return
        if self.utterance is not None:
            self.utterance = np.concatenate((self.utterance, samples))
        tail = samples if self.tail.size == 0 else np.concatenate((self.tail, samples))
        if self.preroll_samples > 0:
            self.tail = tail[-self.preroll_samples :]
        else:
            self.tail = _EMPTY
        self.pos_samples += samples.size
        self.tail_start_samples = self.pos_samples - self.tail.size

    def begin(self, ts_ms: int) -> None:
        """Начать фразу с таймкода ``ts_ms``, прихватив доступный preroll."""
        offset = ms_to_samples(ts_ms) - self.tail_start_samples
        offset = max(0, min(offset, self.tail.size))
        self.utterance = self.tail[offset:].copy()
        self.utterance_start_samples = self.tail_start_samples + offset
        self.last_partial_ms = samples_to_ms(self.utterance_start_samples)

    def trim_to(self, ts_ms: int) -> None:
        """Обрезать фразу по таймкоду конца речи (хвостовая тишина не нужна)."""
        if self.utterance is None:
            return
        keep = ms_to_samples(ts_ms) - self.utterance_start_samples
        if keep <= 0:
            self.utterance = _EMPTY
        elif keep < self.utterance.size:
            self.utterance = self.utterance[:keep]

    def restart(self) -> None:
        """Закрыть текущий кусок фразы, но остаться в состоянии речи."""
        self.utterance = _EMPTY
        self.utterance_start_samples = self.pos_samples
        self.last_partial_ms = self.now_ms

    def end(self) -> None:
        """Завершить фразу."""
        self.utterance = None

    def bounds_ms(self) -> tuple[int, int]:
        """Таймкоды ``(начало, конец)`` накопленной фразы от начала потока."""
        start = samples_to_ms(self.utterance_start_samples)
        size = 0 if self.utterance is None else self.utterance.size
        return start, samples_to_ms(self.utterance_start_samples + size)

    def snapshot(self) -> Int16Array:
        """Копия накопленного аудио фразы (для транскрипции в другом потоке)."""
        return _EMPTY if self.utterance is None else self.utterance.copy()


class SttEngine:
    """Движок распознавания: поток аудиочанков -> события ``stt.*``.

    Пример::

        from engine.stt import create_engine
        from engine.stt.base import SttConfig

        engine = create_engine(SttConfig.from_env(), vad_backend="energy",
                               transcriber_backend="fake")
        async for envelope in engine.run(source, stream="in", lang="ru"):
            print(envelope.type, envelope.payload)
    """

    def __init__(
        self,
        vad: VoiceActivityDetector,
        transcriber: Transcriber,
        config: SttConfig | None = None,
    ) -> None:
        """Собрать движок из готовых VAD и распознавателя.

        Args:
            vad: детектор речи (:class:`~engine.stt.base.VoiceActivityDetector`).
            transcriber: распознаватель (:class:`~engine.stt.base.Transcriber`).
            config: настройки; по умолчанию :class:`SttConfig` со значениями
                по умолчанию.
        """
        self._vad = vad
        self._transcriber = transcriber
        self._config = config or SttConfig()
        self.last_latency_ms: int | None = None

    @property
    def config(self) -> SttConfig:
        """Настройки движка."""
        return self._config

    @property
    def vad(self) -> VoiceActivityDetector:
        """Используемый детектор речи."""
        return self._vad

    @property
    def transcriber(self) -> Transcriber:
        """Используемый распознаватель."""
        return self._transcriber

    async def run(
        self,
        source: AsyncIterator[Any],
        stream: Stream | str = Stream.IN,
        lang: str | None = None,
    ) -> AsyncIterator[Envelope]:
        """Прогнать поток аудиочанков через VAD и распознавание.

        Args:
            source: async-итератор чанков с полями ``pcm`` (PCM 16 kHz mono
                int16 или ``bytes``) и ``ts_ms``.
            stream: ``"in"`` — речь собеседника, ``"out"`` — речь пользователя.
            lang: фиксированный язык или ``None`` — берётся
                :attr:`SttConfig.language`, иначе автоопределение модели.

        Yields:
            Конверты :class:`~engine.contracts.events.Envelope` с событиями
            ``stt.partial`` и ``stt.final``, уже провалидированные схемами.
        """
        stream_enum = Stream(stream)
        target_lang = lang or self._config.language
        self._vad.reset()

        state = _RunState(preroll_samples=ms_to_samples(self._config.preroll_ms))
        pending: asyncio.Task[Envelope | None] | None = None

        try:
            async for chunk in source:
                samples = chunk_samples(chunk).reshape(-1)
                state.append(samples)
                events = self._vad.process(samples)

                if pending is not None and pending.done():
                    ready = await self._collect(pending)
                    pending = None
                    if ready is not None:
                        yield ready

                for event in events:
                    if event.is_start:
                        if not state.active:
                            state.begin(event.ts_ms)
                        continue
                    if not state.active:
                        continue
                    state.trim_to(event.ts_ms)
                    if pending is not None:
                        ready = await self._collect(pending)
                        pending = None
                        if ready is not None:
                            yield ready
                    final = await self._final_event(state, stream_enum, target_lang)
                    state.end()
                    if final is not None:
                        yield final

                if self._should_force_final(state):
                    logger.info(
                        "фраза длиннее max_utterance_ms=%d — закрываю принудительно",
                        self._config.max_utterance_ms,
                    )
                    if pending is not None:
                        ready = await self._collect(pending)
                        pending = None
                        if ready is not None:
                            yield ready
                    final = await self._final_event(state, stream_enum, target_lang)
                    state.restart()
                    if final is not None:
                        yield final

                if pending is None and self._should_emit_partial(state):
                    state.last_partial_ms = state.now_ms
                    pending = asyncio.create_task(
                        self._partial_event(state.snapshot(), stream_enum, target_lang)
                    )

            if pending is not None:
                ready = await self._collect(pending)
                pending = None
                if ready is not None:
                    yield ready

            for event in self._flush_vad():
                if event.is_end and state.active:
                    state.trim_to(event.ts_ms)

            if state.active:
                final = await self._final_event(state, stream_enum, target_lang)
                state.end()
                if final is not None:
                    yield final
        finally:
            if pending is not None and not pending.done():
                pending.cancel()

    # --- внутреннее --------------------------------------------------------

    def _flush_vad(self) -> list[Any]:
        """Добрать события VAD в конце потока, если он это умеет."""
        flush = getattr(self._vad, "flush", None)
        if not callable(flush):
            return []
        events: list[Any] = list(flush())
        return events

    def _should_force_final(self, state: _RunState) -> bool:
        limit = self._config.max_utterance_ms
        return bool(state.active and limit and state.utterance_ms >= limit)

    def _should_emit_partial(self, state: _RunState) -> bool:
        interval = self._config.partial_interval_ms
        if not state.active or not interval or state.utterance_ms <= 0:
            return False
        return state.now_ms - state.last_partial_ms >= interval

    async def _collect(self, task: asyncio.Task[Envelope | None]) -> Envelope | None:
        """Дождаться задачи partial-транскрипции и вернуть её событие."""
        try:
            return await task
        except asyncio.CancelledError:  # pragma: no cover — гонка при остановке
            return None

    async def _transcribe(self, pcm: Int16Array, lang: str | None) -> TranscriptResult | None:
        """Распознать буфер в отдельном потоке, замерив время."""
        if pcm.size == 0:
            return None
        started = time.perf_counter()
        try:
            result = await asyncio.to_thread(self._transcriber.transcribe, pcm, lang)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("транскрипция упала, событие пропущено")
            return None
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        self.last_latency_ms = elapsed_ms
        logger.debug(
            "транскрипция %d мс аудио заняла %d мс (RTF %.2f)",
            samples_to_ms(pcm.size),
            elapsed_ms,
            elapsed_ms / max(1, samples_to_ms(pcm.size)),
        )
        return result

    async def _partial_event(
        self, pcm: Int16Array, stream: Stream, lang: str | None
    ) -> Envelope | None:
        """Посчитать промежуточную гипотезу и собрать ``stt.partial``."""
        result = await self._transcribe(pcm, lang)
        if result is None or not result.text.strip():
            return None
        event_lang = self._resolve_lang(lang, result)
        if event_lang is None:
            return None
        return self._envelope(SttPartial(stream=stream, lang=event_lang, text=result.text.strip()))

    async def _final_event(
        self, state: _RunState, stream: Stream, lang: str | None
    ) -> Envelope | None:
        """Посчитать окончательный текст фразы и собрать ``stt.final``."""
        pcm = state.snapshot()
        t_start_ms, t_end_ms = state.bounds_ms()
        if samples_to_ms(pcm.size) < self._config.min_speech_ms:
            logger.debug("фраза короче min_speech_ms — пропускаю")
            return None
        result = await self._transcribe(pcm, lang)
        if result is None or not result.text.strip():
            return None
        event_lang = self._resolve_lang(lang, result)
        if event_lang is None:
            return None
        logger.info(
            "stt.final [%s] %d-%d мс: %s",
            stream.value,
            t_start_ms,
            t_end_ms,
            result.text.strip(),
        )
        return self._envelope(
            SttFinal(
                stream=stream,
                lang=event_lang,
                text=result.text.strip(),
                t_start_ms=t_start_ms,
                t_end_ms=t_end_ms,
            )
        )

    def _resolve_lang(self, requested: str | None, result: TranscriptResult) -> Lang | None:
        """Свести язык к enum контрактов (``ru`` / ``en`` / ``kk``)."""
        candidates = (requested, result.lang, self._config.language, self._config.fallback_lang)
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return Lang(str(candidate).lower())
            except ValueError:
                continue
        logger.warning(
            "язык %r вне контракта ru/en/kk и fallback не задан — событие пропущено",
            result.lang,
        )
        return None

    @staticmethod
    def _envelope(event: Event) -> Envelope:
        """Обернуть событие в конверт и проверить схемами контрактов."""
        envelope = Envelope.wrap(event)
        validate_envelope(envelope.to_dict())
        return envelope
