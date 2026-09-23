"""Перевод своей речи: микрофон → whisper → NLLB → озвучка, плюс панель в UI.

Зачем отдельный режим. В live-режиме (``python -m engine.orchestrator``) всегда
поднимаются **оба** конвейера: ``in`` (WASAPI loopback — звук из Zoom) и ``out``
(микрофон). Пока не установлен VB-Audio Virtual Cable, озвучка уходит в колонки,
loopback ловит её обратно, whisper распознаёт уже собственный перевод — и
разговор зацикливается сам на себе. Здесь работает только микрофон, поэтому
такой петли нет.

Что делает режим:

* слушает только микрофон (поток ``out``), loopback не открывается вовсе;
* озвучивает перевод в наушники или колонки (XTTS-v2). ``--silent`` выключает
  озвучку совсем — тогда XTTS не загружается;
* на колонках эхо собственной озвучки вычитается из микрофона адаптивным
  фильтром (``scripts/aec.py``), поэтому вход не глушится и речь поверх
  перевода не теряется — наушники не нужны. Запасной вариант ``--half-duplex``
  просто закрывает микрофон на время озвучки (и съедает сказанное в этот
  момент), ``--no-aec`` отключает фильтр;
* шлёт по WebSocket те же события контрактов (``stt.partial`` / ``stt.final`` /
  ``translation.ready`` / ``metrics.latency`` / ``session.state``) на том же
  порту, что и движок, поэтому UI подключается без изменений: страница
  ``/pipeline`` показывает цепочку «что услышал whisper → что ушло в NLLB →
  что NLLB вернул».

Задержка. Фраза закрывается либо паузой (``--silence-ms``), либо принудительно
через ``--max-phrase-ms`` — чтобы при непрерывной речи перевод появлялся
примерно раз в две секунды, а не в конце монолога.

Примеры::

    # говорю по-русски, слышу английский перевод; панель: localhost:3000/pipeline
    python scripts/mic_subtitles.py --from ru --to en

    # только субтитры, без звука
    python scripts/mic_subtitles.py --silent

    # другой микрофон (номера: python scripts/audio_smoke.py --list) и тихий вход
    python scripts/mic_subtitles.py --mic-index 9 --gain 3
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import logging
import re
import sys
import time
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import websockets
from websockets.asyncio.server import ServerConnection
from websockets.asyncio.server import serve as ws_serve

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

from aec import EchoCanceller  # noqa: E402  — сосед по scripts/

from engine.audio_io import (  # noqa: E402
    TTS_FORMAT,
    AudioChunk,
    AudioSink,
    AudioSource,
    create_sink,
    create_source,
)
from engine.audio_io.config import AudioConfig  # noqa: E402
from engine.audio_io.pcm import resample  # noqa: E402
from engine.contracts.events import (  # noqa: E402  — после правки sys.path
    COMMAND_SESSION_START,
    COMMAND_SESSION_STOP,
    EVENT_STT_FINAL,
    ContractError,
    Envelope,
    Lang,
    Langs,
    MetricsLatency,
    SessionStart,
    SessionState,
    SessionStatus,
    Stream,
    SttFinal,
    TranslationReady,
    parse_event,
    to_json,
)
from engine.stt import create_engine  # noqa: E402
from engine.stt.base import SttConfig  # noqa: E402
from engine.stt.engine import SttEngine  # noqa: E402
from engine.translate import create_provider  # noqa: E402
from engine.translate.base import TranslateConfig  # noqa: E402
from engine.translate.service import TranslationService  # noqa: E402
from engine.tts import create_tts  # noqa: E402
from engine.tts.base import TtsConfig  # noqa: E402
from engine.tts.router import TtsRouter  # noqa: E402

__all__ = [
    "HalfDuplexGate",
    "MicSubtitles",
    "build_parser",
    "is_hallucination",
    "is_repetition_spam",
    "looks_untranslated",
    "main",
]

logger = logging.getLogger("mic_subtitles")

#: Частота потока внутри движка, Гц (CLAUDE.md, технический стандарт).
ENGINE_RATE: Final[int] = 16_000

#: Сколько событий держим на одного клиента UI, прежде чем ронять старые.
CLIENT_QUEUE_SIZE: Final[int] = 512

#: Сторож цикла событий: период проверки и пороги, с которых пишем сводку.
WATCHDOG_TICK_S: Final[float] = 0.1
LAG_REPORT_MS: Final[float] = 150.0
STAGE_REPORT_MS: Final[float] = 20.0

#: Пик сырого входа ниже этого — цифровая тишина, а не тихая комната: даже
#: шум вентилятора даёт заметно больше. Так выглядит аппаратный mute
#: микрофона (клавиша Fn на ноутбуке) или ползунок «Ввод» в нуле.
DEAD_INPUT_PEAK: Final[float] = 1e-4

#: Сколько секунд цифровой тишины подряд считаем «микрофон выключен».
DEAD_INPUT_S: Final[float] = 5.0

#: Порт WebSocket — тот же, что у движка, чтобы UI не перенастраивать.
DEFAULT_WS_PORT: Final[int] = 8765
DEFAULT_WS_HOST: Final[str] = "127.0.0.1"

#: Пауза, закрывающая фразу, мс. Короче штатных 500 мс: субтитрам важнее темп.
DEFAULT_SILENCE_MS: Final[int] = 450

#: Сколько аудио до срабатывания VAD прихватить в фразу, мс. Спасает первое
#: слово, но НЕ должно превышать паузу ``DEFAULT_SILENCE_MS``: движок берёт
#: предзахват из общего хвоста аудио, и при большем значении в новую фразу
#: попадает конец предыдущей — whisper повторяет уже сказанное.
DEFAULT_PREROLL_MS: Final[int] = 250

#: Предельная длина фразы, мс. Обычно фраза закрывается паузой (см.
#: ``DEFAULT_SILENCE_MS``), а этот потолок нужен для непрерывной речи. Ставить
#: его в 2-4 с нельзя: длинное предложение режется на обрывки посреди слова,
#: whisper на обрывках выдумывает, а NLLB переводит куски без контекста.
DEFAULT_MAX_PHRASE_MS: Final[int] = 7_000

#: Сколько держать микрофон закрытым после конца озвучки, мс (эхо колонок).
DEFAULT_TAIL_MS: Final[int] = 500

#: Если фраза дождалась своей очереди позже этого, озвучивать её уже поздно:
#: синтез идёт последовательно, и на непрерывной речи очередь копится, а звук
#: отстаёт всё сильнее. Текст такой фразы в панель всё равно уходит.
DEFAULT_STALE_MS: Final[int] = 4_000

#: ``stream_chunk_size`` XTTS: чем меньше, тем раньше слышен первый звук.
#: Штатные 20 дают ~1.9 с до первого чанка — для разговора это много.
DEFAULT_TTS_CHUNK_SIZE: Final[int] = 10

#: Порог Silero VAD. Штатные 0.5 на тихом входе с усилением пропускают шум:
#: whisper получает не-речь и выдаёт подписи звуков вместо слов.
DEFAULT_VAD_THRESHOLD: Final[float] = 0.7

#: Минимальная длина речи, мс. Штатные 250 мс пропускают всплески шума
#: вентилятора — whisper потом пишет их как «ссссссс».
DEFAULT_MIN_SPEECH_MS: Final[int] = 400

#: Как часто считать промежуточную гипотезу, мс. Каждая — это whisper на всём
#: накопленном аудио фразы; на 7-секундной фразе раз в секунду это заметная
#: нагрузка на GPU и на GIL, из-за которой цикл событий подтормаживает.
DEFAULT_PARTIAL_MS: Final[int] = 1_500

#: beam search whisper. 1 — штатный быстрый режим движка; 2-3 точнее, но каждая
#: фраза распознаётся заметно дольше, а задержка в разговоре важнее.
DEFAULT_BEAM_SIZE: Final[int] = 1

#: Пауза перед переоткрытием пропавшего устройства, с.
RESTART_DELAY_S: Final[float] = 1.0

#: Ниже этого пика (доля полной шкалы) вход слишком тих для whisper: нормальная
#: речь в микрофон даёт 0.1-0.5, ниже 0.05 модель уже домысливает слова.
QUIET_MIC_PEAK: Final[float] = 0.05

#: Заученные «титры», которые whisper выдаёт на шуме и тишине (см.
#: scripts/testdrive.py — там тот же фильтр и объяснение, откуда они берутся).
HALLUCINATIONS: Final[tuple[str, ...]] = (
    "субтитр",
    "редактор субтитров",
    "корректор",
    "dimatorzok",
    "спасибо за просмотр",
    "продолжение следует",
    "amara.org",
    "субтитры сделал",
    "субтитры создавал",
)


#: Слова-«звуковые теги». Whisper учился на ютуб-субтитрах, где звуки
#: подписывают отдельной строкой («СПОКОЙНАЯ МУЗЫКА», «АПЛОДИСМЕНТЫ»), и на
#: шуме воспроизводит именно такие подписи.
SOUND_TAGS: Final[frozenset[str]] = frozenset(
    {
        "музыка",
        "музыки",
        "музыкальная",
        "мелодия",
        "мелодии",
        "спокойная",
        "тревожная",
        "динамичная",
        "аплодисменты",
        "смех",
        "стон",
        "вздох",
        "кашель",
        "шум",
        "тишина",
        "звонок",
        "звук",
        "звуки",
        "играет",
        "звучит",
        "music",
        "applause",
        "laughter",
        "silence",
    }
)

#: Сколько слов максимум может быть в «звуковой» подписи.
SOUND_TAG_MAX_WORDS: Final[int] = 4

#: Пять и больше одинаковых букв подряд: в речи так не бывает, в шуме — всегда.
RUN_OF_ONE_LETTER: Final[re.Pattern[str]] = re.compile(r"(\w)\1{4,}", re.UNICODE)

#: Группа из 1-4 букв, повторённая четыре и больше раз подряд («іңіңіңіңіңің»,
#: «lalalala»): так whisper зацикливается, пока не кончится лимит токенов.
RUN_OF_GROUP: Final[re.Pattern[str]] = re.compile(r"(\w{1,4}?)\1{3,}", re.UNICODE)

#: Длиннее этого перевод не озвучиваем: живая фраза в 7 с даёт 150-200 знаков,
#: всё, что заметно больше, — размноженный повторами мусор, XTTS читал бы его
#: полминуты, а микрофон всё это время слушал бы динамик.
MAX_SPOKEN_CHARS: Final[int] = 350

#: Сколько последних озвученных переводов помним для защиты от эха по тексту.
SPOKEN_MEMORY: Final[int] = 6

#: Похожесть распознанного на недавно озвученное (0..1), с которой считаем
#: его эхом динамика, а не речью человека.
ECHO_SIMILARITY: Final[float] = 0.55


def is_hallucination(text: str) -> bool:
    """Похоже ли распознанное на артефакт whisper, а не на речь.

    Три признака: заученные «титры», подпись звука целиком («СПОКОЙНАЯ
    МУЗЫКА») и короткая фраза капсом — так модель помечает не-речь.
    """
    stripped = text.strip()
    low = stripped.lower()
    if any(marker in low for marker in HALLUCINATIONS):
        return True

    words = re.findall(r"\w+", low, flags=re.UNICODE)
    if not words:
        return False
    if len(words) <= SOUND_TAG_MAX_WORDS and all(word in SOUND_TAGS for word in words):
        return True
    # «ссссссс», «Ааааааа», «шшшшш» — так whisper записывает шум вентилятора,
    # «іңіңіңіңің» — так он зацикливается на группе букв.
    if any(RUN_OF_ONE_LETTER.search(word) or RUN_OF_GROUP.search(word) for word in words):
        return True
    # Капсом whisper пишет именно звуки: живая реплика так не выглядит.
    letters = [ch for ch in stripped if ch.isalpha()]
    return bool(letters) and len(words) <= 3 and all(ch.isupper() for ch in letters)


def is_repetition_spam(text: str, *, min_words: int = 6, max_unique_share: float = 0.25) -> bool:
    """Зациклилась ли модель на одном слове («No, no, no…» ×100).

    Whisper так залипает на шуме и на обрывках фраз. Переводить и озвучивать
    это нельзя: NLLB честно повторит то же самое по-русски, XTTS будет читать
    десять секунд, а микрофон за это время услышит сам себя.
    """
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    if len(words) < min_words:
        return False
    return len(set(words)) / len(words) <= max_unique_share


def is_garbage(text: str) -> bool:
    """Мусор whisper любого вида: титры, звуковые теги, повторы букв и слов."""
    return is_hallucination(text) or is_repetition_spam(text)


def similarity(a: str, b: str) -> float:
    """Похожесть двух фраз (0..1) — по общим словам и по символам.

    Берётся максимум из доли общих слов и посимвольной близости: эхо через
    динамик теряет окончания и обрывает края, поэтому одного признака мало.
    """
    la, lb = a.lower().strip(), b.lower().strip()
    if not la or not lb:
        return 0.0
    wa, wb = (
        set(re.findall(r"\w+", la, flags=re.UNICODE)),
        set(re.findall(r"\w+", lb, flags=re.UNICODE)),
    )
    jaccard = len(wa & wb) / len(wa | wb) if (wa or wb) else 0.0
    return max(jaccard, difflib.SequenceMatcher(None, la, lb).ratio())


def looks_foreign(text: str, src: Lang) -> bool:
    """Не на том ли языке распознанное, на котором человек говорит.

    Whisper работает с зафиксированным языком, но если микрофон поймал остаток
    собственной английской озвучки, он честно запишет английские слова. Для
    ``ru``/``kk`` латиница в распознанном — почти наверняка эхо, а не речь.
    """
    stripped = text.strip()
    letters = [ch for ch in stripped if ch.isalpha()]
    if not letters:
        return False
    cyrillic = sum(1 for ch in letters if "\u0400" <= ch <= "\u04ff")
    share = cyrillic / len(letters)
    return share < 0.5 if src in (Lang.RU, Lang.KK) else share > 0.5


def looks_untranslated(text: str, dst: Lang) -> bool:
    """Похоже ли, что NLLB вернул исходную фразу вместо перевода.

    Дешёвая проверка по алфавиту: для ``en`` в ответе не должно быть кириллицы,
    для ``ru``/``kk`` — наоборот, она обязана быть. Так ловятся случаи, когда
    модель «переводит» обрывок фразы копированием.
    """
    stripped = text.strip()
    if not stripped:
        return False
    cyrillic = sum(1 for ch in stripped if "Ѐ" <= ch <= "ӿ")
    letters = sum(1 for ch in stripped if ch.isalpha())
    if letters == 0:
        return False
    share = cyrillic / letters
    return share > 0.5 if dst is Lang.EN else share < 0.5


async def measure_input(source: AudioSource, seconds: float = 1.5) -> tuple[float, float]:
    """Померить уровень входа: ``(RMS, пик)`` в долях полной шкалы.

    Тихий вход — самая частая причина, по которой whisper «выдумывает» слова:
    Silero VAD пропускает начала фраз, а модель достраивает их по догадке.
    """
    peak = 0.0
    total = 0.0
    frames = 0
    deadline = time.perf_counter() + seconds
    async for chunk in source:
        samples = chunk.samples.astype(np.float32) / 32768.0
        peak = max(peak, float(np.abs(samples).max(initial=0.0)))
        total += float(np.sum(samples * samples))
        frames += samples.size
        if time.perf_counter() >= deadline:
            break
    rms = (total / frames) ** 0.5 if frames else 0.0
    return rms, peak


def with_gain(source: AudioSource, gain: float) -> AsyncIterator[AudioChunk]:
    """Усилить поток перед распознаванием (тихий ноутбучный микрофон)."""

    async def gained() -> AsyncIterator[AudioChunk]:
        async for chunk in source:
            samples = chunk.samples.astype(np.float32) * gain
            np.clip(samples, -32768.0, 32767.0, out=samples)
            yield AudioChunk.from_array(samples.astype(np.int16), ts_ms=chunk.ts_ms, fmt=chunk.fmt)

    return gained()


def build_mic(config: AudioConfig, index: int | None) -> AudioSource:
    """Источник микрофона: по индексу устройства или подбором по имени.

    Windows показывает один микрофон под несколькими host API с почти
    одинаковыми именами, поэтому иногда нужен явный номер из
    ``python scripts/audio_smoke.py --list`` (так же, как в testdrive.py).
    """
    if index is None:
        return create_source("mic", config)

    from engine.audio_io.devices import list_devices
    from engine.audio_io.windows import MicSource

    device = next((d for d in list_devices(include_loopback=False) if d.id == index), None)
    if device is None:
        raise SystemExit(f"устройства с id={index} нет — см. python scripts/audio_smoke.py --list")
    return MicSource(
        device,
        fmt=config.format,
        chunk_ms=config.chunk_ms,
        timeout_s=config.device_timeout_s,
    )


@dataclass(slots=True)
class SpeechJob:
    """Переведённая фраза, ожидающая озвучки."""

    index: int
    source: str
    text: str
    t_end_ms: int
    stt_ms: int
    mt_ms: int
    queued_at: float


class Broadcaster:
    """Раздаёт конверты всем подключённым клиентам UI, не тормозя конвейер.

    ``publish`` никогда не ждёт сеть: у каждого клиента своя очередь и своя
    задача-отправитель. Если браузер не успевает (вкладка в фоне, панель
    перерисовывается), его очередь переполняется и самые старые события
    выбрасываются — так же, как это делает шина движка. Иначе ``await send``
    в цикле захвата встал бы вместе с браузером, а микрофон за это время
    переполнил бы очередь и потерял чанки.
    """

    __slots__ = ("_queues", "_senders", "dropped")

    def __init__(self) -> None:
        self._queues: dict[ServerConnection, asyncio.Queue[str]] = {}
        self._senders: dict[ServerConnection, asyncio.Task[None]] = {}
        #: Сколько событий выброшено из-за медленных клиентов.
        self.dropped = 0

    @property
    def clients(self) -> int:
        """Сколько UI сейчас подключено."""
        return len(self._queues)

    def add(self, connection: ServerConnection) -> None:
        """Запомнить нового клиента и завести ему отправителя."""
        queue: asyncio.Queue[str] = asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE)
        self._queues[connection] = queue
        self._senders[connection] = asyncio.create_task(
            self._pump(connection, queue), name="mic-subtitles-sender"
        )

    def discard(self, connection: ServerConnection) -> None:
        """Забыть отключившегося клиента."""
        self._queues.pop(connection, None)
        sender = self._senders.pop(connection, None)
        if sender is not None and not sender.done():
            sender.cancel()

    async def publish(self, envelope: Envelope) -> None:
        """Положить конверт в очереди всех клиентов (без ожидания сети)."""
        raw = to_json(envelope)
        for queue in list(self._queues.values()):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
                self.dropped += 1
            queue.put_nowait(raw)

    async def _pump(self, connection: ServerConnection, queue: asyncio.Queue[str]) -> None:
        """Отправлять события клиенту по одному, пока он жив."""
        try:
            while True:
                raw = await queue.get()
                await connection.send(raw)
        except asyncio.CancelledError:
            raise
        except Exception:  # клиент мог закрыться в любой момент
            self.discard(connection)


class LoopWatchdog:
    """Сторож цикла событий: меряет, насколько поздно просыпается задача.

    Если ``asyncio.sleep(0.1)`` просыпается через 0.5 с — цикл был занят
    чем-то синхронным 400 мс, и микрофонная очередь за это время росла.
    Раз в ``report_every_s`` печатает сводку, если было за что зацепиться.
    """

    __slots__ = (
        "_dead_reported",
        "_last_sound_at",
        "_max_lag_ms",
        "_report_every_s",
        "_stage_ms",
        "_task",
        "printer",
    )

    def __init__(self, printer: Any = print, report_every_s: float = 10.0) -> None:
        self.printer = printer
        self._report_every_s = report_every_s
        self._max_lag_ms = 0.0
        #: Максимальное время синхронных этапов за период, мс.
        self._stage_ms: dict[str, float] = {}
        self._task: asyncio.Task[None] | None = None
        #: Когда вход последний раз содержал хоть какой-то сигнал.
        self._last_sound_at: float | None = None
        self._dead_reported = False

    def note_input(self, peak: float) -> None:
        """Сообщить пик сырого чанка микрофона (до усиления и AEC)."""
        now = time.perf_counter()
        if peak >= DEAD_INPUT_PEAK:
            if self._dead_reported:
                self.printer("[сторож] микрофон снова отдаёт сигнал")
            self._last_sound_at = now
            self._dead_reported = False
        elif self._last_sound_at is None:
            self._last_sound_at = now

    @property
    def input_dead(self) -> bool:
        """Отдаёт ли микрофон цифровой ноль дольше ``DEAD_INPUT_S``."""
        if self._last_sound_at is None:
            return False
        return time.perf_counter() - self._last_sound_at >= DEAD_INPUT_S

    def note(self, stage: str, elapsed_ms: float) -> None:
        """Запомнить длительность синхронного этапа (AEC, ресемплинг…)."""
        if elapsed_ms > self._stage_ms.get(stage, 0.0):
            self._stage_ms[stage] = elapsed_ms

    def start(self) -> None:
        """Запустить сторожа."""
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="mic-subtitles-watchdog")

    def stop(self) -> None:
        """Остановить сторожа."""
        task, self._task = self._task, None
        if task is not None:
            task.cancel()

    async def _run(self) -> None:
        last_report = time.perf_counter()
        while True:
            before = time.perf_counter()
            await asyncio.sleep(WATCHDOG_TICK_S)
            lag_ms = (time.perf_counter() - before - WATCHDOG_TICK_S) * 1000
            self._max_lag_ms = max(self._max_lag_ms, lag_ms)
            if time.perf_counter() - last_report >= self._report_every_s:
                self._report()
                last_report = time.perf_counter()

    def _report(self) -> None:
        if self.input_dead and not self._dead_reported:
            self._dead_reported = True
            self.printer(
                "[сторож] МИКРОФОН ОТДАЁТ ЦИФРОВОЙ НОЛЬ уже "
                f"{DEAD_INPUT_S:.0f}+ с: проверьте клавишу выключения микрофона "
                "(Fn + перечёркнутый микрофон) и ползунок «Ввод» в настройках звука"
            )
        slow_stage = any(v >= STAGE_REPORT_MS for v in self._stage_ms.values())
        if self._max_lag_ms < LAG_REPORT_MS and not slow_stage:
            self._max_lag_ms = 0.0
            self._stage_ms.clear()
            return
        stages = ", ".join(f"{k} {v:.0f} мс" for k, v in sorted(self._stage_ms.items()))
        self.printer(
            f"[сторож] лаг цикла событий до {self._max_lag_ms:.0f} мс"
            + (f"; этапы: {stages}" if stages else "")
        )
        self._max_lag_ms = 0.0
        self._stage_ms.clear()


class HalfDuplexGate:
    """Глушит микрофон на то время, пока играет озвученный перевод.

    Без этого на колонках whisper слышит собственный синтезированный голос,
    распознаёт его как новую фразу и переводит её снова — получается, что
    программа разговаривает сама с собой.

    Окно глушения задаётся **в таймкодах потока** (``chunk.ts_ms``), а не по
    часам: whisper может отставать от микрофона на секунду-другую, и всё, что
    записано во время проигрывания, дойдёт до потребителя уже после того, как
    динамик замолчал. Таймкод присваивается в колбэке захвата, поэтому окно по
    ``ts_ms`` ловит именно ту запись, в которую попала озвучка. Чанк не
    выбрасывается, а заменяется тишиной — таймлайн потока остаётся непрерывным,
    а VAD видит паузу.
    """

    __slots__ = ("_from_ms", "_tail_ms", "_to_ms", "muted_chunks")

    def __init__(self, tail_ms: int = DEFAULT_TAIL_MS) -> None:
        self._tail_ms = max(0, tail_ms)
        self._from_ms: int | None = None
        self._to_ms: int | None = None
        #: Сколько чанков заменено тишиной (для отладки и тестов).
        self.muted_chunks = 0

    def mute_from(self, ts_ms: int) -> None:
        """Закрыть микрофон начиная с таймкода ``ts_ms`` и до :meth:`release_at`."""
        self._from_ms = ts_ms
        self._to_ms = None

    def release_at(self, ts_ms: int) -> None:
        """Озвучка кончилась на таймкоде ``ts_ms``: закрыть окно с запасом ``tail_ms``."""
        if self._from_ms is None:
            return
        self._to_ms = ts_ms + self._tail_ms

    def blocks(self, ts_ms: int) -> bool:
        """Попадает ли чанк с таймкодом ``ts_ms`` в окно озвучки."""
        if self._from_ms is None:
            return False
        if ts_ms < self._from_ms:
            return False
        return self._to_ms is None or ts_ms <= self._to_ms

    def wrap(self, source: Any) -> AsyncIterator[AudioChunk]:
        """Обернуть поток: записанное во время озвучки заменить тишиной."""

        async def gated() -> AsyncIterator[AudioChunk]:
            async for chunk in source:
                if self.blocks(chunk.ts_ms):
                    self.muted_chunks += 1
                    yield AudioChunk.from_array(
                        np.zeros(chunk.n_frames, dtype=np.int16), ts_ms=chunk.ts_ms, fmt=chunk.fmt
                    )
                else:
                    yield chunk

        return gated()


class MicSubtitles:
    """Захват микрофона → распознавание → перевод → озвучка, события в UI и консоль.

    Args:
        stt: распознаватель (свой экземпляр на процесс — состояние VAD внутри).
        translation: сервис перевода поверх провайдера NLLB.
        broadcaster: раздача событий в WebSocket.
        source_factory: чем открыть микрофон (вызывается на каждый запуск).
        lang_src: язык, на котором говорит пользователь.
        lang_dst: язык перевода.
        tts: синтез перевода; ``None`` — режим субтитров, ничего не звучит.
        sink_factory: куда играть перевод (наушники/колонки); нужен вместе с ``tts``.
        voice_id: профиль голоса; ``None`` — встроенный голос XTTS.
        gate: полудуплекс — глушит микрофон на время озвучки. С эхоподавлением
            не нужен: микрофон остаётся открытым, и речь поверх перевода не
            теряется.
        aec: эхоподавление — вычитает из микрофона собственную озвучку, чтобы
            работать на динамиках без наушников и без глушения входа.
        gain: усиление входа (1.0 — без изменений).
        skip_hallucinations: отбрасывать заученные «титры» whisper.
        stale_ms: фразу, дождавшуюся очереди позже этого, не озвучиваем —
            в панель она всё равно попадёт.
        printer: куда печатать ленту (по умолчанию ``print``).
    """

    def __init__(
        self,
        stt: SttEngine,
        translation: TranslationService,
        broadcaster: Broadcaster,
        source_factory: Any,
        lang_src: Lang,
        lang_dst: Lang,
        *,
        tts: TtsRouter | None = None,
        sink_factory: Any = None,
        voice_id: str | None = None,
        gate: HalfDuplexGate | None = None,
        aec: EchoCanceller | None = None,
        gain: float = 1.0,
        skip_hallucinations: bool = True,
        stale_ms: int = DEFAULT_STALE_MS,
        printer: Any = print,
    ) -> None:
        self._stt = stt
        self._translation = translation
        self._bus = broadcaster
        self._source_factory = source_factory
        self._lang_src = lang_src
        self._lang_dst = lang_dst
        self._tts = tts
        self._sink_factory = sink_factory
        self._voice_id = voice_id
        self._gate = gate
        self._aec = aec
        self._gain = gain
        self._skip_hallucinations = skip_hallucinations
        self._stale_ms = stale_ms
        self._print = printer
        self._queue: asyncio.Queue[Any] | None = None
        self._watchdog = LoopWatchdog(printer)
        self._task: asyncio.Task[None] | None = None
        self._source: AudioSource | None = None
        self._sink: AudioSink | None = None
        # Момент ``perf_counter``, соответствующий нулю таймлайна потока.
        # Ставится по первому чанку, а не по началу запуска: между ними
        # открывается устройство и лениво грузится Silero VAD (секунды), и
        # если считать от запуска, задержка фразы завышается на это время.
        self._t0: float | None = None
        #: Сколько фраз обработано с начала запуска.
        self.index = 0
        #: Сколько раз whisper поймал остаток собственной озвучки.
        self.echo_leaks = 0
        #: Последние озвученные переводы — эхо динамика узнаётся по тексту.
        self._spoken: deque[str] = deque(maxlen=SPOKEN_MEMORY)

    @property
    def speaking(self) -> bool:
        """Озвучивает ли режим перевод (или только пишет субтитры)."""
        return self._tts is not None and self._sink_factory is not None

    # --- состояние ---------------------------------------------------------

    @property
    def running(self) -> bool:
        """Идёт ли захват прямо сейчас."""
        return self._task is not None and not self._task.done()

    def state(self, status: SessionStatus | None = None) -> Envelope:
        """Конверт ``session.state``.

        Языки раскладываются по контракту: ``langs.out`` — язык пользователя
        (микрофон), ``langs.in`` — язык собеседника, он же язык перевода.
        """
        resolved = status or (SessionStatus.RUNNING if self.running else SessionStatus.IDLE)
        return Envelope.wrap(
            SessionState(
                status=resolved,
                session_id=None,
                langs=Langs(in_=self._lang_dst, out=self._lang_src),
                voice_id=None,
            )
        )

    # --- управление --------------------------------------------------------

    async def start(self, lang_src: Lang | None = None, lang_dst: Lang | None = None) -> None:
        """Запустить захват (идущий сначала останавливается).

        Промежуточный ``idle`` наружу не уходит: UI увидел бы «сессия
        остановлена» на долю секунды и зря почистил бы ленты.
        """
        await self._halt()
        if lang_src is not None:
            self._lang_src = lang_src
        if lang_dst is not None:
            self._lang_dst = lang_dst
        self._translation.reset(Stream.OUT)
        self._watchdog.start()
        self._task = asyncio.create_task(self._capture(), name="mic-subtitles")
        await self._bus.publish(self.state(SessionStatus.RUNNING))
        mode = "перевод озвучивается" if self.speaking else "только текст, без звука"
        self._print(
            f"\n=== Говорите ({self._lang_src.value} → {self._lang_dst.value}): {mode}. ==="
        )

    async def stop(self) -> None:
        """Остановить захват и сообщить UI ``idle``."""
        self._watchdog.stop()
        await self._halt()
        await self._bus.publish(self.state(SessionStatus.IDLE))

    async def _halt(self) -> None:
        """Остановить захват молча (без ``session.state``)."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await self._close_source()

    async def _close_source(self) -> None:
        source, self._source = self._source, None
        if source is not None:
            with contextlib.suppress(Exception):
                await source.close()
        sink, self._sink = self._sink, None
        if sink is not None:
            with contextlib.suppress(Exception):
                await sink.drain()
                await sink.close()

    # --- основной цикл -----------------------------------------------------

    async def _capture(self) -> None:
        """Читать микрофон и публиковать события распознавания."""
        queue: asyncio.Queue[tuple[Envelope, int, float] | None] = asyncio.Queue()
        self._queue = queue
        speech: asyncio.Queue[SpeechJob | None] = asyncio.Queue()
        worker = asyncio.create_task(self._worker(queue, speech), name="mic-subtitles-worker")
        speaker = asyncio.create_task(self._speak_worker(speech), name="mic-subtitles-speaker")
        self._t0 = None
        failed = False
        try:
            source = self._source_factory()
            self._source = source
            await source.open()
            if self.speaking:
                sink = self._sink_factory()
                await sink.open()
                self._sink = sink
            stream: Any = self._timed(source)
            if self._gain != 1.0:
                stream = with_gain(stream, self._gain)
            if self._gate is not None:
                stream = self._gate.wrap(stream)
            async for envelope in self._stt.run(stream, Stream.OUT, self._lang_src.value):
                if self._skip_hallucinations and self._is_noise(envelope):
                    # Артефакт whisper на шуме: в панель не показываем вовсе,
                    # иначе лента забита «субтитрами» и «ссссссс».
                    continue
                await self._bus.publish(envelope)
                if envelope.type == EVENT_STT_FINAL:
                    queue.put_nowait(
                        (envelope, int(self._stt.last_latency_ms or 0), time.perf_counter())
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            # Типичный случай — выдернули гарнитуру: устройство исчезло, поток
            # оборвался. Молча умирать нельзя: UI продолжал бы считать сессию
            # живой, а микрофон уже никто не слушает.
            logger.exception("захват прерван (устройство пропало?) — перезапускаю")
            failed = True
        finally:
            queue.put_nowait(None)
            with contextlib.suppress(asyncio.CancelledError):
                await worker
            speech.put_nowait(None)
            with contextlib.suppress(asyncio.CancelledError):
                await speaker
        if failed:
            await self._close_source()
            await asyncio.sleep(RESTART_DELAY_S)
            self._print("\n=== Источник звука пропал, переоткрываю микрофон… ===")
            self._task = asyncio.create_task(self._capture(), name="mic-subtitles")

    def _is_noise(self, envelope: Envelope) -> bool:
        """Похоже ли событие распознавания на артефакт, а не на речь."""
        text = getattr(envelope.payload, "text", None)
        if not isinstance(text, str):
            return False
        if is_garbage(text):
            if envelope.type == EVENT_STT_FINAL:
                logger.info("пропущен артефакт whisper: %r", text[:80])
            return True
        if looks_foreign(text, self._lang_src) or self._sounds_like_speaker(text):
            if envelope.type == EVENT_STT_FINAL:
                logger.info("похоже на эхо озвучки, пропущено: %r", text[:80])
                self.echo_leaks += 1
            return True
        return False

    def _sounds_like_speaker(self, text: str) -> bool:
        """Не наш ли это собственный перевод, вернувшийся через динамик.

        Работает для любой пары языков — в отличие от проверки по алфавиту,
        которая бессильна, когда оба языка кириллические (kk ↔ ru).
        """
        return any(similarity(text, spoken) >= ECHO_SIMILARITY for spoken in self._spoken)

    def _timed(self, source: Any) -> AsyncIterator[AudioChunk]:
        """Привязать таймлайн потока к часам процесса и мерить уровень входа.

        Уровень считается здесь, до полудуплекса: дальше по конвейеру чанк уже
        может быть заменён тишиной, а знать, говорит ли человек прямо сейчас,
        нужно именно во время озвучки.
        """

        async def timed() -> AsyncIterator[AudioChunk]:
            async for chunk in source:
                if self._t0 is None:
                    self._t0 = time.perf_counter() - chunk.ts_ms / 1000.0
                raw = chunk.samples
                self._watchdog.note_input(float(np.abs(raw).max()) / 32768.0 if raw.size else 0.0)
                if self._aec is None:
                    yield chunk
                    continue
                started = time.perf_counter()
                cleaned = self._aec.process(chunk.samples, chunk.ts_ms)
                self._watchdog.note("aec", (time.perf_counter() - started) * 1000)
                yield AudioChunk.from_array(cleaned, ts_ms=chunk.ts_ms, fmt=chunk.fmt)

        return timed()

    async def _worker(
        self,
        queue: asyncio.Queue[tuple[Envelope, int, float] | None],
        speech: asyncio.Queue[SpeechJob | None],
    ) -> None:
        """Переводит фразы по очереди: распознавание не ждёт NLLB, NLLB не ждёт TTS."""
        while True:
            item = await queue.get()
            if item is None:
                return
            try:
                job = await self._handle_final(*item)
                if job is not None:
                    speech.put_nowait(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("фраза пропущена")

    async def _speak_worker(self, speech: asyncio.Queue[SpeechJob | None]) -> None:
        """Озвучивает переводы по очереди и публикует метрики."""
        while True:
            job = await speech.get()
            if job is None:
                return
            try:
                await self._deliver(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("озвучка пропущена")

    async def _handle_final(
        self, envelope: Envelope, stt_ms: int, queued_at: float
    ) -> SpeechJob | None:
        payload = envelope.payload
        if not isinstance(payload, SttFinal):  # pragma: no cover — чужое событие
            return None

        self.index += 1
        ready = await self._translation.handle(envelope, self._lang_dst)
        await self._bus.publish(ready)

        translated = ready.payload
        text = translated.text if isinstance(translated, TranslationReady) else ""
        mt_ms = int(self._translation.last_latency_ms)

        self._print(f"\n[{self.index:02d}] whisper : {payload.text}")
        self._print(f"     в NLLB  : {payload.text}")
        self._print(f"     перевод : {text}")
        return SpeechJob(
            index=self.index,
            source=payload.text,
            text=text,
            t_end_ms=payload.t_end_ms,
            stt_ms=stt_ms,
            mt_ms=mt_ms,
            queued_at=queued_at,
        )

    async def _deliver(self, job: SpeechJob) -> None:
        """Озвучить перевод (если он ещё актуален) и опубликовать метрики."""
        # Сколько фраза прождала от распознавания до своей очереди на озвучку.
        waited_ms = round((time.perf_counter() - job.queued_at) * 1000)
        if waited_ms > self._stale_ms:
            # Отстаём: синтез идёт последовательно, и озвучка этой фразы
            # прозвучала бы через несколько секунд после сказанного. Текст в
            # панели уже есть — голос пропускаем, чтобы догнать разговор.
            logger.info("фраза ждала %d мс — не озвучиваю, только текст", waited_ms)
            tts_ms, spoken_s, first_at = 0, 0.0, None
        else:
            tts_ms, spoken_s, first_at = await self._speak(job.text, job.source)
        # Бюджет ARCHITECTURE.md раздела 2 — «конец фразы → озвучка», поэтому
        # засекаем момент ПЕРВОГО чанка синтеза, а не конца проигрывания: пока
        # перевод дочитывается, задержка уже не растёт (так же в pipeline.py).
        ready_at = first_at if first_at is not None else time.perf_counter()
        elapsed_ms = round((ready_at - (self._t0 or ready_at)) * 1000)
        total_ms = max(0, elapsed_ms - job.t_end_ms)
        stt_ms, mt_ms = job.stt_ms, job.mt_ms

        await self._bus.publish(
            Envelope.wrap(
                MetricsLatency(
                    stream=Stream.OUT,
                    stt_ms=max(0, stt_ms),
                    mt_ms=max(0, mt_ms),
                    tts_ms=max(0, tts_ms),
                    total_ms=total_ms,
                )
            )
        )

        voiced = f", озвучено {spoken_s:.1f} с" if spoken_s > 0 else ""
        if self._aec is not None and self._aec.stats.adapted > 0:
            st = self._aec.stats
            erle = st.erle_db
            voiced += f", эхо {erle:+.0f} дБ" if erle < 3 else f", эхо подавлено на {erle:.0f} дБ"
            voiced += f" [блоков громче: {st.rejected}, сжатий весов: {st.leaked}]"
            lag_ms, corr = self._aec.estimate_delay()
            voiced += f", реальная задержка эха {lag_ms:+d} мс (корр {corr:.2f})"
        self._print(
            f"     [{job.index:02d}]  (stt {stt_ms} мс, mt {mt_ms} мс, tts {tts_ms} мс, "
            f"итого {total_ms} мс{voiced})"
        )

    async def _speak(self, text: str, source: str = "") -> tuple[int, float, float | None]:
        """Озвучить перевод в приёмник.

        Микрофон на это время глушится полудуплексом, иначе whisper распознает
        собственный синтезированный голос.

        Returns:
            ``(мс до первого чанка, длительность озвучки в секундах, момент
            первого чанка по ``perf_counter``)``. Для режима без озвучки и для
            пропущенной фразы — ``(0, 0.0, None)``.
        """
        sink = self._sink
        if self._tts is None or sink is None or not text.strip():
            return 0, 0.0, None
        if source and text.strip().lower() == source.strip().lower():
            logger.info("перевод совпал с исходником, не озвучиваю: %r", text)
            return 0, 0.0, None
        if is_garbage(text) or len(text) > MAX_SPOKEN_CHARS:
            # NLLB честно размножил повторы whisper: читать это полминуты
            # незачем, а микрофон всё это время слушал бы динамик.
            logger.info("перевод похож на мусор (%d знаков), не озвучиваю", len(text))
            return 0, 0.0, None
        if looks_untranslated(text, self._lang_dst):
            # NLLB иногда возвращает исходную фразу как есть (обрывок, шум).
            # Озвучивать её незачем: в панели она видна, а в динамик уйдёт
            # русский текст «английским» голосом, и микрофон услышит сам себя.
            logger.info("перевод не похож на %s, не озвучиваю: %r", self._lang_dst.value, text)
            return 0, 0.0, None

        if self._gate is not None:
            self._gate.mute_from(self._stream_ms())
        self._spoken.append(text)

        started = time.perf_counter()
        first_ms = 0
        first_at: float | None = None
        frames = 0
        try:
            async for pcm in self._tts.synthesize(text, self._lang_dst, self._voice_id):
                if first_at is None:
                    first_at = time.perf_counter()
                    first_ms = max(0, round((first_at - started) * 1000))
                chunk = AudioChunk.from_array(pcm, ts_ms=0, fmt=TTS_FORMAT)
                frames += chunk.n_frames
                self._note_reference(chunk, sink)
                await sink.write(chunk)
            with contextlib.suppress(Exception):
                await sink.drain()
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("синтез не удался, фраза не озвучена")
        finally:
            if self._gate is not None:
                # drain() возвращается, когда динамик реально отыграл, поэтому
                # окно закрывается здесь, а хвост добавляет сам gate.
                self._gate.release_at(self._stream_ms())

        return first_ms, (frames / TTS_FORMAT.sample_rate if frames else 0.0), first_at

    def _note_reference(self, chunk: AudioChunk, sink: AudioSink) -> None:
        """Сообщить эхоподавителю, что и когда зазвучит в динамике.

        ``play_at`` намеренно занижен: берём «сейчас» плюс то, что ещё лежит в
        буфере приёмника. Раньше этого момента чанк зазвучать не может, а
        всё, что позже (латентность драйвера, путь до микрофона), — уже забота
        адаптивного фильтра.
        """
        aec = self._aec
        if aec is None:
            return
        buffered = int(getattr(sink, "buffered_frames", 0) or 0)
        device_rate = int(getattr(sink, "device_sample_rate", 0) or TTS_FORMAT.sample_rate)
        play_at_ms = self._stream_ms() + 1000.0 * buffered / device_rate
        started = time.perf_counter()
        pcm16k = resample(chunk.samples, chunk.fmt.sample_rate, ENGINE_RATE)
        aec.add_reference(pcm16k, play_at_ms)
        self._watchdog.note("reference", (time.perf_counter() - started) * 1000)

    def _stream_ms(self) -> int:
        """Текущая точка на таймлайне потока, мс (0 — первый чанк микрофона)."""
        if self._t0 is None:
            return 0
        return max(0, round((time.perf_counter() - self._t0) * 1000))

    # --- WebSocket ---------------------------------------------------------

    async def handle_client(self, connection: ServerConnection) -> None:
        """Обслужить подключение UI: отдать состояние и слушать команды."""
        self._bus.add(connection)
        with contextlib.suppress(Exception):
            await connection.send(to_json(self.state()))
        try:
            async for raw in connection:
                await self.on_message(raw)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._bus.discard(connection)

    async def on_message(self, raw: str | bytes) -> None:
        """Разобрать команду UI. Кривое сообщение только пишется в лог."""
        try:
            envelope = parse_event(raw)
        except ContractError as exc:
            logger.warning("некорректная команда UI: %s", exc)
            return

        if envelope.type == COMMAND_SESSION_START and isinstance(envelope.payload, SessionStart):
            command = envelope.payload
            # Контракт: lang_out — язык пользователя (микрофон), lang_in — собеседника.
            await self.start(command.lang_out, command.lang_in)
        elif envelope.type == COMMAND_SESSION_STOP:
            await self.stop()
            self._print("\n=== Остановлено. Start в UI — запустить снова. ===")


def build_parser() -> argparse.ArgumentParser:
    """Аргументы командной строки."""
    parser = argparse.ArgumentParser(
        prog="python scripts/mic_subtitles.py",
        description="Моя речь: микрофон → whisper → NLLB → озвучка перевода + панель в UI",
    )
    parser.add_argument("--from", dest="lang_src", default="ru", choices=[x.value for x in Lang])
    parser.add_argument("--to", dest="lang_dst", default="en", choices=[x.value for x in Lang])
    parser.add_argument("--host", default=DEFAULT_WS_HOST, help="адрес WebSocket для UI")
    parser.add_argument("--port", type=int, default=DEFAULT_WS_PORT, help="порт WebSocket для UI")
    parser.add_argument("--mic-index", type=int, default=None, help="номер устройства микрофона")
    parser.add_argument("--gain", type=float, default=1.0, help="усиление входа, разы")
    parser.add_argument(
        "--preroll-ms",
        type=int,
        default=DEFAULT_PREROLL_MS,
        help="сколько аудио до начала речи прихватывать в фразу (спасает первое слово)",
    )
    parser.add_argument(
        "--min-speech-ms",
        type=int,
        default=DEFAULT_MIN_SPEECH_MS,
        help="короче этого речью не считается (отсекает щелчки и шипение)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=DEFAULT_VAD_THRESHOLD,
        help="порог Silero VAD: выше — меньше шума принимается за речь",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=DEFAULT_BEAM_SIZE,
        help="beam search whisper: больше — точнее и чуть медленнее",
    )
    parser.add_argument(
        "--silence-ms",
        type=int,
        default=DEFAULT_SILENCE_MS,
        help="пауза, закрывающая фразу, мс",
    )
    parser.add_argument(
        "--max-phrase-ms",
        type=int,
        default=DEFAULT_MAX_PHRASE_MS,
        help="предельная длина фразы, мс (при непрерывной речи задаёт темп перевода)",
    )
    parser.add_argument(
        "--keep-hallucinations",
        action="store_true",
        help="не отбрасывать заученные «титры» whisper",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="не озвучивать перевод (режим субтитров: XTTS не загружается вовсе)",
    )
    parser.add_argument("--voice", dest="voice_id", default=None, help="voice_id профиля голоса")
    parser.add_argument(
        "--tts-chunk-size",
        type=int,
        default=DEFAULT_TTS_CHUNK_SIZE,
        help="stream_chunk_size XTTS: меньше — раньше первый звук, но мельче куски",
    )
    parser.add_argument(
        "--stale-ms",
        type=int,
        default=DEFAULT_STALE_MS,
        help="фразу, прождавшую в очереди дольше этого, не озвучивать (текст остаётся)",
    )
    parser.add_argument(
        "--tail-ms",
        type=int,
        default=DEFAULT_TAIL_MS,
        help="сколько держать микрофон закрытым после озвучки, мс",
    )
    parser.add_argument(
        "--half-duplex",
        action="store_true",
        help="вместо эхоподавления глушить микрофон на время озвучки "
        "(речь поверх перевода при этом теряется)",
    )
    parser.add_argument(
        "--no-aec",
        dest="aec",
        action="store_false",
        help="выключить эхоподавление (нужно, только если оно мешает)",
    )
    parser.add_argument("--log-level", default="WARNING")
    return parser


async def _amain(args: argparse.Namespace) -> int:
    lang_src, lang_dst = Lang(args.lang_src), Lang(args.lang_dst)

    stt_config = replace(
        SttConfig.from_env(),
        min_silence_ms=args.silence_ms,
        max_utterance_ms=args.max_phrase_ms,
        language=lang_src.value,
        beam_size=args.beam_size,
        vad_threshold=args.vad_threshold,
        preroll_ms=args.preroll_ms,
        min_speech_ms=args.min_speech_ms,
        partial_interval_ms=DEFAULT_PARTIAL_MS,
    )
    print("Загружаю модели (первый запуск — дольше, веса читаются с диска)…")
    stt = create_engine(stt_config, backend="faster_whisper", vad_backend="silero")
    provider = create_provider(TranslateConfig.from_env())

    step = time.perf_counter()
    await asyncio.to_thread(
        stt.transcriber.transcribe, np.zeros(16_000, dtype=np.int16), lang_src.value
    )
    print(f"  whisper small готов — {time.perf_counter() - step:.1f} с")
    step = time.perf_counter()
    await asyncio.to_thread(provider.warmup)
    print(f"  NLLB-200 (CPU) готов — {time.perf_counter() - step:.1f} с")

    tts: TtsRouter | None = None
    if not args.silent:
        # Порядок как в ARCHITECTURE.md разделе 6: whisper уже в памяти, XTTS встаёт следом.
        tts = create_tts(replace(TtsConfig.from_env(), stream_chunk_size=args.tts_chunk_size))
        step = time.perf_counter()
        await tts.warmup((lang_dst,))
        print(f"  XTTS-v2 готов — {time.perf_counter() - step:.1f} с")
    else:
        print("  XTTS не загружается: --silent, только субтитры")

    audio_config = AudioConfig.from_env()

    # Проверка уровня до старта: с тихим микрофоном whisper выдумывает слова,
    # и это выглядит как «программа несёт чушь», хотя дело во входе.
    probe = build_mic(audio_config, args.mic_index)
    try:
        print("\nПроверка микрофона: скажите любую фразу (3 секунды)…")
        async with probe:
            rms, peak = await measure_input(probe, seconds=3.0)
        print(f"  уровень: RMS {rms:.3f}, пик {peak:.3f} (усиление x{args.gain:g})")
        if peak * args.gain < QUIET_MIC_PEAK:
            advised = min(10.0, max(2.0, round(QUIET_MIC_PEAK / max(peak, 1e-3), 1)))
            print("  ВХОД СЛИШКОМ ТИХИЙ — на таком сигнале whisper выдумывает слова.")
            print("  Что помогает, по убыванию эффекта:")
            print("    1) гарнитура вместо массива микрофонов в крышке ноутбука;")
            print("    2) уровень микрофона на 100% в «Параметры → Звук → Ввод»;")
            print(f"    3) перезапуск с --gain {advised:g}")
    except Exception:
        logger.exception("не удалось померить уровень микрофона")

    broadcaster = Broadcaster()
    speaking = tts is not None
    gate = HalfDuplexGate(args.tail_ms) if (speaking and args.half_duplex) else None
    # Эхоподавление и полудуплекс решают одну задачу; вместе не нужны.
    canceller = EchoCanceller(ENGINE_RATE) if (speaking and args.aec and gate is None) else None
    app = MicSubtitles(
        stt,
        TranslationService(provider),
        broadcaster,
        lambda: build_mic(audio_config, args.mic_index),
        lang_src,
        lang_dst,
        tts=tts,
        sink_factory=(None if tts is None else lambda: create_sink("headphones", audio_config)),
        voice_id=args.voice_id,
        gate=gate,
        aec=canceller,
        gain=args.gain,
        skip_hallucinations=not args.keep_hallucinations,
        stale_ms=args.stale_ms,
    )

    async with ws_serve(app.handle_client, args.host, args.port):
        print(f"\nWebSocket для UI: ws://{args.host}:{args.port}")
        print("Панель: http://localhost:3000/pipeline  (Ctrl+C — выход)")
        if tts is not None:
            voice = args.voice_id or "XTTS по умолчанию"
            print(f"Перевод звучит в наушники/колонки, голос: {voice}")
            if canceller is not None:
                print(
                    f"Эхоподавление включено (хвост {canceller.tail_ms} мс): микрофон слушает "
                    "и во время озвучки, говорить можно поверх"
                )
            elif gate is not None:
                print("Полудуплекс: микрофон молчит, пока играет перевод")
        await app.start()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            pass
        finally:
            await app.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Точка входа скрипта."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # Эти двое печатают строку на каждый чанк и на каждое подключение UI —
    # в терминале от них ничего не видно.
    for noisy in ("faster_whisper", "websockets.server", "websockets.client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\nостановлено")
        return 0


if __name__ == "__main__":
    sys.exit(main())
