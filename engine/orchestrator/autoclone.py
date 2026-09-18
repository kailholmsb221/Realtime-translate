"""Автоклон голоса собеседника (ARCHITECTURE.md раздел 1: «клон голоса говорящего»).

Голос пользователя задаётся заранее (``voice_id`` в ``session.start``, профиль
создаётся через ``POST /api/voices`` или ``scripts/tts_smoke.py``), а вот голоса
собеседника до звонка нет. :class:`AutoCloner` собирает его сам:

1. inbound-конвейер отдаёт сюда каждый чанк потока ``in`` (окно последних
   ``window_seconds``);
2. по каждому ``stt.final`` из окна вырезается **речевой** участок
   ``[t_start_ms, t_end_ms]`` — тишина и шум в сэмпл не попадают;
3. как только накоплено ``min_seconds`` речи, в фоне (``asyncio.to_thread``)
   создаётся профиль в :class:`~engine.tts.voices.VoiceStore` (латенты XTTS —
   через ``latents_fn``), строка пишется в таблицу ``voices``;
4. конвейер начинает синтезировать перевод этим голосом; до готовности звучит
   встроенный голос XTTS.

Клон для казахского не создаётся: XTTS работает только с ru/en, а kk-провайдер
``voice_id`` игнорирует (engine/tts/README.md).

**Правовая заметка (ARCHITECTURE.md раздел 8).** Голос — биометрический
признак. Клонировать голос собеседника допустимо только с его согласия;
уведомить его — обязанность пользователя. Поэтому автопрофили по умолчанию
удаляются вместе с сессией (``RT_AUTO_CLONE_KEEP=0``), а выключается механизм
целиком переменной ``RT_AUTO_CLONE=0``.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import tempfile
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

import numpy as np

from engine.audio_io import AudioChunk
from engine.contracts.events import Lang
from engine.tts.audio import write_wav
from engine.tts.voices import (
    MAX_SAMPLE_SECONDS,
    MIN_SAMPLE_SECONDS,
    LatentsFn,
    VoiceError,
    VoiceProfile,
    VoiceStore,
)

if TYPE_CHECKING:  # избегаем цикла импортов (db -> config -> ... )
    from engine.orchestrator.db import Database

__all__ = [
    "AUTO_NAME_PREFIX",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_MIN_SECONDS",
    "DEFAULT_WAIT_SECONDS",
    "AutoCloneConfig",
    "AutoCloner",
]

logger: Final[logging.Logger] = logging.getLogger("engine.orchestrator.autoclone")

#: Частота потока движка (CLAUDE.md, технический стандарт).
SAMPLE_RATE: Final[int] = 16_000

#: Сколько речи копим до создания профиля и сколько максимум кладём в сэмпл.
DEFAULT_MIN_SECONDS: Final[float] = 12.0
DEFAULT_MAX_SECONDS: Final[float] = 30.0

#: Окно потока, из которого вырезаются речевые участки, с.
DEFAULT_WINDOW_SECONDS: Final[float] = 60.0

#: Сколько ждать фонового создания профиля при остановке сессии, с.
DEFAULT_WAIT_SECONDS: Final[float] = 15.0

#: Префикс имени автопрофиля — по нему видно, что голос собран движком,
#: а не заведён пользователем (схему БД менять нельзя, CLAUDE.md закон 2).
AUTO_NAME_PREFIX: Final[str] = "auto_"

#: Языки, для которых XTTS умеет клонировать (engine/tts/router.py).
CLONING_LANGS: Final[frozenset[Lang]] = frozenset({Lang.RU, Lang.EN})

ENV_ENABLED: Final[str] = "RT_AUTO_CLONE"
ENV_MIN_S: Final[str] = "RT_AUTO_CLONE_MIN_S"
ENV_MAX_S: Final[str] = "RT_AUTO_CLONE_MAX_S"
ENV_KEEP: Final[str] = "RT_AUTO_CLONE_KEEP"

_TRUE: Final[frozenset[str]] = frozenset({"1", "true", "yes", "on"})
_FALSE: Final[frozenset[str]] = frozenset({"0", "false", "no", "off"})


def _env_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = (env.get(key) or "").strip().lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{key}={raw!r}: ожидалось булево значение")


def _env_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = (env.get(key) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{key}={raw!r}: ожидалось число") from exc


@dataclass(frozen=True, slots=True)
class AutoCloneConfig:
    """Настройки автоклона голоса собеседника.

    Attributes:
        enabled: включён ли механизм (``RT_AUTO_CLONE``).
        min_seconds: сколько чистой речи накопить до создания профиля.
            Не меньше ``VoiceStore`` -овского минимума в 5 секунд.
        max_seconds: сколько речи максимум класть в сэмпл (не больше 60 с).
        keep: оставлять ли автопрофиль на диске после ``session.stop``.
            По умолчанию ``False`` — чужие голоса не копятся.
        window_seconds: окно потока, из которого вырезаются речевые участки.
    """

    enabled: bool = True
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_seconds: float = DEFAULT_MAX_SECONDS
    keep: bool = False
    window_seconds: float = DEFAULT_WINDOW_SECONDS

    def __post_init__(self) -> None:
        if self.min_seconds < MIN_SAMPLE_SECONDS:
            raise ValueError(
                f"min_seconds={self.min_seconds}: сэмпл короче {MIN_SAMPLE_SECONDS:.0f} с "
                "не принимает VoiceStore (engine/tts/voices.py)"
            )
        if self.max_seconds > MAX_SAMPLE_SECONDS:
            raise ValueError(
                f"max_seconds={self.max_seconds}: длиннее {MAX_SAMPLE_SECONDS:.0f} с "
                "не принимает VoiceStore"
            )
        if self.max_seconds < self.min_seconds:
            raise ValueError("max_seconds не может быть меньше min_seconds")
        if self.window_seconds < self.max_seconds:
            raise ValueError("window_seconds должен покрывать max_seconds")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AutoCloneConfig:
        """Собрать настройки из ``RT_AUTO_CLONE*``."""
        source: Mapping[str, str] = os.environ if env is None else env
        min_seconds = _env_float(source, ENV_MIN_S, DEFAULT_MIN_SECONDS)
        max_seconds = _env_float(source, ENV_MAX_S, DEFAULT_MAX_SECONDS)
        return cls(
            enabled=_env_bool(source, ENV_ENABLED, True),
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            keep=_env_bool(source, ENV_KEEP, False),
            window_seconds=max(DEFAULT_WINDOW_SECONDS, max_seconds),
        )


def supports_auto_clone(lang_src: Lang, lang_dst: Lang) -> bool:
    """Можно ли клонировать голос в этой паре языков.

    Нужны оба условия: сэмпл на языке, который понимает XTTS (``lang_src``),
    и синтез на таком же языке (``lang_dst``) — казахский провайдер клон не
    использует вовсе.
    """
    return lang_src in CLONING_LANGS and lang_dst in CLONING_LANGS


class AutoCloner:
    """Накапливает речь собеседника и один раз создаёт из неё голосовой профиль.

    Args:
        config: настройки автоклона.
        voices: хранилище профилей на диске (источник правды).
        lang: язык сэмпла — он же ``voices.lang`` (язык собеседника).
        session_id: id сессии, попадает в имя профиля.
        db: база движка; профиль дублируется строкой в таблице ``voices``.
        latents_fn: чем считать латенты XTTS (``XttsProvider.compute_latents``);
            ``None`` — латенты посчитаются при первом синтезе (и так работает
            фейковый бэкенд).
        stream: имя потока для имени профиля (``in`` — собеседник).
    """

    __slots__ = (
        "_collected",
        "_collected_samples",
        "_config",
        "_db",
        "_failed",
        "_lang",
        "_latents_fn",
        "_session_id",
        "_stream",
        "_task",
        "_voices",
        "_window",
        "_window_samples",
        "profile",
    )

    def __init__(
        self,
        config: AutoCloneConfig,
        voices: VoiceStore,
        lang: Lang,
        session_id: int | None = None,
        db: Database | None = None,
        latents_fn: LatentsFn | None = None,
        stream: str = "in",
    ) -> None:
        self._config = config
        self._voices = voices
        self._lang = lang
        self._session_id = session_id
        self._db = db
        self._latents_fn = latents_fn
        self._stream = stream
        self._window: deque[tuple[int, np.ndarray]] = deque()
        self._window_samples = 0
        self._collected: list[np.ndarray] = []
        self._collected_samples = 0
        self._task: asyncio.Task[None] | None = None
        self._failed = False
        #: Созданный профиль или ``None``, пока он не готов.
        self.profile: VoiceProfile | None = None

    # --- состояние ---------------------------------------------------------

    @property
    def voice_id(self) -> str | None:
        """``voice_id`` готового автопрофиля или ``None``."""
        return self.profile.id if self.profile is not None else None

    @property
    def voices(self) -> VoiceStore:
        """Хранилище профилей, в котором создаётся автоклон."""
        return self._voices

    @property
    def collected_seconds(self) -> float:
        """Сколько чистой речи уже накоплено, с."""
        return self._collected_samples / SAMPLE_RATE

    @property
    def started(self) -> bool:
        """Запущено ли (или уже завершено) создание профиля."""
        return self._task is not None

    @property
    def name(self) -> str:
        """Имя профиля: по префиксу видно, что голос собран автоматически."""
        session = self._session_id if self._session_id is not None else "file"
        return f"{AUTO_NAME_PREFIX}session{session}_{self._stream}"

    # --- накопление --------------------------------------------------------

    def feed(self, chunk: AudioChunk) -> None:
        """Положить чанк потока в окно, из которого вырезается речь."""
        if not self._config.enabled or self._failed or self.started:
            return
        self._window.append((chunk.ts_ms, chunk.samples))
        self._window_samples += chunk.n_frames
        limit = int(self._config.window_seconds * SAMPLE_RATE)
        while self._window_samples > limit and len(self._window) > 1:
            _, dropped = self._window.popleft()
            self._window_samples -= int(dropped.size)

    def note_utterance(self, t_start_ms: int, t_end_ms: int) -> None:
        """Забрать речевой участок фразы (по таймкодам ``stt.final``)."""
        if not self._config.enabled or self._failed or self.started:
            return
        speech = self._slice(t_start_ms, t_end_ms)
        if speech.size == 0:
            return
        room = int(self._config.max_seconds * SAMPLE_RATE) - self._collected_samples
        if room <= 0:
            return
        piece = speech[:room]
        self._collected.append(piece)
        self._collected_samples += int(piece.size)

    def _slice(self, t_start_ms: int, t_end_ms: int) -> np.ndarray:
        """Вырезать из окна участок ``[t_start_ms, t_end_ms)``."""
        if t_end_ms <= t_start_ms:
            return np.zeros(0, dtype=np.int16)
        parts: list[np.ndarray] = []
        for ts_ms, samples in self._window:
            chunk_start = ts_ms
            chunk_end = ts_ms + int(samples.size * 1000 // SAMPLE_RATE)
            if chunk_end <= t_start_ms or chunk_start >= t_end_ms:
                continue
            begin = max(0, (t_start_ms - chunk_start) * SAMPLE_RATE // 1000)
            end = min(samples.size, (t_end_ms - chunk_start) * SAMPLE_RATE // 1000)
            if end > begin:
                parts.append(samples[begin:end])
        if not parts:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(parts).astype(np.int16)

    # --- создание профиля --------------------------------------------------

    @property
    def ready_to_create(self) -> bool:
        """Накоплено ли достаточно речи для профиля."""
        if not self._config.enabled or self._failed or self.started:
            return False
        return self.collected_seconds >= self._config.min_seconds

    def maybe_start(self) -> bool:
        """Запустить фоновое создание профиля, если речи уже достаточно.

        Returns:
            ``True``, если задача создана именно сейчас.
        """
        if not self.ready_to_create:
            return False
        self._task = asyncio.create_task(self._create(), name="autoclone")
        return True

    async def _create(self) -> None:
        """Создать профиль в отдельном потоке и записать строку в БД."""
        seconds = self.collected_seconds
        logger.info(
            "накоплено %.1f с речи собеседника — создаю автопрофиль голоса (%s)",
            seconds,
            self._lang.value,
        )
        try:
            profile = await asyncio.to_thread(self._create_sync)
        except VoiceError as exc:
            self._failed = True
            logger.warning("автоклон голоса не удался: %s", exc)
            return
        except Exception:
            self._failed = True
            logger.exception("автоклон голоса упал")
            return

        self.profile = profile
        self._window.clear()
        self._window_samples = 0
        self._collected.clear()
        logger.info(
            "автопрофиль голоса собеседника готов: %s (%s, %.1f с)",
            profile.id,
            profile.name,
            seconds,
        )
        if self._db is not None:
            try:
                await self._db.upsert_voice(
                    profile.id,
                    profile.name,
                    profile.lang,
                    profile.sample_path,
                    profile.created_at_ms,
                )
            except Exception:
                logger.exception("автопрофиль %s не попал в таблицу voices", profile.id)

    def _create_sync(self) -> VoiceProfile:
        """Собрать WAV из накопленной речи и отдать его ``VoiceStore``."""
        samples = np.concatenate(self._collected).astype(np.int16)
        limit = int(self._config.max_seconds * SAMPLE_RATE)
        samples = samples[:limit]
        with tempfile.TemporaryDirectory(prefix="rt-autoclone-") as tmp:
            sample_path = Path(tmp) / "sample.wav"
            write_wav(sample_path, samples, SAMPLE_RATE)
            return self._voices.create_voice(
                sample_path,
                name=self.name,
                lang=self._lang,
                latents_fn=self._latents_fn,
            )

    async def wait(self, timeout: float = DEFAULT_WAIT_SECONDS) -> None:
        """Дождаться фонового создания профиля (остановка сессии, тесты).

        Ждём ограниченное время, чтобы ``session.stop`` не висел из-за
        затянувшегося клонирования: по таймауту задача отменяется. Отменить
        уже начатый ``to_thread`` нельзя, поэтому профиль может дописаться на
        диск после остановки — узнать его можно по префиксу ``auto_``.
        """
        task = self._task
        if task is None or task.done():
            return
        with contextlib.suppress(TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        if not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    # --- уборка ------------------------------------------------------------

    async def cleanup(self) -> None:
        """Удалить автопрофиль, если его не просили сохранять.

        Правовая заметка (ARCHITECTURE.md раздел 8): чужой голос не должен
        оставаться на диске дольше разговора, если пользователь не попросил
        обратного (``RT_AUTO_CLONE_KEEP=1``).
        """
        await self.wait()
        profile = self.profile
        if profile is None:
            return
        if self._config.keep:
            logger.info("автопрофиль %s сохранён (RT_AUTO_CLONE_KEEP=1)", profile.id)
            return
        try:
            await asyncio.to_thread(self._voices.delete, profile.id)
        except VoiceError as exc:
            logger.warning("не удалось удалить автопрофиль %s: %s", profile.id, exc)
        if self._db is not None:
            with contextlib.suppress(Exception):
                await self._db.delete_voice(profile.id)
        logger.info("автопрофиль %s удалён вместе с сессией", profile.id)
        self.profile = None
