"""Профили голосов для клонирования XTTS-v2 (ARCHITECTURE.md 4.4).

Хранилище на диске: один каталог на голос::

    <voices_dir>/<voice_id>/
        meta.json    id, name, lang, sample_path, created_at_ms
        sample.wav   нормализованный сэмпл: mono, 24 kHz, int16
        latents.pt   кэш латентов XTTS (создаёт engine.tts.xtts, опционален)

Структура профиля 1:1 совпадает с таблицей ``voices`` из ``db/schema.sql``
(``id`` TEXT = ``voice_id`` из контрактов, ``created_at_ms`` -> ``created_at``);
писать в SQLite — работа orchestrator, этот модуль про диск.

Правовая заметка (ARCHITECTURE.md раздел 8): клонировать можно только
собственный голос пользователя или голос с явного согласия владельца.
"""

from __future__ import annotations

import json
import logging
import secrets
import shutil
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from engine.contracts.events import Lang
from engine.tts.audio import Pcm16, read_wav, resample_pcm, write_wav
from engine.tts.base import OUTPUT_SAMPLE_RATE

__all__ = [
    "LATENTS_FILENAME",
    "MAX_SAMPLE_SECONDS",
    "META_FILENAME",
    "MIN_SAMPLE_SECONDS",
    "RECOMMENDED_SAMPLE_SECONDS",
    "SAMPLE_FILENAME",
    "LatentsFn",
    "VoiceError",
    "VoiceNotFoundError",
    "VoiceProfile",
    "VoiceStore",
]

log: Final[logging.Logger] = logging.getLogger(__name__)

META_FILENAME: Final[str] = "meta.json"
SAMPLE_FILENAME: Final[str] = "sample.wav"
LATENTS_FILENAME: Final[str] = "latents.pt"

#: Границы длительности сэмпла (ARCHITECTURE.md 4.4: клон по 15-30 сек).
MIN_SAMPLE_SECONDS: Final[float] = 5.0
RECOMMENDED_SAMPLE_SECONDS: Final[float] = 15.0
MAX_SAMPLE_SECONDS: Final[float] = 60.0

VOICE_ID_PREFIX: Final[str] = "v_"
VOICE_ID_HEX: Final[int] = 8

#: Функция вычисления латентов: ``(sample_wav, voice_dir) -> что угодно``.
#: Результат игнорируется, побочный эффект — файл ``latents.pt`` в ``voice_dir``.
LatentsFn = Callable[[Path, Path], Any]


class VoiceError(ValueError):
    """Некорректный сэмпл, дубликат имени или битый профиль голоса."""


class VoiceNotFoundError(VoiceError):
    """Профиль с таким ``voice_id`` не найден."""


def _new_voice_id() -> str:
    """Сгенерировать ``voice_id`` вида ``v_1a2b3c4d`` (8 hex)."""
    return f"{VOICE_ID_PREFIX}{secrets.token_hex(VOICE_ID_HEX // 2)}"


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """Профиль голоса. Поля совпадают со строкой таблицы ``voices``."""

    id: str
    name: str
    lang: Lang
    sample_path: Path
    created_at_ms: int

    @property
    def dir(self) -> Path:
        """Каталог профиля на диске."""
        return self.sample_path.parent

    @property
    def latents_path(self) -> Path:
        """Путь к кэшу латентов XTTS (может не существовать)."""
        return self.dir / LATENTS_FILENAME

    def to_meta(self) -> dict[str, Any]:
        """JSON-совместимый ``meta.json`` (``sample_path`` — относительный)."""
        return {
            "id": self.id,
            "name": self.name,
            "lang": self.lang.value,
            "sample_path": self.sample_path.name,
            "created_at_ms": self.created_at_ms,
        }

    def to_row(self) -> dict[str, Any]:
        """Строка для таблицы ``voices`` (для orchestrator; путь абсолютный)."""
        return {
            "id": self.id,
            "name": self.name,
            "lang": self.lang.value,
            "sample_path": str(self.sample_path),
            "created_at": self.created_at_ms,
        }

    @classmethod
    def from_meta(cls, data: dict[str, Any], voice_dir: Path) -> VoiceProfile:
        """Собрать профиль из ``meta.json``, лежащего в ``voice_dir``.

        Raises:
            VoiceError: в meta.json нет обязательного поля или язык неизвестен.
        """
        try:
            sample = Path(data["sample_path"])
            return cls(
                id=str(data["id"]),
                name=str(data["name"]),
                lang=Lang(data["lang"]),
                sample_path=sample if sample.is_absolute() else (voice_dir / sample),
                created_at_ms=int(data["created_at_ms"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise VoiceError(f"битый {META_FILENAME} в {voice_dir}: {exc}") from exc


class VoiceStore:
    """Каталог голосовых профилей на диске.

    Пример::

        store = VoiceStore(Path("./voices"))
        profile = store.create_voice(Path("me.wav"), name="me", lang=Lang.RU)
        print(profile.id)  # v_1a2b3c4d
    """

    __slots__ = ("_root",)

    def __init__(self, voices_dir: Path) -> None:
        self._root = Path(voices_dir).expanduser()

    @property
    def root(self) -> Path:
        """Корневой каталог хранилища."""
        return self._root

    # --- чтение ------------------------------------------------------------

    def path_for(self, voice_id: str) -> Path:
        """Каталог профиля (может не существовать)."""
        if not voice_id or "/" in voice_id or "\\" in voice_id or voice_id.startswith("."):
            raise VoiceError(f"недопустимый voice_id: {voice_id!r}")
        return self._root / voice_id

    def exists(self, voice_id: str) -> bool:
        """Есть ли такой профиль на диске."""
        return (self.path_for(voice_id) / META_FILENAME).is_file()

    def get(self, voice_id: str) -> VoiceProfile:
        """Прочитать профиль.

        Raises:
            VoiceNotFoundError: профиля нет на диске.
            VoiceError: ``meta.json`` битый.
        """
        voice_dir = self.path_for(voice_id)
        meta = voice_dir / META_FILENAME
        if not meta.is_file():
            raise VoiceNotFoundError(f"голос {voice_id!r} не найден в {self._root}")
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise VoiceError(f"не читается {meta}: {exc}") from exc
        if not isinstance(data, dict):
            raise VoiceError(f"{meta}: ожидался JSON-объект")
        return VoiceProfile.from_meta(data, voice_dir)

    def list(self) -> list[VoiceProfile]:
        """Все профили, отсортированные по времени создания (старые первыми)."""
        if not self._root.is_dir():
            return []
        profiles: list[VoiceProfile] = []
        for entry in sorted(self._root.iterdir()):
            if not (entry / META_FILENAME).is_file():
                continue
            try:
                profiles.append(self.get(entry.name))
            except VoiceError as exc:  # битый профиль не должен ронять список
                log.warning("пропускаю профиль %s: %s", entry.name, exc)
        profiles.sort(key=lambda profile: (profile.created_at_ms, profile.id))
        return profiles

    # --- запись ------------------------------------------------------------

    def create_voice(
        self,
        sample_wav: Path,
        name: str,
        lang: Lang,
        latents_fn: LatentsFn | None = None,
    ) -> VoiceProfile:
        """Создать профиль голоса из WAV-сэмпла.

        Сэмпл проверяется (WAV, 5-60 с), сводится в моно и ресемплится в
        24 kHz, затем сохраняется как ``sample.wav``. Если передан
        ``latents_fn`` (обычно ``XttsProvider.compute_latents``), он вызывается
        с ``(sample_path, voice_dir)`` и должен положить рядом ``latents.pt``.

        Raises:
            VoiceError: сэмпл не WAV, неподходящей длительности или имя занято.
        """
        clean_name = name.strip()
        if not clean_name:
            raise VoiceError("имя голоса не может быть пустым")
        if any(profile.name == clean_name for profile in self.list()):
            raise VoiceError(
                f"голос с именем {clean_name!r} уже есть (name UNIQUE в db/schema.sql)"
            )

        samples, sample_rate = self._load_sample(Path(sample_wav))
        if sample_rate != OUTPUT_SAMPLE_RATE:
            samples = resample_pcm(samples, sample_rate, OUTPUT_SAMPLE_RATE)

        voice_id = self._reserve_id()
        voice_dir = self._root / voice_id
        voice_dir.mkdir(parents=True, exist_ok=True)
        sample_path = voice_dir / SAMPLE_FILENAME

        profile = VoiceProfile(
            id=voice_id,
            name=clean_name,
            lang=lang,
            sample_path=sample_path,
            created_at_ms=int(time.time() * 1000),
        )
        try:
            write_wav(sample_path, samples, OUTPUT_SAMPLE_RATE)
            (voice_dir / META_FILENAME).write_text(
                json.dumps(profile.to_meta(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if latents_fn is not None:
                latents_fn(sample_path, voice_dir)
        except Exception:
            shutil.rmtree(voice_dir, ignore_errors=True)
            raise
        log.info("создан голос %s (%s, %s)", voice_id, clean_name, lang.value)
        return profile

    def delete(self, voice_id: str) -> None:
        """Удалить профиль вместе с сэмплом и латентами.

        Raises:
            VoiceNotFoundError: профиля нет.
        """
        voice_dir = self.path_for(voice_id)
        if not (voice_dir / META_FILENAME).is_file():
            raise VoiceNotFoundError(f"голос {voice_id!r} не найден в {self._root}")
        shutil.rmtree(voice_dir)
        log.info("удалён голос %s", voice_id)

    # --- внутреннее --------------------------------------------------------

    def _reserve_id(self) -> str:
        """Уникальный ``voice_id``, которого ещё нет на диске."""
        for _ in range(16):
            voice_id = _new_voice_id()
            if not (self._root / voice_id).exists():
                return voice_id
        raise VoiceError("не удалось подобрать свободный voice_id")

    @staticmethod
    def _load_sample(path: Path) -> tuple[Pcm16, int]:
        """Прочитать и проверить сэмпл: WAV, моно int16, 5-60 секунд."""
        if not path.is_file():
            raise VoiceError(f"файл сэмпла не найден: {path}")
        try:
            samples, sample_rate = read_wav(path)
        except (wave.Error, ValueError, OSError) as exc:
            raise VoiceError(
                f"{path}: ожидался несжатый WAV (PCM). Перекодируйте, например "
                f"ffmpeg -i вход -ac 1 -ar 24000 -c:a pcm_s16le sample.wav. Ошибка: {exc}"
            ) from exc

        duration = samples.size / sample_rate if sample_rate else 0.0
        if duration < MIN_SAMPLE_SECONDS:
            raise VoiceError(
                f"{path}: сэмпл {duration:.1f} с — короче минимума {MIN_SAMPLE_SECONDS:.0f} с"
            )
        if duration > MAX_SAMPLE_SECONDS:
            raise VoiceError(
                f"{path}: сэмпл {duration:.1f} с — длиннее максимума {MAX_SAMPLE_SECONDS:.0f} с"
            )
        if duration < RECOMMENDED_SAMPLE_SECONDS:
            log.warning(
                "сэмпл %s короткий (%.1f с): для качественного клона нужно %.0f-30 с",
                path,
                duration,
                RECOMMENDED_SAMPLE_SECONDS,
            )
        return samples, sample_rate
