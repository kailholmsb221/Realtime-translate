"""Маршрутизатор TTS: язык -> провайдер, чанки -> события `tts.chunk`.

Единая точка входа модуля (ARCHITECTURE.md 4.4):

* ru/en -> XTTS-v2 с клоном голоса (``voice_id``); без ``voice_id`` — встроенный
  голос XTTS из ``speakers_xtts.pth`` (по умолчанию «Ana Florence»);
* kk -> MMS-TTS (или KazakhTTS2), клонирования нет, ``voice_id`` игнорируется;
* ``backend=fake`` -> :class:`~engine.tts.fake.FakeTts` на все языки.

Провайдеры создаются лениво: пока язык не понадобился, его модель не грузится.
"""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Final

import numpy as np

from engine.contracts.events import Envelope, Lang, Stream, TtsChunk
from engine.tts.audio import Pcm16
from engine.tts.base import Backend, KkBackend, TtsConfig, TtsError, TtsProvider
from engine.tts.fake import FakeTts
from engine.tts.voices import VoiceStore

__all__ = ["TtsRouter", "chunk_to_event", "pcm_from_event"]

log: Final[logging.Logger] = logging.getLogger(__name__)

#: Языки, которые синтезирует XTTS (остальное — казахский провайдер).
CLONING_LANGS: Final[frozenset[Lang]] = frozenset({Lang.RU, Lang.EN})


def chunk_to_event(stream: Stream, seq: int, pcm: Pcm16) -> Envelope:
    """Собрать конверт события ``tts.chunk`` из чанка PCM 24 kHz mono int16.

    Args:
        stream: ``in`` — перевод собеседника, ``out`` — перевод пользователя.
        seq: номер чанка в потоке, с нуля.
        pcm: чанк int16; сериализуется как little-endian PCM в base64.

    Returns:
        Провалидированный конверт (`engine/contracts/tts.chunk.json`).
    """
    raw = np.asarray(pcm).reshape(-1).astype("<i2").tobytes()
    payload = TtsChunk(
        stream=stream,
        seq=seq,
        pcm_base64=base64.b64encode(raw).decode("ascii"),
    )
    return Envelope.wrap(payload)


def pcm_from_event(event: TtsChunk) -> Pcm16:
    """Обратное преобразование: payload ``tts.chunk`` -> PCM int16 (для тестов/UI)."""
    return np.frombuffer(base64.b64decode(event.pcm_base64), dtype="<i2")


class TtsRouter:
    """Выбирает провайдера по языку и отдаёт единый поток чанков/событий.

    Пример::

        router = TtsRouter(TtsConfig.from_env())
        async for chunk in router.synthesize("привет", Lang.RU, voice_id="v_1a2b3c4d"):
            ...
    """

    __slots__ = ("_config", "_providers", "_voices")

    def __init__(
        self,
        config: TtsConfig,
        providers: Mapping[Lang, TtsProvider] | None = None,
        voices: VoiceStore | None = None,
    ) -> None:
        """
        Args:
            config: конфигурация модуля.
            providers: готовые провайдеры по языкам — так тесты подменяют
                тяжёлые модели фейками; чего нет в словаре, создаётся лениво.
            voices: хранилище голосов (по умолчанию ``config.voices_dir``).
        """
        self._config = config
        self._voices = voices if voices is not None else VoiceStore(config.voices_dir)
        self._providers: dict[Lang, TtsProvider] = dict(providers or {})

    @property
    def config(self) -> TtsConfig:
        """Конфигурация, с которой создан роутер."""
        return self._config

    @property
    def voices(self) -> VoiceStore:
        """Хранилище голосовых профилей."""
        return self._voices

    # --- провайдеры --------------------------------------------------------

    def _create_provider(self, lang: Lang) -> TtsProvider:
        """Создать провайдера для языка (модели грузятся позже, при warmup/синтезе)."""
        if self._config.backend is Backend.FAKE:
            # Фейк повторяет правила маршрутизации: клон только для ru/en.
            return FakeTts(chunk_ms=self._config.chunk_ms, supports_cloning=lang in CLONING_LANGS)
        if lang in CLONING_LANGS:
            from engine.tts.xtts import XttsProvider

            return XttsProvider(self._config, self._voices)
        if self._config.kk_backend is KkBackend.KAZAKHTTS2:
            from engine.tts.kk_kazakhtts2 import KazakhTts2Provider

            return KazakhTts2Provider(self._config)
        from engine.tts.kk_mms import MmsKazakhTts

        return MmsKazakhTts(self._config)

    def provider_for(self, lang: Lang) -> TtsProvider:
        """Провайдер для языка; создаётся при первом обращении.

        Raises:
            TtsError: провайдер не поддерживает язык (например, подменён в тестах).
        """
        provider = self._providers.get(lang)
        if provider is None:
            provider = self._create_provider(lang)
            self._providers[lang] = provider
        if not provider.supports(lang):
            raise TtsError(f"провайдер {type(provider).__name__} не поддерживает язык {lang.value}")
        return provider

    async def warmup(self, langs: Iterable[Lang] = (Lang.RU, Lang.EN, Lang.KK)) -> None:
        """Прогреть провайдеров нужных языков (загрузка моделей при старте движка)."""
        for lang in langs:
            await self.provider_for(lang).warmup()

    # --- синтез ------------------------------------------------------------

    def effective_voice_id(self, lang: Lang, voice_id: str | None) -> str | None:
        """Какой ``voice_id`` реально уедет в провайдера.

        Для языков без клонирования (kk) всегда ``None``: контракт разрешает
        ``voice_id`` в ``session.start``, но клона там нет.
        """
        if voice_id is None:
            return None
        if not self.provider_for(lang).supports_cloning:
            log.debug("язык %s без клонирования — voice_id=%s игнорируется", lang.value, voice_id)
            return None
        return voice_id

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        """Синтезировать текст выбранным по языку провайдером.

        Yields:
            Чанки PCM 24 kHz mono int16 по 20-100 мс.

        Raises:
            TtsError: язык не поддержан или упал инференс.
        """
        provider = self.provider_for(lang)
        effective = self.effective_voice_id(lang, voice_id)
        async for chunk in provider.synthesize(text, lang, effective):
            yield chunk

    async def synthesize_events(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
        *,
        stream: Stream = Stream.IN,
        start_seq: int = 0,
    ) -> AsyncIterator[Envelope]:
        """То же, что :meth:`synthesize`, но сразу событиями ``tts.chunk``."""
        seq = start_seq
        async for chunk in self.synthesize(text, lang, voice_id):
            yield chunk_to_event(stream, seq, chunk)
            seq += 1

    async def synthesize_array(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> Pcm16:
        """Собрать весь синтез в один массив (файловый режим, smoke-тесты)."""
        chunks: list[Pcm16] = [chunk async for chunk in self.synthesize(text, lang, voice_id)]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks).astype(np.int16)
