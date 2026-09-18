"""Казахский TTS без клонирования: `facebook/mms-tts-kaz` (VITS, CPU, ~150 MB).

ARCHITECTURE.md 4.4: для kk клона голоса нет — используется стандартный голос.
Раздел 6 (бюджет VRAM): модель работает на CPU, GPU остаётся под whisper и XTTS.

Модель отдаёт 16 kHz float32; здесь он переводится в int16 и ресемплится до
24 kHz — единого формата выхода TTS. transformers и torch импортируются лениво,
внутри методов, чтобы модуль (и юнит-тесты) работали без тяжёлых пакетов.
"""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Any, Final

import numpy as np

from engine.contracts.events import Lang
from engine.tts.audio import Pcm16, float_to_int16, iter_chunks, resample_pcm
from engine.tts.base import OUTPUT_SAMPLE_RATE, TtsConfig, TtsError

__all__ = ["MMS_SAMPLE_RATE", "MmsKazakhTts"]

log: Final[logging.Logger] = logging.getLogger(__name__)

#: Частота дискретизации MMS-TTS (VITS) — берётся из конфига модели, это дефолт.
MMS_SAMPLE_RATE: Final[int] = 16_000

#: Длинный текст режем на куски по предложениям — так первый звук приходит раньше.
MAX_PIECE_CHARS: Final[int] = 220
_SENTENCE_SPLIT: Final[re.Pattern[str]] = re.compile(r"(?<=[.!?…])\s+")

# Одна модель на процесс: (id модели) -> (tokenizer, model).
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MODEL_LOCK: Final[threading.Lock] = threading.Lock()


def split_text(text: str, limit: int = MAX_PIECE_CHARS) -> list[str]:
    """Разбить текст на куски не длиннее ``limit`` символов по границам предложений."""
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT.split(text.strip()):
        if not sentence:
            continue
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= limit:
            current = f"{current} {sentence}"
        else:
            pieces.append(current)
            current = sentence
    if current:
        pieces.append(current)
    return pieces or ([text.strip()] if text.strip() else [])


class MmsKazakhTts:
    """Провайдер казахского TTS на `facebook/mms-tts-kaz`. Клонирования нет."""

    __slots__ = ("_config", "_loaded", "_lock")

    def __init__(self, config: TtsConfig) -> None:
        self._config = config
        self._loaded: tuple[Any, Any] | None = None
        self._lock = asyncio.Lock()

    @property
    def supports_cloning(self) -> bool:
        """MMS-TTS — один встроенный голос, клонирование не поддерживается."""
        return False

    def supports(self, lang: Lang) -> bool:
        """Только казахский."""
        return lang is Lang.KK

    # --- загрузка ----------------------------------------------------------

    def _model_source(self) -> str:
        """Локальный каталог модели, если он есть, иначе id репозитория."""
        model_id = self._config.kk_model_id
        for candidate in (
            self._config.models_dir / "mms-tts-kaz",
            self._config.models_dir / model_id,
            self._config.models_dir / model_id.split("/")[-1],
        ):
            if (candidate / "config.json").is_file():
                return str(candidate)
        return model_id

    def _load(self) -> tuple[Any, Any]:
        """Загрузить токенизатор и модель (блокирующая операция, CPU)."""
        source = self._model_source()
        with _MODEL_LOCK:
            cached = _MODEL_CACHE.get(source)
            if cached is not None:
                return cached
            try:
                from transformers import AutoTokenizer, VitsModel
            except ImportError as exc:  # pragma: no cover — требует тяжёлых пакетов
                raise TtsError(
                    "не установлен transformers: pip install -r engine/tts/requirements-tts.txt"
                ) from exc

            log.info("загружаю казахский TTS %s (CPU)", source)
            try:
                tokenizer = AutoTokenizer.from_pretrained(source)
                model = VitsModel.from_pretrained(source)
            except Exception as exc:
                raise TtsError(f"не удалось загрузить {source}: {exc}") from exc
            model.eval()
            _MODEL_CACHE[source] = (tokenizer, model)
            return tokenizer, model

    async def _ensure_model(self) -> tuple[Any, Any]:
        if self._loaded is not None:
            return self._loaded
        async with self._lock:
            if self._loaded is None:
                self._loaded = await asyncio.to_thread(self._load)
        return self._loaded

    async def warmup(self) -> None:
        """Загрузить модель заранее."""
        await self._ensure_model()

    # --- синтез ------------------------------------------------------------

    def _render(self, piece: str) -> Pcm16:
        """Синтезировать кусок текста целиком: PCM int16 24 kHz (блокирующе)."""
        import torch

        tokenizer, model = self._load()
        inputs = tokenizer(piece, return_tensors="pt")
        with torch.no_grad():
            waveform = model(**inputs).waveform
        pcm = float_to_int16(np.asarray(waveform.detach().cpu().numpy(), dtype=np.float32))
        source_rate = int(getattr(model.config, "sampling_rate", MMS_SAMPLE_RATE))
        return resample_pcm(pcm, source_rate, OUTPUT_SAMPLE_RATE)

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        """Синтез kk: модель отдаёт фразу целиком, чанки нарезаем сами.

        ``voice_id`` игнорируется — клонирования у MMS нет (ARCHITECTURE.md 4.4).
        """
        if not self.supports(lang):
            raise TtsError(f"MmsKazakhTts: язык {lang.value} не поддерживается (только kk)")
        if voice_id is not None:
            log.debug("kk: voice_id=%s игнорируется, клонирования нет", voice_id)
        if not text.strip():
            return

        await self._ensure_model()
        for piece in split_text(text):
            pcm = await asyncio.to_thread(self._render, piece)
            for chunk in self._chunks(pcm):
                yield chunk

    def _chunks(self, pcm: Pcm16) -> Iterator[Pcm16]:
        return iter_chunks(pcm, OUTPUT_SAMPLE_RATE, self._config.chunk_ms)
