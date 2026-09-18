"""Адаптер KazakhTTS2 (ISSAI) — заготовка, инференс не реализован.

ARCHITECTURE.md 4.4 допускает два бэкенда для казахского: `facebook/mms-tts-kaz`
(работает «из коробки», см. :mod:`engine.tts.kk_mms`) и KazakhTTS2 от ISSAI —
он звучит лучше, но требует ESPnet и ручной установки чекпоинтов, поэтому
включается осознанно (``RT_TTS_KK_BACKEND=kazakhtts2``).

Как подключить KazakhTTS2
-------------------------

1. Репозиторий рецептов: https://github.com/IS2AI/Kazakh_TTS (ISSAI, Назарбаев
   Университет; статья «KazakhTTS2», Apache-2.0 для кода). Датасет —
   https://huggingface.co/datasets/issai/KazakhTTS.
2. Скачать чекпоинты Tacotron2 и вокодеров Parallel WaveGAN (пять дикторов:
   female1-3, male1-2), ссылки лежат в README репозитория, например::

       https://issai.nu.edu.kz/wp-content/uploads/2022/03/kaztts_female2_tacotron2_train.loss.ave.zip
       https://issai.nu.edu.kz/wp-content/uploads/2022/03/parallelwavegan_female2_checkpoint.zip

   Распаковать в ``$RT_MODELS_DIR/kazakhtts2/<speaker>/`` (по умолчанию
   ``./models/kazakhtts2/``).
3. Поставить ESPnet2 и parallel_wavegan (в ``requirements-tts.txt`` их нет:
   ESPnet тянет много зависимостей и в бюджет «лёгкой» установки не входит)::

       pip install espnet espnet_model_zoo parallel_wavegan

4. Реализовать :meth:`KazakhTts2Provider._render` через
   ``espnet2.bin.tts_inference.Text2Speech`` примерно так::

       from espnet2.bin.tts_inference import Text2Speech
       tts = Text2Speech.from_pretrained(
           train_config=str(model_dir / "config.yaml"),
           model_file=str(model_dir / "train.loss.ave.pth"),
           vocoder_config=str(vocoder_dir / "config.yml"),
           vocoder_file=str(vocoder_dir / "checkpoint-400000steps.pkl"),
           device="cpu",
       )
       wav = tts(text)["wav"].view(-1).cpu().numpy()  # 22.05 kHz float32

   затем ``float_to_int16`` + ``resample_pcm(..., 22_050, OUTPUT_SAMPLE_RATE)``
   и нарезка через ``iter_chunks`` — как сделано в :mod:`engine.tts.kk_mms`.
5. Текст нормализовать под кириллическую казахскую орфографию рецепта (числа
   прописью, латиница транслитерацией) — в рецепте это делает препроцессор ESPnet.

Ограничение по VRAM (ARCHITECTURE.md раздел 6): держать модель на CPU.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

from engine.contracts.events import Lang
from engine.tts.audio import Pcm16
from engine.tts.base import TtsConfig

__all__ = ["KAZAKHTTS2_REPO", "KazakhTts2Provider"]

log: Final[logging.Logger] = logging.getLogger(__name__)

#: Репозиторий рецептов и ссылок на чекпоинты.
KAZAKHTTS2_REPO: Final[str] = "https://github.com/IS2AI/Kazakh_TTS"

#: Каталог с распакованными чекпоинтами внутри ``models_dir``.
KAZAKHTTS2_SUBDIR: Final[str] = "kazakhtts2"

_NOT_IMPLEMENTED_MSG: Final[str] = (
    "Бэкенд kazakhtts2 не реализован: нужен ESPnet и чекпоинты ISSAI. "
    f"Инструкция — в docstring engine/tts/kk_kazakhtts2.py и {KAZAKHTTS2_REPO}. "
    "Пока используйте RT_TTS_KK_BACKEND=mms (facebook/mms-tts-kaz)."
)


class KazakhTts2Provider:
    """Заготовка провайдера KazakhTTS2 (ESPnet). Все методы синтеза падают.

    Интерфейс совпадает с :class:`~engine.tts.base.TtsProvider`, чтобы адаптер
    можно было включить, не трогая роутер.
    """

    __slots__ = ("_config",)

    def __init__(self, config: TtsConfig) -> None:
        self._config = config

    @property
    def supports_cloning(self) -> bool:
        """Клонирования нет: это многодикторная модель с фиксированными голосами."""
        return False

    def supports(self, lang: Lang) -> bool:
        """Только казахский."""
        return lang is Lang.KK

    @property
    def models_dir(self) -> Path:
        """Куда распаковывать чекпоинты ISSAI."""
        return self._config.models_dir / KAZAKHTTS2_SUBDIR

    async def warmup(self) -> None:
        """Не реализовано.

        Raises:
            NotImplementedError: всегда.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)

    async def synthesize(
        self,
        text: str,
        lang: Lang,
        voice_id: str | None = None,
    ) -> AsyncIterator[Pcm16]:
        """Не реализовано.

        Raises:
            NotImplementedError: всегда.
        """
        raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
        yield  # pragma: no cover — делает функцию асинхронным генератором
