"""faster-whisper (ARCHITECTURE.md 4.2): распознавание фраз.

Модель — строго ``small`` с ``compute_type=int8_float16`` (≈1 GB VRAM,
ARCHITECTURE.md раздел 6, CLAUDE.md закон 4). Для казахского можно подменить
модель на файнтюн через :attr:`SttConfig.model_overrides` (env
``RT_STT_KK_MODEL``) — см. ``engine/stt/README.md``.

``faster_whisper`` и ``torch`` импортируются лениво: модуль можно импортировать
и типизировать без установленных моделей.

Установка::

    pip install -r engine/stt/requirements-stt.txt
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np

from engine.stt.base import (
    Device,
    Int16Array,
    SttConfig,
    SttError,
    TranscriptResult,
    TranscriptSegment,
    samples_to_ms,
    to_float32,
)

__all__ = ["FasterWhisperTranscriber", "cuda_available", "resolve_runtime"]

logger = logging.getLogger("engine.stt")

_CPU_COMPUTE_TYPE = "int8"


def cuda_available() -> bool:
    """Есть ли работающая CUDA (проверка через ``torch``, без исключений)."""
    try:
        import torch
    except ImportError:
        logger.debug("torch не установлен — считаю, что CUDA недоступна")
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover — сломанный драйвер/сборка torch
        logger.warning("torch.cuda.is_available() упал, считаю CUDA недоступной", exc_info=True)
        return False


def resolve_runtime(config: SttConfig) -> tuple[str, str]:
    """Выбрать устройство и тип вычислений с учётом наличия CUDA.

    Args:
        config: конфиг модуля (``device``, ``compute_type``).

    Returns:
        Пара ``(device, compute_type)`` для ``WhisperModel``. Если CUDA
        недоступна, возвращается ``("cpu", "int8")`` с предупреждением в лог.
    """
    requested: Device = config.device
    if requested in ("cuda", "auto") and cuda_available():
        return "cuda", config.compute_type

    if requested == "cuda":
        logger.warning(
            "запрошен device=cuda, но CUDA недоступна — перехожу на cpu/%s; "
            "для RTX 4050 поставьте CUDA-сборку torch (см. engine/stt/README.md)",
            _CPU_COMPUTE_TYPE,
        )
    else:
        logger.info("CUDA недоступна — работаю на cpu/%s (медленнее)", _CPU_COMPUTE_TYPE)
    return "cpu", _CPU_COMPUTE_TYPE


class FasterWhisperTranscriber:
    """Распознаватель на faster-whisper под протокол ``Transcriber``.

    Модель грузится лениво при первом :meth:`transcribe` и живёт в памяти до
    :meth:`close` (ARCHITECTURE.md раздел 6: модели загружаются один раз при
    старте движка). Модели для разных языков кэшируются по имени/пути, так что
    kk-файнтюн не вытесняет основную модель.
    """

    def __init__(self, config: SttConfig | None = None) -> None:
        """Создать распознаватель (модель ещё не грузится)."""
        self._config = config or SttConfig()
        self._device, self._compute_type = resolve_runtime(self._config)
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        """Фактическое устройство после проверки CUDA."""
        return self._device

    @property
    def compute_type(self) -> str:
        """Фактический тип вычислений CTranslate2."""
        return self._compute_type

    def model_name(self, lang: str | None) -> str:
        """Имя или путь модели, которая будет использована для языка."""
        return self._config.model_for(lang)

    def transcribe(self, pcm_int16_16k: Int16Array, lang: str | None) -> TranscriptResult:
        """Распознать буфер PCM 16 kHz mono int16.

        Args:
            pcm_int16_16k: аудио фразы целиком.
            lang: код языка или ``None`` — автоопределение whisper.

        Returns:
            Текст, язык, сегменты с таймкодами от начала буфера.

        Raises:
            SttError: не установлен ``faster-whisper`` или модель не грузится.
        """
        samples = np.asarray(pcm_int16_16k, dtype=np.int16).reshape(-1)
        duration_ms = samples_to_ms(samples.size)
        target_lang = lang or self._config.language
        model = self._model_for(target_lang)

        segments_iter, info = model.transcribe(
            to_float32(samples),
            language=target_lang or None,
            beam_size=self._config.beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0,
        )

        segments = tuple(
            TranscriptSegment(
                text=str(segment.text).strip(),
                t_start_ms=max(0, int(segment.start * 1000)),
                t_end_ms=max(0, int(segment.end * 1000)),
            )
            for segment in segments_iter
        )
        text = " ".join(segment.text for segment in segments if segment.text).strip()
        detected = target_lang or str(getattr(info, "language", "") or "")

        return TranscriptResult(
            text=text,
            lang=detected,
            segments=segments,
            duration_ms=duration_ms,
        )

    def close(self) -> None:
        """Выгрузить модели из памяти (освободить VRAM)."""
        with self._lock:
            self._models.clear()

    def _model_for(self, lang: str | None) -> Any:
        name = self._config.model_for(lang)
        with self._lock:
            model = self._models.get(name)
            if model is None:
                model = self._load_model(name)
                self._models[name] = model
            return model

    def _load_model(self, name: str) -> Any:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover — зависит от окружения
            raise SttError(
                "не установлен пакет faster-whisper; поставьте зависимости модуля: "
                "pip install -r engine/stt/requirements-stt.txt"
            ) from exc

        models_dir: Path | None = self._config.models_dir
        logger.info(
            "Загружаю faster-whisper %r на %s/%s (download_root=%s)",
            name,
            self._device,
            self._compute_type,
            models_dir or "кэш huggingface по умолчанию",
        )
        try:
            return WhisperModel(
                name,
                device=self._device,
                compute_type=self._compute_type,
                download_root=str(models_dir) if models_dir else None,
            )
        except Exception as exc:
            raise SttError(f"не удалось загрузить модель whisper {name!r}: {exc}") from exc
