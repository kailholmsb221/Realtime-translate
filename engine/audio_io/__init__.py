"""audio_io — захват и вывод звука (ARCHITECTURE.md 4.1, зона агента A).

Публичный API пакета — фабрики :func:`create_source` / :func:`create_sink`
и абстракции из :mod:`engine.audio_io.base`. Остальные модули движка не
должны знать, какой бэкенд используется.

Маршрутизация (Windows):

* ``loopback``   — WASAPI loopback, речь собеседника из Zoom (вход пайплайна ``in``);
* ``mic``        — физический микрофон пользователя (вход пайплайна ``out``);
* ``headphones`` — наушники, туда играет перевод речи собеседника;
* ``cable``      — вход VB-Audio Virtual Cable, туда играет перевод речи
  пользователя (в Zoom микрофоном выбран ``CABLE Output``);
* ``file``       — WAV на входе/выходе (файловый режим, этап 1; работает везде).

На не-Windows платформах всё, кроме ``file``, бросает
:class:`~engine.audio_io.base.AudioBackendUnavailable` с инструкцией.

Пример::

    from engine.audio_io import create_sink, create_source

    src = create_source("file", path="sample.wav")
    sink = create_sink("file", path="out.wav")
    async with src, sink:
        async for chunk in src:
            await sink.write(chunk)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final, Literal

from engine.audio_io.base import (
    CHUNK_FRAMES,
    CHUNK_MS,
    ENGINE_FORMAT,
    SAMPLE_RATE_ENGINE,
    SAMPLE_RATE_TTS,
    TTS_FORMAT,
    AudioBackendUnavailable,
    AudioChunk,
    AudioDeviceInfo,
    AudioFormat,
    AudioIOError,
    AudioSink,
    AudioSource,
    ChunkAssembler,
    DeviceKind,
    DeviceNotFound,
)
from engine.audio_io.config import AudioConfig
from engine.audio_io.devices import (
    backend_report,
    find_default_mic,
    find_default_output,
    find_loopback_device,
    find_virtual_cable_output,
    is_windows,
    list_devices,
)
from engine.audio_io.fake import FakeSink, FakeSource

__all__ = [
    "CHUNK_FRAMES",
    "CHUNK_MS",
    "ENGINE_FORMAT",
    "SAMPLE_RATE_ENGINE",
    "SAMPLE_RATE_TTS",
    "SINK_KINDS",
    "SOURCE_KINDS",
    "TTS_FORMAT",
    "AudioBackendUnavailable",
    "AudioChunk",
    "AudioConfig",
    "AudioDeviceInfo",
    "AudioFormat",
    "AudioIOError",
    "AudioSink",
    "AudioSource",
    "ChunkAssembler",
    "DeviceKind",
    "DeviceNotFound",
    "FakeSink",
    "FakeSource",
    "SinkKind",
    "SourceKind",
    "backend_report",
    "create_sink",
    "create_source",
    "find_default_mic",
    "find_default_output",
    "find_loopback_device",
    "find_virtual_cable_output",
    "is_windows",
    "list_devices",
]

SourceKind = Literal["loopback", "mic", "file"]
SinkKind = Literal["headphones", "cable", "file"]

SOURCE_KINDS: Final[tuple[str, ...]] = ("loopback", "mic", "file")
SINK_KINDS: Final[tuple[str, ...]] = ("headphones", "cable", "file")

_WINDOWS_ONLY_HINT: Final[str] = (
    "Живой звук (WASAPI loopback, микрофон, наушники, VB-Cable) работает только "
    "на Windows. На других платформах используйте файловый режим: "
    "create_source('file', path=...) / create_sink('file', path=...). "
    "Установка зависимостей: pip install -r engine/audio_io/requirements-audio_io.txt"
)


def _require_windows(kind: str) -> None:
    """Проверить платформу перед открытием живого устройства."""
    if not is_windows():
        raise AudioBackendUnavailable(
            f"источник/приёмник {kind!r} недоступен. {_WINDOWS_ONLY_HINT}"
        )


def create_source(
    kind: SourceKind | str,
    config: AudioConfig | None = None,
    path: str | Path | None = None,
    **kwargs: Any,
) -> AudioSource:
    """Создать источник аудио.

    Args:
        kind: ``"loopback"`` (речь собеседника), ``"mic"`` (микрофон
            пользователя) или ``"file"`` (WAV, файловый режим).
        config: настройки устройств; по умолчанию :meth:`AudioConfig.from_env`.
        path: путь к WAV для ``kind="file"``.
        **kwargs: прокидываются в конструктор конкретной реализации
            (``realtime``, ``chunk_ms``, ``queue_chunks`` и т. п.).

    Returns:
        Объект, реализующий протокол :class:`~engine.audio_io.base.AudioSource`:
        async-итератор чанков 30 мс / 16 kHz mono int16.

    Raises:
        AudioBackendUnavailable: живое устройство запрошено не на Windows или
            без установленных пакетов.
        ValueError: неизвестный ``kind`` или не задан ``path`` для файла.
    """
    cfg = config or AudioConfig.from_env()

    if kind == "file":
        if path is None:
            raise ValueError("create_source('file') требует path к WAV-файлу")
        return FakeSource.from_wav(path, fmt=cfg.format, chunk_ms=cfg.chunk_ms, **kwargs)

    if kind == "loopback":
        _require_windows(kind)
        from engine.audio_io.windows import make_loopback_source

        return make_loopback_source(cfg, **kwargs)

    if kind == "mic":
        _require_windows(kind)
        from engine.audio_io.windows import make_mic_source

        return make_mic_source(cfg, **kwargs)

    raise ValueError(f"неизвестный kind источника: {kind!r}; доступны {SOURCE_KINDS}")


def create_sink(
    kind: SinkKind | str,
    config: AudioConfig | None = None,
    path: str | Path | None = None,
    **kwargs: Any,
) -> AudioSink:
    """Создать приёмник аудио.

    Args:
        kind: ``"headphones"`` (перевод собеседника пользователю),
            ``"cable"`` (перевод пользователя в Zoom через VB-Cable)
            или ``"file"`` (WAV, файловый режим).
        config: настройки устройств; по умолчанию :meth:`AudioConfig.from_env`.
        path: путь к WAV для ``kind="file"``.
        **kwargs: прокидываются в конструктор конкретной реализации.

    Returns:
        Объект, реализующий протокол :class:`~engine.audio_io.base.AudioSink`.
        Приёмник сам ресемплит входящие чанки (16 kHz речь, 24 kHz выход TTS)
        под частоту устройства.

    Raises:
        AudioBackendUnavailable: живое устройство запрошено не на Windows или
            без установленных пакетов.
        ValueError: неизвестный ``kind`` или не задан ``path`` для файла.
    """
    cfg = config or AudioConfig.from_env()

    if kind == "file":
        if path is None:
            raise ValueError("create_sink('file') требует path к WAV-файлу")
        return FakeSink(fmt=cfg.format, path=path, **kwargs)

    if kind == "headphones":
        _require_windows(kind)
        from engine.audio_io.windows import make_headphones_sink

        return make_headphones_sink(cfg, **kwargs)

    if kind == "cable":
        _require_windows(kind)
        from engine.audio_io.windows import make_cable_sink

        return make_cable_sink(cfg, **kwargs)

    raise ValueError(f"неизвестный kind приёмника: {kind!r}; доступны {SINK_KINDS}")
