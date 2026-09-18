"""Реестр аудиоустройств: перечисление и поиск по имени.

Источники устройств:

* ``sounddevice`` — обычные входы/выходы (микрофон, наушники, вход
  виртуального кабеля);
* ``pyaudiowpatch`` — дополнительно WASAPI loopback-устройства (захват того,
  что ОС играет в наушники, — речь собеседника из Zoom).

Оба пакета импортируются лениво внутри функций: на Linux/в CI модуль
импортируется без них, :func:`list_devices` просто вернёт пустой список,
а ``find_*`` бросят :class:`~engine.audio_io.base.DeviceNotFound`
или :class:`~engine.audio_io.base.AudioBackendUnavailable`.

Все ``find_*`` принимают готовый список устройств параметром ``devices`` —
так их можно тестировать без железа.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from typing import Any, Final

from engine.audio_io.base import (
    VIRTUAL_CABLE_MARKER,
    AudioBackendUnavailable,
    AudioDeviceInfo,
    DeviceKind,
    DeviceNotFound,
)
from engine.audio_io.config import AudioConfig

__all__ = [
    "BACKEND_FAKE",
    "BACKEND_PYAUDIOWPATCH",
    "BACKEND_SOUNDDEVICE",
    "backend_report",
    "find_default_mic",
    "find_default_output",
    "find_loopback_device",
    "find_virtual_cable_output",
    "import_pyaudiowpatch",
    "import_sounddevice",
    "is_windows",
    "list_devices",
    "match_device",
]

log = logging.getLogger(__name__)

BACKEND_SOUNDDEVICE: Final[str] = "sounddevice"
BACKEND_PYAUDIOWPATCH: Final[str] = "pyaudiowpatch"
BACKEND_FAKE: Final[str] = "fake"

_INSTALL_SOUNDDEVICE: Final[str] = (
    "pip install -r engine/audio_io/requirements-audio_io.txt  (нужен пакет sounddevice)"
)
_INSTALL_PYAUDIOWPATCH: Final[str] = (
    "pip install -r engine/audio_io/requirements-audio_io.txt  "
    "(нужен пакет PyAudioWPatch, только Windows)"
)


def is_windows() -> bool:
    """Запущены ли мы на Windows (единственная ОС с WASAPI loopback)."""
    return sys.platform == "win32"


def import_sounddevice() -> Any:
    """Импортировать ``sounddevice`` или бросить понятную ошибку.

    Raises:
        AudioBackendUnavailable: пакет не установлен или нет системных
            библиотек PortAudio.
    """
    try:
        import sounddevice
    except (ImportError, OSError) as exc:  # OSError: не найдена libportaudio
        raise AudioBackendUnavailable(
            f"sounddevice недоступен ({exc}). Установите: {_INSTALL_SOUNDDEVICE}"
        ) from exc
    return sounddevice


def import_pyaudiowpatch() -> Any:
    """Импортировать ``pyaudiowpatch`` (WASAPI loopback) или бросить ошибку.

    Raises:
        AudioBackendUnavailable: не Windows или пакет не установлен.
    """
    if not is_windows():
        raise AudioBackendUnavailable(
            "WASAPI loopback есть только на Windows. На других ОС используйте "
            "файловый режим: create_source('file', path=...)"
        )
    try:
        import pyaudiowpatch
    except ImportError as exc:
        raise AudioBackendUnavailable(
            f"pyaudiowpatch недоступен ({exc}). Установите: {_INSTALL_PYAUDIOWPATCH}"
        ) from exc
    return pyaudiowpatch


def _sounddevice_devices() -> list[AudioDeviceInfo]:
    """Входы и выходы из ``sounddevice``; пустой список, если бэкенд недоступен."""
    try:
        sd = import_sounddevice()
    except AudioBackendUnavailable as exc:
        log.debug("sounddevice недоступен: %s", exc)
        return []

    try:
        raw_devices = list(sd.query_devices())
        hostapis = list(sd.query_hostapis())
        default_in, default_out = sd.default.device
    except Exception as exc:  # pragma: no cover - зависит от драйверов
        log.warning("не удалось опросить sounddevice: %s", exc)
        return []

    out: list[AudioDeviceInfo] = []
    for index, dev in enumerate(raw_devices):
        hostapi_index = int(dev.get("hostapi", -1))
        hostapi_name = ""
        if 0 <= hostapi_index < len(hostapis):
            hostapi_name = str(hostapis[hostapi_index].get("name", ""))
        rate = float(dev.get("default_samplerate", 0.0))
        name = str(dev.get("name", f"device {index}"))
        if int(dev.get("max_input_channels", 0)) > 0:
            out.append(
                AudioDeviceInfo(
                    id=index,
                    name=name,
                    kind=DeviceKind.INPUT,
                    default_sample_rate=rate,
                    channels=int(dev["max_input_channels"]),
                    backend=BACKEND_SOUNDDEVICE,
                    is_default=index == default_in,
                    hostapi=hostapi_name,
                )
            )
        if int(dev.get("max_output_channels", 0)) > 0:
            out.append(
                AudioDeviceInfo(
                    id=index,
                    name=name,
                    kind=DeviceKind.OUTPUT,
                    default_sample_rate=rate,
                    channels=int(dev["max_output_channels"]),
                    backend=BACKEND_SOUNDDEVICE,
                    is_default=index == default_out,
                    hostapi=hostapi_name,
                )
            )
    return out


def _loopback_devices() -> list[AudioDeviceInfo]:
    """WASAPI loopback-устройства из ``pyaudiowpatch``; пусто, если недоступен."""
    try:
        pyaudio = import_pyaudiowpatch()
    except AudioBackendUnavailable as exc:
        log.debug("pyaudiowpatch недоступен: %s", exc)
        return []

    out: list[AudioDeviceInfo] = []
    audio = pyaudio.PyAudio()
    try:
        default_name = ""
        try:
            wasapi = audio.get_host_api_info_by_type(pyaudio.paWASAPI)
            speakers = audio.get_device_info_by_index(int(wasapi["defaultOutputDevice"]))
            default_name = str(speakers["name"])
        except Exception as exc:  # pragma: no cover - нет WASAPI
            log.debug("нет WASAPI-устройства по умолчанию: %s", exc)

        for dev in audio.get_loopback_device_info_generator():
            name = str(dev["name"])
            out.append(
                AudioDeviceInfo(
                    id=int(dev["index"]),
                    name=name,
                    kind=DeviceKind.LOOPBACK,
                    default_sample_rate=float(dev["defaultSampleRate"]),
                    channels=int(dev["maxInputChannels"]),
                    backend=BACKEND_PYAUDIOWPATCH,
                    is_default=bool(default_name) and default_name in name,
                    hostapi="WASAPI",
                )
            )
    finally:
        audio.terminate()
    return out


def list_devices(*, include_loopback: bool = True) -> list[AudioDeviceInfo]:
    """Перечислить все доступные устройства.

    Args:
        include_loopback: добавлять ли WASAPI loopback-устройства.

    Returns:
        Список устройств; пустой, если ни один бэкенд не доступен
        (например, Linux-контейнер без PortAudio).
    """
    devices = _sounddevice_devices()
    if include_loopback:
        devices.extend(_loopback_devices())
    return devices


def backend_report() -> dict[str, str]:
    """Статус бэкендов: имя -> ``"ok"`` или текст ошибки (для CLI и логов)."""
    report: dict[str, str] = {}
    for name, importer in (
        (BACKEND_SOUNDDEVICE, import_sounddevice),
        (BACKEND_PYAUDIOWPATCH, import_pyaudiowpatch),
    ):
        try:
            importer()
        except AudioBackendUnavailable as exc:
            report[name] = str(exc)
        else:
            report[name] = "ok"
    return report


def _known_names(devices: Sequence[AudioDeviceInfo], kind: DeviceKind | None) -> str:
    if not devices:
        return (
            "список устройств пуст — нет доступных аудио-бэкендов "
            "(нужна Windows + sounddevice/PyAudioWPatch, см. engine/audio_io/README.md)"
        )
    names = [d.name for d in devices if kind is None or d.kind is kind]
    if not names:
        return "устройств такого типа не найдено"
    head = names[:12]
    tail = "" if len(names) == len(head) else f" … (+{len(names) - len(head)})"
    return "; ".join(head) + tail


def match_device(
    devices: Sequence[AudioDeviceInfo],
    kind: DeviceKind,
    name: str | None = None,
    *,
    what: str = "устройство",
) -> AudioDeviceInfo:
    """Выбрать устройство заданного типа по подстроке имени.

    Регистр не важен. Если ``name`` не задан — берётся устройство
    по умолчанию, иначе первое подходящее.

    Args:
        devices: из чего выбирать (обычно результат :func:`list_devices`).
        kind: вход / выход / loopback.
        name: подстрока имени; ``None`` — устройство по умолчанию.
        what: как назвать устройство в тексте ошибки.

    Raises:
        DeviceNotFound: подходящего устройства нет.
    """
    candidates = [d for d in devices if d.kind is kind]
    if not candidates:
        raise DeviceNotFound(
            f"{what}: нет устройств типа {kind.value}. Доступно: {_known_names(devices, None)}"
        )

    if name:
        needle = name.strip().lower()
        matched = [d for d in candidates if needle in d.name.lower()]
        if not matched:
            raise DeviceNotFound(
                f"{what}: не найдено устройство {kind.value} с именем, содержащим {name!r}. "
                f"Доступно: {_known_names(devices, kind)}"
            )
        matched.sort(key=lambda d: (not d.is_default, d.id))
        return matched[0]

    for dev in candidates:
        if dev.is_default:
            return dev
    return candidates[0]


def find_default_mic(
    devices: Sequence[AudioDeviceInfo] | None = None,
    name: str | None = None,
) -> AudioDeviceInfo:
    """Физический микрофон пользователя (по имени или устройство по умолчанию)."""
    pool = list_devices(include_loopback=False) if devices is None else devices
    return match_device(pool, DeviceKind.INPUT, name, what="микрофон")


def find_default_output(
    devices: Sequence[AudioDeviceInfo] | None = None,
    name: str | None = None,
) -> AudioDeviceInfo:
    """Устройство вывода — наушники (по имени или устройство по умолчанию)."""
    pool = list_devices(include_loopback=False) if devices is None else devices
    return match_device(pool, DeviceKind.OUTPUT, name, what="наушники")


def find_loopback_device(
    devices: Sequence[AudioDeviceInfo] | None = None,
    name: str | None = None,
) -> AudioDeviceInfo:
    """WASAPI loopback-устройство — захват звука собеседника из Zoom.

    Без ``name`` берётся loopback устройства вывода по умолчанию.
    """
    pool = list_devices() if devices is None else devices
    return match_device(pool, DeviceKind.LOOPBACK, name, what="loopback")


def find_virtual_cable_output(
    devices: Sequence[AudioDeviceInfo] | None = None,
    name: str | None = None,
) -> AudioDeviceInfo:
    """Вход VB-Audio Virtual Cable (``CABLE Input``) — туда пишем перевод.

    В Zoom микрофоном при этом выбирается ``CABLE Output``.

    Raises:
        DeviceNotFound: кабель не установлен (см. README: установка VB-Cable).
    """
    pool = list_devices(include_loopback=False) if devices is None else devices
    needle = name or AudioConfig().cable_name
    try:
        return match_device(pool, DeviceKind.OUTPUT, needle, what="виртуальный кабель")
    except DeviceNotFound as exc:
        cables = [d.name for d in pool if d.kind is DeviceKind.OUTPUT and d.is_virtual_cable]
        hint = (
            f" Похожие устройства с '{VIRTUAL_CABLE_MARKER}': {'; '.join(cables)}"
            if cables
            else " Установите VB-Audio Virtual Cable (см. engine/audio_io/README.md)."
        )
        raise DeviceNotFound(f"{exc}{hint}") from exc
