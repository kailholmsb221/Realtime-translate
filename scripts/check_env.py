#!/usr/bin/env python3
"""Проверка окружения перед запуском движка.

Печатает:

* версию Python и платформу (нужен Python 3.11);
* наличие PyTorch, CUDA и объём VRAM (бюджет — 6 GB, ARCHITECTURE.md раздел 6);
* список аудиоустройств через ``sounddevice``;
* есть ли устройство с «CABLE» в названии — VB-Audio Virtual Cable, через который
  переведённый голос уходит в Zoom (ARCHITECTURE.md раздел 4.1).

Все импорты защищены: скрипт не падает, если библиотека ещё не установлена, а
пишет, что и как поставить. Код возврата 0 — всё критичное на месте, 1 — нет.

Запуск::

    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import shutil
import sys

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"

MIN_PYTHON = (3, 11)
CABLE_HINT = "CABLE"


def line(title: str) -> None:
    width = min(shutil.get_terminal_size((80, 20)).columns, 80)
    print("\n" + title)
    print("-" * min(len(title), width))


def check_python() -> bool:
    """Версия Python и платформа."""
    line("Python")
    print(f"       {sys.version.splitlines()[0]}")
    print(f"       {platform.platform()}  ({platform.machine()})")
    print(f"       исполняемый файл: {sys.executable}")

    ok = sys.version_info[:2] >= MIN_PYTHON
    if ok:
        print(f"{OK} версия >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]}")
    else:
        print(f"{FAIL} нужен Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+")
    if platform.system() != "Windows":
        print(f"{WARN} целевая ОС проекта — Windows (WASAPI loopback, VB-Audio Virtual Cable)")
    return ok


def check_torch() -> bool:
    """PyTorch, CUDA и VRAM."""
    line("PyTorch / CUDA")
    try:
        import torch
    except ImportError:
        print(f"{WARN} torch не установлен")
        print("       поставьте сборку с CUDA, напр.:")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False

    print(f"       torch {torch.__version__}")
    print(f"       собран с CUDA: {torch.version.cuda or 'нет (CPU-сборка)'}")

    try:
        available = torch.cuda.is_available()
    except Exception as exc:  # драйвер/железо могут бросить что угодно
        print(f"{WARN} torch.cuda.is_available() упал: {exc}")
        return False

    if not available:
        print(f"{WARN} CUDA недоступна — STT и TTS пойдут на CPU, задержка > 2.5 с")
        return False

    print(f"{OK} CUDA доступна, устройств: {torch.cuda.device_count()}")
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        vram_gb = props.total_memory / (1024**3)
        print(f"       [{idx}] {props.name}: {vram_gb:.1f} GB VRAM")
        if vram_gb < 5.5:
            print(f"{WARN}      меньше расчётных 6 GB — держите бюджет раздела 6 ARCHITECTURE.md")
    return True


def check_audio() -> bool:
    """Аудиоустройства и VB-Audio Virtual Cable."""
    line("Аудиоустройства (sounddevice)")
    try:
        import sounddevice as sd
    except ImportError:
        print(f"{WARN} sounddevice не установлен — список устройств недоступен")
        print("       pip install -r engine/audio_io/requirements-audio_io.txt")
        print("       (или pip install sounddevice)")
        return False
    except OSError as exc:  # нет PortAudio / нет звуковой подсистемы
        print(f"{WARN} sounddevice не запустился: {exc}")
        return False

    try:
        devices = list(sd.query_devices())
    except Exception as exc:
        print(f"{WARN} не удалось получить список устройств: {exc}")
        return False

    if not devices:
        print(f"{WARN} аудиоустройств не найдено")
        return False

    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "?"))
        ins = dev.get("max_input_channels", 0)
        outs = dev.get("max_output_channels", 0)
        rate = dev.get("default_samplerate", 0)
        kind = f"in:{ins} out:{outs}"
        print(f"       [{idx:>2}] {name}  ({kind}, {rate:.0f} Hz)")

    cable = [
        str(d.get("name", "")) for d in devices if CABLE_HINT in str(d.get("name", "")).upper()
    ]
    if cable:
        print(f"{OK} найден VB-Audio Virtual Cable: {', '.join(sorted(set(cable)))}")
        print("       в Zoom выберите его микрофоном, чтобы собеседник слышал перевод")
        return True

    print(f"{WARN} устройства с «{CABLE_HINT}» в названии нет")
    print("       поставьте VB-Audio Virtual Cable вручную: https://vb-audio.com/Cable/")
    print("       (установку системных программ агенты не делают — CLAUDE.md, закон 5)")
    return False


def main() -> int:
    print("Realtime Translator — проверка окружения")
    python_ok = check_python()
    torch_ok = check_torch()
    audio_ok = check_audio()

    line("Итог")
    print(f"       Python 3.11+ : {'да' if python_ok else 'НЕТ'}")
    print(f"       CUDA         : {'да' if torch_ok else 'нет'}")
    print(f"       VB-Cable     : {'да' if audio_ok else 'нет'}")
    if not python_ok:
        print(f"{FAIL} окружение не готово: обновите Python")
        return 1
    if not (torch_ok and audio_ok):
        print(f"{WARN} окружение готово частично — см. предупреждения выше")
    else:
        print(f"{OK} окружение готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
