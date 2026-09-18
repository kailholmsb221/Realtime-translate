#!/usr/bin/env python3
"""Скачивание всех моделей проекта в локальный кэш.

Только бесплатные open-source модели из ARCHITECTURE.md (CLAUDE.md, закон 3):

============  ==========================================  ==========================
ключ          репозиторий                                 роль
============  ==========================================  ==========================
``whisper``   ``Systran/faster-whisper-small``            STT, int8_float16, ~1 GB VRAM
``nllb``      ``facebook/nllb-200-distilled-600M``        перевод, CPU
``vad``       pip-пакет ``silero-vad``                    нарезка потока на фразы
``xtts``      ``coqui/XTTS-v2``                           TTS ru/en с клоном голоса
``kazakhtts`` ``issai/...`` (см. ``--kazakhtts-repo``)    TTS kk, без клона
============  ==========================================  ==========================

Кэш — каталог ``./models`` рядом с репозиторием, переопределяется переменной
окружения ``RT_MODELS_DIR``.

Запуск::

    python scripts/download_models.py                  # всё
    python scripts/download_models.py --only whisper nllb
    python scripts/download_models.py --list
    python scripts/download_models.py --only kazakhtts --kazakhtts-repo issai/<точный-id>

Модель ``kazakhtts`` по умолчанию только **проверяется на доступность**
(``--check-only`` включён для неё автоматически, см. ниже): точный id репозитория
ISSAI KazakhTTS2 на Hugging Face не зафиксирован в ARCHITECTURE.md, поэтому он
вынесен в параметр. Уточните его у владельца проекта и передайте
``--kazakhtts-repo``, либо задайте ``RT_KAZAKHTTS_REPO``.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "models"

# Точный id KazakhTTS2 на Hugging Face не подтверждён владельцем — параметр.
# TODO(владелец): зафиксировать репозиторий KazakhTTS2 (ISSAI) в ARCHITECTURE.md.
DEFAULT_KAZAKHTTS_REPO = "issai/KazakhTTS2"

OK = "[ OK ]"
WARN = "[WARN]"
FAIL = "[FAIL]"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """Описание модели для скачивания."""

    key: str
    repo_id: str
    role: str
    repo_type: str = "model"
    allow_patterns: tuple[str, ...] | None = None
    #: только проверить доступность, не качать (id не подтверждён владельцем)
    check_only: bool = False
    notes: str = ""


def model_specs(kazakhtts_repo: str) -> list[ModelSpec]:
    """Список моделей проекта."""
    return [
        ModelSpec(
            key="whisper",
            repo_id="Systran/faster-whisper-small",
            role="STT: faster-whisper small, compute_type=int8_float16 (~1.0 GB VRAM)",
        ),
        ModelSpec(
            key="nllb",
            repo_id="facebook/nllb-200-distilled-600M",
            role="Перевод: NLLB-200-distilled-600M, запускается на CPU",
        ),
        ModelSpec(
            key="xtts",
            repo_id="coqui/XTTS-v2",
            role="TTS ru/en: XTTS-v2 с клонированием голоса (~2.5 GB VRAM)",
        ),
        ModelSpec(
            key="kazakhtts",
            repo_id=kazakhtts_repo,
            role="TTS kk: KazakhTTS2 (ISSAI), без клонирования голоса",
            check_only=True,
            notes="id не зафиксирован в ARCHITECTURE.md — уточните --kazakhtts-repo",
        ),
    ]


# Silero VAD ставится pip-пакетом, а не через huggingface_hub.
VAD_KEY = "vad"
VAD_PIP_PACKAGE = "silero-vad"

ALL_KEYS: tuple[str, ...] = ("whisper", "nllb", VAD_KEY, "xtts", "kazakhtts")


@dataclass(slots=True)
class Result:
    """Итог по одному ключу."""

    key: str
    status: str
    detail: str = ""
    path: str = ""


@dataclass(slots=True)
class Report:
    """Сводка запуска."""

    results: list[Result] = field(default_factory=list)

    def add(self, key: str, status: str, detail: str = "", path: str = "") -> None:
        self.results.append(Result(key=key, status=status, detail=detail, path=path))
        print(f"{status} {key}: {detail}" if detail else f"{status} {key}")

    @property
    def failed(self) -> bool:
        return any(r.status == FAIL for r in self.results)


def models_dir() -> Path:
    """Каталог кэша моделей (``RT_MODELS_DIR`` или ``./models``)."""
    raw = os.environ.get("RT_MODELS_DIR")
    return Path(raw).expanduser().resolve() if raw else DEFAULT_MODELS_DIR


def download_hf(spec: ModelSpec, cache_dir: Path, report: Report) -> None:
    """Скачать (или проверить) репозиторий Hugging Face."""
    try:
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError
    except ImportError:
        report.add(spec.key, FAIL, "нет huggingface_hub — pip install huggingface_hub")
        return

    target = cache_dir / spec.key

    if spec.check_only:
        try:
            from huggingface_hub import HfApi

            info = HfApi().repo_info(spec.repo_id, repo_type=spec.repo_type)
            report.add(
                spec.key,
                OK,
                f"{spec.repo_id} доступен ({len(info.siblings or [])} файлов), "
                f"скачивание отключено (--check-only)",
            )
        except RepositoryNotFoundError:
            report.add(
                spec.key,
                WARN,
                f"{spec.repo_id} не найден на Hugging Face. {spec.notes}",
            )
        except HfHubHTTPError as exc:
            report.add(spec.key, WARN, f"{spec.repo_id}: ошибка HTTP {exc}")
        except Exception as exc:
            report.add(spec.key, WARN, f"{spec.repo_id}: {exc}")
        return

    print(f"       качаю {spec.repo_id} -> {target}")
    try:
        path = snapshot_download(
            repo_id=spec.repo_id,
            repo_type=spec.repo_type,
            local_dir=str(target),
            cache_dir=str(cache_dir / ".hf-cache"),
            allow_patterns=list(spec.allow_patterns) if spec.allow_patterns else None,
        )
    except RepositoryNotFoundError:
        report.add(spec.key, FAIL, f"{spec.repo_id} не найден (нужен ли доступ/логин?)")
        return
    except Exception as exc:
        report.add(spec.key, FAIL, f"{spec.repo_id}: {exc}")
        return

    report.add(spec.key, OK, f"{spec.repo_id}", path=path)


def ensure_vad(report: Report) -> None:
    """Проверить наличие pip-пакета silero-vad (веса тянутся вместе с ним)."""
    try:
        import silero_vad  # noqa: F401
    except ImportError:
        report.add(
            VAD_KEY,
            WARN,
            f"пакет {VAD_PIP_PACKAGE} не установлен — pip install {VAD_PIP_PACKAGE} "
            f"(добавьте его в engine/stt/requirements-stt.txt)",
        )
        return
    report.add(VAD_KEY, OK, f"пакет {VAD_PIP_PACKAGE} установлен, веса идут в комплекте")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Скачать модели Realtime Translator в локальный кэш",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=ALL_KEYS,
        metavar="KEY",
        help=f"скачать только указанные модели: {', '.join(ALL_KEYS)}",
    )
    parser.add_argument(
        "--kazakhtts-repo",
        default=os.environ.get("RT_KAZAKHTTS_REPO", DEFAULT_KAZAKHTTS_REPO),
        help=f"репозиторий KazakhTTS2 на Hugging Face (по умолчанию {DEFAULT_KAZAKHTTS_REPO})",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="каталог кэша (по умолчанию $RT_MODELS_DIR или ./models)",
    )
    parser.add_argument("--list", action="store_true", help="показать список моделей и выйти")
    args = parser.parse_args()

    specs = model_specs(args.kazakhtts_repo)

    if args.list:
        print("Модели проекта (ARCHITECTURE.md разделы 4.2-4.4, 6):\n")
        for spec in specs:
            flag = "  [только проверка]" if spec.check_only else ""
            print(f"  {spec.key:<10} {spec.repo_id}{flag}\n             {spec.role}")
        print(f"  {VAD_KEY:<10} pip {VAD_PIP_PACKAGE}\n             VAD: нарезка потока на фразы")
        return 0

    cache_dir = (args.models_dir or models_dir()).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only) if args.only else set(ALL_KEYS)
    print(f"Кэш моделей: {cache_dir}")
    print(f"Выбрано: {', '.join(k for k in ALL_KEYS if k in selected)}\n")

    report = Report()
    for spec in specs:
        if spec.key in selected:
            download_hf(spec, cache_dir, report)
    if VAD_KEY in selected:
        ensure_vad(report)

    print("\nИтог:")
    for res in report.results:
        print(f"  {res.status} {res.key:<10} {res.path or res.detail}")
    print(f"\nПуть к моделям: {cache_dir}  (переопределяется RT_MODELS_DIR)")
    return 1 if report.failed else 0


if __name__ == "__main__":
    sys.exit(main())
