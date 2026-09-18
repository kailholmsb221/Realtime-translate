#!/usr/bin/env python3
"""Смоук-тест и бенчмарк модуля перевода (ARCHITECTURE.md 4.3).

Запуск::

    python scripts/translate_smoke.py --from ru --to en "Привет, как дела?"
    python scripts/translate_smoke.py --bench            # 6 направлений, таблица задержек
    python scripts/translate_smoke.py --fake --from ru --to kk "Привет"   # без весов

По умолчанию берётся настоящий NLLB-200-distilled-600M на CPU: первый запуск
качает ~2.5 GB с Hugging Face (или используйте
``python scripts/download_models.py --only nllb``). Флаг ``--fake`` подменяет
провайдера заглушкой — удобно проверить обвязку без моделей.

Конфиг читается из окружения (``RT_MT_*``, ``RT_MODELS_DIR``) и
переопределяется флагами ниже.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from dataclasses import replace
from itertools import permutations
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

from engine.contracts.events import Lang  # noqa: E402  — после правки sys.path
from engine.translate import (  # noqa: E402
    TranslateConfig,
    TranslationProvider,
    create_provider,
)

#: Фразы по 15 слов для бенчмарка — типичная длина реплики в созвоне.
BENCH_PHRASES: dict[Lang, str] = {
    Lang.RU: (
        "Сегодня мы обсудим план работы на следующую неделю и договоримся "
        "о времени следующего короткого созвона."
    ),
    Lang.EN: ("Today we discuss the work plan for next week and agree on the next call."),
    Lang.KK: (
        "Бүгін біз келесі аптаның жұмыс жоспарын талқылап, келесі қысқа "
        "қоңыраудың нақты уақытын өзара келісіп аламыз."
    ),
}

#: Все 6 направлений перевода проекта (ru↔en, ru↔kk, en↔kk).
DIRECTIONS: tuple[tuple[Lang, Lang], ...] = tuple(permutations((Lang.RU, Lang.EN, Lang.KK), 2))

BENCH_RUNS = 3


def build_config(args: argparse.Namespace) -> TranslateConfig:
    """Конфиг из окружения с поправками на флаги командной строки."""
    config = TranslateConfig.from_env()
    if args.fake:
        config = replace(config, backend="fake")
    if args.model:
        config = replace(config, model_id=args.model)
    if args.threads:
        config = replace(config, num_threads=args.threads)
    if args.beams:
        config = replace(config, num_beams=args.beams)
    if args.no_int8:
        config = replace(config, quantize_int8=False)
    return config


def describe(config: TranslateConfig) -> str:
    """Однострочное описание конфигурации для шапки вывода."""
    return (
        f"backend={config.backend} model={config.model_source()} device={config.device} "
        f"threads={config.threads} int8={config.quantize_int8} beams={config.num_beams}"
    )


async def run_once(
    provider: TranslationProvider,
    text: str,
    src: Lang,
    dst: Lang,
) -> int:
    """Перевести одну фразу и напечатать результат. Вернуть латентность, мс."""
    result = await provider.translate(text, src, dst)
    print(f"{src.value} -> {dst.value}")
    print(f"  вход:   {text}")
    print(f"  выход:  {result.text}")
    print(f"  латентность: {result.latency_ms} мс")
    return result.latency_ms


async def run_bench(provider: TranslationProvider, runs: int) -> None:
    """Прогнать 6 направлений на фразе из 15 слов и напечатать таблицу."""
    started = time.perf_counter()
    await asyncio.to_thread(provider.warmup)
    print(f"Прогрев: {int((time.perf_counter() - started) * 1000)} мс\n")

    print(f"{'напр.':<10}{'мин, мс':>10}{'сред, мс':>10}{'макс, мс':>10}  перевод")
    print("-" * 78)
    for src, dst in DIRECTIONS:
        text = BENCH_PHRASES[src]
        latencies: list[int] = []
        translated = ""
        for _ in range(runs):
            result = await provider.translate(text, src, dst)
            latencies.append(result.latency_ms)
            translated = result.text
        average = sum(latencies) // len(latencies)
        head = translated if len(translated) <= 34 else translated[:31] + "..."
        print(
            f"{src.value + '->' + dst.value:<10}{min(latencies):>10}"
            f"{average:>10}{max(latencies):>10}  {head}"
        )
    print(
        "\nБюджет задержки всей цепочки — 2.5 с (ARCHITECTURE.md раздел 2); "
        "на перевод закладываем не более ~700 мс."
    )


async def main_async(args: argparse.Namespace) -> int:
    config = build_config(args)
    if args.bench:
        # В бенчмарке кэш только мешает: повторный прогон вернул бы 0 мс.
        config = replace(config, cache_size=0)
    print(f"Конфиг: {describe(config)}\n")

    provider = create_provider(config)
    if args.bench:
        await run_bench(provider, args.runs)
        return 0

    text = " ".join(args.text).strip()
    if not text:
        print("Нечего переводить: передайте текст или используйте --bench", file=sys.stderr)
        return 2
    await run_once(provider, text, Lang(args.src), Lang(args.dst))
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Разобрать аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Смоук-тест перевода NLLB (CPU) для Realtime Translator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    langs = [lang.value for lang in Lang]
    parser.add_argument("--from", dest="src", choices=langs, default="ru", help="язык оригинала")
    parser.add_argument("--to", dest="dst", choices=langs, default="en", help="язык перевода")
    parser.add_argument("text", nargs="*", help="текст для перевода")
    parser.add_argument("--bench", action="store_true", help="прогнать 6 направлений и таблицу")
    parser.add_argument("--fake", action="store_true", help="FakeProvider, без загрузки весов")
    parser.add_argument("--model", default=None, help="id модели или путь (иначе RT_MT_MODEL)")
    parser.add_argument("--threads", type=int, default=0, help="потоков CPU (иначе RT_MT_THREADS)")
    parser.add_argument("--beams", type=int, default=0, help="num_beams (1 = greedy)")
    parser.add_argument("--no-int8", action="store_true", help="без динамической квантизации")
    parser.add_argument(
        "--runs",
        type=int,
        default=BENCH_RUNS,
        help=f"прогонов на направление в --bench (по умолчанию {BENCH_RUNS})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Точка входа скрипта."""
    return asyncio.run(main_async(parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
