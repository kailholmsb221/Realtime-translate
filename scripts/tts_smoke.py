#!/usr/bin/env python3
"""Smoke-тест модуля tts: создать голос, синтезировать фразу, записать WAV.

Примеры::

    # создать профиль голоса из своего сэмпла (15-30 с, моно, тихое помещение)
    python scripts/tts_smoke.py --create-voice sample.wav --name me --lang ru

    # синтез конкретной фразы выбранным голосом
    python scripts/tts_smoke.py --say "привет, это тест" --lang ru \\
        --voice v_1a2b3c4d --out out.wav

    # без аргументов: по одному WAV на ru/en/kk в ./out_tts/
    python scripts/tts_smoke.py

    # то же, но без тяжёлых моделей (тон вместо речи) — проверка обвязки
    python scripts/tts_smoke.py --fake

Выход — WAV 24 kHz mono int16 (формат TTS из контрактов). Модели берутся из
``$RT_MODELS_DIR`` (`python scripts/download_models.py`), голоса — из
``$RT_VOICES_DIR`` (по умолчанию ``./voices``).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Final

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # запуск без установки пакета
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402  — после правки sys.path

from engine.contracts.events import Lang  # noqa: E402
from engine.tts import (  # noqa: E402
    OUTPUT_SAMPLE_RATE,
    Backend,
    Pcm16,
    TtsConfig,
    TtsRouter,
    VoiceStore,
    create_tts,
    write_wav,
)

DEFAULT_OUT_DIR: Final[Path] = Path("./out_tts")

#: Демо-фразы для режима «по WAV на каждый язык».
DEMO_TEXTS: Final[dict[Lang, str]] = {
    Lang.RU: "Привет! Это проверка синтеза речи на русском языке.",
    Lang.EN: "Hello! This is a text to speech smoke test in English.",
    Lang.KK: "Сәлеметсіз бе! Бұл қазақ тіліндегі синтез сынағы.",
}


def build_config(args: argparse.Namespace) -> TtsConfig:
    """Конфигурация из окружения плюс переопределения из аргументов."""
    config = TtsConfig.from_env()
    changes: dict[str, object] = {}
    if args.fake:
        changes["backend"] = Backend.FAKE
    if args.voices_dir is not None:
        changes["voices_dir"] = args.voices_dir
    if args.models_dir is not None:
        changes["models_dir"] = args.models_dir
    if not changes:
        return config
    from dataclasses import replace

    return replace(config, **changes)  # type: ignore[arg-type]


def create_voice(config: TtsConfig, sample: Path, name: str, lang: Lang, fake: bool) -> int:
    """Создать профиль голоса и напечатать его ``voice_id``."""
    store = VoiceStore(config.voices_dir)
    latents_fn = None
    if not fake:
        from engine.tts.xtts import XttsProvider

        latents_fn = XttsProvider(config, store).compute_latents

    profile = store.create_voice(sample, name=name, lang=lang, latents_fn=latents_fn)
    print(f"voice_id: {profile.id}")
    print(f"каталог:  {profile.dir}")
    print(f"сэмпл:    {profile.sample_path}")
    if latents_fn is None:
        print("латенты не считались (--fake): XTTS посчитает их при первом синтезе")
    return 0


async def synthesize_to_file(
    router: TtsRouter,
    text: str,
    lang: Lang,
    voice_id: str | None,
    out: Path,
) -> None:
    """Синтезировать фразу и записать WAV, напечатав время и длительность."""
    started = time.perf_counter()
    first_chunk_ms: float | None = None
    chunks: list[Pcm16] = []
    async for chunk in router.synthesize(text, lang, voice_id):
        if first_chunk_ms is None:
            first_chunk_ms = (time.perf_counter() - started) * 1000
        chunks.append(chunk)

    pcm = np.concatenate(chunks).astype(np.int16) if chunks else np.zeros(0, dtype=np.int16)
    write_wav(out, pcm, OUTPUT_SAMPLE_RATE)
    total_ms = (time.perf_counter() - started) * 1000
    audio_ms = 1000 * pcm.size / OUTPUT_SAMPLE_RATE
    print(
        f"[{lang.value}] {out}: {audio_ms:.0f} мс аудио, {len(chunks)} чанков, "
        f"первый чанк через {first_chunk_ms or 0:.0f} мс, всего {total_ms:.0f} мс"
    )


async def run(args: argparse.Namespace) -> int:
    """Основной сценарий smoke-теста."""
    config = build_config(args)
    router = create_tts(config)
    print(f"backend={config.backend.value} device={config.resolve_device().value}")
    print(f"models_dir={config.models_dir} voices_dir={config.voices_dir}")

    if args.say is not None:
        out = args.out or (DEFAULT_OUT_DIR / f"say_{args.lang.value}.wav")
        await synthesize_to_file(router, args.say, args.lang, args.voice, out)
        return 0

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for lang, text in DEMO_TEXTS.items():
        voice = args.voice if lang is not Lang.KK else None
        await synthesize_to_file(router, text, lang, voice, out_dir / f"{lang.value}.wav")
    print(f"готово: {out_dir.resolve()}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-тест синтеза речи (engine/tts)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--create-voice",
        type=Path,
        metavar="SAMPLE.WAV",
        dest="create_voice",
        help="создать профиль голоса из WAV-сэмпла (15-30 с) и напечатать voice_id",
    )
    parser.add_argument("--name", default="me", help="имя голоса для --create-voice")
    parser.add_argument(
        "--lang",
        type=Lang,
        choices=list(Lang),
        default=Lang.RU,
        help="язык синтеза или язык голоса (по умолчанию ru)",
    )
    parser.add_argument("--say", help="синтезировать эту фразу")
    parser.add_argument("--voice", default=None, help="voice_id профиля (ru/en)")
    parser.add_argument("--out", type=Path, default=None, help="куда писать WAV для --say")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"каталог для режима по умолчанию (по умолчанию {DEFAULT_OUT_DIR})",
    )
    parser.add_argument("--voices-dir", type=Path, default=None, help="вместо RT_VOICES_DIR")
    parser.add_argument("--models-dir", type=Path, default=None, help="вместо RT_MODELS_DIR")
    parser.add_argument(
        "--fake",
        action="store_true",
        help="использовать FakeTts (тон вместо речи) — проверка обвязки без моделей",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)
    try:
        if args.create_voice is not None:
            return create_voice(
                build_config(args), args.create_voice, args.name, args.lang, args.fake
            )
        return asyncio.run(run(args))
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        print(f"ошибка: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
