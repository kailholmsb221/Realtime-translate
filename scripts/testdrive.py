"""Личный тест-драйв: говорите в микрофон — движок отвечает вашим переведённым голосом.

Отличие от ``python -m engine.orchestrator``: здесь поднимается **только** конвейер
``out`` (микрофон → STT → MT → TTS → вывод). WASAPI loopback и VB-Audio Virtual Cable
не нужны, поэтому проверить «сказал фразу → услышал её перевод своим голосом» можно
до установки кабеля.

Каждая озвученная фраза дополнительно сохраняется в каталог ``--out-dir``
(по умолчанию ``voice_out/``) вместе с ``transcript.txt``.

Полудуплекс (``--tail-ms``, по умолчанию включён): пока играет перевод, микрофон
подменяется тишиной. Без этого на ноутбучных динамиках встроенный микрофон слышит
синтезированную речь, движок распознаёт её как новую фразу и уходит в петлю.
С наушниками полудуплекс можно выключить (``--no-half-duplex``).

Примеры::

    # 1. Записать 20 секунд своей речи и создать голосовой профиль
    python scripts/testdrive.py --record-sample --name me --lang ru

    # 2. Тест-драйв: говорите по-русски, слышите свой голос по-английски
    python scripts/testdrive.py --from ru --to en

    # 3. Список голосов / выбор конкретного
    python scripts/testdrive.py --list-voices
    python scripts/testdrive.py --voice v_1a2b3c4d
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import re
import sys
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Final

import numpy as np

from engine.audio_io import (
    TTS_FORMAT,
    AudioChunk,
    AudioSink,
    AudioSource,
    create_sink,
    create_source,
)
from engine.audio_io.config import AudioConfig
from engine.contracts.events import (
    EVENT_STT_FINAL,
    EVENT_STT_PARTIAL,
    Lang,
    Stream,
    SttFinal,
    SttPartial,
)
from engine.stt import create_engine
from engine.stt.base import SttConfig
from engine.translate import create_provider
from engine.translate.base import TranslateConfig
from engine.tts import create_tts, create_voice_store
from engine.tts.audio import write_wav
from engine.tts.base import OUTPUT_SAMPLE_RATE, TtsConfig
from engine.tts.voices import VoiceProfile, VoiceStore

logger = logging.getLogger("testdrive")

#: Сколько тишины подмешивать после конца проигрывания перевода, мс.
DEFAULT_TAIL_MS: Final[int] = 500

#: Длительность сэмпла для клонирования голоса по умолчанию, с.
DEFAULT_SAMPLE_SECONDS: Final[float] = 20.0

#: Частота записи сэмпла голоса (XTTS всё равно сведёт к 24 kHz).
SAMPLE_RATE_RECORD: Final[int] = 48_000


#: Заученные «титры», которые whisper выдаёт на не-речи.
#: Модель обучалась на ютубовских субтитрах, где такие строки идут поверх
#: тишины в конце роликов, поэтому на шуме она воспроизводит именно их.
#: Штатной защиты в движке нет: в ``whisper_fw.py`` стоит ``temperature=0``
#: (fallback выключен) и не заданы ``no_speech_threshold`` /
#: ``hallucination_silence_threshold``.
HALLUCINATIONS: Final[tuple[str, ...]] = (
    "субтитр",
    "редактор субтитров",
    "корректор",
    "dimatorzok",
    "спасибо за просмотр",
    "продолжение следует",
    "amara.org",
    "субтитры сделал",
    "субтитры создавал",
)


def is_hallucination(text: str) -> bool:
    """Похоже ли распознанное на заученный артефакт whisper, а не на речь."""
    low = text.strip().lower()
    return any(marker in low for marker in HALLUCINATIONS)


def _slug(text: str, limit: int = 40) -> str:
    """Кусок текста, пригодный для имени файла."""
    cleaned = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE).strip()
    cleaned = re.sub(r"\s+", "_", cleaned)
    return cleaned[:limit] or "phrase"


def patch_torchaudio_load() -> bool:
    """Починить чтение WAV, если ``torchaudio.load`` не работает.

    С PyTorch 2.9 ``torchaudio.load`` декодирует через ``torchcodec``, а его
    нативная библиотека на Windows/Python 3.14 может не загрузиться (ей не
    хватает зависимостей). Тогда падает клонирование голоса: XTTS читает
    сэмпл именно этим вызовом (``TTS/tts/models/xtts.py::load_audio``).

    Сэмплы голоса — обычные PCM-WAV, поэтому подменяем загрузчик на
    ``soundfile``. Патч живёт здесь, а не в ``engine/`` (CLAUDE.md, закон 1):
    модули движка не трогаем.

    Returns:
        ``True``, если подмена понадобилась и выполнена.
    """
    import torch
    import torchaudio

    probe = Path(__file__).resolve().parents[1] / "engine/tests/fixtures/tone_440hz_1s.wav"
    try:
        torchaudio.load(str(probe))
    except Exception as exc:
        logger.warning("torchaudio.load сломан (%s) — читаю WAV через soundfile", exc)
    else:
        return False

    import soundfile as sf

    def load_via_soundfile(path: Any, *args: Any, **kwargs: Any) -> tuple[Any, int]:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
        return torch.from_numpy(np.ascontiguousarray(data.T)), int(rate)

    torchaudio.load = load_via_soundfile
    return True


# --- запись сэмпла голоса --------------------------------------------------


def record_sample(path: Path, seconds: float) -> Path:
    """Записать сэмпл своей речи с микрофона в WAV (mono 48 kHz)."""
    try:
        import sounddevice as sd
    except ImportError as exc:  # pragma: no cover — зависит от окружения
        raise SystemExit(
            "не установлен sounddevice: pip install -r engine/audio_io/requirements-audio_io.txt"
        ) from exc

    device = sd.query_devices(kind="input")
    print(f"Микрофон: {device['name']}")
    print(f"Говорите {seconds:.0f} секунд — читайте что угодно связным текстом.")
    for i in (3, 2, 1):
        print(f"  {i}...", flush=True)
        time.sleep(1)
    print("  ЗАПИСЬ ПОШЛА", flush=True)

    frames = int(seconds * SAMPLE_RATE_RECORD)
    data = sd.rec(frames, samplerate=SAMPLE_RATE_RECORD, channels=1, dtype="int16")
    sd.wait()
    print("  готово")

    samples = np.asarray(data, dtype=np.int16).reshape(-1)
    peak = int(np.abs(samples).max()) if samples.size else 0
    rms = (
        float(np.sqrt(np.mean((samples.astype(np.float32) / 32768.0) ** 2)))
        if samples.size
        else 0.0
    )
    print(f"  пик {peak}/32767, RMS {rms:.4f}")
    if rms < 0.005:
        print("  ВНИМАНИЕ: сигнал очень тихий — проверьте, что говорили в микрофон")

    write_wav(path, samples, SAMPLE_RATE_RECORD)
    print(f"  сэмпл: {path}")
    return path


def create_profile(
    store: VoiceStore, sample: Path, name: str, lang: Lang, config: TtsConfig
) -> VoiceProfile:
    """Создать голосовой профиль и сразу посчитать латенты XTTS.

    Латенты считаются здесь, а не лениво при первом синтезе: так профиль
    готов и для ``python -m engine.orchestrator`` (у движка нет обходного
    патча ``torchaudio.load``, см. :func:`patch_torchaudio_load`).
    """
    patch_torchaudio_load()
    print("\nСчитаю латенты XTTS (грузится модель, ~30 секунд)...")
    provider = create_tts(config).provider_for(lang)
    latents_fn = getattr(provider, "compute_latents", None)
    profile = store.create_voice(sample, name, lang, latents_fn)
    print(f"\nГолос создан: id={profile.id}  имя={profile.name}  язык={profile.lang.value}")
    print(f"  профиль: {profile.dir}")
    print(f"  latents: {'посчитаны' if profile.latents_path.is_file() else 'будут при синтезе'}")
    return profile


def pick_voice(store: VoiceStore, voice_id: str | None) -> str | None:
    """Выбрать профиль: явный id, иначе первый не-автоклон, иначе встроенный голос."""
    profiles = list(store.list())
    if voice_id is not None:
        if not any(p.id == voice_id for p in profiles):
            raise SystemExit(f"голос {voice_id!r} не найден. Список: --list-voices")
        return voice_id
    own = [p for p in profiles if not p.name.startswith("auto_")]
    if own:
        print(f"Голос: {own[0].name} ({own[0].id})")
        return own[0].id
    print("Голос: встроенный XTTS (своего профиля нет — сделайте --record-sample)")
    return None


def build_mic(config: AudioConfig, index: int | None) -> AudioSource:
    """Источник микрофона: по индексу устройства или штатным подбором по имени.

    ``RT_AUDIO_MIC`` ищет подстроку в имени, а Windows показывает один и тот же
    микрофон под несколькими host API (MME, DirectSound, WASAPI) с почти
    одинаковыми именами. Если рабочий только один из них, выбрать его по имени
    нельзя — отсюда ``--mic-index``. Номера берутся из
    ``python scripts/audio_smoke.py --list``.
    """
    if index is None:
        return create_source("mic", config)

    from engine.audio_io.devices import list_devices
    from engine.audio_io.windows import MicSource

    device = next((d for d in list_devices(include_loopback=False) if d.id == index), None)
    if device is None:
        raise SystemExit(f"устройства с id={index} нет — см. python scripts/audio_smoke.py --list")
    rate = int(device.default_sample_rate)
    print(f"Микрофон: id={device.id} {device.name!r} ({rate} Hz)")
    return MicSource(
        device,
        fmt=config.format,
        chunk_ms=config.chunk_ms,
        timeout_s=config.device_timeout_s,
    )


def with_gain(source: AudioSource, gain: float) -> AsyncIterator[AudioChunk]:
    """Усилить поток перед распознаванием.

    Встроенный микрофон ноутбука часто отдаёт пик ~0.03 вместо привычных
    0.1-0.5. На таком уровне Silero VAD пропускает начала фраз, а whisper
    путает слова и выдумывает текст на паузах. Правильное лечение — поднять
    уровень входа в настройках Windows; ``--gain`` даёт то же самое здесь,
    с защитой от клиппинга.
    """

    async def gained() -> AsyncIterator[AudioChunk]:
        async for chunk in source:
            samples = chunk.samples.astype(np.float32) * gain
            np.clip(samples, -32768.0, 32767.0, out=samples)
            yield AudioChunk.from_array(samples.astype(np.int16), ts_ms=chunk.ts_ms, fmt=chunk.fmt)

    return gained()


# --- полудуплексный источник -----------------------------------------------


class HalfDuplexGate:
    """Глушит микрофон, пока играет синтезированный перевод.

    Чанк не выбрасывается, а заменяется тишиной: таймлайн потока (и таймкоды
    ``stt.final``) остаётся непрерывным, а VAD видит паузу.
    """

    def __init__(self, tail_ms: int = DEFAULT_TAIL_MS) -> None:
        self._tail_s = max(0, tail_ms) / 1000.0
        self._muted_until = 0.0
        self.muted_chunks = 0

    @property
    def muted(self) -> bool:
        return time.monotonic() < self._muted_until

    def mute_now(self) -> None:
        """Держать микрофон закрытым прямо сейчас (во время синтеза)."""
        self._muted_until = time.monotonic() + self._tail_s + 3600.0

    def release_after_tail(self) -> None:
        """Открыть микрофон через ``tail_ms`` после конца проигрывания."""
        self._muted_until = time.monotonic() + self._tail_s

    def wrap(self, source: AudioSource) -> AsyncIterator[AudioChunk]:
        """Обернуть источник: во время проигрывания отдавать тишину."""

        async def gated() -> AsyncIterator[AudioChunk]:
            async for chunk in source:
                if self.muted:
                    self.muted_chunks += 1
                    yield AudioChunk.from_array(
                        np.zeros(chunk.n_frames, dtype=np.int16), ts_ms=chunk.ts_ms, fmt=chunk.fmt
                    )
                else:
                    yield chunk

        return gated()


# --- тест-драйв ------------------------------------------------------------


class TestDrive:
    """Микрофон → STT → MT → TTS → вывод, с сохранением каждой фразы в WAV."""

    def __init__(
        self,
        lang_src: Lang,
        lang_dst: Lang,
        voice_id: str | None,
        out_dir: Path,
        gate: HalfDuplexGate | None,
        gain: float = 0.0,
    ) -> None:
        self.lang_src = lang_src
        self.lang_dst = lang_dst
        self.voice_id = voice_id
        self.out_dir = out_dir
        self.gate = gate
        #: 0 — подобрать усиление по замеру уровня, иначе фиксированный множитель.
        self.gain = gain
        self.index = 0
        self.transcript = out_dir / "transcript.txt"

    # Подбирать усиление по фоновому уровню нельзя: микрофонные массивы
    # ноутбуков (AMD/Intel) глушат фон шумодавом в цифровой ноль и
    # открываются только на речь. Фон ничего не говорит об уровне голоса,
    # поэтому усиление задаётся явно через --gain.

    async def check_mic(self, source: AudioSource, seconds: float = 1.5) -> float:
        """Снять уровень со входа: мёртвый микрофон должен быть виден сразу.

        Без этой проверки замьюченный вход выглядит как «движок ничего не
        делает»: VAD просто никогда не срабатывает.
        """
        deadline = time.monotonic() + seconds
        peak = 0.0
        async for chunk in source:
            samples = chunk.samples.astype(np.float32) / 32768.0
            if samples.size:
                peak = max(peak, float(np.abs(samples).max()))
            if time.monotonic() >= deadline:
                break
        if peak < 1e-4:
            print(f"  фон на входе: {peak:.5f} — почти ноль.")
            print("  Это норма для массивов с шумодавом: фон глушится, гейт")
            print("  открывается только на речь. Если движок не реагирует и")
            print("  на голос — проверьте мьют и выбор входа (RT_AUDIO_MIC).")
        else:
            print(f"  фон на входе: {peak:.4f}")
        return peak

    async def run(self, source: AudioSource, sink: AudioSink, stt: Any, mt: Any, tts: Any) -> None:
        """Основной цикл: слушать микрофон и отвечать переведённым голосом."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        await self.check_mic(source)
        gain = self.gain if self.gain > 0 else 1.0
        boosted: Any = with_gain(source, gain) if gain != 1.0 else source
        if gain != 1.0:
            print(f"  усиление входа: x{gain:.1f}")
        stream = self.gate.wrap(boosted) if self.gate is not None else boosted

        print("\n" + "=" * 72)
        print(f"  ГОВОРИТЕ  ({self.lang_src.value} → {self.lang_dst.value}).  Ctrl+C — выход.")
        print(f"  Перевод звучит в динамик и пишется в {self.out_dir}")
        if self.gate is not None:
            print("  Полудуплекс включён: микрофон молчит, пока играет перевод.")
        print("=" * 72 + "\n")

        # Перевод и синтез уходят в фоновый воркер, как в engine/orchestrator/
        # pipeline.py: иначе чтение микрофона встаёт на всё время озвучки
        # (несколько секунд), очередь захвата переполняется и аудио теряется —
        # «MicSource: очередь захвата переполнена, чанк отброшен».
        queue: asyncio.Queue[SttFinal | None] = asyncio.Queue()
        worker = asyncio.create_task(self._worker(queue, sink, mt, tts), name="testdrive-worker")
        try:
            async for envelope in stt.run(stream, Stream.OUT, self.lang_src.value):
                payload = envelope.payload
                if envelope.type == EVENT_STT_PARTIAL and isinstance(payload, SttPartial):
                    print(f"  … {payload.text}", end="\r", flush=True)
                elif envelope.type == EVENT_STT_FINAL and isinstance(payload, SttFinal):
                    queue.put_nowait(payload)
        finally:
            queue.put_nowait(None)
            with contextlib.suppress(asyncio.CancelledError):
                await worker

    async def _worker(
        self,
        queue: asyncio.Queue[SttFinal | None],
        sink: AudioSink,
        mt: Any,
        tts: Any,
    ) -> None:
        """Разбирает фразы по очереди: перевод → синтез → файл."""
        while True:
            payload = await queue.get()
            if payload is None:
                return
            try:
                await self._handle_final(payload, sink, mt, tts)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("фраза пропущена")

    async def _handle_final(self, payload: SttFinal, sink: AudioSink, mt: Any, tts: Any) -> None:
        if is_hallucination(payload.text):
            print(f"\n  (пропущен артефакт whisper: {payload.text!r})")
            return

        self.index += 1
        started = time.perf_counter()
        print(f"\n[{self.index:02d}] вы сказали : {payload.text}")

        result = await mt.translate(payload.text, payload.lang, self.lang_dst)
        mt_ms = int((time.perf_counter() - started) * 1000)
        print(f"     перевод   : {result.text}   ({mt_ms} мс)")
        if not result.text.strip():
            return

        await self._speak(result.text, sink, tts, payload.text)

    async def _speak(self, text: str, sink: AudioSink, tts: Any, source_text: str) -> None:
        """Синтезировать перевод, проиграть и сохранить в WAV."""
        if self.gate is not None:
            self.gate.mute_now()

        started = time.perf_counter()
        first_ms: int | None = None
        pieces: list[np.ndarray[Any, Any]] = []
        try:
            async for pcm in tts.synthesize(text, self.lang_dst, self.voice_id):
                if first_ms is None:
                    first_ms = int((time.perf_counter() - started) * 1000)
                pieces.append(np.asarray(pcm, dtype=np.int16).reshape(-1))
                await sink.write(AudioChunk.from_array(pcm, ts_ms=0, fmt=TTS_FORMAT))
            with contextlib.suppress(Exception):
                await sink.drain()
        except Exception:
            logger.exception("синтез не удался")
        finally:
            if self.gate is not None:
                self.gate.release_after_tail()

        if not pieces:
            print("     (синтезировать нечего)")
            return

        audio = np.concatenate(pieces)
        name = f"{self.index:02d}_{self.lang_dst.value}_{_slug(text)}.wav"
        path = self.out_dir / name
        write_wav(path, audio, OUTPUT_SAMPLE_RATE)
        seconds = audio.size / OUTPUT_SAMPLE_RATE
        print(f"     озвучено  : {seconds:.1f} с, первый чанк через {first_ms} мс")
        print(f"     файл      : {path.name}")

        with self.transcript.open("a", encoding="utf-8") as fh:
            fh.write(f"[{self.index:02d}] {self.lang_src.value}: {source_text}\n")
            fh.write(f"     {self.lang_dst.value}: {text}\n")
            fh.write(f"     wav: {name}\n\n")


# --- сборка и точка входа --------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/testdrive.py",
        description="Личный тест-драйв: говорите в микрофон, слышите свой переведённый голос",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--from", dest="lang_src", default="ru", choices=[x.value for x in Lang])
    parser.add_argument("--to", dest="lang_dst", default="en", choices=[x.value for x in Lang])
    parser.add_argument("--voice", dest="voice_id", default=None, help="voice_id профиля голоса")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("voice_out"), help="куда складывать WAV-ответы"
    )
    parser.add_argument("--log-level", default="WARNING")
    parser.add_argument(
        "--mic-index",
        type=int,
        default=None,
        help="номер входа из scripts/audio_smoke.py --list (перебивает RT_AUDIO_MIC)",
    )

    group = parser.add_argument_group("голосовой профиль")
    group.add_argument(
        "--record-sample", action="store_true", help="записать свою речь и создать профиль голоса"
    )
    group.add_argument("--seconds", type=float, default=DEFAULT_SAMPLE_SECONDS)
    group.add_argument("--name", default="me", help="имя профиля голоса")
    group.add_argument("--list-voices", action="store_true", help="показать профили и выйти")

    duplex = parser.add_argument_group("эхо")
    duplex.add_argument(
        "--no-half-duplex",
        dest="half_duplex",
        action="store_false",
        help="не глушить микрофон во время проигрывания (только с наушниками!)",
    )
    duplex.add_argument("--tail-ms", type=int, default=DEFAULT_TAIL_MS)
    duplex.add_argument(
        "--gain",
        type=float,
        default=0.0,
        help="усиление микрофона (0 — подобрать автоматически, 1 — не усиливать)",
    )
    return parser


async def _amain(args: argparse.Namespace) -> int:
    lang_src, lang_dst = Lang(args.lang_src), Lang(args.lang_dst)
    tts_config = TtsConfig.from_env()
    store = create_voice_store(tts_config)

    if args.list_voices:
        profiles = list(store.list())
        if not profiles:
            print("профилей нет — создайте: python scripts/testdrive.py --record-sample")
        for profile in profiles:
            print(f"  {profile.id}  {profile.name:20s} {profile.lang.value}  {profile.sample_path}")
        return 0

    if args.record_sample:
        sample = Path("voice_sample.wav")
        record_sample(sample, args.seconds)
        create_profile(store, sample, args.name, lang_src, tts_config)
        print("\nТеперь запускайте тест-драйв: python scripts/testdrive.py")
        return 0

    voice_id = pick_voice(store, args.voice_id)

    print("\nЗагружаю модели (первый раз — долго, веса читаются с диска)...")
    if patch_torchaudio_load():
        print("  (torchaudio.load подменён на soundfile — см. patch_torchaudio_load)")
    stt = create_engine(SttConfig.from_env(), backend="faster_whisper", vad_backend="silero")
    mt = create_provider(TranslateConfig.from_env())
    tts = create_tts(tts_config)

    step = time.perf_counter()
    await asyncio.to_thread(
        stt.transcriber.transcribe, np.zeros(16_000, dtype=np.int16), lang_src.value
    )
    print(f"  STT (faster-whisper small) готов  — {time.perf_counter() - step:.1f} с")

    step = time.perf_counter()
    await tts.warmup((lang_dst,))
    print(f"  TTS (XTTS-v2) готов               — {time.perf_counter() - step:.1f} с")

    step = time.perf_counter()
    await asyncio.to_thread(mt.warmup)
    print(f"  MT (NLLB-200, CPU) готов          — {time.perf_counter() - step:.1f} с")

    audio_config = AudioConfig.from_env()
    source = build_mic(audio_config, args.mic_index)
    sink = create_sink("headphones", audio_config)
    gate = HalfDuplexGate(args.tail_ms) if args.half_duplex else None
    drive = TestDrive(lang_src, lang_dst, voice_id, args.out_dir, gate, args.gain)

    try:
        await drive.run(source, sink, stt, mt, tts)
    finally:
        with contextlib.suppress(Exception):
            await source.close()
        with contextlib.suppress(Exception):
            await sink.drain()
            await sink.close()
        print(f"\nВсего фраз: {drive.index}. Файлы: {args.out_dir.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.WARNING),
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    try:
        return asyncio.run(_amain(args))
    except KeyboardInterrupt:
        print("\nостановлено")
        return 0


if __name__ == "__main__":
    sys.exit(main())
