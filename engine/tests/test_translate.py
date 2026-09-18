"""Тесты модуля перевода (engine/translate, ARCHITECTURE.md 4.3).

Юнит-тесты работают без весов и без `torch`: используется `FakeProvider`,
а у `NllbProvider` подменяется единственный метод, который трогает модель.
Тесты с реальной NLLB помечены `real_models` и идут только при `RT_REAL_MODELS=1`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path

import pytest

from engine.contracts.events import (
    EVENT_TRANSLATION_READY,
    Envelope,
    Lang,
    Stream,
    SttFinal,
    TranslationReady,
    parse_event,
    to_json,
)
from engine.translate import (
    DEFAULT_CONTEXT_SIZE,
    DEFAULT_MODEL_ID,
    FakeProvider,
    TranslateConfig,
    TranslationCache,
    TranslationError,
    TranslationProvider,
    TranslationResult,
    TranslationService,
    create_provider,
    default_num_threads,
)
from engine.translate.nllb import NLLB_LANG_CODES, NllbProvider, nllb_code

DIRECTIONS: tuple[tuple[Lang, Lang], ...] = (
    (Lang.RU, Lang.EN),
    (Lang.EN, Lang.RU),
    (Lang.RU, Lang.KK),
    (Lang.KK, Lang.RU),
    (Lang.EN, Lang.KK),
    (Lang.KK, Lang.EN),
)

ENV_VARS = (
    "RT_MT_MODEL",
    "RT_MT_DEVICE",
    "RT_MT_THREADS",
    "RT_MT_INT8",
    "RT_MT_BACKEND",
    "RT_MT_MAX_NEW_TOKENS",
    "RT_MT_BEAMS",
    "RT_MT_CACHE",
    "RT_MODELS_DIR",
)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Убрать RT_*-переменные, чтобы конфиг собирался из умолчаний."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def stt_final(text: str, lang: Lang, stream: Stream = Stream.IN) -> Envelope:
    """Готовый конверт stt.final (валидируется схемой при сборке)."""
    return parse_event(to_json(SttFinal(stream, lang, text, 0, 940)))


class CountingNllb(NllbProvider):
    """NLLB без модели: считает обращения к «генерации» и возвращает заглушку."""

    def __init__(self, config: TranslateConfig | None = None) -> None:
        super().__init__(config)
        self.generate_calls = 0

    def _generate(self, text: str, src_code: str, dst_code: str) -> str:
        self.generate_calls += 1
        return f"{dst_code}:{text}"


# --- коды языков NLLB -------------------------------------------------------


@pytest.mark.parametrize(
    ("lang", "code"),
    [(Lang.RU, "rus_Cyrl"), (Lang.EN, "eng_Latn"), (Lang.KK, "kaz_Cyrl")],
)
def test_nllb_lang_codes(lang: Lang, code: str) -> None:
    """Все три языка проекта имеют корректный код FLORES-200."""
    assert nllb_code(lang) == code
    assert NLLB_LANG_CODES[lang] == code


def test_nllb_codes_cover_all_project_langs() -> None:
    """В таблице нет лишних и недостающих языков."""
    assert set(NLLB_LANG_CODES) == set(Lang)


def test_nllb_code_unknown_lang() -> None:
    """Неизвестный язык — понятная ошибка, а не KeyError."""
    with pytest.raises(TranslationError):
        nllb_code("de")  # type: ignore[arg-type]


# --- сервис: stt.final -> translation.ready ---------------------------------


@pytest.mark.parametrize(("src", "dst"), DIRECTIONS)
async def test_service_builds_valid_translation_ready(src: Lang, dst: Lang) -> None:
    """Для всех 6 направлений собирается валидный translation.ready."""
    provider = FakeProvider()
    service = TranslationService(provider)

    envelope = await service.handle(stt_final("одна фраза", src), dst, ref_utterance_id=17)

    assert envelope.type == EVENT_TRANSLATION_READY
    payload = envelope.payload
    assert isinstance(payload, TranslationReady)
    assert payload.stream is Stream.IN
    assert payload.src_lang is src
    assert payload.dst_lang is dst
    assert payload.src_text == "одна фраза"
    assert payload.text == f"[{dst.value}] одна фраза"
    assert payload.ref_utterance_id == 17
    # сериализация проходит валидацию схемой контракта
    assert parse_event(to_json(envelope)).payload == payload


async def test_service_keeps_stream_and_null_ref() -> None:
    """Поток out и ref_utterance_id=None проходят контракт."""
    service = TranslationService(FakeProvider())
    envelope = await service.handle(stt_final("hello", Lang.EN, Stream.OUT), Lang.RU)
    payload = envelope.payload
    assert isinstance(payload, TranslationReady)
    assert payload.stream is Stream.OUT
    assert payload.ref_utterance_id is None
    assert parse_event(to_json(envelope)).type == EVENT_TRANSLATION_READY


async def test_service_rejects_wrong_event() -> None:
    """Сервис принимает только stt.final."""
    service = TranslationService(FakeProvider())
    ready = Envelope.wrap(TranslationReady(Stream.IN, Lang.RU, Lang.EN, "a", "b", None))
    with pytest.raises(TranslationError):
        await service.handle(ready, Lang.EN)


async def test_same_lang_short_circuit() -> None:
    """src == dst: текст без изменений, латентность 0, провайдер не вызывается."""
    provider = FakeProvider()
    service = TranslationService(provider)

    envelope = await service.handle(stt_final("не трогай меня", Lang.RU), Lang.RU)

    payload = envelope.payload
    assert isinstance(payload, TranslationReady)
    assert payload.text == "не трогай меня"
    assert provider.calls == 0
    assert service.last_latency_ms == 0


async def test_empty_text_not_sent_to_provider() -> None:
    """Пустая (и пробельная) фраза не доходит до модели."""
    provider = FakeProvider()
    service = TranslationService(provider)

    envelope = await service.handle(stt_final("   ", Lang.RU), Lang.EN)

    payload = envelope.payload
    assert isinstance(payload, TranslationReady)
    assert payload.text == ""
    assert payload.src_text == "   "
    assert provider.calls == 0
    assert service.last_latency_ms == 0
    assert service.context(Stream.IN) == ()


async def test_provider_empty_text_short_circuit() -> None:
    """Сам провайдер тоже защищён от пустой строки."""
    provider = FakeProvider()
    result = await provider.translate("", Lang.RU, Lang.EN)
    assert result == TranslationResult(text="", src=Lang.RU, dst=Lang.EN, latency_ms=0)
    assert provider.calls == 0


async def test_fake_provider_delay_is_awaited() -> None:
    """Задержка FakeProvider реально ждётся — тесты таймингов работают."""
    provider = FakeProvider(delay=0.05)
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await provider.translate("привет", Lang.RU, Lang.EN)
    assert loop.time() - started >= 0.05
    assert result.latency_ms >= 40


async def test_fake_provider_table_overrides_prefix() -> None:
    """Словарь пар важнее префикса по умолчанию."""
    provider = FakeProvider(table={(Lang.RU, Lang.EN, "привет"): "hello"})
    assert (await provider.translate("привет", Lang.RU, Lang.EN)).text == "hello"
    assert (await provider.translate("пока", Lang.RU, Lang.EN)).text == "[en] пока"


# --- LRU-кэш ----------------------------------------------------------------


async def test_nllb_cache_skips_second_call() -> None:
    """Повтор той же фразы берётся из кэша — модель не дёргается."""
    provider = CountingNllb(TranslateConfig(backend="nllb"))

    first = await provider.translate("Привет, как дела?", Lang.RU, Lang.EN)
    second = await provider.translate("Привет, как дела?", Lang.RU, Lang.EN)

    assert provider.generate_calls == 1
    assert second.text == first.text
    assert provider.cache.hits == 1

    # другое направление — отдельный ключ
    await provider.translate("Привет, как дела?", Lang.RU, Lang.KK)
    assert provider.generate_calls == 2


async def test_nllb_no_model_call_on_empty_and_same_lang() -> None:
    """Пустой текст и src == dst не доходят до модели и стоят 0 мс."""
    provider = CountingNllb()

    empty = await provider.translate("  ", Lang.RU, Lang.EN)
    same = await provider.translate("как есть", Lang.RU, Lang.RU)

    assert provider.generate_calls == 0
    assert empty.text == ""
    assert empty.latency_ms == 0
    assert same.text == "как есть"
    assert same.latency_ms == 0


async def test_nllb_cache_size_zero_disables_cache() -> None:
    """cache_size=0 выключает кэш (нужно для честного --bench)."""
    provider = CountingNllb(TranslateConfig(cache_size=0))
    await provider.translate("фраза", Lang.RU, Lang.EN)
    await provider.translate("фраза", Lang.RU, Lang.EN)
    assert provider.generate_calls == 2


def test_cache_evicts_least_recently_used() -> None:
    """LRU: вытесняется самый давно не использованный элемент."""
    cache = TranslationCache(maxsize=2)
    cache.put(Lang.RU, Lang.EN, "a", "A")
    cache.put(Lang.RU, Lang.EN, "b", "B")
    assert cache.get(Lang.RU, Lang.EN, "a") == "A"  # «a» снова свежий
    cache.put(Lang.RU, Lang.EN, "c", "C")

    assert len(cache) == 2
    assert cache.get(Lang.RU, Lang.EN, "b") is None
    assert cache.get(Lang.RU, Lang.EN, "a") == "A"
    assert cache.get(Lang.RU, Lang.EN, "c") == "C"


def test_cache_keeps_last_256_by_default() -> None:
    """Размер кэша по умолчанию — 256 последних переводов."""
    cache = TranslationCache(maxsize=256)
    for i in range(300):
        cache.put(Lang.RU, Lang.EN, f"фраза {i}", f"phrase {i}")
    assert len(cache) == 256
    assert cache.get(Lang.RU, Lang.EN, "фраза 43") is None
    assert cache.get(Lang.RU, Lang.EN, "фраза 299") == "phrase 299"


# --- окно контекста ---------------------------------------------------------


class RecordingProvider:
    """Провайдер, запоминающий контекст каждого вызова."""

    def __init__(self) -> None:
        self.contexts: list[tuple[str, ...]] = []

    def warmup(self) -> None:
        return None

    async def translate(
        self,
        text: str,
        src: Lang,
        dst: Lang,
        context: Sequence[str] = (),
    ) -> TranslationResult:
        self.contexts.append(tuple(context))
        return TranslationResult(text=f"<{text}>", src=src, dst=dst, latency_ms=1)


async def test_context_window_is_bounded() -> None:
    """Окно контекста хранит ровно N последних фраз потока."""
    provider = RecordingProvider()
    service = TranslationService(provider, context_size=DEFAULT_CONTEXT_SIZE)

    for i in range(5):
        await service.handle(stt_final(f"фраза {i}", Lang.RU), Lang.EN)

    assert service.context_size == 3
    assert service.context(Stream.IN) == ("фраза 2", "фраза 3", "фраза 4")
    # контекст последнего вызова — три фразы до него
    assert provider.contexts[-1] == ("фраза 1", "фраза 2", "фраза 3")
    assert provider.contexts[0] == ()


async def test_context_is_per_stream_and_resettable() -> None:
    """Потоки in/out не смешиваются, reset чистит окно."""
    service = TranslationService(RecordingProvider())

    await service.handle(stt_final("собеседник", Lang.EN, Stream.IN), Lang.RU)
    await service.handle(stt_final("пользователь", Lang.RU, Stream.OUT), Lang.EN)

    assert service.context(Stream.IN) == ("собеседник",)
    assert service.context(Stream.OUT) == ("пользователь",)

    service.reset(Stream.IN)
    assert service.context(Stream.IN) == ()
    assert service.context(Stream.OUT) == ("пользователь",)

    service.reset()
    assert service.context(Stream.OUT) == ()


def test_negative_context_size_rejected() -> None:
    """Отрицательное окно — ошибка конфигурации."""
    with pytest.raises(TranslationError):
        TranslationService(FakeProvider(), context_size=-1)


# --- конфигурация -----------------------------------------------------------


@pytest.mark.usefixtures("clean_env")
def test_config_defaults() -> None:
    """Умолчания: NLLB-600M, CPU, greedy, int8, min(4, cpu_count) потоков."""
    config = TranslateConfig.from_env()
    assert config.model_id == DEFAULT_MODEL_ID
    assert config.device == "cpu"
    assert config.backend == "nllb"
    assert config.num_beams == 1
    assert config.max_new_tokens == 128
    assert config.quantize_int8 is True
    assert config.cache_size == 256
    assert config.threads == default_num_threads()
    assert config.threads <= 4


@pytest.mark.usefixtures("clean_env")
def test_config_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Все документированные переменные окружения читаются."""
    monkeypatch.setenv("RT_MT_MODEL", "some/other-mt")
    monkeypatch.setenv("RT_MT_DEVICE", "cuda")
    monkeypatch.setenv("RT_MT_THREADS", "2")
    monkeypatch.setenv("RT_MT_INT8", "0")
    monkeypatch.setenv("RT_MT_BACKEND", "fake")
    monkeypatch.setenv("RT_MT_BEAMS", "2")
    monkeypatch.setenv("RT_MT_MAX_NEW_TOKENS", "64")
    monkeypatch.setenv("RT_MT_CACHE", "8")
    monkeypatch.setenv("RT_MODELS_DIR", str(tmp_path))

    config = TranslateConfig.from_env()

    assert config.model_id == "some/other-mt"
    assert config.device == "cuda"
    assert config.num_threads == 2
    assert config.threads == 2
    assert config.quantize_int8 is False
    assert config.backend == "fake"
    assert config.num_beams == 2
    assert config.max_new_tokens == 64
    assert config.cache_size == 8
    assert config.models_dir == tmp_path
    assert config.local_model_dir == tmp_path / "nllb"
    # локальных весов нет — источником остаётся id модели
    assert config.model_source() == "some/other-mt"


@pytest.mark.usefixtures("clean_env")
def test_config_rejects_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Мусор в переменных окружения — понятная ошибка."""
    monkeypatch.setenv("RT_MT_THREADS", "много")
    with pytest.raises(TranslationError):
        TranslateConfig.from_env()
    monkeypatch.delenv("RT_MT_THREADS")

    monkeypatch.setenv("RT_MT_BACKEND", "openai")
    with pytest.raises(TranslationError):
        TranslateConfig.from_env()


@pytest.mark.usefixtures("clean_env")
def test_config_uses_local_weights_when_downloaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Если scripts/download_models.py скачал nllb — берём локальный каталог."""
    local = tmp_path / "nllb"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("RT_MODELS_DIR", str(tmp_path))

    assert TranslateConfig.from_env().model_source() == str(local)


# --- фабрика ----------------------------------------------------------------


def test_create_provider_fake() -> None:
    """backend=fake отдаёт FakeProvider и не тянет transformers."""
    provider = create_provider(TranslateConfig(backend="fake"))
    assert isinstance(provider, FakeProvider)
    assert isinstance(provider, TranslationProvider)


def test_create_provider_nllb_is_lazy() -> None:
    """backend=nllb создаёт провайдера, но веса ещё не грузит."""
    provider = create_provider(TranslateConfig(backend="nllb"))
    assert isinstance(provider, NllbProvider)
    assert isinstance(provider, TranslationProvider)
    assert provider.loaded is False


def test_create_provider_unknown_backend() -> None:
    """Неизвестный бэкенд — TranslationError."""
    with pytest.raises(TranslationError):
        create_provider(TranslateConfig(backend="llm"))  # type: ignore[arg-type]


# --- реальная модель --------------------------------------------------------

REAL_CASES: tuple[tuple[Lang, Lang, str, tuple[str, ...]], ...] = (
    (Lang.RU, Lang.EN, "Привет, как дела?", ("how", "hello", "hi")),
    (Lang.EN, Lang.RU, "Hello, how are you?", ("прив", "как", "здрав")),
    (Lang.RU, Lang.KK, "Привет, как дела?", ("сәлем", "қалай", "халің")),
    (Lang.KK, Lang.RU, "Сәлем, қалың қалай?", ("прив", "как", "здрав")),
    (Lang.EN, Lang.KK, "Good morning, my friend.", ("таң", "дос", "қайырлы")),
    (Lang.KK, Lang.EN, "Сәлем, қалың қалай?", ("hello", "how", "hi")),
)


@pytest.fixture(scope="module")
def real_provider() -> NllbProvider:
    """Одна прогретая NLLB на весь модуль: загрузка весов стоит секунды."""
    provider = NllbProvider(TranslateConfig(backend="nllb"))
    provider.warmup()
    return provider


@pytest.mark.real_models
@pytest.mark.parametrize(("src", "dst", "text", "expected"), REAL_CASES)
async def test_real_nllb_translations(
    real_provider: NllbProvider,
    src: Lang,
    dst: Lang,
    text: str,
    expected: tuple[str, ...],
) -> None:
    """Настоящая NLLB переводит все 6 направлений осмысленно (ключевые слова)."""
    result = await real_provider.translate(text, src, dst)
    lowered = result.text.lower()
    assert lowered, "перевод пустой"
    assert any(word in lowered for word in expected), f"{src}->{dst}: {result.text!r}"


@pytest.mark.real_models
async def test_real_service_roundtrip(real_provider: NllbProvider) -> None:
    """Сервис с реальной моделью отдаёт валидный translation.ready."""
    service = TranslationService(real_provider)
    envelope = await service.handle(stt_final("Доброе утро, коллеги.", Lang.RU), Lang.EN, 1)
    payload = envelope.payload
    assert isinstance(payload, TranslationReady)
    assert payload.text.strip()
    assert parse_event(to_json(envelope)).type == EVENT_TRANSLATION_READY
