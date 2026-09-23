"""Тесты акустического эхоподавления — ``scripts/aec.py``.

Реального динамика и микрофона тут нет: эхо строится синтетически — опорный
сигнал сворачивается с импульсной характеристикой «комнаты» и задерживается,
как это делает звуковой тракт. Проверяется то, ради чего фильтр и нужен:

* эхо давится (ERLE), значит whisper перестаёт слышать собственную озвучку;
* речь человека поверх эха (двойной разговор) не выгрызается — иначе фраза
  теряется, а ровно от этого режим и страдал;
* пока динамик молчит, сигнал проходит нетронутым.

``scripts/`` лежит на ``sys.path`` — это делает ``engine/tests/conftest.py``.
"""

from __future__ import annotations

import numpy as np
from aec import EchoCanceller, erle_db

RATE = 16_000
CHUNK = 480  # 30 мс — чанк движка


def room_impulse(delay_ms: float = 45.0, taps: int = 400, decay: float = 25.0) -> np.ndarray:
    """Импульсная характеристика: задержка вывода плюс затухающие отражения."""
    rng = np.random.default_rng(7)
    delay = int(RATE * delay_ms / 1000)
    response = np.zeros(delay + taps, dtype=np.float32)
    tail = rng.normal(0.0, 1.0, taps) * np.exp(-np.arange(taps) / decay)
    response[delay:] = tail.astype(np.float32)
    response[delay] = 1.0
    return response * 0.5


def speech_like(seconds: float, seed: int = 1) -> np.ndarray:
    """Речеподобный сигнал: шум с медленной огибающей и окраской."""
    rng = np.random.default_rng(seed)
    n = int(RATE * seconds)
    noise = rng.normal(0.0, 0.3, n).astype(np.float32)
    # Простейшая окраска — скользящее среднее (низкочастотный акцент речи).
    kernel = np.ones(8, dtype=np.float32) / 8
    coloured = np.convolve(noise, kernel, mode="same")
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / (RATE * 0.35))
    return (coloured * envelope).astype(np.float32)


def to_int16(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 32768.0, -32768.0, 32767.0).astype(np.int16)


def run(
    canceller: EchoCanceller,
    reference: np.ndarray,
    mic: np.ndarray,
) -> np.ndarray:
    """Прогнать сигналы чанками, как это делает живой конвейер."""
    out: list[np.ndarray] = []
    for start in range(0, len(mic) - CHUNK + 1, CHUNK):
        ts_ms = start * 1000 / RATE
        canceller.add_reference(to_int16(reference[start : start + CHUNK]), ts_ms)
        cleaned = canceller.process(to_int16(mic[start : start + CHUNK]), ts_ms)
        out.append(cleaned.astype(np.float32) / 32768.0)
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def test_echo_is_suppressed() -> None:
    """Эхо собственной озвучки давится — микрофон можно не глушить."""
    reference = speech_like(6.0, seed=2)
    echo = np.convolve(reference, room_impulse(), mode="full")[: len(reference)]

    canceller = EchoCanceller(RATE)
    cleaned = run(canceller, reference, echo)

    # Первые секунды фильтр сходится, меряем по установившемуся режиму.
    tail_from = int(RATE * 3.0)
    gain = erle_db(echo[tail_from : len(cleaned)], cleaned[tail_from:])
    assert gain > 12.0, f"эхо ослаблено всего на {gain:.1f} дБ"
    assert canceller.stats.adapted > 0


def test_double_talk_keeps_human_voice() -> None:
    """Речь поверх озвучки сохраняется — ради этого всё и затевалось."""
    reference = speech_like(8.0, seed=3)
    echo = np.convolve(reference, room_impulse(), mode="full")[: len(reference)]
    near = np.zeros_like(reference)
    # Человек начинает говорить на пятой секунде, когда фильтр уже сошёлся.
    speech_from = int(RATE * 5.0)
    near[speech_from:] = speech_like(len(reference[speech_from:]) / RATE, seed=9)[
        : len(reference) - speech_from
    ]

    canceller = EchoCanceller(RATE)
    cleaned = run(canceller, reference, echo + near)

    window = slice(speech_from, len(cleaned))
    kept = float(np.corrcoef(cleaned[window], near[window])[0, 1])
    assert kept > 0.8, f"голос человека искажён, корреляция {kept:.2f}"

    residual_echo = erle_db(echo[window], cleaned[window] - near[window])
    assert residual_echo > 6.0, f"эхо при двойном разговоре ослаблено на {residual_echo:.1f} дБ"
    assert canceller.stats.double_talk > 0, "двойной разговор должен распознаваться"


def test_silent_speaker_passes_audio_through() -> None:
    """Пока динамик молчит, микрофон не трогаем вовсе."""
    mic = speech_like(1.0, seed=4)
    canceller = EchoCanceller(RATE)
    cleaned = run(canceller, np.zeros_like(mic), mic)

    assert np.allclose(cleaned, to_int16(mic[: len(cleaned)]).astype(np.float32) / 32768.0)
    assert canceller.stats.adapted == 0


def test_filter_never_makes_input_louder() -> None:
    """Расходящийся фильтр не подмешивает шум: вывод не громче входа.

    Ровно так режим и «глючил со временем»: веса уползали, эхоподавитель
    начинал добавлять в микрофон собственную оценку, whisper слышал шум и
    выдумывал слова. Теперь блок, где стало громче, отбрасывается, а веса
    забываются.
    """
    reference = speech_like(4.0, seed=11)
    # Микрофон почти молчит: эха нет вовсе, только слабый шум.
    mic = (np.random.default_rng(12).normal(0.0, 0.002, len(reference))).astype(np.float32)

    canceller = EchoCanceller(RATE)
    # Испортим веса, как если бы фильтр уже разошёлся.
    canceller._weights[:] = 5.0
    cleaned = run(canceller, reference, mic)

    out_power = float(np.mean(np.square(cleaned, dtype=np.float64)))
    in_power = float(np.mean(np.square(mic[: len(cleaned)], dtype=np.float64)))
    assert out_power <= in_power * 1.6, "вывод громче входа — фильтр добавляет шум"
    assert canceller.stats.rejected > 0, "расхождение должно быть замечено"


def test_quiet_reference_does_not_blow_up_weights() -> None:
    """Тихие бины опорного сигнала не раздувают шаг адаптации."""
    reference = speech_like(5.0, seed=13) * 0.02  # очень тихая озвучка
    echo = np.convolve(reference, room_impulse(), mode="full")[: len(reference)]
    canceller = EchoCanceller(RATE)
    cleaned = run(canceller, reference, echo)

    assert np.isfinite(canceller._weights).all()
    tail_from = int(RATE * 2.5)
    assert erle_db(echo[tail_from : len(cleaned)], cleaned[tail_from:]) > 6.0


def test_delay_estimate_finds_the_echo_lag() -> None:
    """Оценка задержки находит, на сколько эхо отстаёт от опорного сигнала."""
    reference = speech_like(4.0, seed=21)
    echo = np.convolve(reference, room_impulse(delay_ms=180.0), mode="full")[: len(reference)]
    canceller = EchoCanceller(RATE)
    run(canceller, reference, echo)

    lag_ms, corr = canceller.estimate_delay()
    assert corr > 0.3, f"эхо должно коррелировать с опорным сигналом, корр {corr:.2f}"
    assert 120 <= lag_ms <= 260, f"ожидал лаг около 180 мс, получил {lag_ms}"


def test_reset_forgets_everything() -> None:
    """После reset фильтр чист — например, сменили устройство вывода."""
    reference = speech_like(2.0, seed=5)
    echo = np.convolve(reference, room_impulse(), mode="full")[: len(reference)]
    canceller = EchoCanceller(RATE)
    run(canceller, reference, echo)
    assert canceller.stats.blocks > 0

    canceller.reset()
    assert canceller.stats.blocks == 0
    assert canceller.stats.adapted == 0


def test_erle_metric() -> None:
    """Метрика ERLE: вдвое тише по амплитуде — примерно 6 дБ."""
    signal = speech_like(0.5, seed=6)
    assert erle_db(signal, signal * 0.5) == np.float32(erle_db(signal, signal * 0.5))
    assert 5.5 < erle_db(signal, signal * 0.5) < 6.5
    assert erle_db(np.zeros(10), np.zeros(10)) == 0.0
