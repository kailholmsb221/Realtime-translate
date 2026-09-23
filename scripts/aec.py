"""Акустическое эхоподавление: слушать микрофон, пока играет перевод.

Задача. На динамиках микрофон слышит и человека, и собственную озвучку.
Полудуплекс (подмена входа тишиной на время проигрывания) решает петлю, но
съедает речь: человек говорит поверх перевода, и эта часть фразы теряется.

Идея. Сигнал, который уходит в динамик, известен точно — это чанки TTS. Значит
эхо можно вычесть: адаптивный фильтр оценивает, как сигнал динамика доходит до
микрофона (задержка вывода, отражения комнаты, АЧХ), и вычитает эту оценку из
записи. Остаётся голос человека, и распознавание работает не прерываясь.

Как устроено:

* :meth:`EchoCanceller.add_reference` кладёт то, что уходит в динамик, на
  временную ось потока — по моменту, когда чанк реально заиграет (вызывающий
  считает его по заполненности буфера приёмника);
* :meth:`EchoCanceller.process` чистит микрофонный блок: partitioned block
  frequency-domain adaptive filter (PBFDAF) — тот же NLMS, но по блокам через
  FFT, поэтому хвост в 300+ мс стоит доли процента CPU;
* адаптация замирает, когда человек говорит одновременно с динамиком (double
  talk): иначе фильтр «разъедется» и начнёт выгрызать живую речь. Вычитание
  при этом продолжается уже настроенным фильтром.

Модуль самостоятельный (numpy и стандартная библиотека), поэтому его легко
проверить синтетикой: эхо = свёртка опорного сигнала с импульсной
характеристикой, метрика — ERLE (насколько тише стало эхо).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

__all__ = ["AecStats", "EchoCanceller", "erle_db"]

Float32Array = npt.NDArray[np.float32]
Int16Array = npt.NDArray[np.int16]

#: Размер блока обработки, сэмплов при 16 кГц (15 мс — делитель чанка 30 мс).
DEFAULT_BLOCK: Final[int] = 240

#: Длина хвоста фильтра, мс: задержка вывода + отражения комнаты.
DEFAULT_TAIL_MS: Final[int] = 300

#: Насколько раньше расчётного момента может заиграть чанк, мс.
#:
#: Фильтр причинный: «отрицательную» задержку (звук заиграл раньше, чем мы
#: пометили) он не покроет. Поэтому вызывающий обязан подавать ``play_at_ms``
#: с недооценкой — момент, раньше которого чанк точно не зазвучит (для
#: приёмника это «сейчас плюс то, что ещё лежит в его буфере»). При таком
#: контракте запас не нужен и по умолчанию равен нулю; параметр оставлен на
#: случай приёмника, который о своей буферизации не сообщает.
DEFAULT_PRE_ROLL_MS: Final[int] = 0

#: Сколько секунд опорного сигнала держим в кольцевом буфере.
DEFAULT_REF_SECONDS: Final[float] = 6.0

#: Шаг адаптации NLMS (0..1): больше — быстрее сходится, но шумнее.
DEFAULT_MU: Final[float] = 0.4

#: Ниже этой мощности опорный сигнал считаем тишиной — эха нет, чистить нечего.
REF_SILENCE_POWER: Final[float] = 1e-7

#: С какой ERLE фильтр считается сошедшимся: до этого детектор двойного
#: разговора не включаем, иначе он заморозит обучение на первом же блоке.
CONVERGED_ERLE_DB: Final[float] = 4.0

#: Какая доля энергии входа должна остаться после вычитания, чтобы считать это
#: речью человека поверх эха, а не плохо настроенным фильтром.
DOUBLE_TALK_SHARE: Final[float] = 0.5


def erle_db(echo: npt.NDArray[np.floating], residual: npt.NDArray[np.floating]) -> float:
    """Echo Return Loss Enhancement, дБ: насколько тише стало эхо.

    10 дБ — эхо ослаблено в 10 раз по мощности; для распознавания речи этого
    уже достаточно, 20 дБ — эха практически не слышно.
    """
    before = float(np.mean(np.square(echo, dtype=np.float64)))
    after = float(np.mean(np.square(residual, dtype=np.float64)))
    if before <= 0.0:
        return 0.0
    return float(10.0 * np.log10(before / max(after, 1e-20)))


@dataclass(slots=True)
class AecStats:
    """Счётчики для логов и тестов."""

    #: Обработано блоков.
    blocks: int = 0
    #: Блоков, на которых фильтр адаптировался.
    adapted: int = 0
    #: Блоков, распознанных как двойной разговор (адаптация заморожена).
    double_talk: int = 0
    #: Скользящая оценка подавления эха, дБ.
    erle_db: float = 0.0


class EchoCanceller:
    """Вычитает из микрофона то, что играет в динамике.

    Args:
        sample_rate: частота потока (движок работает на 16 кГц).
        block: размер блока обработки, сэмплов.
        tail_ms: длина хвоста фильтра — задержка вывода плюс реверберация.
        pre_roll_ms: запас назад по оси времени на случай, если чанк заиграл
            раньше расчётного момента.
        mu: шаг адаптации NLMS.
        ref_seconds: глубина кольцевого буфера опорного сигнала.
    """

    __slots__ = (
        "_block",
        "_freq_bins",
        "_mu",
        "_partitions",
        "_pending",
        "_pending_pos",
        "_pre_roll",
        "_ref",
        "_ref_end",
        "_ref_len",
        "_ref_powers",
        "_sample_rate",
        "_spectra",
        "_weights",
        "stats",
    )

    def __init__(
        self,
        sample_rate: int = 16_000,
        *,
        block: int = DEFAULT_BLOCK,
        tail_ms: int = DEFAULT_TAIL_MS,
        pre_roll_ms: int = DEFAULT_PRE_ROLL_MS,
        mu: float = DEFAULT_MU,
        ref_seconds: float = DEFAULT_REF_SECONDS,
    ) -> None:
        if block <= 0:
            raise ValueError(f"block должен быть > 0, получено {block}")
        if not 0.0 < mu <= 2.0:
            raise ValueError(f"mu должен быть в (0, 2], получено {mu}")
        self._sample_rate = sample_rate
        self._block = block
        self._mu = mu
        self._pre_roll = int(sample_rate * max(0, pre_roll_ms) / 1000)
        tail = max(block, int(sample_rate * tail_ms / 1000) + self._pre_roll)
        self._partitions = max(1, -(-tail // block))  # ceil
        self._freq_bins = block + 1  # rfft длины 2*block

        self._weights = np.zeros((self._partitions, self._freq_bins), dtype=np.complex128)
        self._spectra = np.zeros((self._partitions, self._freq_bins), dtype=np.complex128)
        #: Мощность каждого запомненного окна опорного сигнала (временная область).
        self._ref_powers = np.zeros(self._partitions, dtype=np.float64)

        self._ref_len = max(int(sample_rate * ref_seconds), 4 * block)
        self._ref = np.zeros(self._ref_len, dtype=np.float32)
        #: До какой абсолютной позиции опорный сигнал уже записан (в сэмплах).
        self._ref_end = 0
        #: Необработанный хвост входа и его абсолютная позиция.
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_pos = 0
        self.stats = AecStats()

    # --- опорный сигнал ----------------------------------------------------

    @property
    def tail_ms(self) -> int:
        """Длина хвоста фильтра, мс."""
        return round(1000 * self._partitions * self._block / self._sample_rate)

    def add_reference(self, pcm: npt.NDArray[np.number], play_at_ms: float) -> None:
        """Запомнить, что уйдёт в динамик и когда это начнёт звучать.

        Args:
            pcm: моно int16/float сигнал в частоте потока.
            play_at_ms: момент на оси потока (та же ось, что у ``ts_ms``
                микрофонных чанков), когда первый сэмпл станет слышен.
        """
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        if np.issubdtype(np.asarray(pcm).dtype, np.integer):
            samples = samples / 32768.0

        start = max(round(play_at_ms * self._sample_rate / 1000), 0)
        end = start + samples.size

        # Разрыв между прошлой записью и этой — тишина: динамик молчал.
        if start > self._ref_end:
            self._zero_range(self._ref_end, start)
        self._write_range(start, samples)
        self._ref_end = max(self._ref_end, end)

    def _write_range(self, start: int, samples: Float32Array) -> None:
        for offset, value in self._slices(start, samples.size):
            self._ref[value] = samples[offset : offset + (value.stop - value.start)]

    def _zero_range(self, start: int, end: int) -> None:
        if end - start >= self._ref_len:
            self._ref[:] = 0.0
            return
        for _, value in self._slices(start, end - start):
            self._ref[value] = 0.0

    def _slices(self, start: int, length: int) -> list[tuple[int, slice]]:
        """Разбить абсолютный отрезок на куски кольцевого буфера."""
        out: list[tuple[int, slice]] = []
        offset = 0
        while offset < length:
            pos = (start + offset) % self._ref_len
            take = min(length - offset, self._ref_len - pos)
            out.append((offset, slice(pos, pos + take)))
            offset += take
        return out

    def _reference(self, start: int, length: int) -> Float32Array:
        """Прочитать отрезок опорного сигнала по абсолютной позиции."""
        out = np.zeros(length, dtype=np.float32)
        if start < 0:
            skip = min(-start, length)
            start += skip
            if skip >= length:
                return out
            length -= skip
            for offset, value in self._slices(start, length):
                out[skip + offset : skip + offset + (value.stop - value.start)] = self._ref[value]
            return out
        for offset, value in self._slices(start, length):
            out[offset : offset + (value.stop - value.start)] = self._ref[value]
        return out

    # --- обработка ---------------------------------------------------------

    def process(self, mic: npt.NDArray[np.number], ts_ms: float) -> Int16Array:
        """Вычистить эхо из микрофонного чанка.

        Args:
            mic: моно int16 PCM микрофона.
            ts_ms: таймкод первого сэмпла на оси потока.

        Returns:
            Чанк той же длины и типа ``int16``. Пока опорный сигнал молчит,
            вход возвращается без изменений.
        """
        source = np.asarray(mic)
        samples = source.astype(np.float32).reshape(-1)
        if np.issubdtype(source.dtype, np.integer):
            samples = samples / 32768.0
        if samples.size == 0:
            return np.zeros(0, dtype=np.int16)

        position = round(ts_ms * self._sample_rate / 1000)
        if self._pending.size == 0:
            self._pending_pos = position
        self._pending = np.concatenate([self._pending, samples])

        cleaned: list[Float32Array] = []
        while self._pending.size >= self._block:
            block = self._pending[: self._block]
            cleaned.append(self._process_block(block, self._pending_pos))
            self._pending = self._pending[self._block :]
            self._pending_pos += self._block

        if not cleaned:
            # Меньше блока — отдать как есть, остаток дочистится следующим чанком.
            out = np.zeros(samples.size, dtype=np.float32)
        else:
            joined = np.concatenate(cleaned)
            out = joined[-samples.size :] if joined.size >= samples.size else joined
            if out.size < samples.size:
                out = np.concatenate([np.zeros(samples.size - out.size, dtype=np.float32), out])

        return np.clip(out * 32768.0, -32768.0, 32767.0).astype(np.int16)

    def _process_block(self, near: Float32Array, position: int) -> Float32Array:
        """Один блок PBFDAF: оценить эхо, вычесть, при возможности подстроиться."""
        self.stats.blocks += 1

        # Окно читается со сдвигом вперёд на ``pre_roll``: так фильтр покрывает
        # задержки от ``-pre_roll`` (чанк заиграл раньше расчётного момента) до
        # ``tail - pre_roll``. Сдвигать саму запись нельзя — опорный сигнал
        # уехал бы за пределы уже записанной области, и фильтр видел бы нули.
        window = self._reference(position - self._block + self._pre_roll, 2 * self._block)
        window_power = float(np.mean(np.square(window, dtype=np.float64)))

        spectrum = np.fft.rfft(window)
        self._spectra = np.roll(self._spectra, 1, axis=0)
        self._spectra[0] = spectrum
        self._ref_powers = np.roll(self._ref_powers, 1)
        self._ref_powers[0] = window_power

        # Активность динамика считается по всей памяти фильтра, а не по
        # текущему окну: опорный сигнал лежит на оси со сдвигом назад
        # (``pre_roll``), а эхо приходит с задержкой тракта, поэтому нужный
        # кусок находится в прошлых партициях. Если смотреть только на «сейчас»,
        # фильтр решит, что динамик молчит, и никогда не обучится.
        ref_power = float(self._ref_powers.max())
        if ref_power < REF_SILENCE_POWER and not np.any(self._weights):
            # Динамик молчит и фильтр ещё пуст — чистить нечего.
            return near

        estimate_spectrum = np.sum(self._weights * self._spectra, axis=0)
        estimate = np.fft.irfft(estimate_spectrum, n=2 * self._block)[self._block :]
        residual = near - estimate.astype(np.float32)

        near_power = float(np.mean(np.square(near, dtype=np.float64)))
        residual_power = float(np.mean(np.square(residual, dtype=np.float64)))
        echo_power = float(np.mean(np.square(estimate, dtype=np.float64)))

        if echo_power > 0.0:
            measured = 10.0 * np.log10(max(near_power, 1e-20) / max(residual_power, 1e-20))
            self.stats.erle_db = 0.9 * self.stats.erle_db + 0.1 * float(measured)

        if ref_power < REF_SILENCE_POWER:
            return residual

        # Двойной разговор: человек заговорил поверх динамика. Подстраиваться
        # под его голос нельзя — фильтр разъедется и начнёт выгрызать живую
        # речь; вычитание при этом продолжается уже настроенным фильтром.
        # Но пока фильтр не сошёлся, остаток и так равен входу, и этот признак
        # сработал бы всегда — поэтому он включается только после сходимости.
        converged = self.stats.erle_db > CONVERGED_ERLE_DB
        if converged and residual_power > DOUBLE_TALK_SHARE * near_power:
            self.stats.double_talk += 1
            return residual

        # Нормировка NLMS — по каждому частотному бину (сумма мощностей всех
        # партиций). Скалярная нормировка на всю энергию окна делает шаг на
        # три порядка меньше нужного, и фильтр не успевает сойтись за фразу.
        power = np.sum(np.abs(self._spectra) ** 2, axis=0) + 1e-9
        error_spectrum = np.fft.rfft(np.concatenate([np.zeros(self._block), residual]))
        self._weights += (2.0 * self._mu) * np.conj(self._spectra) * error_spectrum / power
        self.stats.adapted += 1
        return residual

    def reset(self) -> None:
        """Забыть накопленное (смена устройства, новая сессия)."""
        self._weights[:] = 0
        self._spectra[:] = 0
        self._ref_powers[:] = 0.0
        self._ref[:] = 0.0
        self._ref_end = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_pos = 0
        self.stats = AecStats()
