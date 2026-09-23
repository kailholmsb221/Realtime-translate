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

#: Длина хвоста фильтра, мс: задержка вывода + отражения комнаты. На ноутбуке
#: с Realtek измеренная задержка «опорный сигнал → микрофон» доходила до
#: 1.1 с (буферы WASAPI и драйвера сверх того, что видно приёмнику), поэтому
#: хвост короче 1.5 с эхо просто не достаёт. 100 партиций по 15 мс — это всё
#: ещё доли миллисекунды на блок.
DEFAULT_TAIL_MS: Final[int] = 1_500

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

#: Регуляризация NLMS: доля средней мощности опорного сигнала, добавляемая в
#: знаменатель. Без неё в тихих бинах шаг взлетает до бесконечности, и фильтр
#: расходится за несколько блоков — вместо эха он начинает добавлять шум.
REGULARIZATION: Final[float] = 0.05

#: Сглаживание оценки мощности по блокам (0 — только текущий блок).
POWER_SMOOTHING: Final[float] = 0.7

#: Если после вычитания стало громче во столько раз — фильтр ошибся. Его
#: вывод не используем, а веса плавно забываем.
DIVERGENCE_RATIO: Final[float] = 1.5

#: Предел модуля веса. Путь «динамик → микрофон» ослабляет сигнал, поэтому
#: честные веса по модулю невелики; если вес перевалил за этот порог, фильтр
#: расходится — по энергии одного блока это не отличить от сходимости, а по
#: норме весов видно сразу.
WEIGHT_LIMIT: Final[float] = 8.0

#: Как сжимаем веса при расхождении (множитель).
LEAK: Final[float] = 0.5

#: Сколько секунд сырого микрофона держим для оценки задержки эха и в каком
#: диапазоне лагов её искать (отрицательный лаг — эхо пришло раньше, чем мы
#: пометили опорный сигнал, и причинный фильтр его не достанет).
DELAY_PROBE_S: Final[float] = 3.0
DELAY_LAG_MS: Final[tuple[int, int]] = (-800, 2500)

#: Пределы измеренной ERLE, дБ: один нелепый блок не должен утащить оценку.
ERLE_CLIP_DB: Final[tuple[float, float]] = (-20.0, 60.0)


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
    #: Блоков, где фильтр сделал громче и его вывод отброшен.
    rejected: int = 0
    #: Сколько раз веса сжимались из-за расхождения.
    leaked: int = 0
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
        "_mic_tail",
        "_mic_tail_pos",
        "_mu",
        "_partitions",
        "_pending",
        "_pending_pos",
        "_power",
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
        #: Сглаженная мощность опорного сигнала по бинам — знаменатель NLMS.
        self._power = np.zeros(self._freq_bins, dtype=np.float64)

        self._ref_len = max(int(sample_rate * ref_seconds), 4 * block)
        self._ref = np.zeros(self._ref_len, dtype=np.float32)
        #: До какой абсолютной позиции опорный сигнал уже записан (в сэмплах).
        self._ref_end = 0
        #: Необработанный хвост входа и его абсолютная позиция.
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_pos = 0
        #: Последние секунды сырого микрофона — для оценки реальной задержки эха.
        self._mic_tail = np.zeros(int(sample_rate * DELAY_PROBE_S), dtype=np.float32)
        self._mic_tail_pos = 0
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
        self._remember_mic(samples, position)
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

        if ref_power < REF_SILENCE_POWER:
            return residual

        # Фильтр не имеет права делать сигнал громче: если остаток громче
        # входа, в whisper уходит сырой микрофон, а не наша ошибка. Обучение
        # при этом НЕ останавливается — на ранней сходимости такие блоки
        # нормальны, и именно на них фильтр учится быстрее всего. Забываем веса
        # только при устойчивом расхождении: много таких блоков подряд.
        louder = near_power > 0.0 and residual_power > DIVERGENCE_RATIO * near_power
        output = near if louder else residual
        if louder:
            self.stats.rejected += 1
        else:
            if echo_power > 0.0:
                measured = 10.0 * np.log10(max(near_power, 1e-20) / max(residual_power, 1e-20))
                measured = float(np.clip(measured, *ERLE_CLIP_DB))
                self.stats.erle_db = 0.9 * self.stats.erle_db + 0.1 * measured

        # Двойной разговор: человек заговорил поверх динамика. Подстраиваться
        # под его голос нельзя — фильтр разъедется и начнёт выгрызать живую
        # речь; вычитание при этом продолжается уже настроенным фильтром.
        # Но пока фильтр не сошёлся, остаток и так равен входу, и этот признак
        # сработал бы всегда — поэтому он включается только после сходимости.
        converged = self.stats.erle_db > CONVERGED_ERLE_DB
        if converged and residual_power > DOUBLE_TALK_SHARE * near_power:
            self.stats.double_talk += 1
            return output

        # Нормировка NLMS — по каждому частотному бину (сумма мощностей всех
        # партиций). Скалярная нормировка на всю энергию окна делает шаг на
        # три порядка меньше нужного, и фильтр не успевает сойтись за фразу.
        instant = np.sum(np.abs(self._spectra) ** 2, axis=0)
        if not self._power.any():
            self._power[:] = instant
        self._power = POWER_SMOOTHING * self._power + (1.0 - POWER_SMOOTHING) * instant
        # Регуляризация относительно средней мощности: тихий бин не получает
        # гигантский шаг только потому, что в нём почти нет сигнала.
        delta = REGULARIZATION * float(np.mean(self._power)) + 1e-12
        error_spectrum = np.fft.rfft(np.concatenate([np.zeros(self._block), residual]))
        step = (2.0 * self._mu) * np.conj(self._spectra) * error_spectrum / (self._power + delta)
        self._weights += step
        self.stats.adapted += 1

        # Настоящее расхождение видно по весам: они уходят за физически
        # возможные значения. Сжимаем, а не обнуляем — направление обычно
        # верное, велик только масштаб.
        peak = float(np.abs(self._weights).max())
        if peak > WEIGHT_LIMIT:
            self._weights *= LEAK
            self.stats.leaked += 1
            self.stats.erle_db = max(ERLE_CLIP_DB[0], self.stats.erle_db - 3.0)
        return output

    def _remember_mic(self, samples: Float32Array, position: int) -> None:
        """Положить сырой микрофон в кольцо для оценки задержки."""
        n = self._mic_tail.size
        for offset in range(0, samples.size, n):
            piece = samples[offset : offset + n]
            start = (position + offset) % n
            take = min(piece.size, n - start)
            self._mic_tail[start : start + take] = piece[:take]
            if take < piece.size:
                self._mic_tail[: piece.size - take] = piece[take:]
        self._mic_tail_pos = position + samples.size

    def estimate_delay(self) -> tuple[int, float]:
        """Оценить реальную задержку эха по огибающим микрофона и опорного сигнала.

        Returns:
            ``(лаг в мс, коэффициент корреляции)``. Лаг — на сколько микрофон
            отстаёт от опорного сигнала; корреляция ниже ~0.2 значит, что эхо в
            микрофоне вообще не похоже на то, что мы считаем опорным сигналом.
        """
        n = self._mic_tail.size
        end = self._mic_tail_pos
        start = end - n
        if start < 0 or end <= 0:
            return 0, 0.0
        roll = end % n
        mic = np.concatenate([self._mic_tail[roll:], self._mic_tail[:roll]])
        lo, hi = DELAY_LAG_MS
        lo_s, hi_s = lo * self._sample_rate // 1000, hi * self._sample_rate // 1000
        ref = self._reference(start - hi_s, n + hi_s - lo_s)

        # Огибающие по 4 мс — лаг ищется по энергии, а не по фазе: устойчиво к
        # искажениям тракта и дёшево.
        hop = max(1, self._sample_rate // 250)

        def envelope(x: Float32Array) -> Float32Array:
            m = (x.size // hop) * hop
            e = np.sqrt(np.mean(x[:m].reshape(-1, hop) ** 2, axis=1))
            return (e - e.mean()) / (e.std() + 1e-9)

        em = envelope(mic)
        er = envelope(ref)
        if em.size == 0 or er.size <= em.size:
            return 0, 0.0
        # er покрывает [start - hi, end - lo): сдвиг k соответствует лагу hi - k*hop.
        best_lag, best_corr = 0, 0.0
        for k in range(0, er.size - em.size + 1):
            corr = float(np.dot(em, er[k : k + em.size])) / em.size
            if corr > best_corr:
                best_corr = corr
                best_lag = hi_s - k * hop
        return round(best_lag * 1000 / self._sample_rate), best_corr

    def reset(self) -> None:
        """Забыть накопленное (смена устройства, новая сессия)."""
        self._weights[:] = 0
        self._spectra[:] = 0
        self._ref_powers[:] = 0.0
        self._power[:] = 0.0
        self._mic_tail[:] = 0.0
        self._mic_tail_pos = 0
        self._ref[:] = 0.0
        self._ref_end = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._pending_pos = 0
        self.stats = AecStats()
