/**
 * Редьюсер состояния UI: превращает поток событий движка в две ленты субтитров,
 * состояние сессии, метрики задержки и живой транскрипт для панели ассистента.
 *
 * Чистые функции, без React — чтобы тестировать без рендера.
 */

import type {
  Envelope,
  Lang,
  MetricsLatency,
  SessionState,
  Stream,
} from "@/lib/contracts";
import { STREAMS } from "@/lib/contracts";

/** Окно живого транскрипта для панели ассистента (5 минут). */
export const TRANSCRIPT_WINDOW_MS = 5 * 60 * 1000;

/** Сколько зафиксированных реплик держим в каждой ленте. */
export const MAX_FINALS_PER_LANE = 200;

/**
 * Сколько промежуточных гипотез whisper храним на одну реплику.
 * Нужны только панели пайплайна («как распознавал»), поэтому держим немного.
 */
export const MAX_PARTIALS_PER_UTTERANCE = 8;

/** Сколько последних реплик показывает панель пайплайна. */
export const PIPELINE_TRACE_SIZE = 30;

/** Зафиксированная реплика (stt.final) + прикреплённый к ней перевод. */
export interface Utterance {
  /** Локальный стабильный ключ для React. */
  key: string;
  /** id строки в `utterances` из БД, когда движок его прислал (translation.ready). */
  refUtteranceId: number | null;
  stream: Stream;
  lang: Lang;
  text: string;
  t_start_ms: number;
  t_end_ms: number;
  /** `ts` конверта, по нему считается окно транскрипта. */
  ts: number;
  translation: string | null;
  translationLang: Lang | null;
  /**
   * Промежуточные гипотезы whisper (`stt.partial`), накопленные до этого
   * `stt.final`, в порядке поступления. Для панели пайплайна.
   */
  partials: string[];
  /**
   * Текст, который движок реально отдал в NLLB (`src_text` из
   * `translation.ready`). Обычно совпадает с `text`; расхождение видно в панели
   * пайплайна и означает, что между STT и MT текст изменили.
   */
  srcText: string | null;
  /** Язык перевода, запрошенный у NLLB (`dst_lang`). */
  translationLangRequested: Lang | null;
  /** `ts` конверта `translation.ready`; `null` — перевод ещё не пришёл. */
  translationTs: number | null;
  /** `metrics.latency`, привязанная к этой реплике (порядок FIFO внутри потока). */
  metrics: MetricsLatency | null;
}

/** Одна лента субтитров. */
export interface Lane {
  /** "Живая" строка из stt.partial; заменяется целиком, пропадает на stt.final. */
  live: { text: string; lang: Lang } | null;
  finals: Utterance[];
  /**
   * Гипотезы `stt.partial`, пришедшие после последнего `stt.final`: очередной
   * final забирает их себе и очищает буфер.
   */
  partials: string[];
}

export interface AppState {
  session: SessionState;
  lanes: Record<Stream, Lane>;
  metrics: Record<Stream, MetricsLatency | null>;
  /** `ts` последнего обработанного события (null — событий не было). */
  lastEventTs: number | null;
  /** Счётчик для генерации стабильных ключей. */
  seq: number;
}

const emptyLane = (): Lane => ({ live: null, finals: [], partials: [] });

export const initialState: AppState = {
  session: { status: "idle", session_id: null, langs: { in: "en", out: "ru" }, voice_id: null },
  lanes: { in: emptyLane(), out: emptyLane() },
  metrics: { in: null, out: null },
  lastEventTs: null,
  seq: 0,
};

function withLane(state: AppState, stream: Stream, lane: Lane): AppState {
  return { ...state, lanes: { ...state.lanes, [stream]: lane } };
}

function trim(finals: Utterance[]): Utterance[] {
  return finals.length > MAX_FINALS_PER_LANE
    ? finals.slice(finals.length - MAX_FINALS_PER_LANE)
    : finals;
}

/**
 * Добавляет гипотезу whisper в буфер ленты.
 *
 * Повтор той же строки не пишем (движок может прислать одинаковый partial
 * дважды), длину буфера ограничиваем — панель пайплайна показывает только
 * последние шаги распознавания.
 */
function pushPartial(partials: string[], text: string): string[] {
  if (partials.length > 0 && partials[partials.length - 1] === text) return partials;
  const next = [...partials, text];
  return next.length > MAX_PARTIALS_PER_UTTERANCE
    ? next.slice(next.length - MAX_PARTIALS_PER_UTTERANCE)
    : next;
}

/**
 * Куда прикрепить перевод.
 *
 * Движок обрабатывает фразы одного потока строго по очереди (FIFO, один воркер
 * на конвейер — engine/orchestrator/pipeline.py), поэтому очередной
 * `translation.ready` относится к САМОЙ СТАРОЙ ещё не переведённой реплике.
 * Искать с конца нельзя: перевод и синтез идут дольше распознавания, и к
 * моменту `translation.ready` в ленте уже лежат более свежие `stt.final`.
 *
 * Порядок поиска:
 *  1. реплика с уже известным `refUtteranceId === ref`;
 *  2. самая старая реплика без перевода, чей текст совпал с `src_text`
 *     (`stt.final` не несёт id реплики — см. README, вопрос к владельцу
 *      контрактов);
 *  3. самая старая реплика без перевода;
 *  4. последняя реплика ленты.
 */
function findTarget(
  finals: Utterance[],
  refUtteranceId: number | null,
  srcText: string,
): number {
  if (refUtteranceId !== null) {
    const byId = finals.findIndex((u) => u.refUtteranceId === refUtteranceId);
    if (byId !== -1) return byId;
  }
  const byText = finals.findIndex((u) => u.translation === null && u.text === srcText);
  if (byText !== -1) return byText;
  const byOrder = finals.findIndex((u) => u.translation === null);
  if (byOrder !== -1) return byOrder;
  return finals.length - 1;
}

/** Применяет одно событие к состоянию. Неизвестные/служебные события не ломают состояние. */
export function reduce(state: AppState, envelope: Envelope): AppState {
  const next: AppState = { ...state, lastEventTs: envelope.ts };

  switch (envelope.type) {
    case "stt.partial": {
      const { stream, lang, text } = envelope.payload;
      const lane = next.lanes[stream];
      return withLane(next, stream, {
        ...lane,
        live: { text, lang },
        partials: pushPartial(lane.partials, text),
      });
    }

    case "stt.final": {
      const { stream, lang, text, t_start_ms, t_end_ms } = envelope.payload;
      const lane = next.lanes[stream];
      const seq = next.seq + 1;
      const utterance: Utterance = {
        key: `${stream}-${seq}`,
        refUtteranceId: null,
        stream,
        lang,
        text,
        t_start_ms,
        t_end_ms,
        ts: envelope.ts,
        translation: null,
        translationLang: null,
        // Гипотезы, накопленные до этой фразы, принадлежат именно ей.
        partials: lane.partials,
        srcText: null,
        translationLangRequested: null,
        translationTs: null,
        metrics: null,
      };
      return withLane({ ...next, seq }, stream, {
        live: null,
        finals: trim([...lane.finals, utterance]),
        partials: [],
      });
    }

    case "translation.ready": {
      const { stream, src_text, text, dst_lang, src_lang, ref_utterance_id } = envelope.payload;
      const lane = next.lanes[stream];
      const index = findTarget(lane.finals, ref_utterance_id, src_text);

      if (index === -1) {
        // Перевод пришёл раньше, чем stt.final (или после перезапуска UI) —
        // создаём реплику из src_text, чтобы ничего не потерять.
        const seq = next.seq + 1;
        const utterance: Utterance = {
          key: `${stream}-${seq}`,
          refUtteranceId: ref_utterance_id,
          stream,
          lang: src_lang,
          text: src_text,
          t_start_ms: 0,
          t_end_ms: 0,
          ts: envelope.ts,
          translation: text,
          translationLang: dst_lang,
          partials: [],
          srcText: src_text,
          translationLangRequested: dst_lang,
          translationTs: envelope.ts,
          metrics: null,
        };
        return withLane({ ...next, seq }, stream, {
          ...lane,
          finals: trim([...lane.finals, utterance]),
        });
      }

      const finals = lane.finals.slice();
      finals[index] = {
        ...finals[index],
        refUtteranceId: ref_utterance_id ?? finals[index].refUtteranceId,
        translation: text,
        translationLang: dst_lang,
        // Что именно ушло в NLLB и что он вернул — для панели пайплайна.
        srcText: src_text,
        translationLangRequested: dst_lang,
        translationTs: envelope.ts,
      };
      return withLane(next, stream, { ...lane, finals });
    }

    case "session.state": {
      const session = envelope.payload;
      // Новая сессия — чистим ленты, чтобы не смешивать разговоры.
      const startedNewSession =
        session.status === "running" &&
        (state.session.status !== "running" || state.session.session_id !== session.session_id);
      if (startedNewSession) {
        return {
          ...next,
          session,
          lanes: { in: emptyLane(), out: emptyLane() },
          metrics: { in: null, out: null },
        };
      }
      if (session.status === "idle") {
        // Сессия остановлена: незакрытая гипотеза stt.partial уже никогда не
        // станет stt.final — гасим живые строки, чтобы они не висели в окне.
        return {
          ...next,
          session,
          lanes: {
            in: { ...next.lanes.in, live: null },
            out: { ...next.lanes.out, live: null },
          },
        };
      }
      return { ...next, session };
    }

    case "metrics.latency": {
      const metrics = envelope.payload;
      const withStreamMetrics: AppState = {
        ...next,
        metrics: { ...next.metrics, [metrics.stream]: metrics },
      };

      // Движок публикует metrics.latency сразу после translation.ready той же
      // фразы (engine/orchestrator/pipeline.py, один воркер на поток), поэтому
      // замер относится к самой старой переведённой реплике без метрик.
      const lane = withStreamMetrics.lanes[metrics.stream];
      const index = lane.finals.findIndex((u) => u.translation !== null && u.metrics === null);
      if (index === -1) return withStreamMetrics;

      const finals = lane.finals.slice();
      finals[index] = { ...finals[index], metrics };
      return withLane(withStreamMetrics, metrics.stream, { ...lane, finals });
    }

    // tts.chunk — служебное аудио, UI игнорирует (но парсер его понимает).
    case "tts.chunk":
    default:
      return next;
  }
}

/** Свёртка пачки событий (удобно в тестах и при восстановлении буфера). */
export function reduceAll(state: AppState, envelopes: readonly Envelope[]): AppState {
  return envelopes.reduce(reduce, state);
}

/**
 * Живой транскрипт последних N минут (обе ленты, по возрастанию времени).
 * `now` по умолчанию — время последнего события, чтобы транскрипт не "протухал"
 * из-за расхождения часов UI и движка.
 */
export function selectTranscript(
  state: AppState,
  options: { now?: number; windowMs?: number } = {},
): Utterance[] {
  const windowMs = options.windowMs ?? TRANSCRIPT_WINDOW_MS;
  const now = options.now ?? state.lastEventTs ?? 0;
  const from = now - windowMs;
  return STREAMS.flatMap((stream) => state.lanes[stream].finals)
    .filter((u) => u.ts >= from)
    .sort((a, b) => a.ts - b.ts || a.t_start_ms - b.t_start_ms);
}

/**
 * Трасса пайплайна: последние реплики обоих потоков, новые сверху.
 *
 * Панель `/pipeline` показывает по каждой реплике всю цепочку — гипотезы и
 * финал whisper, текст, ушедший в NLLB, и его перевод, — поэтому порядок здесь
 * обратный ленте субтитров: свежее интереснее.
 */
export function selectPipelineTrace(
  state: AppState,
  options: { limit?: number } = {},
): Utterance[] {
  const limit = options.limit ?? PIPELINE_TRACE_SIZE;
  const all = STREAMS.flatMap((stream) => state.lanes[stream].finals).sort(
    (a, b) => b.ts - a.ts || b.t_start_ms - a.t_start_ms,
  );
  return limit >= 0 ? all.slice(0, limit) : all;
}

/** Сколько реплик всего зафиксировано (для отладочных индикаторов). */
export function countUtterances(state: AppState): number {
  return STREAMS.reduce((sum, stream) => sum + state.lanes[stream].finals.length, 0);
}
