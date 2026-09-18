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
}

/** Одна лента субтитров. */
export interface Lane {
  /** "Живая" строка из stt.partial; заменяется целиком, пропадает на stt.final. */
  live: { text: string; lang: Lang } | null;
  finals: Utterance[];
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

const emptyLane = (): Lane => ({ live: null, finals: [] });

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
 * Куда прикрепить перевод.
 *
 * Порядок поиска:
 *  1. реплика с уже известным `refUtteranceId === ref`;
 *  2. последняя реплика без перевода, чей текст совпал с `src_text`
 *     (единственный надёжный ключ: `stt.final` не несёт id реплики — см. README,
 *      вопрос к владельцу контрактов);
 *  3. последняя реплика без перевода;
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
  for (let i = finals.length - 1; i >= 0; i -= 1) {
    if (finals[i].translation === null && finals[i].text === srcText) return i;
  }
  for (let i = finals.length - 1; i >= 0; i -= 1) {
    if (finals[i].translation === null) return i;
  }
  return finals.length - 1;
}

/** Применяет одно событие к состоянию. Неизвестные/служебные события не ломают состояние. */
export function reduce(state: AppState, envelope: Envelope): AppState {
  const next: AppState = { ...state, lastEventTs: envelope.ts };

  switch (envelope.type) {
    case "stt.partial": {
      const { stream, lang, text } = envelope.payload;
      return withLane(next, stream, { ...next.lanes[stream], live: { text, lang } });
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
      };
      return withLane({ ...next, seq }, stream, {
        live: null,
        finals: trim([...lane.finals, utterance]),
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
      return { ...next, session };
    }

    case "metrics.latency": {
      const metrics = envelope.payload;
      return { ...next, metrics: { ...next.metrics, [metrics.stream]: metrics } };
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

/** Сколько реплик всего зафиксировано (для отладочных индикаторов). */
export function countUtterances(state: AppState): number {
  return STREAMS.reduce((sum, stream) => sum + state.lanes[stream].finals.length, 0);
}
