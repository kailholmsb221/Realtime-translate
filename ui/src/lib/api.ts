/**
 * REST-клиент истории и голосов.
 *
 * Формы ответов — 1:1 со строками таблиц `db/schema.sql` (sessions, utterances,
 * voices) плюс два производных поля у сессии (`utterance_count`, `duration_ms`).
 * Контракт описан в ui/README.md как ПРЕДЛОЖЕНИЕ для оркестратора — пока движка
 * нет, запросы обслуживают заглушки в `src/app/api/**` поверх `mocks/data.json`.
 *
 * По умолчанию клиент ходит на свой же origin (`/api/...`). Задайте
 * `NEXT_PUBLIC_ENGINE_HTTP=http://localhost:8766`, чтобы обращаться к движку
 * напрямую, минуя заглушки.
 */

import type { Lang, Stream } from "@/lib/contracts";

export const HISTORY_BASE_URL = process.env.NEXT_PUBLIC_ENGINE_HTTP ?? "";

/** Строка таблицы `sessions` + производные поля для списка истории. */
export interface SessionRow {
  id: number;
  started_at: number;
  ended_at: number | null;
  lang_from: Lang;
  lang_to: Lang;
  audio_path: string | null;
  /** Производное: COUNT(utterances). */
  utterance_count: number;
  /** Производное: ended_at - started_at (null, если сессия ещё идёт). */
  duration_ms: number | null;
}

/** Строка таблицы `utterances`. */
export interface UtteranceRow {
  id: number;
  session_id: number;
  t_start_ms: number;
  t_end_ms: number;
  speaker: Stream;
  lang: Lang;
  text: string;
  translation: string | null;
  translation_lang: Lang | null;
}

/** Строка таблицы `voices`. */
export interface VoiceRow {
  id: string;
  name: string;
  lang: Lang;
  sample_path: string;
  created_at: number;
}

export interface SessionsResponse {
  sessions: SessionRow[];
}

export interface SessionDetailResponse {
  session: SessionRow;
  utterances: UtteranceRow[];
}

export interface VoicesResponse {
  voices: VoiceRow[];
}

async function getJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(`${HISTORY_BASE_URL}${path}`, { signal, cache: "no-store" });
  if (!response.ok) {
    throw new Error(`${path}: HTTP ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchSessions(signal?: AbortSignal): Promise<SessionsResponse> {
  return getJson<SessionsResponse>("/api/sessions", signal);
}

export function fetchSession(id: string | number, signal?: AbortSignal): Promise<SessionDetailResponse> {
  return getJson<SessionDetailResponse>(`/api/sessions/${id}`, signal);
}

export function fetchVoices(signal?: AbortSignal): Promise<VoicesResponse> {
  return getJson<VoicesResponse>("/api/voices", signal);
}

export function sessionAudioUrl(id: string | number): string {
  return `${HISTORY_BASE_URL}/api/sessions/${id}/audio`;
}
