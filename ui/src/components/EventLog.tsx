"use client";

/**
 * Сырой лог событий движка — вторая половина отладочной страницы `/pipeline`.
 *
 * Панель пайплайна показывает цепочку по фразам, а лог — поток конвертов как
 * есть (`{type, ts, payload}` из раздела 5 ARCHITECTURE.md), чтобы было видно
 * и порядок событий, и служебные `tts.chunk`.
 */

import { useState } from "react";

import type { Envelope, EventType } from "@/lib/contracts";
import { LANG_LABEL, formatClock, formatSeconds } from "@/lib/format";

export interface EventLogProps {
  /** Буфер `useEngineSocket().events` — от старых к новым. */
  events: Envelope[];
  /** Сколько последних событий показывать. */
  limit?: number;
  /** Сколько сообщений не прошло валидацию контракта. */
  invalidCount?: number;
}

export const DEFAULT_LOG_SIZE = 80;

const TYPE_COLOR: Record<EventType, string> = {
  "stt.partial": "text-slate-500",
  "stt.final": "text-sky-300",
  "translation.ready": "text-emerald-300",
  "tts.chunk": "text-slate-600",
  "session.state": "text-amber-300",
  "metrics.latency": "text-violet-300",
  "session.start": "text-amber-300",
  "session.stop": "text-amber-300",
};

/** Короткая человекочитаемая сводка payload-а. */
export function describeEvent(envelope: Envelope): string {
  switch (envelope.type) {
    case "stt.partial": {
      const { stream, lang, text } = envelope.payload;
      return `${stream} ${lang}: ${text}`;
    }
    case "stt.final": {
      const { stream, lang, text, t_start_ms, t_end_ms } = envelope.payload;
      return `${stream} ${lang}: ${text} [${formatSeconds(t_start_ms)} → ${formatSeconds(t_end_ms)}]`;
    }
    case "translation.ready": {
      const { stream, src_lang, dst_lang, src_text, text } = envelope.payload;
      return `${stream} ${src_lang}→${dst_lang}: ${src_text}  ⇒  ${text}`;
    }
    case "tts.chunk": {
      const { stream, seq, pcm_base64 } = envelope.payload;
      return `${stream} seq=${seq}, ${pcm_base64.length} символов base64`;
    }
    case "session.state": {
      const { status, session_id, langs, voice_id } = envelope.payload;
      const id = session_id === null ? "—" : `#${session_id}`;
      return `${status} ${id}, ${LANG_LABEL[langs.in]} / ${LANG_LABEL[langs.out]}, голос ${voice_id ?? "по умолчанию"}`;
    }
    case "metrics.latency": {
      const { stream, stt_ms, mt_ms, tts_ms, total_ms } = envelope.payload;
      return `${stream} stt ${stt_ms} · mt ${mt_ms} · tts ${tts_ms} · итого ${total_ms} мс`;
    }
    case "session.start": {
      const { lang_in, lang_out, voice_id, record } = envelope.payload;
      return `${lang_in} / ${lang_out}, голос ${voice_id ?? "по умолчанию"}, запись ${record ? "вкл" : "выкл"}`;
    }
    default:
      return "";
  }
}

export function EventLog({ events, limit = DEFAULT_LOG_SIZE, invalidCount = 0 }: EventLogProps) {
  const [showChunks, setShowChunks] = useState(false);

  const filtered = showChunks ? events : events.filter((e) => e.type !== "tts.chunk");
  // Новые сверху: при живой сессии интересен конец потока, а не начало.
  const rows = filtered.slice(Math.max(0, filtered.length - limit)).reverse();
  const chunks = events.length - events.filter((e) => e.type !== "tts.chunk").length;

  return (
    <section aria-label="Лог событий движка" className="flex min-h-0 flex-col">
      <header className="flex flex-wrap items-center gap-x-3 gap-y-1 pb-2">
        <h3 className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
          Лог событий
        </h3>
        <span className="text-[11px] tabular-nums text-slate-600" data-testid="log-count">
          {events.length} событий
        </span>
        {invalidCount > 0 ? (
          <span className="text-[11px] text-rose-400" data-testid="log-invalid">
            невалидных: {invalidCount}
          </span>
        ) : null}
        <label className="ml-auto flex items-center gap-1.5 text-[11px] text-slate-400">
          <input
            type="checkbox"
            className="h-3.5 w-3.5 accent-sky-500"
            checked={showChunks}
            onChange={(e) => setShowChunks(e.target.checked)}
          />
          показывать tts.chunk{chunks > 0 ? ` (${chunks})` : ""}
        </label>
      </header>

      <ol className="min-h-0 flex-1 space-y-0.5 overflow-y-auto rounded border border-slate-800 bg-slate-950/60 p-2 font-mono text-[11px] leading-relaxed">
        {rows.length === 0 ? (
          <li className="text-slate-600">событий пока нет</li>
        ) : (
          rows.map((envelope, index) => (
            <li
              key={`${envelope.ts}-${index}-${envelope.type}`}
              className="flex gap-2"
              data-testid="log-row"
            >
              <span className="shrink-0 tabular-nums text-slate-600">
                {formatClock(envelope.ts)}
              </span>
              <span className={`w-[9.5rem] shrink-0 ${TYPE_COLOR[envelope.type]}`}>
                {envelope.type}
              </span>
              <span className="min-w-0 break-words text-slate-400">{describeEvent(envelope)}</span>
            </li>
          ))
        )}
      </ol>
    </section>
  );
}
