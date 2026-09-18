"use client";

import { useState } from "react";

import { STREAM_LABEL, formatTimecode } from "@/lib/format";
import type { Utterance } from "@/lib/state";
import { TRANSCRIPT_WINDOW_MS } from "@/lib/state";

export interface AssistantPanelProps {
  transcript: Utterance[];
  windowMs?: number;
}

export function AssistantPanel({ transcript, windowMs = TRANSCRIPT_WINDOW_MS }: AssistantPanelProps) {
  const [open, setOpen] = useState(true);
  const minutes = Math.round(windowMs / 60000);

  return (
    <section
      aria-label="Панель ассистента"
      data-testid="assistant-panel"
      className="border-t border-slate-800 bg-slate-950/60"
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-3 py-1.5 text-[11px] font-semibold uppercase tracking-wide text-slate-500 hover:text-slate-300"
      >
        <span>Ассистент · транскрипт {minutes} мин</span>
        <span aria-hidden="true">{open ? "▾" : "▸"}</span>
      </button>

      {open ? (
        <div className="space-y-2 px-3 pb-2">
          <div className="max-h-28 space-y-1 overflow-y-auto text-[11px] leading-snug">
            {transcript.length === 0 ? (
              <p className="text-slate-600">Транскрипт пуст.</p>
            ) : (
              transcript.map((utterance) => (
                <p key={utterance.key} className="text-slate-400">
                  <span className="tabular-nums text-slate-600">
                    {formatTimecode(utterance.t_start_ms)}{" "}
                  </span>
                  <span className="text-slate-500">{STREAM_LABEL[utterance.stream]}: </span>
                  {utterance.text}
                </p>
              ))
            )}
          </div>

          <button
            type="button"
            disabled
            title="TODO: LLM-подсказки появятся позже"
            className="w-full cursor-not-allowed rounded border border-dashed border-slate-700 px-2 py-1 text-[11px] text-slate-500"
          >
            Подсказать ответ
          </button>
        </div>
      ) : null}
    </section>
  );
}
