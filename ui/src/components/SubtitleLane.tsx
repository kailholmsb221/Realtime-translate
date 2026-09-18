"use client";

import { useEffect, useRef } from "react";

import type { Stream } from "@/lib/contracts";
import { STREAM_LABEL } from "@/lib/format";
import type { Lane } from "@/lib/state";

export interface SubtitleLaneProps {
  stream: Stream;
  lane: Lane;
}

export function SubtitleLane({ stream, lane }: SubtitleLaneProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [lane.finals.length, lane.live?.text]);

  const isIncoming = stream === "in";

  return (
    <section
      aria-label={`Субтитры: ${STREAM_LABEL[stream]}`}
      data-testid={`lane-${stream}`}
      className="flex min-h-0 flex-1 flex-col"
    >
      <h2 className="flex items-center gap-2 px-3 pt-2 text-[11px] font-semibold uppercase tracking-wide text-slate-500">
        <span
          className={`inline-block h-2 w-2 rounded-full ${isIncoming ? "bg-sky-400" : "bg-violet-400"}`}
          aria-hidden="true"
        />
        {STREAM_LABEL[stream]}
      </h2>

      <div className="min-h-0 flex-1 space-y-2 overflow-y-auto px-3 py-2">
        {lane.finals.length === 0 && lane.live === null ? (
          <p className="text-sm text-slate-600">Ожидание речи…</p>
        ) : null}

        {lane.finals.map((utterance) => (
          <article key={utterance.key} className="leading-snug">
            <p className="text-[15px] text-slate-400">{utterance.text}</p>
            {utterance.translation === null ? (
              <p className="text-[15px] italic text-slate-600" aria-label="перевод готовится">
                перевод…
              </p>
            ) : (
              <p
                className={`text-[17px] font-medium ${isIncoming ? "text-sky-100" : "text-violet-100"}`}
                data-testid="translation"
              >
                {utterance.translation}
              </p>
            )}
          </article>
        ))}

        {lane.live !== null ? (
          <p
            className="text-[15px] italic text-slate-500"
            data-testid={`live-${stream}`}
          >
            {lane.live.text}
            <span className="ml-0.5 animate-pulse" aria-hidden="true">
              ▌
            </span>
          </p>
        ) : null}

        <div ref={bottomRef} />
      </div>
    </section>
  );
}
