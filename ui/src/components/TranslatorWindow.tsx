"use client";

import { useMemo, useState } from "react";

import type { SessionStart } from "@/lib/contracts";
import type { VoiceRow } from "@/lib/api";
import type { AppState } from "@/lib/state";
import { selectTranscript } from "@/lib/state";
import type { ConnectionStatus } from "@/lib/ws";

import { AssistantPanel } from "@/components/AssistantPanel";
import { Controls } from "@/components/Controls";
import { LatencyBar } from "@/components/LatencyBar";
import { StatusBar } from "@/components/StatusBar";
import { SubtitleLane } from "@/components/SubtitleLane";

export interface TranslatorWindowProps {
  status: ConnectionStatus;
  state: AppState;
  voices: VoiceRow[];
  onStart: (payload: SessionStart) => void;
  onStop: () => void;
  /** Только для тестов: момент времени для окна транскрипта. */
  now?: number;
}

/**
 * Презентационная оболочка главного окна: всё состояние приходит пропсами,
 * поэтому её можно рендерить в тестах без WebSocket.
 */
export function TranslatorWindow({
  status,
  state,
  voices,
  onStart,
  onStop,
  now,
}: TranslatorWindowProps) {
  const [form, setForm] = useState<SessionStart>({
    lang_in: state.session.langs.in,
    lang_out: state.session.langs.out,
    voice_id: state.session.voice_id,
    record: false,
  });

  const running = state.session.status === "running";
  const connected = status === "open";
  const transcript = useMemo(() => selectTranscript(state, { now }), [state, now]);

  return (
    <main className="mx-auto flex h-[100dvh] w-full max-w-[480px] flex-col overflow-hidden bg-slate-950 text-slate-100">
      <StatusBar status={status} session={state.session} recording={running && form.record} />
      <LatencyBar metrics={state.metrics} />
      <Controls
        value={form}
        onChange={setForm}
        voices={voices}
        running={running}
        connected={connected}
        onStart={() => onStart(form)}
        onStop={onStop}
      />

      <div className="flex min-h-0 flex-1 flex-col divide-y divide-slate-800">
        <SubtitleLane stream="in" lane={state.lanes.in} />
        <SubtitleLane stream="out" lane={state.lanes.out} />
      </div>

      <AssistantPanel transcript={transcript} />
    </main>
  );
}
