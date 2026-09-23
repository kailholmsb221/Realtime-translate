"use client";

/**
 * Страница `/pipeline` — что происходит внутри перевода: гипотезы и финал
 * whisper, текст, ушедший в NLLB, перевод NLLB, задержки по стадиям и сырой
 * лог событий движка.
 *
 * Отдельная страница, а не панель в окне перевода: окно узкое (480 px), а здесь
 * нужна ширина. Сессию можно запустить прямо отсюда — теми же командами
 * `session.start` / `session.stop`.
 */

import Link from "next/link";
import { useMemo, useState } from "react";

import { EventLog } from "@/components/EventLog";
import { LatencyBar } from "@/components/LatencyBar";
import { PipelineInspector } from "@/components/PipelineInspector";
import { SpeechDirection } from "@/components/SpeechDirection";
import type { SessionStart } from "@/lib/contracts";
import { LANG_LABEL } from "@/lib/format";
import { selectPipelineTrace } from "@/lib/state";
import { useEngineSocket } from "@/lib/ws";

const CONNECTION_LABEL = {
  connecting: "подключение…",
  open: "движок на связи",
  closed: "нет связи",
} as const;

const CONNECTION_COLOR = {
  connecting: "bg-amber-400",
  open: "bg-emerald-400",
  closed: "bg-rose-500",
} as const;

export default function PipelinePage() {
  const engine = useEngineSocket();
  // lang_out — язык, на котором говорит пользователь; lang_in — язык перевода.
  const [form, setForm] = useState<SessionStart>({
    lang_in: "en",
    lang_out: "ru",
    voice_id: null,
    record: false,
  });

  const { state } = engine;
  const trace = useMemo(() => selectPipelineTrace(state), [state]);
  const running = state.session.status === "running";
  const live = { in: state.lanes.in.live, out: state.lanes.out.live };

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-4">
      <header className="mb-4 flex flex-wrap items-center gap-x-3 gap-y-2">
        <h1 className="text-lg font-semibold">Пайплайн: whisper → NLLB → озвучка</h1>
        <span className="flex items-center gap-1.5 text-xs" data-testid="connection-status">
          <span
            className={`inline-block h-2.5 w-2.5 rounded-full ${CONNECTION_COLOR[engine.status]}`}
            aria-hidden="true"
          />
          <span className="text-slate-300">{CONNECTION_LABEL[engine.status]}</span>
        </span>
        <span className="text-xs text-slate-400">
          {running ? "сессия идёт" : "сессия не запущена"} · говорю на{" "}
          {LANG_LABEL[state.session.langs.out]}, перевод на{" "}
          {LANG_LABEL[state.session.langs.in]}
        </span>
        <nav className="ml-auto flex gap-3 text-sm">
          <Link href="/" className="text-sky-300 underline-offset-2 hover:underline">
            ← Окно перевода
          </Link>
          <Link href="/history" className="text-sky-300 underline-offset-2 hover:underline">
            История
          </Link>
        </nav>
      </header>

      <div className="grid gap-4 lg:grid-cols-[20rem_minmax(0,1fr)]">
        <aside className="flex min-w-0 flex-col gap-4">
          <div className="rounded-lg border border-slate-800 bg-slate-900/40">
            <SpeechDirection
              value={form}
              onChange={setForm}
              running={running}
              connected={engine.status === "open"}
              onStart={() => engine.start(form)}
              onStop={engine.stop}
            />
            <LatencyBar metrics={state.metrics} />
          </div>

          <div className="rounded-lg border border-slate-800 bg-slate-900/40 p-3 text-[11px] leading-relaxed text-slate-400">
            <h2 className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-slate-300">
              Как читать карточку
            </h2>
            <p>
              <span className="text-slate-200">1 Whisper</span> — промежуточные гипотезы и финальная
              фраза (`stt.partial` / `stt.final`).
            </p>
            <p>
              <span className="text-slate-200">2 NLLB · вход</span> — текст, который движок реально
              передал в перевод (`src_text`), и коды FLORES-200.
            </p>
            <p>
              <span className="text-slate-200">3 NLLB · перевод</span> — что модель вернула (`text`).
            </p>
            <p>
              <span className="text-slate-200">4 Озвучка</span> — время до первого чанка TTS и общая
              задержка от конца фразы.
            </p>
          </div>

          <EventLog events={engine.events} invalidCount={engine.invalidCount} />
        </aside>

        <div className="min-w-0">
          <PipelineInspector trace={trace} live={live} />
        </div>
      </div>
    </main>
  );
}
