import Link from "next/link";

import type { SessionState } from "@/lib/contracts";
import { LANG_LABEL } from "@/lib/format";
import type { ConnectionStatus } from "@/lib/ws";

const CONNECTION_LABEL: Record<ConnectionStatus, string> = {
  connecting: "подключение…",
  open: "движок на связи",
  closed: "нет связи",
};

const CONNECTION_COLOR: Record<ConnectionStatus, string> = {
  connecting: "bg-amber-400",
  open: "bg-emerald-400",
  closed: "bg-rose-500",
};

export interface StatusBarProps {
  status: ConnectionStatus;
  session: SessionState;
  recording: boolean;
}

export function StatusBar({ status, session, recording }: StatusBarProps) {
  return (
    <header className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-slate-800 px-3 py-2 text-xs">
      <span className="flex items-center gap-1.5" data-testid="connection-status">
        <span
          className={`inline-block h-2.5 w-2.5 shrink-0 rounded-full ${CONNECTION_COLOR[status]}`}
          aria-hidden="true"
        />
        <span className="text-slate-300">{CONNECTION_LABEL[status]}</span>
      </span>

      <span className="text-slate-500" aria-hidden="true">
        ·
      </span>

      <span data-testid="session-status" className="text-slate-300">
        {session.status === "running" ? "сессия идёт" : "сессия не запущена"}
      </span>

      <span className="text-slate-400">
        {LANG_LABEL[session.langs.in]} → {LANG_LABEL[session.langs.out]}
      </span>

      {recording ? (
        <span
          className="rounded bg-rose-600/20 px-1.5 py-0.5 font-medium text-rose-300"
          data-testid="recording-badge"
          title="Собеседник должен быть уведомлён о записи"
        >
          ● запись
        </span>
      ) : null}

      <Link
        href="/history"
        className="ml-auto rounded px-1.5 py-0.5 text-slate-300 underline-offset-2 hover:text-sky-300 hover:underline"
      >
        История
      </Link>
    </header>
  );
}
