import Link from "next/link";

import type { SessionRow } from "@/lib/api";
import { LANG_LABEL, formatDateTime, formatDuration, pluralUtterances } from "@/lib/format";
import { getSessions } from "@/lib/server/history";

/** История живёт в БД движка — кэшировать её на сборке нечего. */
export const dynamic = "force-dynamic";

export default async function HistoryPage() {
  let sessions: SessionRow[] = [];
  let error: string | null = null;
  try {
    sessions = await getSessions();
  } catch (err: unknown) {
    error = err instanceof Error ? err.message : "не удалось загрузить историю";
  }

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-5">
      <header className="mb-4 flex items-center justify-between gap-3">
        <h1 className="text-lg font-semibold">История сессий</h1>
        <Link href="/" className="text-sm text-sky-300 underline-offset-2 hover:underline">
          ← К окну перевода
        </Link>
      </header>

      {error !== null ? (
        <p className="rounded border border-rose-800 bg-rose-950/40 px-3 py-2 text-sm text-rose-200">
          Ошибка: {error}
        </p>
      ) : null}

      {error === null && sessions.length === 0 ? (
        <p className="text-sm text-slate-500">Записанных сессий пока нет.</p>
      ) : null}

      <ul className="space-y-2">
        {sessions.map((session) => (
          <li key={session.id}>
            <Link
              href={`/history/${session.id}`}
              className="flex flex-wrap items-baseline gap-x-3 gap-y-1 rounded border border-slate-800 bg-slate-900/50 px-3 py-2 transition hover:border-sky-700 hover:bg-slate-900"
            >
              <span className="font-medium tabular-nums">{formatDateTime(session.started_at)}</span>
              <span className="text-sm text-slate-400">
                {LANG_LABEL[session.lang_from]} → {LANG_LABEL[session.lang_to]}
              </span>
              <span className="text-sm tabular-nums text-slate-400">
                {formatDuration(session.duration_ms)}
              </span>
              <span className="text-sm text-slate-400">
                {pluralUtterances(session.utterance_count)}
              </span>
              {session.audio_path !== null ? (
                <span className="text-xs text-emerald-400">аудио</span>
              ) : null}
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
