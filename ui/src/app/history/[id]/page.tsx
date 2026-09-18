import Link from "next/link";
import { notFound } from "next/navigation";

import { sessionAudioUrl } from "@/lib/api";
import {
  LANG_LABEL,
  STREAM_LABEL,
  formatDateTime,
  formatDuration,
  formatTimecode,
} from "@/lib/format";
import { getSessionDetail } from "@/lib/server/history";

/** Транскрипт читается из БД движка на каждый запрос. */
export const dynamic = "force-dynamic";

export default async function SessionPage(props: { params: Promise<{ id: string }> }) {
  const { id } = await props.params;
  const data = await getSessionDetail(id);
  if (data === null) notFound();

  return (
    <main className="mx-auto w-full max-w-3xl px-4 py-5">
      <header className="mb-4 flex items-center justify-between gap-3">
        <h1 className="text-lg font-semibold">Сессия #{id}</h1>
        <Link href="/history" className="text-sm text-sky-300 underline-offset-2 hover:underline">
          ← К списку
        </Link>
      </header>

      <section className="mb-4 flex flex-wrap items-baseline gap-x-4 gap-y-1 text-sm text-slate-400">
        <span className="tabular-nums">{formatDateTime(data.session.started_at)}</span>
        <span>
          {LANG_LABEL[data.session.lang_from]} → {LANG_LABEL[data.session.lang_to]}
        </span>
        <span className="tabular-nums">{formatDuration(data.session.duration_ms)}</span>
      </section>

      {data.session.audio_path !== null ? (
        <audio
          className="mb-4 w-full"
          controls
          preload="none"
          src={sessionAudioUrl(data.session.id)}
          data-testid="session-audio"
        >
          Ваш браузер не поддерживает аудио.
        </audio>
      ) : (
        <p className="mb-4 text-xs text-slate-500">Запись аудио для этой сессии не велась.</p>
      )}

      <ol className="space-y-3">
        {data.utterances.map((utterance) => (
          <li key={utterance.id} className="flex gap-3">
            <span className="w-14 shrink-0 pt-0.5 text-xs tabular-nums text-slate-600">
              {formatTimecode(utterance.t_start_ms)}
            </span>
            <div className="min-w-0 flex-1">
              <p className="text-[11px] uppercase tracking-wide text-slate-500">
                {STREAM_LABEL[utterance.speaker]} · {utterance.lang}
              </p>
              <p className="text-slate-300">{utterance.text}</p>
              {utterance.translation !== null ? (
                <p className="text-slate-100">
                  {utterance.translation}
                  {utterance.translation_lang !== null ? (
                    <span className="ml-1 text-[11px] text-slate-500">
                      ({utterance.translation_lang})
                    </span>
                  ) : null}
                </p>
              ) : null}
            </div>
          </li>
        ))}
      </ol>

      {data.utterances.length === 0 ? (
        <p className="text-sm text-slate-500">В этой сессии нет реплик.</p>
      ) : null}
    </main>
  );
}
