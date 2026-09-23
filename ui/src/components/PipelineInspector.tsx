"use client";

/**
 * Панель пайплайна: что услышал whisper, какой текст ушёл в NLLB и что он вернул.
 *
 * Компонент презентационный — всё состояние приходит пропсами, поэтому его
 * можно рендерить в тестах без WebSocket и без движка.
 */

import type { Lang, Stream } from "@/lib/contracts";
import { LATENCY_BUDGET_MS, STREAMS } from "@/lib/contracts";
import {
  LANG_LABEL,
  NLLB_LANG_CODE,
  STREAM_LABEL,
  formatClock,
  formatSeconds,
} from "@/lib/format";
import type { Utterance } from "@/lib/state";

export interface PipelineInspectorProps {
  /** Последние реплики обоих потоков, новые сверху (`selectPipelineTrace`). */
  trace: Utterance[];
  /** Текущая незакрытая гипотеза whisper по потокам. */
  live: Record<Stream, { text: string; lang: Lang } | null>;
}

const STREAM_DOT: Record<Stream, string> = {
  in: "bg-sky-400",
  out: "bg-violet-400",
};

const STREAM_RING: Record<Stream, string> = {
  in: "border-sky-900/70",
  out: "border-violet-900/70",
};

function Badge({ children, title }: { children: React.ReactNode; title?: string }) {
  return (
    <span
      title={title}
      className="rounded bg-slate-800/80 px-1.5 py-0.5 font-mono text-[10px] text-slate-300"
    >
      {children}
    </span>
  );
}

function Stage({
  n,
  title,
  hint,
  children,
  testId,
}: {
  n: number;
  title: string;
  hint?: React.ReactNode;
  children: React.ReactNode;
  testId?: string;
}) {
  return (
    <li className="relative pl-7" data-testid={testId}>
      <span
        aria-hidden="true"
        className="absolute left-0 top-0.5 flex h-5 w-5 items-center justify-center rounded-full bg-slate-800 text-[11px] font-semibold text-slate-300"
      >
        {n}
      </span>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5">
        <h4 className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">
          {title}
        </h4>
        {hint}
      </div>
      <div className="mt-1 space-y-1">{children}</div>
    </li>
  );
}

/** Цепочка гипотез whisper: одна строка со стрелками между шагами. */
function Hypotheses({ partials }: { partials: string[] }) {
  if (partials.length === 0) {
    return (
      <p className="text-[11px] text-slate-600" data-testid="partials-empty">
        промежуточных гипотез не было
      </p>
    );
  }
  return (
    <p className="text-[12px] leading-relaxed text-slate-500" data-testid="partials">
      {partials.map((text, index) => (
        <span key={`${index}-${text}`}>
          {index > 0 ? <span className="mx-1 text-slate-700">→</span> : null}
          <span className="italic">{text}</span>
        </span>
      ))}
    </p>
  );
}

function PipelineCard({ utterance }: { utterance: Utterance }) {
  const { metrics } = utterance;
  const dstLang = utterance.translationLangRequested ?? utterance.translationLang;
  const waitingTranslation = utterance.translation === null;
  const changedBeforeMt = utterance.srcText !== null && utterance.srcText !== utterance.text;
  const overBudget = metrics !== null && metrics.total_ms > LATENCY_BUDGET_MS;

  return (
    <article
      data-testid="pipeline-card"
      data-stream={utterance.stream}
      className={`rounded-lg border ${STREAM_RING[utterance.stream]} bg-slate-900/50 p-3`}
    >
      <header className="mb-2 flex flex-wrap items-center gap-x-2 gap-y-1 text-[11px]">
        <span className="flex items-center gap-1.5 font-semibold text-slate-300">
          <span
            className={`inline-block h-2 w-2 rounded-full ${STREAM_DOT[utterance.stream]}`}
            aria-hidden="true"
          />
          {STREAM_LABEL[utterance.stream]}
        </span>
        <span className="text-slate-500">
          {LANG_LABEL[utterance.lang]} → {dstLang === null ? "?" : LANG_LABEL[dstLang]}
        </span>
        <span className="tabular-nums text-slate-600">{formatClock(utterance.ts)}</span>
        {utterance.refUtteranceId !== null ? (
          <Badge title="id строки в таблице utterances">#{utterance.refUtteranceId}</Badge>
        ) : null}
      </header>

      <ol className="space-y-3">
        <Stage
          n={1}
          title="Whisper · распознавание"
          hint={<Badge title="ARCHITECTURE.md 4.2">faster-whisper small · int8</Badge>}
          testId="stage-stt"
        >
          <Hypotheses partials={utterance.partials} />
          <p className="text-[15px] leading-snug text-slate-100" data-testid="stt-final">
            {utterance.text}
          </p>
          <p className="text-[11px] tabular-nums text-slate-600">
            фраза {formatSeconds(utterance.t_start_ms)} → {formatSeconds(utterance.t_end_ms)}
            {metrics !== null ? ` · распознавание ${metrics.stt_ms} мс` : null}
          </p>
        </Stage>

        <Stage
          n={2}
          title="NLLB · вход"
          hint={
            dstLang === null ? null : (
              <Badge title="коды FLORES-200, engine/translate/nllb.py">
                {NLLB_LANG_CODE[utterance.lang]} → {NLLB_LANG_CODE[dstLang]}
              </Badge>
            )
          }
          testId="stage-mt-in"
        >
          {utterance.srcText === null ? (
            <p className="text-[13px] italic text-slate-600">ждёт перевода…</p>
          ) : (
            <>
              <p className="text-[15px] leading-snug text-slate-200" data-testid="mt-source">
                {utterance.srcText}
              </p>
              <p
                className={`text-[11px] ${changedBeforeMt ? "text-amber-400" : "text-slate-600"}`}
                data-testid="mt-source-diff"
              >
                {changedBeforeMt
                  ? "⚠ отличается от финала whisper"
                  : "совпадает с финалом whisper — текст передан как есть"}
              </p>
            </>
          )}
        </Stage>

        <Stage
          n={3}
          title="NLLB · перевод"
          hint={
            metrics !== null ? (
              <Badge title="mt_ms из metrics.latency">{metrics.mt_ms} мс</Badge>
            ) : null
          }
          testId="stage-mt-out"
        >
          {waitingTranslation ? (
            <p className="text-[13px] italic text-slate-600" data-testid="mt-pending">
              перевод…
            </p>
          ) : (
            <p
              className="text-[17px] font-medium leading-snug text-emerald-100"
              data-testid="mt-result"
            >
              {utterance.translation}
            </p>
          )}
        </Stage>

        <Stage n={4} title="Озвучка и задержка" testId="stage-tts">
          {metrics === null ? (
            <p className="text-[11px] text-slate-600">метрики ещё не пришли</p>
          ) : (
            <p className="flex flex-wrap gap-x-3 text-[11px] tabular-nums text-slate-400">
              {/* Режим субтитров (scripts/mic_subtitles.py) ничего не озвучивает. */}
              <span data-testid="tts-stage">
                {metrics.tts_ms === 0 ? "озвучки нет" : `синтез ${metrics.tts_ms} мс`}
              </span>
              <span
                data-testid="total-latency"
                data-over-budget={overBudget ? "true" : "false"}
                className={`rounded px-1 ${
                  overBudget ? "bg-rose-600/25 font-semibold text-rose-300" : "text-emerald-300"
                }`}
                title={`Бюджет ${LATENCY_BUDGET_MS} мс (ARCHITECTURE.md раздел 2)`}
              >
                итого {metrics.total_ms} мс
              </span>
            </p>
          )}
        </Stage>
      </ol>
    </article>
  );
}

/** Незакрытая гипотеза: whisper ещё слушает, до NLLB фраза не дошла. */
function LiveCard({ stream, text }: { stream: Stream; text: string }) {
  return (
    <article
      data-testid={`pipeline-live-${stream}`}
      className="rounded-lg border border-dashed border-slate-800 bg-slate-900/30 p-3"
    >
      <header className="mb-1 flex items-center gap-1.5 text-[11px] font-semibold text-slate-400">
        <span
          className={`inline-block h-2 w-2 rounded-full ${STREAM_DOT[stream]}`}
          aria-hidden="true"
        />
        {STREAM_LABEL[stream]} · whisper слушает
      </header>
      <p className="text-[15px] italic leading-snug text-slate-400">
        {text}
        <span className="ml-0.5 animate-pulse" aria-hidden="true">
          ▌
        </span>
      </p>
      <p className="mt-1 text-[11px] text-slate-600">фраза не закрыта — в NLLB ещё не ушла</p>
    </article>
  );
}

export function PipelineInspector({ trace, live }: PipelineInspectorProps) {
  const liveStreams = STREAMS.filter((stream) => live[stream] !== null);

  return (
    <section aria-label="Пайплайн распознавания и перевода" className="space-y-3">
      {liveStreams.map((stream) => {
        const current = live[stream];
        return current === null ? null : (
          <LiveCard key={stream} stream={stream} text={current.text} />
        );
      })}

      {trace.length === 0 && liveStreams.length === 0 ? (
        <p
          className="rounded-lg border border-dashed border-slate-800 px-3 py-6 text-center text-sm text-slate-600"
          data-testid="pipeline-empty"
        >
          Пока пусто. Запустите сессию — здесь появятся распознанные фразы, текст для NLLB и его
          перевод.
        </p>
      ) : null}

      {trace.map((utterance) => (
        <PipelineCard key={utterance.key} utterance={utterance} />
      ))}
    </section>
  );
}
