import type { MetricsLatency, Stream } from "@/lib/contracts";
import { LATENCY_BUDGET_MS, STREAMS } from "@/lib/contracts";
import { STREAM_LABEL } from "@/lib/format";

export interface LatencyBarProps {
  metrics: Record<Stream, MetricsLatency | null>;
}

function Cell({ label, value }: { label: string; value: number }) {
  return (
    <span className="whitespace-nowrap text-slate-400">
      {label} <span className="tabular-nums text-slate-200">{value}</span>
    </span>
  );
}

function Row({ stream, metrics }: { stream: Stream; metrics: MetricsLatency | null }) {
  const overBudget = metrics !== null && metrics.total_ms > LATENCY_BUDGET_MS;
  return (
    <div
      className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 text-[11px]"
      data-testid={`latency-${stream}`}
      data-over-budget={overBudget ? "true" : "false"}
    >
      <span className="w-[5.5rem] shrink-0 text-slate-500">{STREAM_LABEL[stream]}</span>
      {metrics === null ? (
        <span className="text-slate-600">нет данных</span>
      ) : (
        <>
          <Cell label="stt" value={metrics.stt_ms} />
          <Cell label="mt" value={metrics.mt_ms} />
          <Cell label="tts" value={metrics.tts_ms} />
          <span
            className={`whitespace-nowrap rounded px-1 tabular-nums ${
              overBudget ? "bg-rose-600/25 font-semibold text-rose-300" : "text-emerald-300"
            }`}
            title={
              overBudget
                ? `Превышен бюджет задержки ${LATENCY_BUDGET_MS} мс`
                : `Бюджет задержки ${LATENCY_BUDGET_MS} мс`
            }
          >
            total {metrics.total_ms} мс
          </span>
        </>
      )}
    </div>
  );
}

export function LatencyBar({ metrics }: LatencyBarProps) {
  return (
    <section
      aria-label="Задержки пайплайна"
      className="space-y-0.5 border-b border-slate-800 px-3 py-1.5"
    >
      {STREAMS.map((stream) => (
        <Row key={stream} stream={stream} metrics={metrics[stream]} />
      ))}
    </section>
  );
}
