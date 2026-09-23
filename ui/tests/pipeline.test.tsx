/**
 * Панель пайплайна: сборка цепочки whisper → NLLB → метрики в состоянии
 * и её рендер. Без WebSocket и без движка — только чистые функции и React.
 */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import { EventLog, describeEvent } from "@/components/EventLog";
import { PipelineInspector } from "@/components/PipelineInspector";
import type { Envelope, EventType, PayloadMap } from "@/lib/contracts";
import { makeEnvelope } from "@/lib/contracts";
import {
  MAX_PARTIALS_PER_UTTERANCE,
  initialState,
  reduceAll,
  selectPipelineTrace,
} from "@/lib/state";

function ev<K extends EventType>(type: K, payload: PayloadMap[K], ts = 1000): Envelope {
  return makeEnvelope(type, payload, ts);
}

/** Одна фраза «собеседника», прошедшая весь конвейер. */
const inbound: Envelope[] = [
  ev("stt.partial", { stream: "in", lang: "en", text: "Hi, can you" }, 1000),
  ev("stt.partial", { stream: "in", lang: "en", text: "Hi, can you hear me" }, 1100),
  ev(
    "stt.final",
    { stream: "in", lang: "en", text: "Hi, can you hear me well?", t_start_ms: 1200, t_end_ms: 4100 },
    1200,
  ),
  ev(
    "translation.ready",
    {
      stream: "in",
      src_lang: "en",
      dst_lang: "ru",
      src_text: "Hi, can you hear me well?",
      text: "Привет, хорошо меня слышно?",
      ref_utterance_id: 101,
    },
    1500,
  ),
  ev("metrics.latency", { stream: "in", stt_ms: 420, mt_ms: 310, tts_ms: 640, total_ms: 1370 }, 1600),
];

describe("состояние пайплайна", () => {
  it("гипотезы whisper прикрепляются к следующему финалу и буфер очищается", () => {
    const state = reduceAll(initialState, inbound);
    const [utterance] = state.lanes.in.finals;

    expect(utterance.partials).toEqual(["Hi, can you", "Hi, can you hear me"]);
    expect(state.lanes.in.partials).toEqual([]);
    expect(state.lanes.in.live).toBeNull();
  });

  it("повтор той же гипотезы не дублируется, буфер ограничен", () => {
    const many = Array.from({ length: MAX_PARTIALS_PER_UTTERANCE + 3 }, (_, i) =>
      ev("stt.partial", { stream: "in", lang: "en", text: `шаг ${i}` }, 1000 + i),
    );
    const state = reduceAll(initialState, [
      ...many,
      ev("stt.partial", { stream: "in", lang: "en", text: "шаг 10" }, 2000),
      ev("stt.partial", { stream: "in", lang: "en", text: "шаг 10" }, 2001),
    ]);

    expect(state.lanes.in.partials).toHaveLength(MAX_PARTIALS_PER_UTTERANCE);
    expect(state.lanes.in.partials.filter((p) => p === "шаг 10")).toHaveLength(1);
  });

  it("translation.ready запоминает текст, ушедший в NLLB, и время ответа", () => {
    const state = reduceAll(initialState, inbound);
    const [utterance] = state.lanes.in.finals;

    expect(utterance.srcText).toBe("Hi, can you hear me well?");
    expect(utterance.translation).toBe("Привет, хорошо меня слышно?");
    expect(utterance.translationLangRequested).toBe("ru");
    expect(utterance.translationTs).toBe(1500);
  });

  it("metrics.latency ложится на ту фразу, к которой относится (FIFO потока)", () => {
    const state = reduceAll(initialState, [
      ev("stt.final", { stream: "in", lang: "en", text: "one", t_start_ms: 0, t_end_ms: 900 }, 1000),
      ev("stt.final", { stream: "in", lang: "en", text: "two", t_start_ms: 1000, t_end_ms: 1900 }, 1100),
      ev(
        "translation.ready",
        {
          stream: "in",
          src_lang: "en",
          dst_lang: "ru",
          src_text: "one",
          text: "раз",
          ref_utterance_id: 1,
        },
        1200,
      ),
      ev("metrics.latency", { stream: "in", stt_ms: 100, mt_ms: 200, tts_ms: 300, total_ms: 600 }, 1250),
      ev(
        "translation.ready",
        {
          stream: "in",
          src_lang: "en",
          dst_lang: "ru",
          src_text: "two",
          text: "два",
          ref_utterance_id: 2,
        },
        1300,
      ),
      ev("metrics.latency", { stream: "in", stt_ms: 111, mt_ms: 222, tts_ms: 333, total_ms: 666 }, 1350),
    ]);

    expect(state.lanes.in.finals.map((u) => u.metrics?.total_ms)).toEqual([600, 666]);
    // Лента сохраняет и последнюю метрику потока — её показывает LatencyBar.
    expect(state.metrics.in?.total_ms).toBe(666);
  });

  it("selectPipelineTrace отдаёт свежие реплики обоих потоков сверху", () => {
    const state = reduceAll(initialState, [
      ...inbound,
      ev(
        "stt.final",
        { stream: "out", lang: "ru", text: "Да, слышно отлично.", t_start_ms: 5000, t_end_ms: 7400 },
        2000,
      ),
    ]);

    expect(selectPipelineTrace(state).map((u) => u.text)).toEqual([
      "Да, слышно отлично.",
      "Hi, can you hear me well?",
    ]);
    expect(selectPipelineTrace(state, { limit: 1 })).toHaveLength(1);
    expect(selectPipelineTrace(initialState)).toEqual([]);
  });
});

describe("PipelineInspector", () => {
  const noLive = { in: null, out: null };

  it("показывает всю цепочку: гипотезы, финал, вход NLLB, перевод и задержки", () => {
    const trace = selectPipelineTrace(reduceAll(initialState, inbound));
    render(<PipelineInspector trace={trace} live={noLive} />);

    const card = screen.getByTestId("pipeline-card");
    expect(within(card).getByTestId("partials")).toHaveTextContent(
      "Hi, can you→Hi, can you hear me",
    );
    expect(within(card).getByTestId("stt-final")).toHaveTextContent("Hi, can you hear me well?");
    expect(within(card).getByTestId("mt-source")).toHaveTextContent("Hi, can you hear me well?");
    expect(within(card).getByTestId("mt-source-diff")).toHaveTextContent("совпадает с финалом");
    expect(within(card).getByTestId("mt-result")).toHaveTextContent("Привет, хорошо меня слышно?");
    expect(within(card).getByTestId("stage-mt-in")).toHaveTextContent("eng_Latn → rus_Cyrl");
    expect(within(card).getByTestId("stage-stt")).toHaveTextContent("распознавание 420 мс");
    expect(within(card).getByTestId("stage-mt-out")).toHaveTextContent("310 мс");
    expect(within(card).getByTestId("total-latency")).toHaveAttribute("data-over-budget", "false");
  });

  it("при нулевом синтезе пишет «озвучки нет» (режим субтитров)", () => {
    const trace = selectPipelineTrace(
      reduceAll(initialState, [
        // та же фраза, но метрика без синтеза — как её шлёт режим субтитров
        ...inbound.slice(0, 4),
        ev("metrics.latency", { stream: "in", stt_ms: 300, mt_ms: 400, tts_ms: 0, total_ms: 800 }),
      ]),
    );
    render(<PipelineInspector trace={trace} live={noLive} />);

    expect(screen.getByTestId("tts-stage")).toHaveTextContent("озвучки нет");
  });

  it("помечает случай, когда в NLLB ушёл не тот текст, что выдал whisper", () => {
    const trace = selectPipelineTrace(
      reduceAll(initialState, [
        ev("stt.final", { stream: "in", lang: "en", text: "hello there", t_start_ms: 0, t_end_ms: 500 }),
        ev("translation.ready", {
          stream: "in",
          src_lang: "en",
          dst_lang: "ru",
          src_text: "hello there!",
          text: "привет",
          ref_utterance_id: null,
        }),
      ]),
    );
    render(<PipelineInspector trace={trace} live={noLive} />);

    expect(screen.getByTestId("mt-source-diff")).toHaveTextContent("отличается от финала whisper");
  });

  it("фраза без перевода висит в ожидании, превышение бюджета подсвечено", () => {
    const trace = selectPipelineTrace(
      reduceAll(initialState, [
        ev("stt.final", { stream: "out", lang: "ru", text: "тест", t_start_ms: 0, t_end_ms: 700 }),
        ev("translation.ready", {
          stream: "out",
          src_lang: "ru",
          dst_lang: "en",
          src_text: "тест",
          text: "test",
          ref_utterance_id: null,
        }),
        ev("metrics.latency", {
          stream: "out",
          stt_ms: 900,
          mt_ms: 800,
          tts_ms: 1400,
          total_ms: 3100,
        }),
        ev("stt.final", { stream: "out", lang: "ru", text: "ещё", t_start_ms: 800, t_end_ms: 1500 }, 2000),
      ]),
    );
    render(<PipelineInspector trace={trace} live={noLive} />);

    expect(screen.getByTestId("mt-pending")).toBeInTheDocument();
    expect(screen.getByTestId("total-latency")).toHaveAttribute("data-over-budget", "true");
  });

  it("незакрытая гипотеза показана отдельной карточкой", () => {
    render(
      <PipelineInspector trace={[]} live={{ in: { text: "Great, let's", lang: "en" }, out: null }} />,
    );

    expect(screen.getByTestId("pipeline-live-in")).toHaveTextContent("Great, let's");
    expect(screen.queryByTestId("pipeline-empty")).not.toBeInTheDocument();
  });

  it("без событий показывает подсказку", () => {
    render(<PipelineInspector trace={[]} live={noLive} />);
    expect(screen.getByTestId("pipeline-empty")).toBeInTheDocument();
  });
});

describe("EventLog", () => {
  it("описывает события человекочитаемо", () => {
    expect(describeEvent(inbound[3])).toContain("Hi, can you hear me well?  ⇒  Привет");
    expect(describeEvent(inbound[4])).toContain("итого 1370 мс");
  });

  it("новые события сверху, tts.chunk скрыт до явного включения", async () => {
    const user = userEvent.setup();
    const events: Envelope[] = [
      ...inbound,
      ev("tts.chunk", { stream: "in", seq: 0, pcm_base64: "AAAA" }, 1700),
    ];
    render(<EventLog events={events} invalidCount={2} />);

    expect(screen.getByTestId("log-count")).toHaveTextContent("6 событий");
    expect(screen.getByTestId("log-invalid")).toHaveTextContent("невалидных: 2");

    const rows = screen.getAllByTestId("log-row");
    expect(rows).toHaveLength(5);
    expect(rows[0]).toHaveTextContent("metrics.latency");

    await user.click(screen.getByLabelText(/показывать tts.chunk/));
    expect(screen.getAllByTestId("log-row")).toHaveLength(6);
  });
});
