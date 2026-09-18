/** Рендер главного окна с фейковым состоянием — без WebSocket и без движка. */

import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { TranslatorWindow } from "@/components/TranslatorWindow";
import type { Envelope, EventType, PayloadMap } from "@/lib/contracts";
import { makeEnvelope } from "@/lib/contracts";
import type { AppState } from "@/lib/state";
import { initialState, reduceAll } from "@/lib/state";

function ev<K extends EventType>(type: K, payload: PayloadMap[K], ts = 1000): Envelope {
  return makeEnvelope(type, payload, ts);
}

const voices = [
  { id: "v_ab12cd", name: "Мой голос (ru)", lang: "ru" as const, sample_path: "voices/a.wav", created_at: 1 },
];

function fakeState(): AppState {
  return reduceAll(initialState, [
    ev("session.state", {
      status: "running",
      session_id: 7,
      langs: { in: "en", out: "ru" },
      voice_id: "v_ab12cd",
    }),
    ev("stt.final", {
      stream: "in",
      lang: "en",
      text: "Hi, can you hear me well?",
      t_start_ms: 1200,
      t_end_ms: 4100,
    }),
    ev("translation.ready", {
      stream: "in",
      src_lang: "en",
      dst_lang: "ru",
      src_text: "Hi, can you hear me well?",
      text: "Привет, хорошо меня слышно?",
      ref_utterance_id: 101,
    }),
    ev("stt.final", {
      stream: "out",
      lang: "ru",
      text: "Да, слышно отлично.",
      t_start_ms: 5000,
      t_end_ms: 7400,
    }),
    ev("translation.ready", {
      stream: "out",
      src_lang: "ru",
      dst_lang: "en",
      src_text: "Да, слышно отлично.",
      text: "Yes, I hear you perfectly.",
      ref_utterance_id: 102,
    }),
    ev("stt.partial", { stream: "in", lang: "en", text: "Great, let's" }),
    ev("metrics.latency", { stream: "in", stt_ms: 420, mt_ms: 310, tts_ms: 640, total_ms: 1370 }),
    ev("metrics.latency", { stream: "out", stt_ms: 900, mt_ms: 800, tts_ms: 1400, total_ms: 3100 }),
  ]);
}

const noop = () => {};

describe("TranslatorWindow", () => {
  it("показывает обе ленты: оригинал, перевод и живую строку", () => {
    render(
      <TranslatorWindow status="open" state={fakeState()} voices={voices} onStart={noop} onStop={noop} />,
    );

    const incoming = screen.getByTestId("lane-in");
    expect(within(incoming).getByText("Hi, can you hear me well?")).toBeInTheDocument();
    expect(within(incoming).getByText("Привет, хорошо меня слышно?")).toBeInTheDocument();
    expect(within(incoming).getByTestId("live-in")).toHaveTextContent("Great, let's");

    const outgoing = screen.getByTestId("lane-out");
    expect(within(outgoing).getByText("Да, слышно отлично.")).toBeInTheDocument();
    expect(within(outgoing).getByText("Yes, I hear you perfectly.")).toBeInTheDocument();
  });

  it("подсвечивает превышение бюджета задержки", () => {
    render(
      <TranslatorWindow status="open" state={fakeState()} voices={voices} onStart={noop} onStop={noop} />,
    );

    expect(screen.getByTestId("latency-in")).toHaveAttribute("data-over-budget", "false");
    expect(screen.getByTestId("latency-in")).toHaveTextContent("total 1370 мс");
    expect(screen.getByTestId("latency-out")).toHaveAttribute("data-over-budget", "true");
    expect(screen.getByTestId("latency-out")).toHaveTextContent("total 3100 мс");
  });

  it("Start недоступен без соединения", () => {
    render(
      <TranslatorWindow
        status="closed"
        state={initialState}
        voices={voices}
        onStart={noop}
        onStop={noop}
      />,
    );

    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
    expect(screen.getByTestId("connection-status")).toHaveTextContent("нет связи");
  });

  it("Start отправляет выбранные языки, голос и флаг записи", async () => {
    const user = userEvent.setup();
    const onStart = vi.fn();

    render(
      <TranslatorWindow
        status="open"
        state={initialState}
        voices={voices}
        onStart={onStart}
        onStop={noop}
      />,
    );

    await user.selectOptions(screen.getByLabelText("Язык собеседника"), "kk");
    await user.selectOptions(screen.getByLabelText("Голосовой профиль"), "v_ab12cd");
    await user.click(screen.getByLabelText(/Запись включена/));
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStart).toHaveBeenCalledTimes(1);
    expect(onStart).toHaveBeenCalledWith({
      lang_in: "kk",
      lang_out: "ru",
      voice_id: "v_ab12cd",
      record: true,
    });
  });

  it("во время сессии Stop доступен и зовёт обработчик", async () => {
    const user = userEvent.setup();
    const onStop = vi.fn();

    render(
      <TranslatorWindow status="open" state={fakeState()} voices={voices} onStart={noop} onStop={onStop} />,
    );

    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Stop" }));
    expect(onStop).toHaveBeenCalledTimes(1);
  });

  it("в списке «мой голос» нет автоклонов собеседника (auto_*)", () => {
    // Движок сам заводит профиль auto_session<id>_in для голоса собеседника
    // (engine/orchestrator/autoclone.py) и по умолчанию удаляет его вместе с
    // сессией: пользователю выбирать его как свой голос незачем.
    const withAuto = [
      ...voices,
      {
        id: "v_auto01",
        name: "auto_session7_in",
        lang: "en" as const,
        sample_path: "voices/v_auto01/sample.wav",
        created_at: 2,
      },
    ];

    render(
      <TranslatorWindow
        status="open"
        state={initialState}
        voices={withAuto}
        onStart={noop}
        onStop={noop}
      />,
    );

    const select = screen.getByLabelText("Голосовой профиль");
    const options = within(select).getAllByRole("option").map((o) => o.textContent);
    expect(options).toContain("Мой голос (ru) (ru)");
    expect(options.some((label) => label?.includes("auto_"))).toBe(false);
  });

  it("панель ассистента показывает транскрипт, кнопка подсказок — заглушка", () => {
    render(
      <TranslatorWindow
        status="open"
        state={fakeState()}
        voices={voices}
        onStart={noop}
        onStop={noop}
        now={1000}
      />,
    );

    const panel = screen.getByTestId("assistant-panel");
    expect(within(panel).getByText(/Hi, can you hear me well\?/)).toBeInTheDocument();
    expect(within(panel).getByText(/Да, слышно отлично\./)).toBeInTheDocument();

    const hint = within(panel).getByRole("button", { name: "Подсказать ответ" });
    expect(hint).toBeDisabled();
    expect(hint).toHaveAttribute("title", "TODO: LLM-подсказки появятся позже");
  });
});
