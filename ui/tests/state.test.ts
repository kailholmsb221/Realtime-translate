/** Тесты редьюсера: ленты субтитров, привязка переводов, метрики, окно транскрипта. */

import { describe, expect, it } from "vitest";

import type { Envelope, EventType, PayloadMap } from "@/lib/contracts";
import { makeEnvelope } from "@/lib/contracts";
import {
  MAX_FINALS_PER_LANE,
  TRANSCRIPT_WINDOW_MS,
  countUtterances,
  initialState,
  reduce,
  reduceAll,
  selectTranscript,
} from "@/lib/state";

function ev<K extends EventType>(type: K, payload: PayloadMap[K], ts = 1000): Envelope {
  return makeEnvelope(type, payload, ts);
}

describe("stt.partial / stt.final", () => {
  it("partial заменяет живую строку, не накапливая её", () => {
    const state = reduceAll(initialState, [
      ev("stt.partial", { stream: "in", lang: "en", text: "Hi, can" }),
      ev("stt.partial", { stream: "in", lang: "en", text: "Hi, can you hear me" }),
    ]);

    expect(state.lanes.in.live).toEqual({ text: "Hi, can you hear me", lang: "en" });
    expect(state.lanes.in.finals).toHaveLength(0);
    expect(state.lanes.out.live).toBeNull();
  });

  it("final фиксирует строку и гасит живую", () => {
    const state = reduceAll(initialState, [
      ev("stt.partial", { stream: "in", lang: "en", text: "Hi, can" }),
      ev("stt.final", {
        stream: "in",
        lang: "en",
        text: "Hi, can you hear me well?",
        t_start_ms: 1200,
        t_end_ms: 4100,
      }),
    ]);

    expect(state.lanes.in.live).toBeNull();
    expect(state.lanes.in.finals).toHaveLength(1);
    expect(state.lanes.in.finals[0]).toMatchObject({
      text: "Hi, can you hear me well?",
      lang: "en",
      t_start_ms: 1200,
      t_end_ms: 4100,
      translation: null,
    });
  });

  it("потоки in и out независимы", () => {
    const state = reduceAll(initialState, [
      ev("stt.final", { stream: "in", lang: "en", text: "one", t_start_ms: 0, t_end_ms: 1 }),
      ev("stt.partial", { stream: "out", lang: "ru", text: "два" }),
    ]);

    expect(state.lanes.in.finals).toHaveLength(1);
    expect(state.lanes.in.live).toBeNull();
    expect(state.lanes.out.finals).toHaveLength(0);
    expect(state.lanes.out.live?.text).toBe("два");
  });

  it("лента не растёт бесконечно", () => {
    const events = Array.from({ length: MAX_FINALS_PER_LANE + 25 }, (_, i) =>
      ev("stt.final", { stream: "in", lang: "en", text: `line ${i}`, t_start_ms: i, t_end_ms: i + 1 }),
    );
    const state = reduceAll(initialState, events);

    expect(state.lanes.in.finals).toHaveLength(MAX_FINALS_PER_LANE);
    expect(state.lanes.in.finals.at(-1)?.text).toBe(`line ${MAX_FINALS_PER_LANE + 24}`);
  });

  it("ключи реплик уникальны", () => {
    const state = reduceAll(
      initialState,
      Array.from({ length: 5 }, (_, i) =>
        ev("stt.final", { stream: "in", lang: "en", text: `t${i}`, t_start_ms: 0, t_end_ms: 1 }),
      ),
    );
    const keys = state.lanes.in.finals.map((u) => u.key);
    expect(new Set(keys).size).toBe(keys.length);
  });
});

describe("translation.ready", () => {
  const final = (stream: "in" | "out", text: string, ts = 1000) =>
    ev("stt.final", { stream, lang: "en", text, t_start_ms: 0, t_end_ms: 10 }, ts);

  it("прикрепляется к последней final того же потока при ref_utterance_id = null", () => {
    const state = reduceAll(initialState, [
      final("in", "first"),
      final("in", "second"),
      ev("translation.ready", {
        stream: "in",
        src_lang: "en",
        dst_lang: "ru",
        src_text: "second",
        text: "второй",
        ref_utterance_id: null,
      }),
    ]);

    expect(state.lanes.in.finals[0].translation).toBeNull();
    expect(state.lanes.in.finals[1].translation).toBe("второй");
    expect(state.lanes.in.finals[1].translationLang).toBe("ru");
  });

  it("не задевает другой поток", () => {
    const state = reduceAll(initialState, [
      final("out", "mine"),
      final("in", "theirs"),
      ev("translation.ready", {
        stream: "in",
        src_lang: "en",
        dst_lang: "ru",
        src_text: "theirs",
        text: "их реплика",
        ref_utterance_id: 7,
      }),
    ]);

    expect(state.lanes.out.finals[0].translation).toBeNull();
    expect(state.lanes.in.finals[0].translation).toBe("их реплика");
    expect(state.lanes.in.finals[0].refUtteranceId).toBe(7);
  });

  it("переводы, пришедшие не по порядку, ложатся на свои реплики по src_text", () => {
    const state = reduceAll(initialState, [
      final("in", "alpha"),
      final("in", "beta"),
      ev("translation.ready", {
        stream: "in",
        src_lang: "en",
        dst_lang: "ru",
        src_text: "alpha",
        text: "альфа",
        ref_utterance_id: 11,
      }),
      ev("translation.ready", {
        stream: "in",
        src_lang: "en",
        dst_lang: "ru",
        src_text: "beta",
        text: "бета",
        ref_utterance_id: 12,
      }),
    ]);

    expect(state.lanes.in.finals.map((u) => u.translation)).toEqual(["альфа", "бета"]);
    expect(state.lanes.in.finals.map((u) => u.refUtteranceId)).toEqual([11, 12]);
  });

  it("повторный перевод той же реплики находит её по ref_utterance_id", () => {
    const state = reduceAll(initialState, [
      final("in", "alpha"),
      ev("translation.ready", {
        stream: "in",
        src_lang: "en",
        dst_lang: "ru",
        src_text: "alpha",
        text: "черновик",
        ref_utterance_id: 42,
      }),
      final("in", "beta"),
      ev("translation.ready", {
        stream: "in",
        src_lang: "en",
        dst_lang: "ru",
        src_text: "alpha",
        text: "исправленный перевод",
        ref_utterance_id: 42,
      }),
    ]);

    expect(state.lanes.in.finals[0].translation).toBe("исправленный перевод");
    expect(state.lanes.in.finals[1].translation).toBeNull();
  });

  it("перевод раньше stt.final создаёт реплику из src_text", () => {
    const state = reduce(
      initialState,
      ev("translation.ready", {
        stream: "out",
        src_lang: "ru",
        dst_lang: "kk",
        src_text: "Конечно, я готов.",
        text: "Әрине, мен дайынмын.",
        ref_utterance_id: null,
      }),
    );

    expect(state.lanes.out.finals).toHaveLength(1);
    expect(state.lanes.out.finals[0]).toMatchObject({
      text: "Конечно, я готов.",
      translation: "Әрине, мен дайынмын.",
      lang: "ru",
      translationLang: "kk",
    });
  });
});

describe("session.state и метрики", () => {
  it("состояние сессии сохраняется", () => {
    const state = reduce(
      initialState,
      ev("session.state", {
        status: "running",
        session_id: 5,
        langs: { in: "kk", out: "ru" },
        voice_id: "v_ab12cd",
      }),
    );

    expect(state.session.status).toBe("running");
    expect(state.session.session_id).toBe(5);
    expect(state.session.langs).toEqual({ in: "kk", out: "ru" });
  });

  it("старт новой сессии очищает ленты и метрики", () => {
    const withData = reduceAll(initialState, [
      ev("stt.final", { stream: "in", lang: "en", text: "old", t_start_ms: 0, t_end_ms: 1 }),
      ev("metrics.latency", { stream: "in", stt_ms: 1, mt_ms: 2, tts_ms: 3, total_ms: 6 }),
    ]);
    expect(countUtterances(withData)).toBe(1);

    const started = reduce(
      withData,
      ev("session.state", {
        status: "running",
        session_id: 1,
        langs: { in: "en", out: "ru" },
        voice_id: null,
      }),
    );

    expect(countUtterances(started)).toBe(0);
    expect(started.metrics.in).toBeNull();
  });

  it("stop не стирает субтитры (их ещё читают)", () => {
    const running = reduceAll(initialState, [
      ev("session.state", {
        status: "running",
        session_id: 1,
        langs: { in: "en", out: "ru" },
        voice_id: null,
      }),
      ev("stt.final", { stream: "in", lang: "en", text: "kept", t_start_ms: 0, t_end_ms: 1 }),
    ]);

    const stopped = reduce(
      running,
      ev("session.state", {
        status: "idle",
        session_id: null,
        langs: { in: "en", out: "ru" },
        voice_id: null,
      }),
    );

    expect(stopped.session.status).toBe("idle");
    expect(countUtterances(stopped)).toBe(1);
  });

  it("метрики хранятся по потокам", () => {
    const state = reduceAll(initialState, [
      ev("metrics.latency", { stream: "in", stt_ms: 420, mt_ms: 310, tts_ms: 640, total_ms: 1370 }),
      ev("metrics.latency", { stream: "out", stt_ms: 350, mt_ms: 420, tts_ms: 830, total_ms: 1600 }),
      ev("metrics.latency", { stream: "in", stt_ms: 910, mt_ms: 640, tts_ms: 1420, total_ms: 2970 }),
    ]);

    expect(state.metrics.in?.total_ms).toBe(2970);
    expect(state.metrics.out?.total_ms).toBe(1600);
  });

  it("tts.chunk не меняет состояние, кроме отметки времени", () => {
    const before = reduce(
      initialState,
      ev("stt.final", { stream: "in", lang: "en", text: "x", t_start_ms: 0, t_end_ms: 1 }, 100),
    );
    const after = reduce(
      before,
      ev("tts.chunk", { stream: "in", seq: 0, pcm_base64: "AAAA" }, 200),
    );

    expect(after.lanes).toEqual(before.lanes);
    expect(after.session).toEqual(before.session);
    expect(after.lastEventTs).toBe(200);
  });
});

describe("selectTranscript", () => {
  const minute = 60_000;

  it("отдаёт только реплики внутри окна, по возрастанию времени", () => {
    const state = reduceAll(initialState, [
      ev("stt.final", { stream: "in", lang: "en", text: "старое", t_start_ms: 0, t_end_ms: 1 }, 0),
      ev("stt.final", { stream: "out", lang: "ru", text: "свежее", t_start_ms: 2, t_end_ms: 3 }, 4 * minute),
      ev("stt.final", { stream: "in", lang: "en", text: "новое", t_start_ms: 4, t_end_ms: 5 }, 5 * minute),
    ]);

    const transcript = selectTranscript(state, { now: 6 * minute });
    expect(transcript.map((u) => u.text)).toEqual(["свежее", "новое"]);
  });

  it("окно по умолчанию — 5 минут, точка отсчёта — последнее событие", () => {
    expect(TRANSCRIPT_WINDOW_MS).toBe(5 * minute);

    const state = reduceAll(initialState, [
      ev("stt.final", { stream: "in", lang: "en", text: "давно", t_start_ms: 0, t_end_ms: 1 }, 0),
      ev("stt.final", { stream: "in", lang: "en", text: "только что", t_start_ms: 0, t_end_ms: 1 }, 10 * minute),
    ]);

    expect(selectTranscript(state).map((u) => u.text)).toEqual(["только что"]);
  });

  it("сужается явным windowMs", () => {
    const state = reduceAll(initialState, [
      ev("stt.final", { stream: "in", lang: "en", text: "a", t_start_ms: 0, t_end_ms: 1 }, 0),
      ev("stt.final", { stream: "in", lang: "en", text: "b", t_start_ms: 0, t_end_ms: 1 }, minute),
    ]);

    expect(selectTranscript(state, { now: minute, windowMs: 30_000 }).map((u) => u.text)).toEqual(["b"]);
  });

  it("на пустом состоянии — пустой список", () => {
    expect(selectTranscript(initialState)).toEqual([]);
  });
});
