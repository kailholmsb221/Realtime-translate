/**
 * Интеграционный тест на живой записи событий движка.
 *
 * `tests/fixtures/engine-live-events.json` — реальный поток конвертов, снятый с
 * `python -m engine.orchestrator --backend fake --fake-in ... --fake-out ... serve`
 * по ws://127.0.0.1:8765 (session.start → 14 секунд → session.stop). Здесь он
 * прогоняется через парсер контрактов и редьюсер UI: так проверяется, что
 * реальные формы событий движка UI понимает и раскладывает по лентам.
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import type { Envelope } from "@/lib/contracts";
import { parseEnvelope } from "@/lib/contracts";
import { countUtterances, initialState, reduceAll, selectTranscript } from "@/lib/state";

const FIXTURE = path.resolve(process.cwd(), "tests", "fixtures", "engine-live-events.json");

const raw = JSON.parse(readFileSync(FIXTURE, "utf8")) as unknown[];

/** Каждый конверт проходит парсер контрактов (он же используется в ws.ts). */
const envelopes: Envelope[] = raw.map((item, index) => {
  const parsed = parseEnvelope(JSON.stringify(item));
  expect(parsed, `событие #${index} не прошло parseEnvelope: ${JSON.stringify(item)}`).not.toBeNull();
  return parsed as Envelope;
});

describe("живой поток движка", () => {
  it("записан со всеми типами событий, которые UI обязан понимать", () => {
    const types = new Set(envelopes.map((e) => e.type));
    expect(types).toContain("session.state");
    expect(types).toContain("stt.partial");
    expect(types).toContain("stt.final");
    expect(types).toContain("translation.ready");
    expect(types).toContain("metrics.latency");
  });

  it("редьюсер раскладывает реплики по обеим лентам и привязывает переводы", () => {
    // Состояние на момент, когда сессия ещё идёт (до финального session.state idle).
    const idleTail = envelopes
      .map((e, i) => ({ e, i }))
      .filter(({ e }) => e.type === "session.state" && e.payload.status === "idle");
    const lastIdle = idleTail[idleTail.length - 1].i;
    const running = reduceAll(initialState, envelopes.slice(0, lastIdle));

    expect(running.session.status).toBe("running");
    expect(running.session.session_id).not.toBeNull();
    expect(running.session.langs).toEqual({ in: "en", out: "ru" });

    expect(running.lanes.in.finals.length).toBeGreaterThan(0);
    expect(running.lanes.out.finals.length).toBeGreaterThan(0);

    const finals = [...running.lanes.in.finals, ...running.lanes.out.finals];
    // Движок публикует translation.ready на каждый stt.final — ни одна реплика
    // не должна остаться без перевода и без ref_utterance_id.
    expect(finals.every((u) => u.translation !== null)).toBe(true);
    expect(finals.every((u) => u.refUtteranceId !== null)).toBe(true);
    // ref_utterance_id — id строки в БД, он уникален для каждой реплики.
    const refs = finals.map((u) => u.refUtteranceId);
    expect(new Set(refs).size).toBe(refs.length);

    // Направления перевода: in — en→ru, out — ru→en.
    expect(running.lanes.in.finals.every((u) => u.lang === "en" && u.translationLang === "ru")).toBe(
      true,
    );
    expect(
      running.lanes.out.finals.every((u) => u.lang === "ru" && u.translationLang === "en"),
    ).toBe(true);

    // Метрики приходят по обоим потокам и укладываются в контракт (>= 0).
    expect(running.metrics.in).not.toBeNull();
    expect(running.metrics.out).not.toBeNull();
    expect(running.metrics.in!.total_ms).toBeGreaterThanOrEqual(0);

    // Транскрипт ассистента отдаёт все реплики по возрастанию времени.
    const transcript = selectTranscript(running);
    expect(transcript).toHaveLength(countUtterances(running));
    for (let i = 1; i < transcript.length; i += 1) {
      expect(transcript[i].ts).toBeGreaterThanOrEqual(transcript[i - 1].ts);
    }
  });

  it("session.stop возвращает состояние в idle, ленты остаются", () => {
    const state = reduceAll(initialState, envelopes);
    expect(state.session.status).toBe("idle");
    expect(state.session.session_id).toBeNull();
    expect(countUtterances(state)).toBeGreaterThan(0);

    // В записи последним пришёл stt.partial без своего stt.final: живая строка
    // не должна пережить остановку сессии.
    expect(state.lanes.in.live).toBeNull();
    expect(state.lanes.out.live).toBeNull();
  });
});
