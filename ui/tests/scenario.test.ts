/**
 * Сценарий mock-движка и фикстуры истории обязаны быть валидны по настоящим
 * контрактам: события — по `engine/contracts/*.json` (ajv, draft 2020-12),
 * строки истории — по ограничениям `db/schema.sql`.
 */

import { readFileSync } from "node:fs";
import path from "node:path";
import Ajv2020 from "ajv/dist/2020";
import type { ValidateFunction } from "ajv/dist/2020";
import { describe, expect, it } from "vitest";

import { LANGS, STREAMS, parseEnvelope } from "@/lib/contracts";
import { initialState, reduceAll } from "@/lib/state";

// cwd под vitest — папка ui/.
const CONTRACTS_DIR = path.resolve(process.cwd(), "..", "engine", "contracts");
const MOCKS_DIR = path.resolve(process.cwd(), "mocks");

interface ScenarioStep {
  delay_ms: number;
  event: { type: string; ts: number; payload: Record<string, unknown> };
}

const scenario = JSON.parse(
  readFileSync(path.join(MOCKS_DIR, "scenario.json"), "utf8"),
) as { steps: ScenarioStep[] };

const mockData = JSON.parse(readFileSync(path.join(MOCKS_DIR, "data.json"), "utf8")) as {
  sessions: { id: number; started_at: number; ended_at: number | null; lang_from: string; lang_to: string; audio_path: string | null }[];
  utterances: {
    id: number;
    session_id: number;
    t_start_ms: number;
    t_end_ms: number;
    speaker: string;
    lang: string;
    text: string;
    translation: string | null;
    translation_lang: string | null;
  }[];
  voices: { id: string; name: string; lang: string; sample_path: string; created_at: number }[];
};

const ajv = new Ajv2020({ strict: false, allErrors: true });

function loadValidator(type: string): ValidateFunction {
  const file = path.join(CONTRACTS_DIR, `${type}.json`);
  const schema = JSON.parse(readFileSync(file, "utf8")) as object;
  return ajv.compile(schema);
}

const envelopeValidator = loadValidator("envelope");
const validators = new Map<string, ValidateFunction>();

function validatorFor(type: string): ValidateFunction {
  const cached = validators.get(type);
  if (cached !== undefined) return cached;
  const validator = loadValidator(type);
  validators.set(type, validator);
  return validator;
}

describe("mocks/scenario.json", () => {
  it("не пустой и у каждого шага есть задержка", () => {
    expect(scenario.steps.length).toBeGreaterThan(10);
    scenario.steps.forEach((step, i) => {
      expect(Number.isInteger(step.delay_ms), `шаг ${i}: delay_ms должен быть целым`).toBe(true);
      expect(step.delay_ms).toBeGreaterThanOrEqual(0);
    });
  });

  it("каждое событие валидно по envelope.json и по схеме своего типа (ajv, draft 2020-12)", () => {
    scenario.steps.forEach((step, i) => {
      const okEnvelope = envelopeValidator(step.event);
      expect(okEnvelope, `шаг ${i}: ${ajv.errorsText(envelopeValidator.errors)}`).toBe(true);

      const validate = validatorFor(step.event.type);
      const ok = validate(step.event);
      expect(ok, `шаг ${i} (${step.event.type}): ${ajv.errorsText(validate.errors)}`).toBe(true);
    });
  });

  it("те же события принимает и парсер UI", () => {
    scenario.steps.forEach((step, i) => {
      expect(parseEnvelope(JSON.stringify(step.event)), `шаг ${i}`).not.toBeNull();
    });
  });

  it("покрывает оба потока, все три языка и полный цикл событий", () => {
    const types = new Set(scenario.steps.map((s) => s.event.type));
    expect(types).toContain("stt.partial");
    expect(types).toContain("stt.final");
    expect(types).toContain("translation.ready");
    expect(types).toContain("metrics.latency");

    const streams = new Set(scenario.steps.map((s) => s.event.payload.stream));
    STREAMS.forEach((stream) => expect(streams).toContain(stream));

    const langs = new Set(
      scenario.steps.flatMap((s) =>
        [s.event.payload.lang, s.event.payload.src_lang, s.event.payload.dst_lang].filter(Boolean),
      ),
    );
    LANGS.forEach((lang) => expect(langs).toContain(lang));
  });

  it("проигранный сценарий даёт осмысленное состояние UI", () => {
    const envelopes = scenario.steps
      .map((s) => parseEnvelope(JSON.stringify(s.event)))
      .filter((e) => e !== null);

    const state = reduceAll(initialState, envelopes);

    expect(state.lanes.in.finals.length).toBeGreaterThan(0);
    expect(state.lanes.out.finals.length).toBeGreaterThan(0);
    expect(state.lanes.in.finals.every((u) => u.translation !== null)).toBe(true);
    expect(state.lanes.out.finals.every((u) => u.translation !== null)).toBe(true);
    expect(state.metrics.in).not.toBeNull();
    expect(state.metrics.out).not.toBeNull();
  });
});

describe("mocks/data.json", () => {
  it("сессии соответствуют ограничениям db/schema.sql", () => {
    expect(mockData.sessions.length).toBeGreaterThan(0);
    mockData.sessions.forEach((session) => {
      expect(Number.isInteger(session.id)).toBe(true);
      expect(Number.isInteger(session.started_at)).toBe(true);
      expect(LANGS).toContain(session.lang_from);
      expect(LANGS).toContain(session.lang_to);
      if (session.ended_at !== null) {
        expect(session.ended_at).toBeGreaterThanOrEqual(session.started_at);
      }
    });
  });

  it("реплики ссылаются на существующие сессии и соблюдают CHECK-и", () => {
    const ids = new Set(mockData.sessions.map((s) => s.id));
    mockData.utterances.forEach((u) => {
      expect(ids).toContain(u.session_id);
      expect(STREAMS).toContain(u.speaker);
      expect(LANGS).toContain(u.lang);
      expect(u.t_start_ms).toBeGreaterThanOrEqual(0);
      expect(u.t_end_ms).toBeGreaterThanOrEqual(u.t_start_ms);
      if (u.translation_lang !== null) {
        expect(LANGS).toContain(u.translation_lang);
      } else {
        expect(u.translation).toBeNull();
      }
    });
  });

  it("голоса уникальны по id и имени", () => {
    const ids = mockData.voices.map((v) => v.id);
    const names = mockData.voices.map((v) => v.name);
    expect(new Set(ids).size).toBe(ids.length);
    expect(new Set(names).size).toBe(names.length);
    mockData.voices.forEach((voice) => {
      expect(LANGS).toContain(voice.lang);
      expect(voice.sample_path.length).toBeGreaterThan(0);
    });
  });
});
