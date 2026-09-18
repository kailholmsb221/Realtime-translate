/**
 * Контракт-тест: TypeScript-типы в `src/lib/contracts.ts` обязаны совпадать с
 * реальными JSON-схемами из `engine/contracts/*.json`. Схемы читаются с диска,
 * поэтому любое расхождение (новый тип события, новое поле payload) уронит тест.
 */

import { readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { describe, expect, it } from "vitest";

import {
  EVENT_TYPES,
  PAYLOAD_FIELDS,
  isEvent,
  isEventType,
  makeEnvelope,
  parseEnvelope,
  sessionStart,
  sessionStop,
} from "@/lib/contracts";

// cwd под vitest — папка ui/, контракты лежат уровнем выше.
const CONTRACTS_DIR = path.resolve(process.cwd(), "..", "engine", "contracts");

interface JsonSchema {
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean;
}

interface JsonSchemaProperty {
  const?: string;
  type?: string | string[];
  enum?: string[];
  properties?: Record<string, JsonSchemaProperty>;
  required?: string[];
  additionalProperties?: boolean;
}

function readSchema(file: string): JsonSchema {
  return JSON.parse(readFileSync(path.join(CONTRACTS_DIR, file), "utf8")) as JsonSchema;
}

const schemaFiles = readdirSync(CONTRACTS_DIR)
  .filter((f) => f.endsWith(".json"))
  .sort();

const eventSchemaFiles = schemaFiles.filter((f) => f !== "envelope.json");

describe("JSON-схемы движка доступны из UI", () => {
  it("папка контрактов не пустая", () => {
    expect(eventSchemaFiles.length).toBeGreaterThan(0);
    expect(schemaFiles).toContain("envelope.json");
  });

  it("конверт описан как {type, ts, payload}", () => {
    const envelope = readSchema("envelope.json");
    expect(envelope.required?.slice().sort()).toEqual(["payload", "ts", "type"]);
    expect(envelope.additionalProperties).toBe(false);
  });
});

describe("набор типов событий совпадает со схемами", () => {
  it("каждый type-константа схемы известна UI и наоборот", () => {
    const fromSchemas = eventSchemaFiles
      .map((file) => readSchema(file).properties?.type?.const)
      .filter((value): value is string => typeof value === "string")
      .sort();

    expect(fromSchemas).toHaveLength(eventSchemaFiles.length);
    expect(fromSchemas).toEqual([...EVENT_TYPES].sort());
    fromSchemas.forEach((type) => expect(isEventType(type)).toBe(true));
  });

  it("имя файла схемы = значение type", () => {
    eventSchemaFiles.forEach((file) => {
      expect(readSchema(file).properties?.type?.const).toBe(file.replace(/\.json$/, ""));
    });
  });
});

describe("поля payload совпадают с TS-типами", () => {
  eventSchemaFiles.forEach((file) => {
    const schema = readSchema(file);
    const type = schema.properties?.type?.const as string;

    it(`${type}: required и properties схемы = ключи TS-типа`, () => {
      const payload = schema.properties?.payload;
      expect(payload, `${file}: нет payload`).toBeDefined();

      const required = [...(payload?.required ?? [])].sort();
      const properties = Object.keys(payload?.properties ?? {}).sort();
      const tsFields = [...PAYLOAD_FIELDS[type as keyof typeof PAYLOAD_FIELDS]].sort();

      // Все поля обязательны (см. engine/contracts/README.md).
      expect(required).toEqual(properties);
      expect(tsFields).toEqual(required);
      expect(payload?.additionalProperties).toBe(false);
    });
  });
});

describe("parseEnvelope", () => {
  const valid = JSON.stringify({
    type: "stt.final",
    ts: 1758240000000,
    payload: { stream: "in", lang: "en", text: "hello there", t_start_ms: 0, t_end_ms: 940 },
  });

  it("разбирает валидный конверт", () => {
    const envelope = parseEnvelope(valid);
    expect(envelope).not.toBeNull();
    expect(envelope?.type).toBe("stt.final");
    if (envelope?.type === "stt.final") {
      expect(envelope.payload.text).toBe("hello there");
      expect(envelope.payload.t_end_ms).toBe(940);
    }
  });

  it.each([
    ["не JSON", "{не json"],
    ["не объект", '"строка"'],
    ["неизвестный type", '{"type":"stt.super","ts":1,"payload":{}}'],
    ["нет ts", '{"type":"stt.partial","payload":{"stream":"in","lang":"en","text":"x"}}'],
    ["ts не integer", '{"type":"stt.partial","ts":1.5,"payload":{"stream":"in","lang":"en","text":"x"}}'],
    ["payload не объект", '{"type":"stt.partial","ts":1,"payload":"x"}'],
    ["нет поля payload", '{"type":"stt.partial","ts":1,"payload":{"stream":"in","lang":"en"}}'],
    [
      "лишнее поле payload",
      '{"type":"stt.partial","ts":1,"payload":{"stream":"in","lang":"en","text":"x","extra":1}}',
    ],
    ["лишнее поле конверта", '{"type":"session.stop","ts":1,"payload":{},"seq":3}'],
    ["неизвестный язык", '{"type":"stt.partial","ts":1,"payload":{"stream":"in","lang":"de","text":"x"}}'],
    ["неизвестный stream", '{"type":"stt.partial","ts":1,"payload":{"stream":"mid","lang":"en","text":"x"}}'],
    [
      "отрицательная длительность",
      '{"type":"stt.final","ts":1,"payload":{"stream":"in","lang":"en","text":"x","t_start_ms":-1,"t_end_ms":5}}',
    ],
    [
      "langs без out",
      '{"type":"session.state","ts":1,"payload":{"status":"idle","session_id":null,"langs":{"in":"ru"},"voice_id":null}}',
    ],
  ])("отбраковывает: %s", (_name, raw) => {
    expect(parseEnvelope(raw)).toBeNull();
  });

  it("пропускает tts.chunk (UI его игнорирует, но парсер не падает)", () => {
    const raw = JSON.stringify({
      type: "tts.chunk",
      ts: 2,
      payload: { stream: "out", seq: 0, pcm_base64: "AAAA" },
    });
    expect(parseEnvelope(raw)?.type).toBe("tts.chunk");
  });

  it("session.state с null-полями валиден", () => {
    const raw = JSON.stringify({
      type: "session.state",
      ts: 3,
      payload: { status: "idle", session_id: null, langs: { in: "ru", out: "kk" }, voice_id: null },
    });
    expect(parseEnvelope(raw)).not.toBeNull();
  });
});

describe("сборка команд", () => {
  it("session.start проходит собственную валидацию", () => {
    const command = sessionStart(
      { lang_in: "en", lang_out: "ru", voice_id: "v_ab12cd", record: true },
      42,
    );
    expect(command).toEqual({
      type: "session.start",
      ts: 42,
      payload: { lang_in: "en", lang_out: "ru", voice_id: "v_ab12cd", record: true },
    });
    expect(isEvent(command)).toBe(true);
  });

  it("session.stop — пустой payload", () => {
    const command = sessionStop(7);
    expect(command).toEqual({ type: "session.stop", ts: 7, payload: {} });
    expect(isEvent(command)).toBe(true);
  });

  it("makeEnvelope бросает на мусорном payload", () => {
    expect(() =>
      makeEnvelope("metrics.latency", {
        stream: "in",
        stt_ms: -1,
        mt_ms: 0,
        tts_ms: 0,
        total_ms: 0,
      }),
    ).toThrow();
  });
});
