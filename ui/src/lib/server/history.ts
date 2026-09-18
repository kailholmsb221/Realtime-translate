/**
 * Серверная часть заглушек истории (`src/app/api/**`).
 *
 * Логика простая: если движок доступен по `ENGINE_HTTP`
 * (по умолчанию `http://localhost:8766`) — проксируем запрос к нему;
 * если нет — отдаём фикстуры из `mocks/data.json`, чтобы UI работал автономно.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";

import type { SessionRow, UtteranceRow, VoiceRow } from "@/lib/api";

/** Базовый URL REST-API движка (серверная переменная, не попадает в бандл клиента). */
export const ENGINE_HTTP =
  process.env.ENGINE_HTTP ?? process.env.NEXT_PUBLIC_ENGINE_HTTP ?? "http://localhost:8766";

/** Таймаут похода в движок, мс: заглушка не должна подвешивать UI. */
const ENGINE_TIMEOUT_MS = 1500;

/** Сырые строки таблиц, как в `db/schema.sql`. */
export interface MockData {
  sessions: Omit<SessionRow, "utterance_count" | "duration_ms">[];
  utterances: UtteranceRow[];
  voices: VoiceRow[];
}

let cache: MockData | null = null;

export async function loadMockData(): Promise<MockData> {
  if (cache !== null) return cache;
  const file = path.join(process.cwd(), "mocks", "data.json");
  const parsed = JSON.parse(await readFile(file, "utf8")) as MockData;
  cache = parsed;
  return parsed;
}

/** Добавляет производные поля списка истории. */
export function withDerived(
  session: MockData["sessions"][number],
  utterances: UtteranceRow[],
): SessionRow {
  return {
    ...session,
    utterance_count: utterances.filter((u) => u.session_id === session.id).length,
    duration_ms: session.ended_at === null ? null : session.ended_at - session.started_at,
  };
}

/**
 * Пытается сходить в движок; при любой ошибке/таймауте возвращает `null`,
 * и вызывающий отдаёт моки.
 */
export async function tryEngine(pathname: string): Promise<Response | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ENGINE_TIMEOUT_MS);
  try {
    const response = await fetch(`${ENGINE_HTTP}${pathname}`, {
      signal: controller.signal,
      cache: "no-store",
    });
    if (!response.ok) return null;
    return response;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Короткий WAV-тон вместо реальной записи — чтобы `<audio>` было что играть в мок-режиме. */
export function makeToneWav(seconds = 2, sampleRate = 22050, frequency = 440): Buffer {
  const samples = Math.floor(seconds * sampleRate);
  const data = Buffer.alloc(samples * 2);
  for (let i = 0; i < samples; i += 1) {
    const fade = Math.min(1, Math.min(i, samples - i) / (sampleRate * 0.05));
    const value = Math.sin((2 * Math.PI * frequency * i) / sampleRate) * 0.25 * fade;
    data.writeInt16LE(Math.round(value * 32767), i * 2);
  }

  const header = Buffer.alloc(44);
  header.write("RIFF", 0);
  header.writeUInt32LE(36 + data.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16); // размер fmt-чанка
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(sampleRate * 2, 28); // byte rate
  header.writeUInt16LE(2, 32); // block align
  header.writeUInt16LE(16, 34); // bits per sample
  header.write("data", 36);
  header.writeUInt32LE(data.length, 40);

  return Buffer.concat([header, data]);
}
