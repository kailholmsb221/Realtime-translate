/**
 * Серверная загрузка истории: страницы `/history` и `/history/[id]` рендерятся
 * на сервере и обязаны отдавать транскрипт прямо в HTML (регрессия: раньше
 * данные тянулись из useEffect, и в ответе был только «Загрузка…»).
 */

import { afterEach, describe, expect, it, vi } from "vitest";

import { getSessionDetail, getSessions } from "@/lib/server/history";

const engineSession = {
  id: 42,
  started_at: 1_758_240_000_000,
  ended_at: 1_758_240_315_000,
  lang_from: "en",
  lang_to: "ru",
  audio_path: "/recordings/42_in.wav",
  utterance_count: 1,
  duration_ms: 315_000,
};

const engineUtterance = {
  id: 101,
  session_id: 42,
  t_start_ms: 1200,
  t_end_ms: 4100,
  speaker: "in",
  lang: "en",
  text: "Hi, can you hear me well?",
  translation: "Привет, хорошо меня слышно?",
  translation_lang: "ru",
};

function stubEngine(body: unknown, ok = true): void {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status: ok ? 200 : 404 })),
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("getSessions", () => {
  it("отдаёт список движка как есть", async () => {
    stubEngine({ sessions: [engineSession] });
    await expect(getSessions()).resolves.toEqual([engineSession]);
  });

  it("без движка падает на фикстуры и сортирует новые сверху", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("движок не запущен");
      }),
    );

    const sessions = await getSessions();
    expect(sessions.length).toBeGreaterThan(0);
    for (let i = 1; i < sessions.length; i += 1) {
      expect(sessions[i - 1].started_at).toBeGreaterThanOrEqual(sessions[i].started_at);
    }
    // Производные поля контракта проставлены и в мок-режиме.
    expect(sessions[0]).toHaveProperty("utterance_count");
    expect(sessions[0]).toHaveProperty("duration_ms");
  });
});

describe("getSessionDetail", () => {
  it("отдаёт сессию с репликами из движка", async () => {
    stubEngine({ session: engineSession, utterances: [engineUtterance] });

    const data = await getSessionDetail("42");
    expect(data?.session.id).toBe(42);
    expect(data?.utterances).toHaveLength(1);
    expect(data?.utterances[0].translation).toBe("Привет, хорошо меня слышно?");
  });

  it("несуществующая сессия — null (страница отдаёт 404)", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("движок не запущен");
      }),
    );

    await expect(getSessionDetail("999999")).resolves.toBeNull();
  });

  it("реплики фикстур отсортированы по t_start_ms", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => {
        throw new Error("движок не запущен");
      }),
    );

    const sessions = await getSessions();
    const data = await getSessionDetail(String(sessions[0].id));
    expect(data).not.toBeNull();
    const starts = data!.utterances.map((u) => u.t_start_ms);
    expect([...starts].sort((a, b) => a - b)).toEqual(starts);
  });
});
