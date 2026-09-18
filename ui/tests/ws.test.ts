/** Экспоненциальная задержка реконнекта: растёт вдвое, потолок — 10 секунд. */

import { describe, expect, it } from "vitest";

import { BACKOFF_MAX_MS, backoffDelay } from "@/lib/ws";

describe("backoffDelay", () => {
  it("удваивается с каждой попыткой", () => {
    expect(backoffDelay(0)).toBe(500);
    expect(backoffDelay(1)).toBe(1000);
    expect(backoffDelay(2)).toBe(2000);
    expect(backoffDelay(3)).toBe(4000);
    expect(backoffDelay(4)).toBe(8000);
  });

  it("не превышает 10 секунд", () => {
    expect(backoffDelay(5)).toBe(BACKOFF_MAX_MS);
    expect(backoffDelay(50)).toBe(BACKOFF_MAX_MS);
    expect(BACKOFF_MAX_MS).toBe(10_000);
  });

  it("устойчива к отрицательной попытке", () => {
    expect(backoffDelay(-3)).toBe(500);
  });
});
