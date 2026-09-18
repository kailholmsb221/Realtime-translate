/** Мелкие форматтеры для субтитров и истории. */

import type { Lang, Stream } from "@/lib/contracts";

export const LANG_LABEL: Record<Lang, string> = {
  ru: "Русский",
  en: "English",
  kk: "Қазақша",
};

export const STREAM_LABEL: Record<Stream, string> = {
  in: "Собеседник",
  out: "Вы",
};

/** `mm:ss` от начала сессии (для транскрипта истории). */
export function formatTimecode(ms: number): string {
  const totalSeconds = Math.max(0, Math.floor(ms / 1000));
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = totalSeconds % 60;
  const mm = String(minutes).padStart(2, "0");
  const ss = String(seconds).padStart(2, "0");
  return hours > 0 ? `${hours}:${mm}:${ss}` : `${mm}:${ss}`;
}

/** Длительность сессии человекочитаемо. */
export function formatDuration(ms: number | null): string {
  if (ms === null || ms < 0) return "—";
  const totalMinutes = Math.floor(ms / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  if (totalMinutes === 0) return `${seconds} с`;
  return `${totalMinutes} мин ${String(seconds).padStart(2, "0")} с`;
}

/** Дата сессии; формат фиксирован, чтобы SSR и клиент не расходились. */
export function formatDateTime(unixMs: number): string {
  const d = new Date(unixMs);
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(d.getDate())}.${pad(d.getMonth() + 1)}.${d.getFullYear()} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

/** Склонение "реплика/реплики/реплик". */
export function pluralUtterances(count: number): string {
  const mod10 = count % 10;
  const mod100 = count % 100;
  if (mod10 === 1 && mod100 !== 11) return `${count} реплика`;
  if (mod10 >= 2 && mod10 <= 4 && (mod100 < 12 || mod100 > 14)) return `${count} реплики`;
  return `${count} реплик`;
}
