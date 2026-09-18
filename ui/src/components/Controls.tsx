"use client";

import type { Lang, SessionStart } from "@/lib/contracts";
import { LANGS } from "@/lib/contracts";
import { LANG_LABEL } from "@/lib/format";
import type { VoiceRow } from "@/lib/api";

export interface ControlsProps {
  value: SessionStart;
  onChange: (value: SessionStart) => void;
  voices: VoiceRow[];
  /** Сессия запущена движком. */
  running: boolean;
  /** Соединение с движком открыто. */
  connected: boolean;
  onStart: () => void;
  onStop: () => void;
}

const selectClass =
  "min-w-0 flex-1 rounded border border-slate-700 bg-slate-900 px-1.5 py-1 text-xs text-slate-100 " +
  "focus:border-sky-500 focus:outline-none disabled:opacity-50";

export function Controls({
  value,
  onChange,
  voices,
  running,
  connected,
  onStart,
  onStop,
}: ControlsProps) {
  const locked = running;

  return (
    <section aria-label="Управление сессией" className="space-y-2 border-b border-slate-800 px-3 py-2">
      <div className="flex flex-wrap items-center gap-2">
        <label className="flex min-w-[8rem] flex-1 items-center gap-1.5 text-[11px] text-slate-400">
          <span className="shrink-0">Вход</span>
          <select
            className={selectClass}
            value={value.lang_in}
            disabled={locked}
            onChange={(e) => onChange({ ...value, lang_in: e.target.value as Lang })}
            aria-label="Язык собеседника"
          >
            {LANGS.map((lang) => (
              <option key={lang} value={lang}>
                {LANG_LABEL[lang]}
              </option>
            ))}
          </select>
        </label>

        <label className="flex min-w-[8rem] flex-1 items-center gap-1.5 text-[11px] text-slate-400">
          <span className="shrink-0">Выход</span>
          <select
            className={selectClass}
            value={value.lang_out}
            disabled={locked}
            onChange={(e) => onChange({ ...value, lang_out: e.target.value as Lang })}
            aria-label="Язык вашей речи"
          >
            {LANGS.map((lang) => (
              <option key={lang} value={lang}>
                {LANG_LABEL[lang]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
        <span className="shrink-0">Голос</span>
        <select
          className={selectClass}
          value={value.voice_id ?? ""}
          disabled={locked}
          onChange={(e) => onChange({ ...value, voice_id: e.target.value === "" ? null : e.target.value })}
          aria-label="Голосовой профиль"
        >
          <option value="">Голос по умолчанию (без клона)</option>
          {voices.map((voice) => (
            <option key={voice.id} value={voice.id}>
              {voice.name} ({voice.lang})
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-start gap-2 text-[11px] leading-snug text-slate-400">
        <input
          type="checkbox"
          className="mt-0.5 h-3.5 w-3.5 shrink-0 accent-rose-500"
          checked={value.record}
          disabled={locked}
          onChange={(e) => onChange({ ...value, record: e.target.checked })}
        />
        <span className={value.record ? "text-rose-300" : undefined}>
          Запись включена — уведомите собеседника
        </span>
      </label>

      <div className="flex gap-2">
        <button
          type="button"
          onClick={onStart}
          disabled={!connected || running}
          title={connected ? undefined : "Нет связи с движком"}
          className="flex-1 rounded bg-emerald-600 px-2 py-1.5 text-sm font-semibold text-white transition hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
        >
          Start
        </button>
        <button
          type="button"
          onClick={onStop}
          disabled={!connected || !running}
          className="flex-1 rounded bg-slate-700 px-2 py-1.5 text-sm font-semibold text-white transition hover:bg-slate-600 disabled:cursor-not-allowed disabled:bg-slate-800 disabled:text-slate-500"
        >
          Stop
        </button>
      </div>
    </section>
  );
}
