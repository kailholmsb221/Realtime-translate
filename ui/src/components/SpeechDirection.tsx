"use client";

/**
 * Выбор направления перевода для панели пайплайна.
 *
 * Зачем отдельный компонент, а не `Controls`: там подписи «Вход» и «Выход»
 * означают язык **собеседника** и язык **пользователя** (ARCHITECTURE.md
 * раздел 5), и это легко прочитать наоборот — «вход = что я говорю». Тогда
 * `session.start` уходит с перевёрнутыми языками, движок ждёт от микрофона не
 * тот язык, и перевод получается бессмысленным. Здесь подписи говорят прямо,
 * что с чем делают, а раскладка по контракту остаётся внутри компонента:
 * `lang_out` — язык пользователя, `lang_in` — язык перевода его речи.
 */

import type { Lang, SessionStart } from "@/lib/contracts";
import { LANGS } from "@/lib/contracts";
import { LANG_LABEL } from "@/lib/format";

export interface SpeechDirectionProps {
  value: SessionStart;
  onChange: (value: SessionStart) => void;
  /** Сессия уже идёт — языки менять поздно. */
  running: boolean;
  /** Есть связь с движком. */
  connected: boolean;
  onStart: () => void;
  onStop: () => void;
}

const selectClass =
  "min-w-0 flex-1 rounded border border-slate-700 bg-slate-900 px-1.5 py-1 text-xs text-slate-100 " +
  "focus:border-sky-500 focus:outline-none disabled:opacity-50";

export function SpeechDirection({
  value,
  onChange,
  running,
  connected,
  onStart,
  onStop,
}: SpeechDirectionProps) {
  const swap = () => onChange({ ...value, lang_in: value.lang_out, lang_out: value.lang_in });

  return (
    <section aria-label="Направление перевода" className="space-y-2 px-3 py-2">
      <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
        <span className="w-[6.5rem] shrink-0">Я говорю на</span>
        <select
          className={selectClass}
          value={value.lang_out}
          disabled={running}
          onChange={(e) => onChange({ ...value, lang_out: e.target.value as Lang })}
          aria-label="Я говорю на"
        >
          {LANGS.map((lang) => (
            <option key={lang} value={lang}>
              {LANG_LABEL[lang]}
            </option>
          ))}
        </select>
      </label>

      <label className="flex items-center gap-1.5 text-[11px] text-slate-400">
        <span className="w-[6.5rem] shrink-0">Переводить на</span>
        <select
          className={selectClass}
          value={value.lang_in}
          disabled={running}
          onChange={(e) => onChange({ ...value, lang_in: e.target.value as Lang })}
          aria-label="Переводить на"
        >
          {LANGS.map((lang) => (
            <option key={lang} value={lang}>
              {LANG_LABEL[lang]}
            </option>
          ))}
        </select>
      </label>

      <button
        type="button"
        onClick={swap}
        disabled={running}
        className="w-full rounded border border-slate-700 px-2 py-1 text-[11px] text-slate-400 hover:text-slate-200 disabled:opacity-50"
      >
        ⇄ поменять языки местами
      </button>

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

      <p className="text-[10px] leading-snug text-slate-600">
        В контракте это `lang_out` (язык пользователя) и `lang_in` (язык собеседника) — панель
        раскладывает их сама.
      </p>
    </section>
  );
}
