/**
 * TypeScript-типы контрактов WebSocket, 1:1 с JSON-схемами из
 * `engine/contracts/*.json` (ARCHITECTURE.md раздел 5).
 *
 * ВАЖНО: схемы неприкосновенны (CLAUDE.md, закон 2). Этот файл — только
 * зеркало схем на стороне UI. Соответствие проверяется тестом
 * `tests/contracts.test.ts`, который читает реальные JSON-схемы через fs.
 */

export const LANGS = ["ru", "en", "kk"] as const;
export type Lang = (typeof LANGS)[number];

export const STREAMS = ["in", "out"] as const;
/** `in` — речь собеседника (loopback из Zoom), `out` — речь пользователя (микрофон). */
export type Stream = (typeof STREAMS)[number];

export const SESSION_STATUSES = ["idle", "running"] as const;
export type SessionStatus = (typeof SESSION_STATUSES)[number];

/* ------------------------------------------------------------------ */
/* Payload-типы                                                        */
/* ------------------------------------------------------------------ */

/** `stt.partial` — промежуточная гипотеза распознавания. */
export interface SttPartial {
  stream: Stream;
  lang: Lang;
  text: string;
}

/** `stt.final` — окончательная фраза с таймкодами. */
export interface SttFinal {
  stream: Stream;
  lang: Lang;
  text: string;
  t_start_ms: number;
  t_end_ms: number;
}

/** `translation.ready` — готовый перевод одной фразы. */
export interface TranslationReady {
  stream: Stream;
  src_lang: Lang;
  dst_lang: Lang;
  src_text: string;
  text: string;
  /** id строки в таблице `utterances`; `null`, если реплика ещё не записана. */
  ref_utterance_id: number | null;
}

/** `tts.chunk` — служебный чанк PCM 24 kHz mono int16 в base64. UI его игнорирует. */
export interface TtsChunk {
  stream: Stream;
  seq: number;
  pcm_base64: string;
}

/** Языки сессии по направлениям потоков. */
export interface SessionLangs {
  in: Lang;
  out: Lang;
}

/** `session.state` — текущее состояние движка. */
export interface SessionState {
  status: SessionStatus;
  session_id: number | null;
  langs: SessionLangs;
  voice_id: string | null;
}

/** `metrics.latency` — задержки по стадиям пайплайна (мс). Бюджет `total_ms <= 2500`. */
export interface MetricsLatency {
  stream: Stream;
  stt_ms: number;
  mt_ms: number;
  tts_ms: number;
  total_ms: number;
}

/** `session.start` — команда UI → движок. */
export interface SessionStart {
  lang_in: Lang;
  lang_out: Lang;
  voice_id: string | null;
  record: boolean;
}

/** `session.stop` — команда UI → движок, payload пустой. */
export type SessionStop = Record<never, never>;

/* ------------------------------------------------------------------ */
/* Конверт                                                             */
/* ------------------------------------------------------------------ */

export interface PayloadMap {
  "stt.partial": SttPartial;
  "stt.final": SttFinal;
  "translation.ready": TranslationReady;
  "tts.chunk": TtsChunk;
  "session.state": SessionState;
  "metrics.latency": MetricsLatency;
  "session.start": SessionStart;
  "session.stop": SessionStop;
}

export type EventType = keyof PayloadMap;

type AnyEnvelope = {
  [T in EventType]: { type: T; ts: number; payload: PayloadMap[T] };
}[EventType];

/** Конверт `{ type, ts, payload }` как размеченное объединение по `type`. */
export type Envelope<K extends EventType = EventType> = Extract<AnyEnvelope, { type: K }>;

/** События движок → UI. */
export type EngineEventType =
  | "stt.partial"
  | "stt.final"
  | "translation.ready"
  | "tts.chunk"
  | "session.state"
  | "metrics.latency";
export type EngineEvent = Envelope<EngineEventType>;

/** Команды UI → движок. */
export type CommandType = "session.start" | "session.stop";
export type UiCommand = Envelope<CommandType>;

/** Бюджет задержки из ARCHITECTURE.md раздела 2. */
export const LATENCY_BUDGET_MS = 2500;

/* ------------------------------------------------------------------ */
/* Валидаторы полей                                                    */
/* ------------------------------------------------------------------ */

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

const isString = (v: unknown): boolean => typeof v === "string";
const isBoolean = (v: unknown): boolean => typeof v === "boolean";
const isNonNegativeInt = (v: unknown): boolean => Number.isInteger(v) && (v as number) >= 0;
const isIntOrNull = (v: unknown): boolean => v === null || Number.isInteger(v);
const isStringOrNull = (v: unknown): boolean => v === null || typeof v === "string";
const isLang = (v: unknown): boolean => LANGS.includes(v as Lang);
const isStream = (v: unknown): boolean => STREAMS.includes(v as Stream);
const isStatus = (v: unknown): boolean => SESSION_STATUSES.includes(v as SessionStatus);
const isLangs = (v: unknown): boolean =>
  isRecord(v) &&
  Object.keys(v).length === 2 &&
  isLang(v.in) &&
  isLang(v.out);

/**
 * Требует ровно по одному валидатору на каждое поле payload-типа: лишний ключ
 * не пройдёт проверку типов, пропущенный — тоже. Это и есть компайл-тайм связь
 * между TS-типами и рантайм-проверкой.
 */
type Validators<T> = { [K in keyof T]-?: (value: unknown) => boolean };

const VALIDATORS: { [K in EventType]: Validators<PayloadMap[K]> } = {
  "stt.partial": { stream: isStream, lang: isLang, text: isString },
  "stt.final": {
    stream: isStream,
    lang: isLang,
    text: isString,
    t_start_ms: isNonNegativeInt,
    t_end_ms: isNonNegativeInt,
  },
  "translation.ready": {
    stream: isStream,
    src_lang: isLang,
    dst_lang: isLang,
    src_text: isString,
    text: isString,
    ref_utterance_id: isIntOrNull,
  },
  "tts.chunk": { stream: isStream, seq: isNonNegativeInt, pcm_base64: isString },
  "session.state": {
    status: isStatus,
    session_id: isIntOrNull,
    langs: isLangs,
    voice_id: isStringOrNull,
  },
  "metrics.latency": {
    stream: isStream,
    stt_ms: isNonNegativeInt,
    mt_ms: isNonNegativeInt,
    tts_ms: isNonNegativeInt,
    total_ms: isNonNegativeInt,
  },
  "session.start": {
    lang_in: isLang,
    lang_out: isLang,
    voice_id: isStringOrNull,
    record: isBoolean,
  },
  "session.stop": {},
};

/** Все известные значения `type` — должно совпадать с набором JSON-схем. */
export const EVENT_TYPES = Object.keys(VALIDATORS) as EventType[];

/** Обязательные поля payload по типам события (для контракт-теста и отладки). */
export const PAYLOAD_FIELDS: Record<EventType, string[]> = Object.fromEntries(
  EVENT_TYPES.map((type) => [type, Object.keys(VALIDATORS[type])]),
) as Record<EventType, string[]>;

export function isEventType(value: unknown): value is EventType {
  return typeof value === "string" && Object.prototype.hasOwnProperty.call(VALIDATORS, value);
}

function validatePayload(type: EventType, payload: Record<string, unknown>): boolean {
  const validators = VALIDATORS[type] as Record<string, (value: unknown) => boolean>;
  const expected = Object.keys(validators);
  const actual = Object.keys(payload);
  // additionalProperties: false + все поля обязательны
  if (actual.length !== expected.length) return false;
  return expected.every((field) => {
    if (!Object.prototype.hasOwnProperty.call(payload, field)) return false;
    return validators[field](payload[field]);
  });
}

/**
 * Type guard конверта: проверяет `type` (известный), `ts` (integer),
 * `payload` (объект нужной формы) и отсутствие лишних полей.
 */
export function isEvent(x: unknown): x is Envelope {
  if (!isRecord(x)) return false;
  if (Object.keys(x).length !== 3) return false;
  if (!isEventType(x.type)) return false;
  if (!Number.isInteger(x.ts)) return false;
  if (!isRecord(x.payload)) return false;
  return validatePayload(x.type, x.payload);
}

/** Разбор сырого сообщения WebSocket. Возвращает `null`, если это не валидный конверт. */
export function parseEnvelope(raw: string): Envelope | null {
  let data: unknown;
  try {
    data = JSON.parse(raw);
  } catch {
    return null;
  }
  return isEvent(data) ? data : null;
}

/** Сборка конверта с валидацией (бросает при неверном payload). */
export function makeEnvelope<K extends EventType>(
  type: K,
  payload: PayloadMap[K],
  ts: number = Date.now(),
): Envelope<K> {
  const envelope = { type, ts, payload } as Envelope<K>;
  if (!isEvent(envelope)) {
    throw new Error(`contracts: невалидный конверт для типа ${type}`);
  }
  return envelope;
}

/** Команда `session.start`. */
export function sessionStart(payload: SessionStart, ts?: number): Envelope<"session.start"> {
  return makeEnvelope("session.start", payload, ts);
}

/** Команда `session.stop`. */
export function sessionStop(ts?: number): Envelope<"session.stop"> {
  return makeEnvelope("session.stop", {}, ts);
}

/** Сужение конверта до конкретного типа события. */
export function isEventOfType<K extends EventType>(
  envelope: Envelope,
  type: K,
): envelope is Envelope<K> {
  return envelope.type === type;
}
