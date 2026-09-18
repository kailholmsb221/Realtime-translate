"use client";

/**
 * WebSocket-клиент движка: авто-реконнект с экспоненциальной задержкой,
 * статус соединения, отправка команд, буфер событий и готовое состояние UI.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { Envelope, SessionStart, UiCommand } from "@/lib/contracts";
import { parseEnvelope, sessionStart, sessionStop } from "@/lib/contracts";
import type { AppState } from "@/lib/state";
import { initialState, reduce } from "@/lib/state";

export const DEFAULT_WS_URL = process.env.NEXT_PUBLIC_ENGINE_WS ?? "ws://localhost:8765";

/** Сколько последних событий держим в буфере (для отладки/панели ассистента). */
export const EVENT_BUFFER_SIZE = 500;

const BACKOFF_BASE_MS = 500;
export const BACKOFF_MAX_MS = 10_000;

export type ConnectionStatus = "connecting" | "open" | "closed";

/** Экспоненциальная задержка реконнекта, потолок — 10 секунд. */
export function backoffDelay(attempt: number): number {
  const delay = BACKOFF_BASE_MS * 2 ** Math.max(0, attempt);
  return Math.min(delay, BACKOFF_MAX_MS);
}

export interface EngineSocket {
  url: string;
  status: ConnectionStatus;
  /** Состояние, собранное редьюсером из всех принятых событий. */
  state: AppState;
  /** Последние события как есть (сырьё для отладки). */
  events: Envelope[];
  /** Сколько сообщений пришло, но не прошло валидацию контракта. */
  invalidCount: number;
  /** Отправка команды движку. `false`, если сокет не открыт. */
  send: (command: UiCommand) => boolean;
  start: (payload: SessionStart) => boolean;
  stop: () => boolean;
}

export function useEngineSocket(url: string = DEFAULT_WS_URL): EngineSocket {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [state, setState] = useState<AppState>(initialState);
  const [events, setEvents] = useState<Envelope[]>([]);
  const [invalidCount, setInvalidCount] = useState(0);

  const socketRef = useRef<WebSocket | null>(null);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const attemptRef = useRef(0);
  const closedByUsRef = useRef(false);

  useEffect(() => {
    closedByUsRef.current = false;

    const clearTimer = () => {
      if (timerRef.current !== null) {
        clearTimeout(timerRef.current);
        timerRef.current = null;
      }
    };

    const scheduleReconnect = () => {
      if (closedByUsRef.current) return;
      const delay = backoffDelay(attemptRef.current);
      attemptRef.current += 1;
      clearTimer();
      timerRef.current = setTimeout(connect, delay);
    };

    function connect() {
      if (closedByUsRef.current) return;
      if (typeof WebSocket === "undefined") return;

      setStatus("connecting");
      let socket: WebSocket;
      try {
        socket = new WebSocket(url);
      } catch {
        scheduleReconnect();
        return;
      }
      socketRef.current = socket;

      socket.onopen = () => {
        attemptRef.current = 0;
        setStatus("open");
      };

      socket.onmessage = (event: MessageEvent) => {
        const raw = typeof event.data === "string" ? event.data : null;
        const envelope = raw === null ? null : parseEnvelope(raw);
        if (envelope === null) {
          setInvalidCount((n) => n + 1);
          return;
        }
        setState((prev) => reduce(prev, envelope));
        setEvents((prev) => {
          const next = [...prev, envelope];
          return next.length > EVENT_BUFFER_SIZE
            ? next.slice(next.length - EVENT_BUFFER_SIZE)
            : next;
        });
      };

      socket.onerror = () => {
        // onclose придёт следом — реконнект планируем там.
      };

      socket.onclose = () => {
        socketRef.current = null;
        setStatus("closed");
        scheduleReconnect();
      };
    }

    connect();

    return () => {
      closedByUsRef.current = true;
      clearTimer();
      const socket = socketRef.current;
      socketRef.current = null;
      if (socket) {
        socket.onopen = null;
        socket.onmessage = null;
        socket.onerror = null;
        socket.onclose = null;
        socket.close();
      }
    };
  }, [url]);

  const send = useCallback((command: UiCommand): boolean => {
    const socket = socketRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) return false;
    socket.send(JSON.stringify(command));
    return true;
  }, []);

  const start = useCallback((payload: SessionStart) => send(sessionStart(payload)), [send]);
  const stop = useCallback(() => send(sessionStop()), [send]);

  return useMemo(
    () => ({ url, status, state, events, invalidCount, send, start, stop }),
    [url, status, state, events, invalidCount, send, start, stop],
  );
}
