#!/usr/bin/env node
/**
 * Mock-движок: WebSocket-сервер на порту 8765, говорящий теми же событиями,
 * что и настоящий orchestrator (engine/contracts/*.json).
 *
 * Запуск: npm run mock
 *
 *   - при подключении шлёт `session.state` со статусом `idle`;
 *   - по команде `session.start` отвечает `session.state` (`running`) и
 *     проигрывает сценарий из `mocks/scenario.json`;
 *   - по команде `session.stop` останавливает сценарий и возвращает `idle`.
 *
 * Настоящих моделей и аудио здесь нет — только события, чтобы UI можно было
 * разрабатывать и демонстрировать без движка.
 */

import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { WebSocketServer } from "ws";

const here = path.dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.MOCK_WS_PORT ?? 8765);
const LOOP = process.env.MOCK_LOOP === "1";

const scenario = JSON.parse(await readFile(path.join(here, "scenario.json"), "utf8"));
const steps = scenario.steps;

let nextSessionId = 1;

/** @param {import('ws').WebSocket} socket @param {string} type @param {object} payload */
function send(socket, type, payload) {
  if (socket.readyState !== socket.OPEN) return;
  socket.send(JSON.stringify({ type, ts: Date.now(), payload }));
}

function sleep(ms, signal) {
  return new Promise((resolve) => {
    const timer = setTimeout(resolve, ms);
    signal.addEventListener("abort", () => {
      clearTimeout(timer);
      resolve();
    }, { once: true });
  });
}

/** Проигрывает сценарий, пока не отменят (stop/disconnect). */
async function play(socket, signal) {
  do {
    for (const step of steps) {
      await sleep(step.delay_ms ?? 0, signal);
      if (signal.aborted || socket.readyState !== socket.OPEN) return;
      send(socket, step.event.type, step.event.payload);
    }
  } while (LOOP && !signal.aborted);
}

const server = new WebSocketServer({ port: PORT });

server.on("connection", (socket) => {
  let session = null;
  let controller = null;

  const stop = () => {
    if (controller !== null) {
      controller.abort();
      controller = null;
    }
  };

  const idle = (langs = { in: "en", out: "ru" }, voiceId = null) => ({
    status: "idle",
    session_id: null,
    langs,
    voice_id: voiceId,
  });

  console.log("[mock] клиент подключился");
  send(socket, "session.state", idle());

  socket.on("message", (raw) => {
    let message;
    try {
      message = JSON.parse(raw.toString());
    } catch {
      console.warn("[mock] не JSON, игнорирую");
      return;
    }

    if (message?.type === "session.start") {
      const { lang_in: langIn, lang_out: langOut, voice_id: voiceId } = message.payload ?? {};
      session = {
        id: nextSessionId,
        langs: { in: langIn ?? "en", out: langOut ?? "ru" },
        voiceId: voiceId ?? null,
      };
      nextSessionId += 1;

      console.log(
        `[mock] session.start #${session.id} ${session.langs.in} -> ${session.langs.out}` +
          ` (record=${message.payload?.record === true})`,
      );
      send(socket, "session.state", {
        status: "running",
        session_id: session.id,
        langs: session.langs,
        voice_id: session.voiceId,
      });

      stop();
      controller = new AbortController();
      play(socket, controller.signal).catch((error) => {
        console.error("[mock] сценарий упал:", error);
      });
      return;
    }

    if (message?.type === "session.stop") {
      console.log("[mock] session.stop");
      stop();
      const langs = session?.langs ?? { in: "en", out: "ru" };
      const voiceId = session?.voiceId ?? null;
      session = null;
      send(socket, "session.state", idle(langs, voiceId));
      return;
    }

    console.warn(`[mock] неизвестная команда: ${message?.type}`);
  });

  socket.on("close", () => {
    stop();
    console.log("[mock] клиент отключился");
  });
});

server.on("listening", () => {
  console.log(`[mock] WebSocket-движок на ws://localhost:${PORT}`);
  console.log(`[mock] шагов в сценарии: ${steps.length}${LOOP ? " (MOCK_LOOP=1, по кругу)" : ""}`);
});

const shutdown = () => {
  console.log("\n[mock] выключаюсь");
  server.close(() => process.exit(0));
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
