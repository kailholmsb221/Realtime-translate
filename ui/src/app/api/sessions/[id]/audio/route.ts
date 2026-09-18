import { NextResponse } from "next/server";

import { loadMockData, makeToneWav, tryEngine } from "@/lib/server/history";

export const dynamic = "force-dynamic";

/** GET /api/sessions/:id/audio -> audio/wav (или 404, если записи нет). */
export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;

  const fromEngine = await tryEngine(`/api/sessions/${encodeURIComponent(id)}/audio`);
  if (fromEngine !== null) {
    return new NextResponse(fromEngine.body, {
      status: fromEngine.status,
      headers: {
        "content-type": fromEngine.headers.get("content-type") ?? "audio/wav",
      },
    });
  }

  const data = await loadMockData();
  const session = data.sessions.find((s) => String(s.id) === id);
  if (session === undefined || session.audio_path === null) {
    return NextResponse.json({ error: "audio not available" }, { status: 404 });
  }

  // Мок-режим: реальных WAV в репозитории нет (*.wav в .gitignore),
  // поэтому синтезируем короткий тон — плеер должен чем-то играть.
  const wav = makeToneWav();
  return new NextResponse(new Uint8Array(wav), {
    headers: {
      "content-type": "audio/wav",
      "content-length": String(wav.length),
      "x-rt-source": "mock",
    },
  });
}
