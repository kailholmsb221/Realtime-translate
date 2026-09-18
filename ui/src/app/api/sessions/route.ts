import { NextResponse } from "next/server";

import { loadMockData, tryEngine, withDerived } from "@/lib/server/history";

export const dynamic = "force-dynamic";

/** GET /api/sessions -> { sessions: SessionRow[] } (новые сверху). */
export async function GET() {
  const fromEngine = await tryEngine("/api/sessions");
  if (fromEngine !== null) {
    return new NextResponse(fromEngine.body, {
      status: fromEngine.status,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  const data = await loadMockData();
  const sessions = data.sessions
    .map((session) => withDerived(session, data.utterances))
    .sort((a, b) => b.started_at - a.started_at);

  return NextResponse.json({ sessions }, { headers: { "x-rt-source": "mock" } });
}
