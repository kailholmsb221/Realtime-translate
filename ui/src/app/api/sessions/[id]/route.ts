import { NextResponse } from "next/server";

import { loadMockData, tryEngine, withDerived } from "@/lib/server/history";

export const dynamic = "force-dynamic";

/** GET /api/sessions/:id -> { session: SessionRow, utterances: UtteranceRow[] }. */
export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;

  const fromEngine = await tryEngine(`/api/sessions/${encodeURIComponent(id)}`);
  if (fromEngine !== null) {
    return new NextResponse(fromEngine.body, {
      status: fromEngine.status,
      headers: { "content-type": "application/json; charset=utf-8" },
    });
  }

  const data = await loadMockData();
  const session = data.sessions.find((s) => String(s.id) === id);
  if (session === undefined) {
    return NextResponse.json({ error: "session not found" }, { status: 404 });
  }

  const utterances = data.utterances
    .filter((u) => u.session_id === session.id)
    .sort((a, b) => a.t_start_ms - b.t_start_ms);

  return NextResponse.json(
    { session: withDerived(session, data.utterances), utterances },
    { headers: { "x-rt-source": "mock" } },
  );
}
