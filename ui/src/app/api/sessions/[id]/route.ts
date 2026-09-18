import { NextResponse } from "next/server";

import { getSessionDetail } from "@/lib/server/history";

export const dynamic = "force-dynamic";

/** GET /api/sessions/:id -> { session: SessionRow, utterances: UtteranceRow[] }. */
export async function GET(_request: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  const data = await getSessionDetail(id);
  if (data === null) {
    return NextResponse.json({ error: "session not found" }, { status: 404 });
  }
  return NextResponse.json(data);
}
