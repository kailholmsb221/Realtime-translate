import { NextResponse } from "next/server";

import { getSessions } from "@/lib/server/history";

export const dynamic = "force-dynamic";

/** GET /api/sessions -> { sessions: SessionRow[] } (новые сверху). */
export async function GET() {
  return NextResponse.json({ sessions: await getSessions() });
}
