"use client";

import { useEffect, useState } from "react";

import { TranslatorWindow } from "@/components/TranslatorWindow";
import type { VoiceRow } from "@/lib/api";
import { fetchVoices } from "@/lib/api";
import type { SessionStart } from "@/lib/contracts";
import { useEngineSocket } from "@/lib/ws";

export default function HomePage() {
  const engine = useEngineSocket();
  const [voices, setVoices] = useState<VoiceRow[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    fetchVoices(controller.signal)
      .then((data) => setVoices(data.voices))
      .catch(() => {
        // Голоса не критичны: без них доступен только голос по умолчанию.
      });
    return () => controller.abort();
  }, []);

  const handleStart = (payload: SessionStart) => {
    engine.start(payload);
  };

  return (
    <TranslatorWindow
      status={engine.status}
      state={engine.state}
      voices={voices}
      onStart={handleStart}
      onStop={engine.stop}
    />
  );
}
