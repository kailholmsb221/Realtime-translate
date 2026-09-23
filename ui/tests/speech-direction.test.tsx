/**
 * Выбор направления перевода в панели пайплайна.
 *
 * Главное, что проверяется: подпись «Я говорю на» уезжает в `lang_out`, а
 * «Переводить на» — в `lang_in`. Перепутанное направление означает, что движок
 * ждёт от микрофона не тот язык, и на выходе получается бессмыслица.
 */

import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { useState } from "react";

import { SpeechDirection } from "@/components/SpeechDirection";
import type { SessionStart } from "@/lib/contracts";

function Harness({ onStart }: { onStart: (value: SessionStart) => void }) {
  const [value, setValue] = useState<SessionStart>({
    lang_in: "en",
    lang_out: "ru",
    voice_id: null,
    record: false,
  });
  return (
    <SpeechDirection
      value={value}
      onChange={setValue}
      running={false}
      connected
      onStart={() => onStart(value)}
      onStop={() => {}}
    />
  );
}

describe("SpeechDirection", () => {
  it("«Я говорю на» → lang_out, «Переводить на» → lang_in", async () => {
    const user = userEvent.setup();
    const onStart = vi.fn();
    render(<Harness onStart={onStart} />);

    await user.selectOptions(screen.getByLabelText("Я говорю на"), "kk");
    await user.selectOptions(screen.getByLabelText("Переводить на"), "ru");
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStart).toHaveBeenCalledWith({
      lang_in: "ru",
      lang_out: "kk",
      voice_id: null,
      record: false,
    });
  });

  it("кнопка ⇄ меняет языки местами", async () => {
    const user = userEvent.setup();
    const onStart = vi.fn();
    render(<Harness onStart={onStart} />);

    await user.click(screen.getByRole("button", { name: /поменять языки местами/ }));
    await user.click(screen.getByRole("button", { name: "Start" }));

    expect(onStart).toHaveBeenCalledWith({
      lang_in: "ru",
      lang_out: "en",
      voice_id: null,
      record: false,
    });
  });

  it("во время сессии языки не меняются, Start заблокирован", () => {
    render(
      <SpeechDirection
        value={{ lang_in: "en", lang_out: "ru", voice_id: null, record: false }}
        onChange={() => {}}
        running
        connected
        onStart={() => {}}
        onStop={() => {}}
      />,
    );

    expect(screen.getByLabelText("Я говорю на")).toBeDisabled();
    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Stop" })).toBeEnabled();
  });

  it("без связи с движком Start недоступен", () => {
    render(
      <SpeechDirection
        value={{ lang_in: "en", lang_out: "ru", voice_id: null, record: false }}
        onChange={() => {}}
        running={false}
        connected={false}
        onStart={() => {}}
        onStop={() => {}}
      />,
    );

    expect(screen.getByRole("button", { name: "Start" })).toBeDisabled();
  });
});
