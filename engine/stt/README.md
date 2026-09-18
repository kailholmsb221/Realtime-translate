# engine/stt

Зона агента B, см. ARCHITECTURE.md 4.2.

**Назначение:** распознавание речи (Silero VAD + faster-whisper small int8_float16).

Заглушка этапа 0 — реализацию пишет владелец зоны.
Он же заполняет этот README инструкцией запуска (CLAUDE.md, правило 6г)
и свои зависимости в `requirements-stt.txt`.

Взаимодействие с другими модулями — только через события
из `engine/contracts/` (см. ARCHITECTURE.md раздел 5).
