# engine/orchestrator

Зона оркестратора (Этап 2), см. ARCHITECTURE.md 4.5.

**Назначение:** склейка пайплайна, WebSocket-сервер localhost:8765, запись в БД.

Заглушка этапа 0 — реализацию пишет владелец зоны.
Он же заполняет этот README инструкцией запуска (CLAUDE.md, правило 6г)
и свои зависимости в `requirements-orchestrator.txt`.

Взаимодействие с другими модулями — только через события
из `engine/contracts/` (см. ARCHITECTURE.md раздел 5).
