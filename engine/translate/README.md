# engine/translate

Зона агента C, см. ARCHITECTURE.md 4.3.

**Назначение:** перевод (NLLB-200-distilled-600M на CPU, абстракция Provider).

Заглушка этапа 0 — реализацию пишет владелец зоны.
Он же заполняет этот README инструкцией запуска (CLAUDE.md, правило 6г)
и свои зависимости в `requirements-translate.txt`.

Взаимодействие с другими модулями — только через события
из `engine/contracts/` (см. ARCHITECTURE.md раздел 5).
