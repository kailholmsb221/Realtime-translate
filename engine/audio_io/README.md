# engine/audio_io

Зона агента A, см. ARCHITECTURE.md 4.1.

**Назначение:** захват/вывод звука (WASAPI loopback, микрофон, наушники, VB-Audio Virtual Cable).

Заглушка этапа 0 — реализацию пишет владелец зоны.
Он же заполняет этот README инструкцией запуска (CLAUDE.md, правило 6г)
и свои зависимости в `requirements-audio_io.txt`.

Взаимодействие с другими модулями — только через события
из `engine/contracts/` (см. ARCHITECTURE.md раздел 5).
