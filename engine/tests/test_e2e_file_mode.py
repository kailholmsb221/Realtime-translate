"""Сквозной тест CLI движка в файловом режиме (ARCHITECTURE.md этап 7.1).

Запускает ``python -m engine.orchestrator --backend fake --file ... --out ...``
отдельным процессом и проверяет, что на выходе появился корректный WAV, а в
stdout — валидные по контрактам события JSON-строками. Реальные модели не
поднимаются (бэкенд ``fake``), поэтому тест работает на Linux без звука.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import wave
from pathlib import Path

import pytest

from engine.contracts.events import (
    EVENT_METRICS_LATENCY,
    EVENT_STT_FINAL,
    EVENT_TRANSLATION_READY,
    Envelope,
    Lang,
    MetricsLatency,
    Stream,
    SttFinal,
    TranslationReady,
    parse_event,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "engine" / "tests" / "fixtures" / "stt_speech_pattern.wav"
TIMEOUT_S = 180


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    """Запустить движок отдельным процессом из корня репозитория."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    env["LOG_LEVEL"] = "WARNING"
    return subprocess.run(
        [sys.executable, "-m", "engine.orchestrator", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
    )


def parse_stdout(stdout: str) -> list[Envelope]:
    """Разобрать события из stdout; строки-комментарии (``#``) пропускаются."""
    events: list[Envelope] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        data = json.loads(stripped)
        events.append(parse_event(data))
    return events


@pytest.fixture(scope="module")
def file_mode_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, list[Envelope]]:
    """Один прогон CLI на всех тестах модуля: выходной WAV и события."""
    out = tmp_path_factory.mktemp("e2e") / "out.wav"
    result = run_cli(
        "--backend",
        "fake",
        "--file",
        str(FIXTURE),
        "--from",
        "en",
        "--to",
        "ru",
        "--out",
        str(out),
    )
    assert result.returncode == 0, f"stderr:\n{result.stderr}"
    return out, parse_stdout(result.stdout)


def test_cli_writes_valid_wav(file_mode_run: tuple[Path, list[Envelope]]) -> None:
    """Выходной WAV существует, читается и не пустой."""
    out, _ = file_mode_run
    assert out.is_file()
    with wave.open(str(out), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16_000
        assert handle.getnframes() > 0


def test_cli_prints_contract_events(file_mode_run: tuple[Path, list[Envelope]]) -> None:
    """В stdout — валидные конверты контрактов, включая обе фразы и переводы."""
    _, events = file_mode_run
    assert events

    finals = [event.payload for event in events if event.type == EVENT_STT_FINAL]
    ready = [event.payload for event in events if event.type == EVENT_TRANSLATION_READY]
    metrics = [event.payload for event in events if event.type == EVENT_METRICS_LATENCY]
    assert len(finals) == 2
    assert len(ready) == 2
    assert len(metrics) == 2

    for payload in finals:
        assert isinstance(payload, SttFinal)
        assert payload.stream is Stream.IN
        assert payload.lang is Lang.EN
        assert payload.t_end_ms > payload.t_start_ms

    for payload in ready:
        assert isinstance(payload, TranslationReady)
        assert payload.src_lang is Lang.EN
        assert payload.dst_lang is Lang.RU
        assert payload.text

    for payload in metrics:
        assert isinstance(payload, MetricsLatency)
        assert payload.total_ms >= 0


def test_cli_order_final_then_translation(file_mode_run: tuple[Path, list[Envelope]]) -> None:
    """Переводы идут строго после своих фраз и в том же порядке."""
    _, events = file_mode_run
    order = [
        event.type
        for event in events
        if event.type in {EVENT_STT_FINAL, EVENT_TRANSLATION_READY, EVENT_METRICS_LATENCY}
    ]
    assert order == [
        EVENT_STT_FINAL,
        EVENT_TRANSLATION_READY,
        EVENT_METRICS_LATENCY,
        EVENT_STT_FINAL,
        EVENT_TRANSLATION_READY,
        EVENT_METRICS_LATENCY,
    ]


def test_cli_requires_langs_in_file_mode(tmp_path: Path) -> None:
    """Файловый режим без ``--from``/``--to`` завершается кодом 1 с подсказкой."""
    result = run_cli("--backend", "fake", "--file", str(FIXTURE), "--out", str(tmp_path / "o.wav"))
    assert result.returncode == 1
    assert "--from" in result.stderr


def test_cli_missing_file_reports_error(tmp_path: Path) -> None:
    """Несуществующий вход — код 1 и понятное сообщение."""
    result = run_cli(
        "--backend",
        "fake",
        "--file",
        str(tmp_path / "нет.wav"),
        "--from",
        "en",
        "--to",
        "ru",
    )
    assert result.returncode == 1
    assert "не найден" in result.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="на Windows live-режим доступен")
def test_cli_live_mode_without_audio_backend() -> None:
    """Live-режим с реальными устройствами вне Windows — код 2 и инструкция."""
    result = run_cli("serve", "--backend", "real")
    assert result.returncode == 2
    assert "--backend fake" in result.stderr
