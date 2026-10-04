"""M1 acceptance: 500 simulated sessions produce valid traces; success from backend state."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from examples.scheduling.backend import ClinicBackend
from examples.simulator.asr_noise import AsrNoise
from examples.simulator.caller import date_phrases, make_persona, time_phrases
from examples.simulator.run import SimConfig, judge, run_simulation
from vatic.trace.schema import TurnTrace
from vatic.trace.store import TraceStore


def test_asr_noise_is_deterministic() -> None:
    noise = AsrNoise(rate=1.0)
    a = [noise.apply("Let's do two pm on Tuesday.", random.Random(i)) for i in range(20)]
    b = [noise.apply("Let's do two pm on Tuesday.", random.Random(i)) for i in range(20)]
    assert a == b
    assert any(x[1] is not None for x in a)
    assert AsrNoise(rate=0).apply("hi", random.Random(1)) == ("hi", None)


def test_phrase_generators_roundtrip() -> None:
    backend = ClinicBackend(seed=11)
    for d in backend.clinic_days(10):
        assert date_phrases(d, backend.today), d
    for t in ["09:00", "13:00", "16:00"]:
        assert time_phrases(t)


def test_judge_uses_backend_state() -> None:
    backend = ClinicBackend(seed=5)
    persona = make_persona(backend, random.Random(3))
    assert judge(backend, persona, finished=True) in ("failure", "success")
    if persona.goal == "book":
        assert judge(backend, persona, finished=True) == "failure"
        backend.book_appointment(persona.patient_id, persona.target_date, persona.target_time)  # type: ignore[arg-type]
        assert judge(backend, persona, finished=True) == "success"


# Throughput harness: the scripted caller and model run flat out with zero simulated
# latency and saturate the CPU, so loop timing here reflects the machine, not the
# runtime. Hot-path loop safety is asserted in test_nonblocking.py.
@pytest.mark.allow_slow_callbacks
async def test_500_sessions_produce_valid_traces(tmp_path: Path) -> None:
    report = await run_simulation(SimConfig(sessions=500, seed=1, store=tmp_path, concurrency=16))
    assert report.sessions == 500
    assert report.dropped_traces == 0
    store = TraceStore(tmp_path)
    sessions = list(store.iter_sessions())
    assert len(sessions) == 500
    assert all(s.outcome in ("success", "failure", "abandoned") for s in sessions)
    raw = (tmp_path / "turns.jsonl").read_text().splitlines()
    assert len(raw) == report.turns
    for line in raw:  # every raw record re-validates against the schema
        TurnTrace.model_validate_json(line)
    by_session: dict[str, list[int]] = {}
    for t in store.iter_turns():
        by_session.setdefault(t.session_id, []).append(t.turn_index)
    assert all(v == list(range(len(v))) for v in by_session.values())
    # Task success is measured (from backend state) and the reference agent is competent.
    assert 0.8 <= report.success_rate <= 1.0
    goals = {s.metadata["goal"] for s in sessions}
    assert goals == {"book", "cancel", "reschedule", "hours"}
