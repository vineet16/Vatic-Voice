"""Shared test helpers: synthetic traces and a simulated corpus."""

from __future__ import annotations

import asyncio
import functools
from pathlib import Path
from typing import Any

from examples.simulator.run import SimConfig, run_simulation
from vatic.compiler.compile import Corpus
from vatic.core.tools import SideEffect, ToolInfo
from vatic.trace.schema import SessionTrace, ToolCallRecord, TurnTrace
from vatic.trace.store import TraceStore

GOLDEN_DIR = Path(__file__).parent / "golden"


def call(tool: str, args: dict[str, Any], output: dict[str, Any] | None) -> ToolCallRecord:
    return ToolCallRecord(tool=tool, args=args, output=output, started_at=0.0, ended_at=0.0)


def session(
    sid: str,
    turns: list[tuple[str, list[ToolCallRecord], str]],
    *,
    today: str = "2026-10-05",
    outcome: str = "success",
) -> tuple[SessionTrace, list[TurnTrace]]:
    st = SessionTrace(session_id=sid, outcome=outcome, metadata={"today": today}, started_at=0.0)  # type: ignore[arg-type]
    tts = [
        TurnTrace(
            trace_id=f"{sid}:{i}",
            session_id=sid,
            turn_index=i,
            user_transcript=user,
            route="llm",
            tool_calls=calls,
            agent_text=agent,
        )
        for i, (user, calls, agent) in enumerate(turns)
    ]
    return st, tts


def catalog(**effects: SideEffect) -> dict[str, ToolInfo]:
    params = {
        "lookup": ("name",),
        "check": ("date",),
        "book": ("pid", "date", "time"),
        "note": ("text",),
        "cancel": ("aid",),
    }
    return {
        name: ToolInfo(
            name=name, side_effect=eff, params=params.get(name, ()), required=params.get(name, ())
        )
        for name, eff in effects.items()
    }


@functools.cache
def simulated_store(seed: int, sessions: int, root: str) -> Path:
    """Run (once per process) a baseline simulation and return its store path."""
    path = Path(root) / f"sim-{seed}-{sessions}"
    if not (path / "index.sqlite").exists():
        asyncio.run(
            run_simulation(SimConfig(sessions=sessions, seed=seed, store=path, concurrency=16))
        )
    return path


def load_corpus(path: Path) -> tuple[Corpus, dict[str, ToolInfo]]:
    store = TraceStore(path)
    return store.load_corpus(outcome=None), store.tool_manifest()


@functools.cache
def trained_flows(root: str) -> Path:
    """Compile the cached simulated corpus and train book_appointment's classifier (1 epoch).

    Thresholds are set by hand (accept 0.5 / reject 0.1): calibration is tested separately.
    """
    from vatic.compiler.compile import CompileConfig, compile_corpus
    from vatic.compiler.emit import write_flows
    from vatic.ir.schema import load_flow, save_flow
    from vatic.lifecycle.train import train_flow

    out = Path(root) / "trained-flows"
    if (out / "classifiers").exists():
        return out
    corpus, cat = load_corpus(simulated_store(11, 800, root))
    res = compile_corpus(corpus, cat, CompileConfig())
    write_flows(res.accepted, out, res.examples)
    flow = load_flow(out / "book_appointment.yaml")
    report = train_flow(flow, out, epochs=1)
    assert report.steps, report.skipped
    flow = load_flow(out / "book_appointment.yaml")
    for step in flow.steps:
        mem = getattr(step, "membership", None)
        if mem is not None and mem.classifier:
            mem.accept_threshold, mem.reject_threshold = 0.5, 0.1
    flow.status = "active"
    save_flow(flow, out / "book_appointment.yaml")
    return out
