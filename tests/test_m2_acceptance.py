"""M2 acceptance: compile -> shadow -> promote -> run, end to end on fresh seeds."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from examples.simulator.run import SimConfig, run_simulation
from vatic.compiler.compile import CompileConfig, compile_corpus
from vatic.compiler.emit import write_flows
from vatic.core.loop_monitor import percentile
from vatic.lifecycle.promote import evaluate, load_flows
from vatic.trace.store import TraceStore

LLM_LATENCY_S = 0.05  # modelled network latency of the agent's LLM


def _compile(store: Path, flows: Path) -> list[str]:
    s = TraceStore(store)
    res = compile_corpus(s.load_corpus(outcome=None), s.tool_manifest(), CompileConfig())
    write_flows(res.accepted, flows, res.examples)
    return [f.flow_id for f in res.accepted]


def _promote(store: Path, flows: Path) -> list[tuple[str, str]]:
    return [(e.flow_id, e.to_status) for e in evaluate(flows, TraceStore(store))]


# Throughput harness (caller + model simulators run flat out); hot-path loop
# safety is asserted in test_nonblocking.py.
@pytest.mark.slow
@pytest.mark.allow_slow_callbacks
async def test_compile_shadow_promote_run(tmp_path: Path) -> None:
    store, flows = tmp_path / "traces", tmp_path / "flows"
    base = await run_simulation(
        SimConfig(
            sessions=500,
            seed=31,
            store=store,
            prefix="base",
            concurrency=24,
            llm_latency_s=LLM_LATENCY_S,
        )
    )

    compiled = await asyncio.to_thread(_compile, store, flows)
    assert {"book_appointment", "cancel_appointment"} <= set(compiled)
    to_shadow = await asyncio.to_thread(_promote, store, flows)
    assert {s for _, s in to_shadow} == {"shadow"}

    await run_simulation(
        SimConfig(
            sessions=500,
            seed=32,
            store=store,
            flows=flows,
            prefix="shadow",
            concurrency=24,
            llm_latency_s=LLM_LATENCY_S,
        )
    )
    promoted = dict(await asyncio.to_thread(_promote, store, flows))
    assert promoted.get("book_appointment") == "active"
    assert promoted.get("cancel_appointment") == "active"

    vatic = await run_simulation(
        SimConfig(
            sessions=500,
            seed=33,
            store=store,
            flows=flows,
            prefix="vatic",
            concurrency=24,
            llm_latency_s=LLM_LATENCY_S,
        )
    )
    # Task success within 2 points of baseline.
    assert abs(vatic.success_rate - base.success_rate) <= 0.02, (
        base.success_rate,
        vatic.success_rate,
    )
    # Decision latency on compiled turns drops substantially.
    compiled_ms = vatic.decision_ms["compiled"]
    llm_ms = vatic.decision_ms["llm"]
    assert len(compiled_ms) > 0.25 * vatic.turns
    assert percentile(compiled_ms, 50) < 0.2 * percentile(llm_ms, 50)
    assert vatic.llm_calls < 0.75 * base.llm_calls
    active = {f.flow_id for f in await asyncio.to_thread(load_flows, flows) if f.status == "active"}
    assert {"book_appointment", "cancel_appointment"} <= active
