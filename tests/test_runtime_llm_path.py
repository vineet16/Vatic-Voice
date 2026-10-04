"""Runtime with no flows: every turn goes to the LLM fallback and is traced."""

from __future__ import annotations

import asyncio
from pathlib import Path

from examples.scheduling.agent_llm import ClinicAgent
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.scripted_llm import ScriptedClinicLLM
from examples.scheduling.tools import build_registry
from vatic.adapters.pipeline import PipelineAdapter, split_for_tts
from vatic.core.runtime import VaticRuntime
from vatic.core.types import TurnContext
from vatic.trace.store import TraceStore


async def test_fallback_called_and_traced(tmp_path: Path) -> None:
    pool = BackendPool()
    registry = build_registry(pool)
    store = TraceStore(tmp_path)
    seen: list[TurnContext] = []

    async def fallback(ctx: TurnContext) -> str:
        seen.append(ctx)
        out = await ctx.call_tool("lookup_patient", {"name": "Nobody Here"})
        return f"found={out['found']}"

    async with VaticRuntime(registry, None, store) as rt:
        pool.add("s1", await asyncio.to_thread(ClinicBackend, 1))
        res = await rt.handle_turn("s1", "hello", fallback)
        assert res.route == "llm" and res.text == "found=False"
        await rt.end_session("s1", "failure")
        await rt.flush()
    assert {t["function"]["name"] for t in seen[0].tools} >= {"lookup_patient", "book_appointment"}
    assert "enter_flow" not in {t["function"]["name"] for t in seen[0].tools}
    turns = list(store.iter_turns())
    assert len(turns) == 1
    assert turns[0].tool_calls[0].tool == "lookup_patient"
    assert turns[0].timings.decision_start and turns[0].timings.decision_end
    assert [s.outcome for s in store.iter_sessions()] == ["failure"]


async def test_pipeline_adapter_chunks_and_timings(tmp_path: Path) -> None:
    pool = BackendPool()
    store = TraceStore(tmp_path)
    agent = ClinicAgent(ScriptedClinicLLM())
    async with VaticRuntime(build_registry(pool), None, store) as rt:
        backend = await asyncio.to_thread(ClinicBackend, 3)
        pool.add("s", backend)
        rt.start_session("s", {"today": backend.today.isoformat()})
        adapter = PipelineAdapter(rt, agent)
        adapter.mark_stt_end("s")
        chunks = [c async for c in adapter.on_transcript("s", "What are your hours?")]
        adapter.mark_first_audio("s")
        await rt.flush()
    assert chunks == [
        "We're open Monday to Friday, 8 AM to 5 PM.",
        "Is there anything else I can help you with?",
    ]
    (turn,) = list(store.iter_turns())
    assert turn.timings.stt_end and turn.timings.tts_first_audio
    assert split_for_tts("One. Two? Three!") == ["One.", "Two?", "Three!"]
