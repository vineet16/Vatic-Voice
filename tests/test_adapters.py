"""M3: the same compiled flows run unchanged across hand-built, LiveKit and Pipecat pipelines.

Each test replays recorded call transcripts (tests/fixtures/calls.json) through a
real framework session in text mode (no audio): LiveKit's ``AgentSession`` and a
Pipecat ``PipelineWorker``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("livekit.agents", reason="pip install vatic[livekit]")
pytest.importorskip("pipecat", reason="pip install vatic[pipecat]")

from livekit.agents import AgentSession

from examples.scheduling.agent_llm import SYSTEM_PROMPT, ClinicAgent
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.livekit_scripted import ScriptedLiveKitLLM
from examples.scheduling.pipecat_scripted import ScriptedPipecatLLM
from examples.scheduling.pipeline_pipecat import TextBot
from examples.scheduling.scripted_llm import ScriptedClinicLLM
from examples.scheduling.tools import build_registry
from tests.test_runtime_flows import flows_dir
from vatic.adapters.livekit import VaticAgent
from vatic.adapters.pipeline import PipelineAdapter
from vatic.core.runtime import VaticRuntime
from vatic.trace.store import TraceStore

CALLS: list[dict[str, Any]] = json.loads(
    (Path(__file__).parent / "fixtures" / "calls.json").read_text()
)
EXPECTED_ROUTES = {
    "book": ["llm", "compiled", "compiled", "compiled", "llm"],
    "book_with_correction": ["llm", "compiled", "llm", "compiled", "compiled", "llm"],
    "cancel": ["llm", "compiled", "llm"],
    # first choice has no openings: the compiled branch handles it
    "book_first_choice_full": ["llm", "compiled", "compiled", "compiled", "compiled", "llm"],
}
Transcript = list[tuple[str, str]]  # (route, text) per turn


async def _setup(
    tmp: Path, call: dict[str, Any]
) -> tuple[VaticRuntime, BackendPool, ClinicBackend]:
    pool = BackendPool()
    await asyncio.to_thread(tmp.mkdir, parents=True)
    fd = await asyncio.to_thread(
        flows_dir, tmp, {"book_appointment": "active", "cancel_appointment": "active"}
    )
    store = await asyncio.to_thread(TraceStore, tmp / "traces")
    backend = await asyncio.to_thread(ClinicBackend, call["seed"])
    return VaticRuntime(build_registry(pool), fd, store), pool, backend


async def run_handbuilt(tmp: Path, call: dict[str, Any]) -> Transcript:
    rt, pool, backend = await _setup(tmp, call)
    out: Transcript = []
    async with rt:
        pool.add("s", backend)
        rt.start_session("s", {"today": backend.today.isoformat()})
        adapter = PipelineAdapter(rt, ClinicAgent(ScriptedClinicLLM()))
        for text in call["turns"]:
            chunks = [c async for c in adapter.on_transcript("s", text)]
            last = adapter.last_result("s")
            assert last is not None
            out.append((last.route, " ".join(chunks)))
    return out


async def run_livekit(tmp: Path, call: dict[str, Any]) -> Transcript:
    rt, pool, backend = await _setup(tmp, call)
    out: Transcript = []
    today = backend.today.isoformat()
    async with rt:
        pool.add("s", backend)
        agent = VaticAgent(
            runtime=rt,
            session_id="s",
            session_metadata={"today": today},
            instructions=SYSTEM_PROMPT.format(today=today),
        )
        async with AgentSession(llm=ScriptedLiveKitLLM()) as session:
            await session.start(agent)
            for text in call["turns"]:
                result = await session.run(user_input=text)
                said = [
                    e.item.text_content or ""
                    for e in result.events
                    if getattr(e, "type", "") == "message" and e.item.role == "assistant"  # type: ignore[union-attr]
                ]
                out.append(("", " ".join(said)))
        await rt.flush()
        turns = await asyncio.to_thread(lambda: list(rt.store.iter_turns(session_id="s")))  # type: ignore[union-attr]
    return [(t.route, text) for t, (_, text) in zip(turns, out, strict=True)]


async def run_pipecat(tmp: Path, call: dict[str, Any]) -> Transcript:
    rt, pool, backend = await _setup(tmp, call)
    out: Transcript = []
    today = backend.today.isoformat()
    async with rt:
        pool.add("s", backend)
        bot = TextBot(rt, "s", ScriptedPipecatLLM(SYSTEM_PROMPT.format(today=today)), today)
        await bot.start()
        for text in call["turns"]:
            res = await bot.say(text)
            out.append((res.route, res.text))
        await bot.stop()
    return out


# The frameworks' own startup does work on the loop in debug mode; this is about
# behavioural equivalence (loop safety is covered by test_nonblocking.py).
@pytest.mark.allow_slow_callbacks
@pytest.mark.parametrize("call", CALLS, ids=[c["name"] for c in CALLS])
async def test_same_flows_across_pipelines(tmp_path: Path, call: dict[str, Any]) -> None:
    hand = await run_handbuilt(tmp_path / "hand", call)
    assert [r for r, _ in hand] == EXPECTED_ROUTES[call["name"]]
    livekit = await run_livekit(tmp_path / "livekit", call)
    pipecat = await run_pipecat(tmp_path / "pipecat", call)
    assert livekit == hand
    assert pipecat == hand


def test_adapters_are_thin() -> None:
    root = Path(__file__).parent.parent / "vatic" / "adapters"
    for name in ("pipeline.py", "livekit.py", "pipecat.py"):
        lines = (root / name).read_text().count("\n")
        assert lines < 300, (name, lines)
