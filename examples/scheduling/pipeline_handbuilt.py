"""Hand-built pipeline example: (fake) STT -> Vatic -> (fake) TTS.

    python -m examples.scheduling.pipeline_handbuilt --flows flows

Type caller utterances on stdin (or pass --script to replay a canned call).
Uses a real LLM if OPENAI_API_KEY is set, otherwise the offline scripted model.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import typer

from examples.scheduling.agent_llm import ClinicAgent
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.scripted_llm import ScriptedClinicLLM
from examples.scheduling.tools import build_registry
from vatic.adapters.pipeline import PipelineAdapter
from vatic.core.runtime import VaticRuntime
from vatic.llm.client import default_client
from vatic.trace.store import TraceStore

CANNED = [
    "Hi! I'd like to book an appointment. My name is Jane Doe.",
    "Next Tuesday would be great.",
    "Let's do 2 pm.",
    "Yes, please.",
    "No, that's all. Thanks, bye!",
]


async def run(flows: Path | None, store: Path, script: bool, seed: int) -> None:
    pool = BackendPool()
    registry = build_registry(pool)
    agent = ClinicAgent(default_client() or ScriptedClinicLLM())
    async with VaticRuntime(registry, flows, TraceStore(store)) as runtime:
        adapter = PipelineAdapter(runtime, agent)
        sid = "demo-session"
        backend = ClinicBackend(seed=seed)
        pool.add(sid, backend)
        runtime.start_session(sid, {"today": backend.today.isoformat()})
        lines = iter(CANNED) if script else (line.strip() for line in sys.stdin)
        print(
            f"[clinic] today is {backend.today}. Patients include: "
            + ", ".join(f"{f} {la}" for _, f, la in backend.patients[:5])
        )
        for text in lines:
            if not text:
                continue
            print(f"caller> {text}")
            adapter.mark_stt_end(sid)  # STT finalised the utterance
            async for chunk in adapter.on_transcript(sid, text):
                print(f" agent> {chunk}")  # a real pipeline streams this to TTS
            adapter.mark_first_audio(sid)
            res = adapter.last_result(sid)
            if res is not None:
                print(f"        [{res.route}{' ' + res.flow_id if res.flow_id else ''}]")
        await runtime.end_session(sid)


def main(
    flows: Path | None = typer.Option(None),
    store: Path = typer.Option(Path(".vatic/demo")),
    script: bool = typer.Option(True, help="replay a canned call instead of reading stdin"),
    seed: int = typer.Option(1),
) -> None:
    asyncio.run(run(flows, store, script, seed))


if __name__ == "__main__":
    typer.run(main)
