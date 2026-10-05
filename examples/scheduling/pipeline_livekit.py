"""Clinic agent on LiveKit Agents (livekit-agents==1.8.4) with Vatic.

Offline text demo (scripted model, no LiveKit server or keys):

    python -m examples.scheduling.pipeline_livekit --flows flows

Real rooms (needs LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET and inference access):

    VATIC_FLOWS=flows python -m livekit.agents start examples/scheduling/pipeline_livekit.py --dev
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from livekit.agents import AgentServer, AgentSession, JobContext

from examples.scheduling.agent_llm import SYSTEM_PROMPT
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.livekit_scripted import ScriptedLiveKitLLM
from examples.scheduling.pipeline_handbuilt import canned_call
from examples.scheduling.tools import build_registry
from vatic.adapters.livekit import VaticAgent
from vatic.core.runtime import VaticRuntime
from vatic.trace.store import TraceStore

server = AgentServer()


@server.rtc_session(agent_name="clinic")
async def entrypoint(ctx: JobContext) -> None:
    pool = BackendPool()
    backend = ClinicBackend(seed=1)
    pool.add(ctx.room.name, backend)
    runtime = VaticRuntime(
        build_registry(pool),
        os.environ.get("VATIC_FLOWS", "flows"),
        TraceStore(os.environ.get("VATIC_STORE", ".vatic/livekit")),
    )
    await runtime.start()
    ctx.add_shutdown_callback(runtime.aclose)
    today = backend.today.isoformat()
    agent = VaticAgent(
        runtime=runtime,
        session_id=ctx.room.name,
        session_metadata={"today": today},
        instructions=SYSTEM_PROMPT.format(today=today),
    )
    session: AgentSession = AgentSession(
        stt="deepgram/nova-3", llm="openai/gpt-4.1-mini", tts="cartesia/sonic-2"
    )
    await session.start(agent, room=ctx.room)


async def demo(flows: Path | None, store: Path) -> None:
    pool = BackendPool()
    backend = ClinicBackend(seed=1)
    pool.add("lk-demo", backend)
    today = backend.today.isoformat()
    async with VaticRuntime(build_registry(pool), flows, TraceStore(store)) as runtime:
        agent = VaticAgent(
            runtime=runtime,
            session_id="lk-demo",
            session_metadata={"today": today},
            instructions=SYSTEM_PROMPT.format(today=today),
        )
        async with AgentSession(llm=ScriptedLiveKitLLM()) as session:
            await session.start(agent)
            for text in canned_call(backend):
                result = await session.run(user_input=text)
                print(f"caller> {text}")
                for event in result.events:
                    item = getattr(event, "item", None)
                    if getattr(item, "role", None) == "assistant":
                        print(f" agent> {item.text_content}")  # type: ignore[union-attr]


def main(
    flows: Path | None = typer.Option(None), store: Path = typer.Option(Path(".vatic/demo"))
) -> None:
    asyncio.run(demo(flows, store))


if __name__ == "__main__":
    typer.run(main)
