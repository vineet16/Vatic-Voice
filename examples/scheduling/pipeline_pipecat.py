"""Clinic agent on Pipecat (pipecat-ai==1.12.0) with Vatic.

Offline text demo (scripted LLM service, no transport or keys):

    python -m examples.scheduling.pipeline_pipecat --flows flows

In a voice bot, place the two Vatic processors around the LLM service:

    vatic = VaticPipecat(runtime, session_id, {"today": ...})
    vatic.register_functions(llm)
    pipeline = Pipeline([transport.input(), stt, pair.user(), vatic.input(), llm,
                         vatic.output(), tts, transport.output(), pair.assistant()])
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from pipecat.frames.frames import EndFrame, LLMMessagesAppendFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import LLMContextAggregatorPair
from pipecat.services.llm_service import LLMService
from pipecat.workers.runner import WorkerRunner

from examples.scheduling.agent_llm import SYSTEM_PROMPT
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.pipecat_scripted import ScriptedPipecatLLM
from examples.scheduling.pipeline_handbuilt import canned_call
from examples.scheduling.tools import build_registry
from vatic.adapters.pipecat import VaticPipecat
from vatic.core.runtime import VaticRuntime
from vatic.core.types import TurnResult
from vatic.trace.store import TraceStore


class TextBot:
    """A text-only Pipecat pipeline: user aggregator -> Vatic -> LLM -> Vatic -> assistant."""

    def __init__(self, runtime: VaticRuntime, session_id: str, llm: LLMService, today: str):
        self.vatic = VaticPipecat(runtime, session_id, {"today": today})
        self.vatic.register_functions(llm)
        pair = LLMContextAggregatorPair(LLMContext())
        self.worker = PipelineWorker(
            Pipeline([pair.user(), self.vatic.input(), llm, self.vatic.output(), pair.assistant()])
        )
        self.runner = WorkerRunner(handle_sigint=False)
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self.runner.add_workers(self.worker)
        self._task = asyncio.create_task(self.runner.run())

    async def say(self, text: str, timeout: float = 15.0) -> TurnResult:
        frame = LLMMessagesAppendFrame([{"role": "user", "content": text}], run_llm=True)
        await self.worker.queue_frame(frame)
        return await asyncio.wait_for(self.vatic.results.get(), timeout)

    async def stop(self) -> None:
        await self.worker.queue_frame(EndFrame())
        if self._task is not None:
            await self._task


async def demo(flows: Path | None, store: Path) -> None:
    pool = BackendPool()
    backend = ClinicBackend(seed=1)
    pool.add("pc-demo", backend)
    today = backend.today.isoformat()
    async with VaticRuntime(build_registry(pool), flows, TraceStore(store)) as runtime:
        bot = TextBot(
            runtime, "pc-demo", ScriptedPipecatLLM(SYSTEM_PROMPT.format(today=today)), today
        )
        await bot.start()
        for text in canned_call(backend):
            result = await bot.say(text)
            print(f"caller> {text}\n agent> {result.text}   [{result.route}]")
        await bot.stop()


def main(
    flows: Path | None = typer.Option(None), store: Path = typer.Option(Path(".vatic/demo"))
) -> None:
    asyncio.run(demo(flows, store))


if __name__ == "__main__":
    typer.run(main)
