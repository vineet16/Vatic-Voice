"""Tool registry, trace store, non-blocking trace writer."""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel

from vatic.core.tools import SideEffect, ToolContext, ToolError, ToolRegistry, ToolSpec
from vatic.trace.schema import SessionTrace, TurnTrace
from vatic.trace.store import AsyncTraceWriter, TraceStore


class EchoIn(BaseModel):
    text: str


class EchoOut(BaseModel):
    text: str


def _registry() -> ToolRegistry:
    def echo(args: EchoIn, ctx: ToolContext) -> EchoOut:
        if args.text == "boom":
            raise ToolError("exploded")
        return EchoOut(text=args.text.upper())

    async def slow(args: EchoIn, ctx: ToolContext) -> EchoOut:
        await asyncio.sleep(1)
        return EchoOut(text="late")

    return ToolRegistry(
        [
            ToolSpec(
                name="echo",
                input_schema=EchoIn,
                output_schema=EchoOut,
                side_effect=SideEffect.READ_ONLY,
                handler=echo,
            ),
            ToolSpec(
                name="slow",
                input_schema=EchoIn,
                output_schema=EchoOut,
                side_effect=SideEffect.IRREVERSIBLE,
                handler=slow,
            ),
        ]
    )


async def test_tool_call_records() -> None:
    reg = _registry()
    ctx = ToolContext("s1")
    with ThreadPoolExecutor(1) as pool:
        ok = await reg.call("echo", {"text": "hi"}, ctx, executor=pool)
        assert ok.output == {"text": "HI"} and ok.error is None
        err = await reg.call("echo", {"text": "boom"}, ctx, executor=pool)
        assert err.output is None and err.error == "exploded"
        bad = await reg.call("echo", {"nope": 1}, ctx, executor=pool)
        assert bad.error is not None and bad.error.startswith("invalid arguments")
        unknown = await reg.call("missing", {}, ctx, executor=pool)
        assert unknown.error is not None
    late = await reg.call("slow", {"text": "x"}, ctx, timeout=0.05)
    assert late.error is not None and "timeout" in late.error


def test_openai_schema() -> None:
    schema = _registry().openai_schemas()[0]
    assert schema["function"]["name"] == "echo"
    assert schema["function"]["parameters"]["required"] == ["text"]


def test_store_roundtrip(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    t = TurnTrace(trace_id="s:0", session_id="s", turn_index=0, user_transcript="hi", route="llm")
    store.write_batch([t, SessionTrace(session_id="s", outcome="success", started_at=1.0)])
    assert [x.trace_id for x in store.iter_turns()] == ["s:0"]
    corpus = store.load_corpus()
    assert len(corpus) == 1 and corpus[0][0].outcome == "success"
    assert (tmp_path / "turns.jsonl").read_text().count("\n") == 1


async def test_async_writer_drops_instead_of_blocking(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    with ThreadPoolExecutor(1) as pool:
        writer = AsyncTraceWriter(store, pool, maxsize=5)
        # Not started: the queue fills and further submits are dropped without blocking.
        t0 = time.perf_counter()
        results = [
            writer.submit(
                TurnTrace(
                    trace_id=f"s:{i}",
                    session_id="s",
                    turn_index=i,
                    user_transcript="x",
                    route="llm",
                )
            )
            for i in range(20)
        ]
        assert time.perf_counter() - t0 < 0.05
        assert results.count(True) == 5 and writer.dropped == 15
        writer.start()
        await writer.aclose()
    assert len(list(store.iter_turns())) == 5


async def test_event_loop_is_in_strict_debug_mode() -> None:
    loop = asyncio.get_running_loop()
    assert loop.get_debug() and loop.slow_callback_duration == 0.005
