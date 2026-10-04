"""Audio-loop safety (spec 6.7): the event loop carrying audio frames is never blocked."""

from __future__ import annotations

import asyncio
import contextlib
import time
from asyncio import events
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from examples.scheduling.agent_llm import ClinicAgent
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.scripted_llm import ScriptedClinicLLM
from examples.scheduling.tools import build_registry
from examples.simulator.caller import date_phrases, time_phrases
from tests.helpers import trained_flows
from vatic.core.loop_monitor import percentile
from vatic.core.runtime import RuntimeConfig, VaticRuntime
from vatic.trace.store import TraceStore

FRAME_S = 0.020
SLOW_CALLBACK_S = 0.005


@contextlib.contextmanager
def production_loop_watch() -> Iterator[list[float]]:
    """Measure like production: debug mode off, every callback timed.

    asyncio debug mode captures a full traceback for every Task/Future; with 100
    concurrent calls that bookkeeping alone stalls the loop and would dominate the
    measurement. Instead, time each callback directly (what debug mode's slow
    callback check does) and report any over 5 ms.
    """
    loop = asyncio.get_running_loop()
    was_debug = loop.get_debug()
    loop.set_debug(False)
    slow: list[float] = []
    slow_what: list[str] = []
    original = events.Handle._run

    def timed(self: events.Handle) -> None:
        t0 = time.perf_counter()
        original(self)
        took = time.perf_counter() - t0
        if took > SLOW_CALLBACK_S:
            slow.append(took)
            slow_what.append(f"{took * 1000:.1f}ms {self!r}"[:300])

    import gc
    import os

    gc_pauses: list[tuple[float, int]] = []
    gc_t0 = [0.0]

    def gc_cb(phase: str, info: dict) -> None:
        if phase == "start":
            gc_t0[0] = time.perf_counter()
        else:
            gc_pauses.append(((time.perf_counter() - gc_t0[0]) * 1000, info["generation"]))

    gc.callbacks.append(gc_cb)
    events.Handle._run = timed  # type: ignore[method-assign]
    try:
        yield slow
        if slow_what:
            print("\nSLOW:", *slow_what, sep="\n  ")
        if os.environ.get("VATIC_GC_DEBUG"):
            print(
                "GC pauses >2ms:",
                [(round(p, 1), g) for p, g in gc_pauses if p > 2],
                "objects",
                len(gc.get_objects()),
            )
    finally:
        gc.callbacks.remove(gc_cb)
        events.Handle._run = original  # type: ignore[method-assign]
        loop.set_debug(was_debug)


class AudioLoop:
    """Emits a frame every 20 ms; a frame later than one full period is 'dropped'."""

    def __init__(self) -> None:
        self.jitter_ms: list[float] = []
        self.dropped = 0
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        deadline = time.perf_counter()
        while True:
            deadline += FRAME_S
            await asyncio.sleep(max(0.0, deadline - time.perf_counter()))
            late = time.perf_counter() - deadline
            self.jitter_ms.append(late * 1000)
            if late > FRAME_S:
                self.dropped += int(late // FRAME_S)
                deadline = time.perf_counter()

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass


def _scripts(n: int) -> list[tuple[ClinicBackend, list[str]]]:
    out = []
    for i in range(n):
        b = ClinicBackend(seed=500 + i)
        _pid, first, last = next(
            p
            for p in b.patients
            if not [a for a in b.appointments_for(p[0]) if a["status"] == "booked"]
        )
        date = next(d for d in b.clinic_days(10) if b.check_availability(d)["times"])
        hhmm = b.check_availability(date)["times"][0]
        out.append(
            (
                b,
                [
                    f"Hi! I'd like to book an appointment. My name is {first} {last}.",
                    f"{date_phrases(date, b.today)[-1]} please.",
                    f"{time_phrases(hhmm)[0]}.",
                    "Yes, please.",
                    "No, that's all. Thanks, bye!",
                ],
            )
        )
    return out


# These tests check slow callbacks themselves, in production mode (see
# production_loop_watch), so the debug-mode catcher in conftest is not used.
@pytest.mark.allow_slow_callbacks
async def test_audio_loop_unaffected_by_100_concurrent_turns(
    tmp_path: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    pytest.importorskip("onnxruntime")
    pytest.importorskip("torch")
    pool = BackendPool()
    store = await asyncio.to_thread(TraceStore, tmp_path / "t")
    # Flows with the per-step ONNX classifier enabled (trained once per test session).
    fd = await asyncio.to_thread(trained_flows, str(tmp_path_factory.getbasetemp()))
    scripts = await asyncio.to_thread(_scripts, 100)
    # The LLM is remote in production: model it as network latency, not local CPU.
    agent = ClinicAgent(ScriptedClinicLLM(latency_s=0.15, jitter_s=0.1))
    audio = AudioLoop()
    routes: list[str] = []
    # Startup (flow loading, model warm-up, worker spawn) happens before audio flows;
    # the measured window is steady-state call handling.
    async with VaticRuntime(build_registry(pool), fd, store, config=RuntimeConfig()) as rt:
        await rt.handle_turn("warmup", "hello", agent)
        with production_loop_watch() as slow:
            lag_p99, compiled = await _drive(rt, pool, agent, audio, scripts, routes)
        await rt.flush()
        scored = await asyncio.to_thread(
            lambda: sum(
                1
                for t in store.iter_turns()
                if t.membership is not None and t.membership.classifier_score is not None
            )
        )
    assert not slow, f"slow callbacks: {[round(x * 1000, 1) for x in slow]}"
    assert scored >= 100, scored  # the classifier really ran on the hot path
    assert compiled >= 150  # most compiled turns still run compiled with the classifier on
    assert audio.dropped == 0
    assert percentile(audio.jitter_ms, 99) <= 5.0, percentile(audio.jitter_ms, 99)
    assert lag_p99 <= 5.0, lag_p99


async def _drive(
    rt: VaticRuntime,
    pool: BackendPool,
    agent: ClinicAgent,
    audio: AudioLoop,
    scripts: list[tuple[ClinicBackend, list[str]]],
    routes: list[str],
) -> tuple[float, int]:
    audio.start()

    async def call(i: int, backend: ClinicBackend, lines: list[str]) -> None:
        await asyncio.sleep(i * 0.005)  # calls arrive over ~0.5 s
        sid = f"nb-{i}"
        pool.add(sid, backend)
        rt.start_session(sid, {"today": backend.today.isoformat()})
        for text in lines:
            res = await rt.handle_turn(sid, text, agent)
            routes.append(res.route)
        await rt.end_session(sid, "success")

    tasks = []
    for i, (b, lines) in enumerate(scripts):
        tasks.append(asyncio.ensure_future(call(i, b, lines)))
        await asyncio.sleep(0)
    await asyncio.gather(*tasks)
    await audio.stop()
    assert rt.dropped_traces == 0
    return rt.loop_monitor.p(99), routes.count("compiled")


@pytest.mark.allow_slow_callbacks
async def test_full_trace_queue_drops_instead_of_blocking(tmp_path: Path) -> None:
    store = await asyncio.to_thread(TraceStore, tmp_path / "t")
    original = store.write_batch

    def slow_write(records: Any) -> None:
        time.sleep(0.05)  # a slow disk, on the writer's executor thread
        original(records)

    store.write_batch = slow_write  # type: ignore[method-assign]
    pool = BackendPool()

    async def fallback(ctx: Any) -> str:
        return "ok"

    cfg = RuntimeConfig(trace_queue_size=5)
    async with VaticRuntime(build_registry(pool), None, store, config=cfg) as rt:
        with production_loop_watch() as slow:
            t0 = time.perf_counter()
            for i in range(200):
                await rt.handle_turn(f"q{i % 10}", "hello", fallback)
            elapsed = time.perf_counter() - t0
        assert rt.dropped_traces > 0
    assert not slow, slow
    # 200 turns never waited on the 50 ms disk writes.
    assert elapsed < 0.5, elapsed
