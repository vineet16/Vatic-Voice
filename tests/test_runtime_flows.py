"""Runtime executing compiled flows: entry, compiled turns, fallback, resume, shadow, demotion."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from examples.scheduling.agent_llm import ClinicAgent
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.scripted_llm import ScriptedClinicLLM
from examples.scheduling.tools import build_registry
from examples.simulator.caller import date_phrases, time_phrases
from tests.helpers import GOLDEN_DIR
from vatic.core.executor import FlowExecutor
from vatic.core.runtime import RuntimeConfig, VaticRuntime
from vatic.core.session import Session
from vatic.core.tools import ToolRegistry
from vatic.core.transforms import TransformContext
from vatic.core.types import TurnContext, TurnResult
from vatic.ir.schema import load_flow, save_flow
from vatic.lifecycle.shadow import ShadowTracker
from vatic.trace.schema import ToolCallRecord, TurnTrace
from vatic.trace.store import TraceStore


def flows_dir(tmp: Path, statuses: dict[str, str]) -> Path:
    out = tmp / "flows"
    out.mkdir()
    for name, status in statuses.items():
        flow = load_flow(GOLDEN_DIR / "flows" / f"{name}.yaml")
        flow.status = status  # type: ignore[assignment]
        save_flow(flow, out / f"{name}.yaml")
    return out


class Call:
    """One simulated call against a runtime with the clinic agent as fallback."""

    def __init__(
        self, rt: VaticRuntime, pool: BackendPool, backend: ClinicBackend, sid: str
    ) -> None:
        self.rt, self.backend, self.sid = rt, backend, sid
        self.agent = ClinicAgent(ScriptedClinicLLM())
        pool.add(sid, backend)
        rt.start_session(sid, {"today": backend.today.isoformat()})

    async def say(self, text: str) -> TurnResult:
        return await self.rt.handle_turn(self.sid, text, self.agent)


async def make_backend(seed: int) -> ClinicBackend:
    return await asyncio.to_thread(ClinicBackend, seed)


async def target(backend: ClinicBackend, skip: int = 0) -> tuple[str, str, str, str, str]:
    return await asyncio.to_thread(booking_target, backend, skip)


def booked(backend: ClinicBackend, name: str) -> list[tuple[str, str]]:
    pid = backend.lookup_patient(name)["patient"]["id"]
    return [
        (a["date"], a["time"]) for a in backend.appointments_for(pid) if a["status"] == "booked"
    ]


def booking_target(backend: ClinicBackend, skip: int = 0) -> tuple[str, str, str, str, str]:
    _pid, first, last = next(
        p
        for p in backend.patients
        if not [a for a in backend.appointments_for(p[0]) if a["status"] == "booked"]
    )
    days = [d for d in backend.clinic_days(10) if backend.check_availability(d)["times"]]
    date = days[skip]
    hhmm = backend.check_availability(date)["times"][0]
    return (
        f"{first} {last}",
        date,
        date_phrases(date, backend.today)[-1],
        hhmm,
        time_phrases(hhmm)[0],
    )


@pytest.fixture
async def env(tmp_path: Path) -> Any:
    pool = BackendPool()
    store = await asyncio.to_thread(TraceStore, tmp_path / "traces")
    fd = await asyncio.to_thread(
        flows_dir, tmp_path, {"book_appointment": "active", "cancel_appointment": "active"}
    )
    rt = VaticRuntime(build_registry(pool), fd, store)
    await rt.start()
    yield rt, pool, store, fd
    await rt.aclose()


async def test_compiled_booking_end_to_end(env: Any) -> None:
    rt, pool, store, _ = env
    backend = await make_backend(101)
    name, date, dphrase, hhmm, tphrase = await target(backend)
    c = Call(rt, pool, backend, "b1")
    r0 = await c.say(f"Hi! I'd like to book an appointment. My name is {name}.")
    assert r0.route == "llm" and r0.flow_id == "book_appointment"
    assert r0.text.startswith("Thanks,") and "What day" in r0.text
    r1 = await c.say(f"{dphrase} please.")
    assert r1.route == "compiled" and "Which time" in r1.text
    r2 = await c.say(f"{tphrase}.")
    assert r2.route == "compiled" and r2.text.endswith("Shall I book it?")
    r3 = await c.say("Yes.")
    assert r3.route == "compiled" and r3.text.startswith("You're all set")
    assert await asyncio.to_thread(booked, backend, name) == [(date, hhmm)]
    r4 = await c.say("No, that's all. Thanks, bye!")
    assert r4.route == "llm" and "Goodbye" in r4.text
    await rt.flush()
    turns = await asyncio.to_thread(lambda: list(store.iter_turns(session_id="b1")))
    assert [t.route for t in turns] == ["llm", "compiled", "compiled", "compiled", "llm"]
    assert turns[1].membership is not None and turns[1].membership.decision == "on_path"
    assert [c.tool for c in turns[3].tool_calls] == ["book_appointment"]


async def test_correction_falls_back_then_resumes(env: Any) -> None:
    rt, pool, store, _ = env
    backend = await make_backend(102)
    name, _, dphrase, _, _ = await target(backend)
    _, date2, dphrase2, hhmm2, tphrase2 = await target(backend, skip=1)
    c = Call(rt, pool, backend, "b2")
    await c.say(f"Hi, this is {name}. I'd like to book an appointment.")
    await c.say(f"{dphrase}.")
    r = await c.say(f"Actually, can we do {dphrase2} instead?")
    assert r.route == "llm" and r.fallback_reason is not None
    assert r.fallback_reason.startswith("membership:")
    assert "Which time" in r.text
    r = await c.say(f"{tphrase2}.")
    assert r.route == "compiled", r
    r = await c.say("Yes, please.")
    assert r.route == "compiled" and r.text.startswith("You're all set")
    assert await asyncio.to_thread(booked, backend, name) == [(date2, hhmm2)]
    await rt.flush()
    turns = await asyncio.to_thread(lambda: list(store.iter_turns(session_id="b2")))
    tools = [c.tool for t in turns for c in t.tool_calls]
    assert "resume_flow" in tools


async def test_unoffered_time_falls_back_on_guard(env: Any) -> None:
    rt, pool, _, _ = env
    backend = await make_backend(103)
    name, date, dphrase, _, _ = await target(backend)
    offered = (await asyncio.to_thread(backend.check_availability, date))["times"]
    missing = next(
        t
        for t in ["09:00", "10:00", "11:00", "13:00", "14:00", "15:00", "16:00"]
        if t not in offered
    )
    c = Call(rt, pool, backend, "b3")
    await c.say(f"Hi, I need an appointment. This is {name}.")
    await c.say(f"{dphrase}.")
    r = await c.say(f"{time_phrases(missing)[0]}.")
    assert (
        r.route == "llm"
        and r.fallback_reason is not None
        and r.fallback_reason.startswith("guard:")
    )
    assert "isn't available" in r.text


async def test_enter_flow_rejects_ungrounded_slots(env: Any) -> None:
    rt, pool, _, _ = env
    backend = await make_backend(104)
    pool.add("b4", backend)
    rt.start_session("b4", {"today": backend.today.isoformat()})
    results: list[dict[str, Any]] = []

    async def llm(ctx: TurnContext) -> str:
        results.append(
            await ctx.call_tool(
                "enter_flow", {"flow_id": "book_appointment", "slots": {"name": "Jane Doe"}}
            )
        )
        return "ok"

    r = await rt.handle_turn("b4", "Hi, I'd like to book an appointment.", llm)
    assert results[0]["status"] == "rejected" and "not found" in results[0]["reason"]
    assert r.route == "llm" and r.text == "ok"


async def test_shadow_never_calls_side_effecting_tools(tmp_path: Path) -> None:
    pool = BackendPool()
    registry = build_registry(pool)
    calls: list[str] = []
    for spec in registry.specs():
        original = spec.handler
        assert original is not None

        def spy(args: Any, ctx: Any, _name: str = spec.name, _orig: Any = original) -> Any:
            calls.append(_name)
            return _orig(args, ctx)

        spec.handler = spy
    store = await asyncio.to_thread(TraceStore, tmp_path / "traces")
    fd = await asyncio.to_thread(
        flows_dir, tmp_path, {"book_appointment": "shadow", "cancel_appointment": "shadow"}
    )
    async with VaticRuntime(registry, fd, store) as rt:
        backend = await make_backend(105)
        name, _, dphrase, _, tphrase = await target(backend)
        c = Call(rt, pool, backend, "s1")
        for text in [
            f"Hi! I'd like to book an appointment. My name is {name}.",
            f"{dphrase}.",
            f"{tphrase}.",
            "Yes.",
            "No, that's all.",
        ]:
            assert (await c.say(text)).route == "llm"
        await rt.end_session("s1", "success")
        await rt.flush()
    # Exactly the LLM's own calls: shadow mode replays recorded outputs.
    assert calls == ["lookup_patient", "check_availability", "book_appointment"]
    turns = await asyncio.to_thread(lambda: list(store.iter_turns(route="shadow")))
    shadow = [t.shadow for t in turns if t.flow_id == "book_appointment"]
    assert shadow and all(s is not None and s.matched for s in shadow)
    assert shadow[-1] is not None and shadow[-1].run_ended


async def test_shadow_tracker_with_raising_handlers() -> None:
    """Even with every handler raising, shadow comparison completes (no tool is invoked)."""
    from examples.scheduling import tools as t
    from vatic.core.tools import SideEffect, ToolSpec

    def explode(args: Any, ctx: Any) -> Any:
        raise AssertionError("tool invoked in shadow mode")

    real = build_registry(BackendPool())
    registry = ToolRegistry(
        [
            ToolSpec(
                name=s.name,
                input_schema=s.input_schema,
                output_schema=s.output_schema,
                side_effect=s.side_effect,
                handler=explode,
            )
            for s in real.specs()
        ]
    )
    assert t and SideEffect
    flow = await asyncio.to_thread(load_flow, GOLDEN_DIR / "flows" / "cancel_appointment.yaml")
    tracker = ShadowTracker(FlowExecutor(registry), membership=_rules_membership)
    import datetime as dt

    session = Session("x", {}, 0.0, TransformContext(dt.date(2026, 10, 5)))
    appt = {"id": "A5001", "date": "2026-10-08", "time": "11:00", "provider": "Dr. Kim"}
    patient = {"id": "P1", "first_name": "Anna", "last_name": "Taylor", "next_appointment": appt}
    turns = [
        (
            "Hi, I need to cancel my appointment. This is Anna Taylor.",
            [_rec("lookup_patient", {"name": "Anna Taylor"}, {"found": True, "patient": patient})],
            "Thanks, Anna. I see your appointment on Thursday, October 8 at 11:00 AM with Dr. Kim. "
            "Shall I cancel it?",
        ),
        (
            "Yes.",
            [
                _rec(
                    "cancel_appointment",
                    {"appointment_id": "A5001"},
                    {"cancelled": True, "appointment_id": "A5001"},
                )
            ],
            "Your appointment on Thursday, October 8 at 11:00 AM has been cancelled. "
            "Is there anything else I can help you with?",
        ),
    ]
    for i, (user, calls, agent) in enumerate(turns):
        session.transcripts.append(user)
        tt = TurnTrace(
            trace_id=f"x:{i}",
            session_id="x",
            turn_index=i,
            user_transcript=user,
            route="llm",
            tool_calls=calls,
            agent_text=agent,
        )
        await tracker.observe(session, [flow], tt)
    records = tracker.finalize(session)
    assert len(records) == 2 and all(r.shadow and r.shadow.matched for r in records)
    assert records[-1].shadow is not None and records[-1].shadow.irreversible


def _rec(tool: str, args: dict[str, Any], out: dict[str, Any]) -> ToolCallRecord:
    return ToolCallRecord(tool=tool, args=args, output=out, started_at=0, ended_at=0)


async def _rules_membership(step: Any, flow: Any, text: str) -> Any:
    from vatic.core.step_check import check_ask

    return check_ask(step, flow, text)


async def test_irreversible_tool_failure_demotes_flow(
    env: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    rt, pool, store, fd = env
    backend = await make_backend(106)
    name, _, dphrase, _, tphrase = await target(backend)

    def taken(*a: Any, **k: Any) -> Any:
        from examples.scheduling.backend import SlotTaken

        raise SlotTaken("slot was just taken")

    monkeypatch.setattr(backend, "book_appointment", taken)
    c = Call(rt, pool, backend, "b6")
    await c.say(f"Hi! I'd like to book an appointment. My name is {name}.")
    await c.say(f"{dphrase}.")
    await c.say(f"{tphrase}.")
    r = await c.say("Yes.")
    assert r.route == "llm" and r.fallback_reason is not None and "tool_error" in r.fallback_reason
    assert rt.flows("active") and all(f.flow_id != "book_appointment" for f in rt.flows("active"))
    await asyncio.sleep(0.2)  # demotion is persisted off the event loop
    await rt.flush()
    flow = await asyncio.to_thread(load_flow, fd / "book_appointment.yaml")
    assert flow.status == "shadow"
    events = await asyncio.to_thread(lambda: list(store.iter_events("book_appointment")))
    assert events and events[-1].to_status == "shadow" and "irreversible" in events[-1].reason


async def test_fallback_rate_spike_demotes(tmp_path: Path) -> None:
    pool = BackendPool()
    fd = await asyncio.to_thread(flows_dir, tmp_path, {"book_appointment": "active"})
    store = await asyncio.to_thread(TraceStore, tmp_path / "t")
    cfg = RuntimeConfig(demotion_min_attempts=5, demotion_window=10, demotion_fallback_rate=0.4)
    async with VaticRuntime(build_registry(pool), fd, store, config=cfg) as rt:
        for i in range(6):
            backend = await make_backend(200 + i)
            name, *_ = await target(backend)
            c = Call(rt, pool, backend, f"d{i}")
            await c.say(f"Hi! I'd like to book an appointment. My name is {name}.")
            await c.say("What are your hours?")  # off-path every time
        assert not rt.flows("active")
