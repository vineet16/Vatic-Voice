"""VaticRuntime: the single turn-handling API.

Hot-path rules (spec Section 6.7): the event loop is never blocked. CPU-bound
work (classifier inference, IR loading) runs on a bounded executor owned by
the runtime with a timeout; trace writes go through a bounded queue drained by
a background task; tool handlers that are synchronous run on the executor.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from vatic.core.classifier import StepClassifier, step_prompt
from vatic.core.executor import FlowExecutor, check_ask_guards
from vatic.core.guards import check_domain, check_guard, references_output
from vatic.core.loop_monitor import LoopLagMonitor
from vatic.core.phrasing import RenderError, render
from vatic.core.router import (
    enter_flow_schema,
    ground_slot,
    premature_slots,
    reachable_from,
    resumable_steps,
    resume_flow_schema,
    resume_problems,
)
from vatic.core.session import FlowState, Session
from vatic.core.step_check import check_ask, classify_confirm
from vatic.core.tools import SideEffect, ToolContext, ToolRegistry
from vatic.core.transforms import TransformContext
from vatic.core.types import (
    CompiledExchange,
    FlowContext,
    LLMFallback,
    TurnContext,
    TurnResult,
)
from vatic.ir.schema import AskStep, ConfirmStep, FlowGraph, Step, ToolStep, load_flow
from vatic.ir.validate import errors as validation_errors
from vatic.lifecycle.shadow import ShadowTracker
from vatic.llm.client import LLMClient
from vatic.trace.schema import (
    MembershipResult,
    SessionOutcome,
    SessionTrace,
    ToolCallRecord,
    TurnTimings,
    TurnTrace,
)
from vatic.trace.store import AsyncTraceWriter, Record, TraceStore

log = logging.getLogger("vatic.runtime")
T = TypeVar("T")


class RuntimeConfig(BaseModel):
    max_workers: int = 2
    membership_budget_ms: float = 30.0
    tool_timeout_s: float = 10.0
    trace_queue_size: int = 10_000
    suspended_flow_ttl_turns: int = 3
    demotion_window: int = 50
    demotion_min_attempts: int = 20
    demotion_fallback_rate: float = 0.6
    loop_monitor: bool = True
    shadow: bool = True
    classifier_threads: int = 1
    hedge_budget_ms: float = 250.0
    # Process-wide: how long another Python thread may hold the GIL before the event
    # loop thread can take it back (CPython default 5 ms). None leaves it unchanged.
    gil_switch_interval_ms: float | None = 1.0


@dataclass
class HedgePlan:
    """An uncertain turn that may still be answered by the flow (see handle_turn)."""

    state: FlowState  # trial copy with the turn's slots applied
    start: str
    step: AskStep
    membership: MembershipResult


@dataclass
class PendingTurn:
    """An LLM turn in progress (see :meth:`VaticRuntime.begin_turn`)."""

    session: Session
    trace: TurnTrace
    ctx: TurnContext
    hedge: HedgePlan | None = None
    gate: asyncio.Event | None = None  # opens side-effecting tools once a hedge resolves
    hedge_won: bool = False

    @property
    def session_id(self) -> str:
        return self.session.session_id


class VaticRuntime:
    def __init__(
        self,
        tools: ToolRegistry,
        flows_dir: str | Path | None = None,
        store: TraceStore | None = None,
        *,
        config: RuntimeConfig | None = None,
        phrasing_llm: LLMClient | None = None,
    ) -> None:
        self.tools = tools
        self.flows_dir = Path(flows_dir) if flows_dir is not None else None
        self.store = store
        self.config = config or RuntimeConfig()
        self._pool = ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="vatic"
        )
        self._flows: dict[str, FlowGraph] = {}
        self._sessions: dict[str, Session] = {}
        self._writer: AsyncTraceWriter | None = None
        self._attempts: dict[str, deque[bool]] = {}
        self._recent: OrderedDict[str, TurnTrace] = OrderedDict()
        self._enter_schema: tuple[tuple[str, ...], dict[str, Any] | None] = ((), None)
        self._classifiers: dict[str, StepClassifier] = {}
        self._prompts: dict[tuple[str, int, str], str] = {}
        self._started = False
        self._start_lock: asyncio.Lock | None = None
        self.loop_monitor = LoopLagMonitor()
        self.executor = FlowExecutor(tools, phrasing_llm=phrasing_llm)
        self.shadow = ShadowTracker(self.executor, self._membership)
        self.offload_timeouts = 0

    # -- lifecycle ---------------------------------------------------------------

    async def start(self) -> None:
        if self._start_lock is None:
            self._start_lock = asyncio.Lock()
        async with self._start_lock:
            if self._started:
                return
            if self.config.gil_switch_interval_ms is not None:
                sys.setswitchinterval(self.config.gil_switch_interval_ms / 1000)
            await self.reload_flows()
            # Warm-up: build tool schemas now so the first live turn doesn't pay for it.
            await self.offload(self.tools.openai_schemas, timeout=10.0)
            if self.store is not None:
                await self.offload(
                    self.store.write_tool_manifest, self.tools.catalog(), timeout=10.0
                )
                self._writer = AsyncTraceWriter(
                    self.store, self._pool, maxsize=self.config.trace_queue_size
                )
                self._writer.start()
            if self.config.loop_monitor:
                self.loop_monitor.start()
            self._started = True

    async def aclose(self) -> None:
        if self._writer is not None:
            await self._writer.aclose()
        await self.loop_monitor.stop()
        self._pool.shutdown(wait=True)
        self._started = False

    async def __aenter__(self) -> VaticRuntime:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def flush(self) -> None:
        if self._writer is not None:
            await self._writer.flush()

    @property
    def dropped_traces(self) -> int:
        return self._writer.dropped if self._writer else 0

    async def offload(self, fn: Callable[..., T], *args: Any, timeout: float) -> T:
        """Run CPU-bound work on the runtime's executor with a hard timeout."""
        loop = asyncio.get_running_loop()
        try:
            return await asyncio.wait_for(loop.run_in_executor(self._pool, fn, *args), timeout)
        except TimeoutError:
            self.offload_timeouts += 1
            raise

    async def reload_flows(self) -> None:
        if self.flows_dir is None or not self.flows_dir.exists():
            self._flows = {}
            return
        loaded = await self.offload(_load_flows, self.flows_dir, self.tools, timeout=30.0)
        self._flows = {f.flow_id: f for f in loaded}
        await self._load_classifiers()

    async def _load_classifiers(self) -> None:
        """Create one ONNX session per classifier at startup (with a warm-up inference)."""
        assert self.flows_dir is not None
        paths = sorted(
            {
                s.membership.classifier
                for f in self._flows.values()
                for s in f.steps
                if isinstance(s, AskStep) and s.membership.classifier
            }
        )
        for path in paths:
            if path in self._classifiers:
                continue
            try:
                self._classifiers[path] = await self.offload(
                    _load_classifier,
                    self.flows_dir / path,
                    self.config.classifier_threads,
                    timeout=120.0,
                )
            except Exception as exc:
                log.error("cannot load classifier %s: %s", path, exc)

    def flows(self, status: str | None = None) -> list[FlowGraph]:
        return [f for f in self._flows.values() if status is None or f.status == status]

    def _emit(self, record: Record) -> None:
        if self._writer is not None:
            self._writer.submit(record)

    # -- sessions ----------------------------------------------------------------

    def start_session(self, session_id: str, metadata: dict[str, Any] | None = None) -> Session:
        md = dict(metadata or {})
        session = Session(
            session_id=session_id,
            metadata=md,
            started_at=time.time(),
            ctx=TransformContext.from_metadata(md),
        )
        self._sessions[session_id] = session
        return session

    async def end_session(self, session_id: str, outcome: SessionOutcome | None = None) -> None:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return
        for rec in self.shadow.finalize(session):
            self._emit(rec)
        self._emit(
            SessionTrace(
                session_id=session_id,
                outcome=outcome,
                metadata=session.metadata,
                started_at=session.started_at,
                ended_at=time.time(),
                n_turns=session.turn_index,
            )
        )

    # -- the turn API ---------------------------------------------------------------

    async def handle_turn(
        self,
        session_id: str,
        transcript: str,
        llm_fallback: LLMFallback,
        *,
        timings: TurnTimings | None = None,
    ) -> TurnResult:
        """Handle one user turn: compiled if possible, else ``llm_fallback``."""
        began = await self._begin(session_id, transcript, timings=timings, hedge=True)
        if isinstance(began, TurnResult):
            return began
        if began.hedge is not None:
            return await self._race(began, llm_fallback)
        try:
            text = await llm_fallback(began.ctx)
        except BaseException:
            await self.complete_turn(began, "", error="llm_fallback raised")
            raise
        return await self.complete_turn(began, text)

    async def begin_turn(
        self,
        session_id: str,
        transcript: str,
        *,
        timings: TurnTimings | None = None,
    ) -> TurnResult | PendingTurn:
        """First half of a turn, for frameworks that run their own LLM loop.

        Returns a :class:`TurnResult` if the compiled flow answered, otherwise a
        :class:`PendingTurn` whose ``ctx`` the framework's LLM must use for tool calls
        (including ``enter_flow``/``resume_flow``); finish it with :meth:`complete_turn`.
        Uncertain membership decisions are not hedged here (the runtime cannot start
        the framework's LLM): they fall back.
        """
        return await self._begin(session_id, transcript, timings=timings, hedge=False)

    async def _begin(
        self,
        session_id: str,
        transcript: str,
        *,
        timings: TurnTimings | None,
        hedge: bool,
    ) -> TurnResult | PendingTurn:
        if not self._started:
            await self.start()
        session = self._sessions.get(session_id) or self.start_session(session_id)
        if session.pending is not None:  # e.g. barge-in cancelled the previous LLM turn
            await self.complete_turn(session.pending, "", error="interrupted")
        async with session.lock:
            tm = timings.model_copy() if timings is not None else TurnTimings()
            tm.decision_start = time.time()
            turn_index = session.turn_index
            session.turn_index += 1
            session.transcripts.append(transcript)
            trace = TurnTrace(
                trace_id=f"{session_id}:{turn_index}",
                session_id=session_id,
                turn_index=turn_index,
                user_transcript=transcript,
                route="llm",
                timings=tm,
            )
            prior: list[ToolCallRecord] = []
            plan: HedgePlan | None = None
            state = session.flow
            if state is not None and state.status == "waiting":
                outcome = await self._compiled_turn(session, state, trace, hedge=hedge)
                if outcome is True:
                    return self._finish(trace)
                if isinstance(outcome, HedgePlan):
                    plan = outcome
                prior = list(trace.tool_calls)
            pending = self._pending_turn(session, trace, prior)
            if plan is not None:
                pending.hedge, pending.gate = plan, asyncio.Event()
            session.pending = pending
            return pending

    async def complete_turn(
        self, pending: PendingTurn, text: str, *, error: str | None = None
    ) -> TurnResult:
        """Second half of an LLM turn: record the reply and update flow/shadow state."""
        session, trace, ctx = pending.session, pending.trace, pending.ctx
        if session.pending is not pending:
            raise RuntimeError("turn already completed")
        session.pending = None
        trace.llm_calls = max(ctx.llm_calls, 1)
        trace.route = "llm"
        trace.agent_text = ctx.handoff_text if ctx.handed_off and ctx.handoff_text else text
        if error is not None:
            trace.outcome = "error"
            trace.fallback_reason = trace.fallback_reason or error

        # Expire a suspended flow the LLM did not resume.
        state = session.flow
        if state is not None and state.status == "suspended":
            state.suspended_turns += 1
            if state.suspended_turns >= self.config.suspended_flow_ttl_turns:
                session.flow = None

        if (
            self.config.shadow
            and error is None
            and session.flow is None
            and not ctx.handed_off
            and trace.flow_id is None
        ):
            shadow_flows = sorted(self.flows("shadow"), key=lambda f: f.flow_id)
            if shadow_flows:
                await self.shadow.observe(session, shadow_flows, trace)
        return self._finish(trace)

    def _finish(self, trace: TurnTrace) -> TurnResult:
        now = time.time()
        trace.timings.decision_end = now
        trace.timings.first_text = trace.timings.first_text or now
        if trace.outcome is None:
            trace.outcome = "ok"
        self._emit(trace)
        self._recent[trace.trace_id] = trace
        while len(self._recent) > 1024:
            self._recent.popitem(last=False)
        return TurnResult(
            route="compiled" if trace.route == "compiled" else "llm",
            text=trace.agent_text,
            trace_id=trace.trace_id,
            flow_id=trace.flow_id,
            flow_step=trace.flow_step,
            fallback_reason=trace.fallback_reason,
            membership=trace.membership,
        )

    def annotate_turn(self, trace_id: str, **timings: float) -> bool:
        """Attach late timings (e.g. ``tts_first_audio``, ``turn_end``) to a recent turn.

        The updated record replaces the earlier one in the SQLite index (the JSONL
        log keeps both lines). Returns False if the turn is no longer buffered.
        """
        trace = self._recent.get(trace_id)
        if trace is None:
            return False
        trace.timings = trace.timings.model_copy(update=timings)
        self._emit(trace)
        return True

    # -- compiled path --------------------------------------------------------------

    async def _membership(
        self, step: AskStep, flow: FlowGraph, transcript: str
    ) -> MembershipResult:
        """Membership: layers 1-3 inline (microseconds), then the step classifier on the
        executor with a timeout. Never calls an LLM."""
        t0 = time.perf_counter()
        m = check_ask(step, flow, transcript)
        m.rules = m.decision
        mem = step.membership
        if m.decision != "on_path" or not mem.classifier or mem.accept_threshold is None:
            return m
        clf = self._classifiers.get(mem.classifier)
        key = (flow.flow_id, flow.version, step.id)
        prompt = self._prompts.get(key) or self._prompts.setdefault(key, step_prompt(step))
        remaining = self.config.membership_budget_ms / 1000 - (time.perf_counter() - t0)
        score: float | None = None
        reason = "classifier:unavailable"
        if clf is not None and remaining > 0:
            try:
                score = await self.offload(clf.score, prompt, transcript, timeout=remaining)
            except TimeoutError:
                reason = "classifier:timeout"
        accept = mem.accept_threshold
        reject = mem.reject_threshold if mem.reject_threshold is not None else accept
        if score is None:
            m.decision, m.reason = "uncertain", reason
        elif score >= accept:
            m.decision, m.reason = "on_path", "classifier"
        elif score <= reject:
            m.decision, m.reason = "off_path", "classifier:reject"
        else:
            m.decision, m.reason = "uncertain", "classifier:uncertain"
        m.classifier_score = score
        m.latency_ms = (time.perf_counter() - t0) * 1000
        return m

    async def _live_call(
        self, session: Session, step: ToolStep, args: dict[str, Any]
    ) -> ToolCallRecord:
        return await self.tools.call(
            step.tool,
            args,
            ToolContext(session.session_id, session.metadata),
            executor=self._pool,
            timeout=self.config.tool_timeout_s,
        )

    def _suspend(self, session: Session, trace: TurnTrace, reason: str) -> bool:
        state = session.flow
        if state is not None:
            state.status = "suspended"
            state.reason = reason
            state.suspended_turns = 0
        trace.fallback_reason = reason
        trace.outcome = "fallback"
        return False

    async def _compiled_turn(
        self, session: Session, state: FlowState, trace: TurnTrace, *, hedge: bool = False
    ) -> bool | HedgePlan:
        flow = state.flow
        step = flow.step(state.step_id)
        trace.flow_id, trace.flow_version, trace.flow_step = flow.flow_id, flow.version, step.id
        transcript = trace.user_transcript
        if isinstance(step, AskStep):
            m = await self._membership(step, flow, transcript)
            trace.membership = m
            if m.decision == "uncertain" and hedge:
                plan = self._hedge_plan(session, state, step, m)
                if plan is not None:
                    self._suspend(session, trace, "membership:uncertain")
                    trace.outcome = None
                    return plan
            if m.decision != "on_path":
                self._record_attempt(flow, fallback=True)
                return self._suspend(session, trace, f"membership:{m.reason}")
            state.slots.update(m.slots)
            guards = check_ask_guards(step, state, session.ctx)
            trace.guard_results.extend(guards)
            failed = [g for g in guards if not g.passed]
            if failed:
                self._record_attempt(flow, fallback=True)
                return self._suspend(session, trace, f"guard:{failed[0].expr}")
            start = step.next
        elif isinstance(step, ConfirmStep):
            label = classify_confirm(transcript)
            trace.membership = MembershipResult(
                decision="on_path" if label == "yes" else "off_path", reason=f"confirm:{label}"
            )
            if label != "yes":
                self._record_attempt(flow, fallback=True)
                return self._suspend(session, trace, f"confirm:{label}")
            start = step.on_yes
        else:
            return self._suspend(session, trace, f"not_waiting:{step.kind}")

        res = await self.executor.run(
            flow,
            state,
            start,
            call_tool=lambda s, a: self._live_call(session, s, a),
            ctx=session.ctx,
        )
        trace.tool_calls.extend(res.tool_calls)
        trace.guard_results.extend(res.guard_results)
        if res.status == "fallback":
            self._record_attempt(
                flow, fallback=True, irreversible_failure=res.irreversible_guard_failure
            )
            return self._suspend(session, trace, res.reason or "fallback")
        self._record_attempt(flow, fallback=False)
        trace.route = "compiled"
        trace.agent_text = res.text
        trace.outcome = "ok"
        if res.status == "ended":
            session.flow = None
        session.history_delta.append(
            CompiledExchange(trace.turn_index, transcript, res.text, list(res.tool_calls))
        )
        return True

    # -- hedged fallback (uncertain band) -----------------------------------------------

    def _hedge_plan(
        self, session: Session, state: FlowState, step: AskStep, m: MembershipResult
    ) -> HedgePlan | None:
        """Hedge only if the flow could still safely answer: the utterance has no
        residual content, the turn's guards pass, and every tool the flow would run
        before its next wait point is read-only (it may run concurrently with the LLM).
        """
        if m.residual:
            return None
        flow = state.flow
        seen: set[str] = set()
        stack = [step.next]
        while stack:
            sid = stack.pop()
            if sid in seen or not flow.has_step(sid):
                continue
            seen.add(sid)
            s = flow.step(sid)
            if isinstance(s, (AskStep, ConfirmStep)):
                continue
            if isinstance(s, ToolStep) and (
                s.side_effect != "read_only"
                or self.executor.side_effect(s.tool) != SideEffect.READ_ONLY
            ):
                return None
            stack.extend(flow.successors(sid))
        trial = FlowState(
            flow=flow,
            step_id=state.step_id,
            slots={**state.slots, **m.slots},
            step_args={k: dict(v) for k, v in state.step_args.items()},
            step_outputs=dict(state.step_outputs),
            entered_turn=state.entered_turn,
        )
        if any(not g.passed for g in check_ask_guards(step, trial, session.ctx)):
            return None
        return HedgePlan(trial, step.next, step, m)

    async def _hedge_gate(self, pending: PendingTurn, tool: str) -> None:
        """While a hedge is undecided, the LLM may only use read-only tools."""
        gate = pending.gate
        if gate is None or gate.is_set():
            return
        if tool in self.tools and self.tools.get(tool).side_effect == SideEffect.READ_ONLY:
            return
        await gate.wait()
        if pending.hedge_won:
            raise asyncio.CancelledError("the compiled flow answered this turn")

    async def _race(self, pending: PendingTurn, llm_fallback: LLMFallback) -> TurnResult:
        """Start the LLM now; let the flow answer instead if it finishes within budget."""
        plan, session, trace, ctx = pending.hedge, pending.session, pending.trace, pending.ctx
        assert plan is not None and pending.gate is not None
        flow = plan.state.flow
        delivered = list(ctx.history_delta)
        llm_task = asyncio.ensure_future(llm_fallback(ctx))
        res = None
        try:
            res = await asyncio.wait_for(
                self.executor.run(
                    flow,
                    plan.state,
                    plan.start,
                    call_tool=lambda s, a: self._live_call(session, s, a),
                    ctx=session.ctx,
                ),
                self.config.hedge_budget_ms / 1000,
            )
        except TimeoutError:
            pass
        trace.membership = plan.membership.model_copy(update={"hedged": True})
        if res is not None:
            trace.tool_calls.extend(res.tool_calls)
            trace.guard_results.extend(res.guard_results)
        if res is not None and res.status in ("waiting", "ended") and not llm_task.done():
            pending.hedge_won = True
            llm_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await llm_task
            session.pending = None
            plan.state.status = "waiting"
            session.flow = plan.state if res.status == "waiting" else None
            session.history_delta = [
                *delivered,
                CompiledExchange(trace.turn_index, trace.user_transcript, res.text, res.tool_calls),
            ]
            self._record_attempt(flow, fallback=False)
            trace.route, trace.agent_text, trace.outcome = "compiled", res.text, "ok"
            trace.fallback_reason = None
            trace.llm_calls = ctx.llm_calls
            return self._finish(trace)
        self._record_attempt(flow, fallback=True)
        pending.gate.set()
        try:
            text = await llm_task
        except BaseException:
            await self.complete_turn(pending, "", error="llm_fallback raised")
            raise
        return await self.complete_turn(pending, text)

    # -- LLM path ----------------------------------------------------------------------

    def _render_prompt(self, state: FlowState, ctx: TransformContext) -> Callable[[Step], str]:
        def render_prompt(step: Step) -> str:
            if isinstance(step, (AskStep, ConfirmStep)) and step.say.template:
                try:
                    return render(step.say.template, state.namespace(), ctx).text
                except RenderError:
                    return ""
            return ""

        return render_prompt

    def _enter_flow_schema(self) -> dict[str, Any] | None:
        active = sorted(self.flows("active"), key=lambda f: f.flow_id)
        key = tuple(f"{f.flow_id}@{f.version}" for f in active)
        if key != self._enter_schema[0]:
            self._enter_schema = (key, enter_flow_schema(active) if active else None)
        return self._enter_schema[1]

    def _flow_context(self, session: Session, state: FlowState) -> FlowContext:
        return FlowContext(
            flow_id=state.flow.flow_id,
            version=state.flow.version,
            description=state.flow.description,
            step_id=state.step_id,
            reason=state.reason or "suspended",
            slots=dict(state.slots),
            resumable_steps=resumable_steps(
                state.flow, state, self._render_prompt(state, session.ctx)
            ),
        )

    def _pending_turn(
        self, session: Session, trace: TurnTrace, prior: list[ToolCallRecord]
    ) -> PendingTurn:
        tools = self.tools.openai_schemas()
        state = session.flow
        fc: FlowContext | None = None
        if state is not None and state.status == "suspended":
            fc = self._flow_context(session, state)
            if fc.resumable_steps:
                tools.append(resume_flow_schema(fc))
            trace.flow_id = trace.flow_id or state.flow.flow_id
        enter = self._enter_flow_schema()
        if enter is not None:
            tools.append(enter)

        ctx: TurnContext

        async def dispatch(name: str, args: dict[str, Any], call_id: str | None) -> dict[str, Any]:
            if name in ("enter_flow", "resume_flow"):
                await self._hedge_gate(pending, name)
            if name == "enter_flow":
                return await self._enter_flow(session, ctx, trace, args, call_id)
            if name == "resume_flow":
                return await self._resume_flow(session, ctx, trace, args, call_id)
            if ctx.handed_off:
                return {"error": "a flow has taken over this turn; stop and reply with nothing"}
            await self._hedge_gate(pending, name)
            rec = await self.tools.call(
                name,
                args,
                ToolContext(session.session_id, session.metadata),
                executor=self._pool,
                timeout=self.config.tool_timeout_s,
                call_id=call_id,
            )
            ctx.tool_calls.append(rec)
            trace.tool_calls.append(rec)
            if rec.error is not None:
                return {"error": rec.error}
            return rec.output or {}

        ctx = TurnContext(
            session_id=session.session_id,
            transcript=trace.user_transcript,
            turn_index=trace.turn_index,
            tools=tools,
            history_delta=list(session.history_delta),
            flow=fc,
            metadata=session.metadata,
            _dispatch=dispatch,
            tool_calls=list(prior),
        )
        session.history_delta.clear()
        pending = PendingTurn(session=session, trace=trace, ctx=ctx)
        return pending

    async def _enter_flow(
        self,
        session: Session,
        ctx: TurnContext,
        trace: TurnTrace,
        args: dict[str, Any],
        call_id: str | None,
    ) -> dict[str, Any]:
        started = time.time()

        def reply(result: dict[str, Any]) -> dict[str, Any]:
            trace.tool_calls.append(
                ToolCallRecord(
                    tool="enter_flow",
                    args=args,
                    output=result,
                    started_at=started,
                    ended_at=time.time(),
                    call_id=call_id,
                )
            )
            return result

        flow = self._flows.get(str(args.get("flow_id")))
        if flow is None or flow.status != "active":
            return reply({"status": "rejected", "reason": "unknown or inactive flow"})
        if ctx.handed_off or (session.flow is not None and session.flow.status == "waiting"):
            return reply({"status": "rejected", "reason": "a flow is already running"})
        given = args.get("slots") or {}
        slots: dict[str, str] = {}
        for name in flow.entry.slots:
            raw = ground_slot(flow, name, given.get(name), session.transcripts)
            if raw is None:
                return reply(
                    {"status": "rejected", "reason": f"slot {name!r} not found in caller's words"}
                )
            slots[name] = raw
        early = premature_slots(flow, trace.user_transcript)
        if early:
            return reply(
                {
                    "status": "rejected",
                    "reason": f"caller already gave {early}; this flow would ask again",
                }
            )
        state = FlowState(
            flow=flow, step_id=flow.first_step, slots=slots, entered_turn=trace.turn_index
        )
        ns = state.namespace()
        checks = [check_guard(g, ns, session.ctx) for g in flow.entry.guards]
        checks += [check_domain(d, ns, session.ctx) for d in flow.entry.learned_guards]
        trace.guard_results.extend(checks)
        if any(not g.passed for g in checks):
            return reply({"status": "rejected", "reason": "entry slots outside learned domain"})
        res = await self.executor.run(
            flow,
            state,
            flow.first_step,
            call_tool=lambda s, a: self._live_call(session, s, a),
            ctx=session.ctx,
        )
        executed = [
            {"tool": c.tool, "args": c.args, "output": c.output, "error": c.error}
            for c in res.tool_calls
        ]
        out = reply({"status": "pending"})
        trace.tool_calls.extend(res.tool_calls)
        trace.guard_results.extend(res.guard_results)
        if res.status == "fallback":
            self._record_attempt(
                flow, fallback=True, irreversible_failure=res.irreversible_guard_failure
            )
            out.update(status="rejected", reason=res.reason, executed=executed)
            return out
        self._record_attempt(flow, fallback=False)
        session.flow = state if res.status == "waiting" else None
        ctx.handed_off = True
        ctx.handoff_text = res.text
        trace.flow_id, trace.flow_version = flow.flow_id, flow.version
        trace.flow_step = state.step_id if res.status == "waiting" else None
        out.update(
            status="entered",
            executed=executed,
            said=res.text,
            note="The flow has replied to the caller. Reply with an empty message.",
        )
        return out

    async def _resume_flow(
        self,
        session: Session,
        ctx: TurnContext,
        trace: TurnTrace,
        args: dict[str, Any],
        call_id: str | None,
    ) -> dict[str, Any]:
        started = time.time()

        def reply(result: dict[str, Any]) -> dict[str, Any]:
            trace.tool_calls.append(
                ToolCallRecord(
                    tool="resume_flow",
                    args=args,
                    output=result,
                    started_at=started,
                    ended_at=time.time(),
                    call_id=call_id,
                )
            )
            return result

        state = session.flow
        if state is None or state.status != "suspended":
            return reply({"status": "rejected", "reason": "no paused flow"})
        flow = state.flow
        if args.get("flow_id") != flow.flow_id:
            return reply({"status": "rejected", "reason": "wrong flow_id"})
        step_id = str(args.get("step_id"))
        if not flow.has_step(step_id) or not isinstance(flow.step(step_id), (AskStep, ConfirmStep)):
            return reply({"status": "rejected", "reason": "can only resume at ask/confirm steps"})

        # Work on a copy so a rejected resume leaves the paused state untouched.
        trial = FlowState(
            flow=flow,
            step_id=state.step_id,
            slots=dict(state.slots),
            step_args={k: dict(v) for k, v in state.step_args.items()},
            step_outputs=dict(state.step_outputs),
            status="suspended",
            entered_turn=state.entered_turn,
        )
        downstream = set(reachable_from(flow, step_id))
        for s in flow.steps:
            if not isinstance(s, ToolStep) or s.id in downstream:
                continue
            latest = next(
                (c for c in reversed(ctx.tool_calls) if c.tool == s.tool and c.error is None),
                None,
            )
            if latest is None or latest.output is None:
                continue
            trial.step_args[s.id] = dict(latest.args)
            trial.step_outputs[s.id] = latest.output
            ns = {**trial.namespace(), "args": latest.args, "output": latest.output}
            post = [check_guard(g, ns, session.ctx) for g in s.guards if references_output(g)]
            post += [check_domain(d, ns, session.ctx) for d in s.learned_guards]
            if any(not g.passed for g in post):
                return reply(
                    {"status": "rejected", "reason": f"refreshed {s.tool} output fails guards"}
                )
        for name, value in (args.get("slots") or {}).items():
            raw = ground_slot(flow, name, value, session.transcripts)
            if raw is None:
                return reply({"status": "rejected", "reason": f"slot {name!r} not grounded"})
            trial.slots[name] = raw
        problems = resume_problems(flow, trial, step_id)
        if problems:
            return reply({"status": "rejected", "reason": problems[0]})
        trial.step_id = step_id
        trial.status = "waiting"
        session.flow = trial
        trace.flow_id, trace.flow_version, trace.flow_step = flow.flow_id, flow.version, step_id
        return reply({"status": "resumed", "step_id": step_id})

    # -- demotion ----------------------------------------------------------------------

    def _record_attempt(
        self, flow: FlowGraph, *, fallback: bool, irreversible_failure: bool = False
    ) -> None:
        window = self._attempts.setdefault(flow.flow_id, deque(maxlen=self.config.demotion_window))
        window.append(fallback)
        reason = None
        if irreversible_failure:
            reason = "guard-triggered error on an irreversible step"
        elif len(window) >= self.config.demotion_min_attempts:
            rate = sum(window) / len(window)
            if rate > self.config.demotion_fallback_rate:
                reason = f"fallback rate {rate:.0%} over last {len(window)} attempts"
        if reason is not None and flow.status == "active":
            self._demote(flow, reason, {"window": list(window)})

    def _demote(self, flow: FlowGraph, reason: str, evidence: dict[str, Any]) -> None:
        flow.status = "shadow"
        self._attempts.pop(flow.flow_id, None)
        log.warning("demoting %s: %s", flow.flow_id, reason)
        if self.flows_dir is None:
            return
        from vatic.lifecycle.promote import transition

        flows_dir, store = self.flows_dir, self.store

        def persist() -> None:
            transition(flows_dir, flow.flow_id, "shadow", reason, store=store, evidence=evidence)

        asyncio.get_running_loop().run_in_executor(self._pool, persist)


def _load_classifier(path: Path, threads: int) -> StepClassifier:
    clf = StepClassifier(path, threads=threads)
    clf.warmup()
    return clf


def _load_flows(flows_dir: Path, registry: ToolRegistry) -> list[FlowGraph]:
    flows = []
    for path in sorted(flows_dir.glob("*.yaml")):
        try:
            flow = load_flow(path)
        except Exception as exc:
            log.error("cannot load %s: %s", path, exc)
            continue
        problems = validation_errors(flow, registry)
        if problems:
            log.error(
                "refusing invalid flow %s: %s", path.name, json.dumps([str(p) for p in problems])
            )
            continue
        flows.append(flow)
    return flows
