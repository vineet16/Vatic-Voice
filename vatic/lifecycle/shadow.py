"""Shadow execution: compute what a compiled flow *would* have done, compare.

The LLM handles every live turn. For each flow in ``shadow`` status, the
tracker replays the flow against the turn using the LLM's *recorded* tool
outputs. It never calls a tool, so reversible/irreversible tools are never run
in shadow.

A run's comparisons only count if the session's LLM tool sequence (from the
entry turn) matches one of the flow's tool paths; otherwise the run is
discarded as "not this task" (e.g. a cancel call that also starts with
``lookup_patient``).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from vatic.core.executor import ExecResult, FlowExecutor, check_ask_guards
from vatic.core.extract import normalize_text, unique_span
from vatic.core.router import premature_slots
from vatic.core.session import FlowState, Session
from vatic.core.step_check import classify_confirm
from vatic.core.tools import SideEffect
from vatic.ir.schema import END, AskStep, ConfirmStep, FlowGraph, ToolStep
from vatic.trace.schema import (
    VIRTUAL_TOOLS,
    MembershipResult,
    ShadowComparison,
    ToolCallRecord,
    TurnTrace,
)

MembershipFn = Callable[[AskStep, FlowGraph, str], Awaitable[MembershipResult]]

_TOOLS_KEY = "_llm_tools"


@dataclass
class ShadowRun:
    flow: FlowGraph
    state: FlowState | None = None
    entry_turn: int | None = None
    records: list[TurnTrace] = field(default_factory=list)
    stopped: bool = False


def normalize_value(v: Any) -> Any:
    if isinstance(v, str):
        return normalize_text(v)
    if isinstance(v, dict):
        return {k: normalize_value(x) for k, x in sorted(v.items())}
    if isinstance(v, list):
        return [normalize_value(x) for x in v]
    return v


def tool_paths(flow: FlowGraph) -> list[list[str]]:
    """Every tool-name sequence along a path from the first step to END."""
    out: list[list[str]] = []

    def walk(sid: str, acc: list[str], seen: frozenset[str]) -> None:
        if sid == END:
            out.append(acc)
            return
        if not flow.has_step(sid) or sid in seen:
            return
        step = flow.step(sid)
        nxt = [*acc, step.tool] if isinstance(step, ToolStep) else acc
        succ = flow.successors(sid)
        if isinstance(step, ConfirmStep):
            succ = [step.on_yes]
        for s in succ:
            walk(s, nxt, seen | {sid})

    walk(flow.first_step, [], frozenset())
    return out


class ShadowTracker:
    def __init__(self, executor: FlowExecutor, membership: MembershipFn) -> None:
        self.executor = executor
        self.membership = membership

    def _replay(
        self, calls: list[ToolCallRecord]
    ) -> Callable[..., Awaitable[ToolCallRecord | None]]:
        pending = list(calls)

        async def call(step: ToolStep, args: dict[str, Any]) -> ToolCallRecord | None:
            for i, rec in enumerate(pending):
                if rec.tool == step.tool:
                    pending.pop(i)
                    return rec.model_copy(update={"args": args})
            return None

        return call

    def _compare(
        self,
        run: ShadowRun,
        turn: TurnTrace,
        llm_calls: list[ToolCallRecord],
        res: ExecResult,
        step_id: str | None,
        membership: MembershipResult | None,
    ) -> None:
        reasons: list[str] = []
        expected = [{"tool": c.tool, "args": c.args} for c in res.tool_calls]
        compared = True
        if res.status == "fallback" and res.reason != "no_recorded_output":
            compared = False
            reasons.append(f"would_fallback:{res.reason}")
        else:
            if res.reason == "no_recorded_output":
                reasons.append(f"llm_did_not_call:{res.step_id}")
            got = [c.tool for c in llm_calls]
            want = [c.tool for c in res.tool_calls]
            if res.status != "fallback" and got != want:
                reasons.append(f"tools:{want}!={got}")
            for mine, theirs in zip(res.tool_calls, llm_calls, strict=False):
                if mine.tool == theirs.tool and normalize_value(mine.args) != normalize_value(
                    theirs.args
                ):
                    reasons.append(f"args:{mine.tool}")
            text = normalize_text(turn.agent_text)
            for value in res.rendered_values:
                if normalize_text(value) not in text:
                    reasons.append(f"text_value:{value}")
        irreversible = any(
            self.executor.side_effect(c.tool) == SideEffect.IRREVERSIBLE
            for c in [*res.tool_calls, *llm_calls]
        )
        cmp = ShadowComparison(
            flow_id=run.flow.flow_id,
            flow_version=run.flow.version,
            step_id=step_id,
            decision=membership.decision if membership else None,
            compared=compared,
            matched=compared and not reasons,
            irreversible=irreversible,
            run_ended=compared and not reasons and res.status == "ended",
            reasons=reasons,
            expected_tools=expected,
        )
        run.records.append(
            TurnTrace(
                trace_id=f"{turn.trace_id}:shadow:{run.flow.flow_id}",
                session_id=turn.session_id,
                turn_index=turn.turn_index,
                user_transcript=turn.user_transcript,
                route="shadow",
                flow_id=run.flow.flow_id,
                flow_version=run.flow.version,
                flow_step=step_id,
                tool_calls=res.tool_calls,
                agent_text=res.text,
                guard_results=res.guard_results,
                membership=membership,
                shadow=cmp,
            )
        )
        if not cmp.matched or res.status != "waiting":
            run.stopped = True

    async def observe(self, session: Session, flows: list[FlowGraph], turn: TurnTrace) -> None:
        llm_calls = [c for c in turn.tool_calls if c.tool not in VIRTUAL_TOOLS]
        tools: list[tuple[int, str]] = session.shadow.setdefault(_TOOLS_KEY, [])
        tools.extend((turn.turn_index, c.tool) for c in llm_calls)
        entering: list[tuple[ShadowRun, dict[str, str]]] = []
        for flow in flows:
            key = f"{flow.flow_id}@{flow.version}"
            run: ShadowRun = session.shadow.setdefault(key, ShadowRun(flow=flow))
            if run.stopped:
                continue
            if run.state is None:
                slots = self._entry_slots(run.flow, session, llm_calls)
                if slots is not None:
                    entering.append((run, slots))
            else:
                await self._step(run, session, turn, llm_calls)
        # Like the LLM choosing between variants of one task, only the most specific
        # flows (maximal entry-slot sets among flows with the same tool signature) enter.
        for run, slots in entering:
            sig = run.flow.provenance.tool_signature
            if any(
                other.flow.provenance.tool_signature == sig and set(slots) < set(other_slots)
                for other, other_slots in entering
            ):
                continue
            await self._enter(run, session, turn, llm_calls, slots)

    def _entry_slots(
        self, flow: FlowGraph, session: Session, llm_calls: list[ToolCallRecord]
    ) -> dict[str, str] | None:
        first = flow.step(flow.first_step)
        if not llm_calls or not isinstance(first, ToolStep) or llm_calls[0].tool != first.tool:
            return None
        if premature_slots(flow, session.transcripts[-1]):
            return None  # the live runtime would refuse enter_flow for this caller
        slots: dict[str, str] = {}
        recent_first = list(reversed(session.transcripts))
        for name in flow.entry.slots:
            span = unique_span(flow.slots[name].extractor, recent_first)
            if span is None:
                return None
            slots[name] = span
        return slots

    async def _enter(
        self,
        run: ShadowRun,
        session: Session,
        turn: TurnTrace,
        llm_calls: list[ToolCallRecord],
        slots: dict[str, str],
    ) -> None:
        flow = run.flow
        state = FlowState(flow=flow, step_id=flow.first_step, slots=slots)
        run.state = state
        run.entry_turn = turn.turn_index
        res = await self.executor.run(
            flow,
            state,
            flow.first_step,
            call_tool=self._replay(llm_calls),
            ctx=session.ctx,
            auto_confirm_inserted=True,
        )
        self._compare(run, turn, llm_calls, res, None, None)

    async def _step(
        self, run: ShadowRun, session: Session, turn: TurnTrace, llm_calls: list[ToolCallRecord]
    ) -> None:
        state = run.state
        assert state is not None
        flow = run.flow
        step = flow.step(state.step_id)
        membership: MembershipResult | None
        if isinstance(step, AskStep):
            membership = await self.membership(step, flow, turn.user_transcript)
            # Compare whenever the rules accept, so calibration gets a ground-truth label
            # for every utterance the classifier has to judge; the final decision is
            # recorded alongside.
            if (membership.rules or membership.decision) == "on_path":
                state.slots.update(membership.slots)
                failed = [g for g in check_ask_guards(step, state, session.ctx) if not g.passed]
                if failed:
                    membership = membership.model_copy(
                        update={
                            "decision": "off_path",
                            "rules": "off_path",
                            "reason": f"guard:{failed[0].expr}",
                        }
                    )
            start = step.next
        elif isinstance(step, ConfirmStep):
            label = classify_confirm(turn.user_transcript)
            membership = MembershipResult(
                decision="on_path" if label == "yes" else "off_path", reason=f"confirm:{label}"
            )
            start = step.on_yes
        else:
            run.stopped = True
            return
        rules_ok = (membership.rules or membership.decision) == "on_path"
        if not rules_ok or membership.reason.startswith("guard:"):
            res = ExecResult(status="fallback", reason=f"membership:{membership.reason}")
            self._compare(run, turn, llm_calls, res, step.id, membership)
            return
        res = await self.executor.run(
            flow,
            state,
            start,
            call_tool=self._replay(llm_calls),
            ctx=session.ctx,
            auto_confirm_inserted=True,
        )
        self._compare(run, turn, llm_calls, res, step.id, membership)
        if membership.decision != "on_path":
            run.stopped = True  # the live flow would have handed this turn to the LLM

    def finalize(self, session: Session) -> list[TurnTrace]:
        tools: list[tuple[int, str]] = session.shadow.get(_TOOLS_KEY, [])
        out: list[TurnTrace] = []
        for key, run in session.shadow.items():
            if key == _TOOLS_KEY or not isinstance(run, ShadowRun) or run.entry_turn is None:
                continue
            seq = [name for ti, name in tools if ti >= run.entry_turn]
            if any(seq[: len(p)] == p and len(seq) >= len(p) for p in tool_paths(run.flow)):
                out.extend(run.records)
        return out
