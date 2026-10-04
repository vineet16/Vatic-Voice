"""Executes compiled flow steps until the next wait point, end, or fallback.

Shared by the live runtime and shadow mode; the difference is the
``call_tool`` callable (live registry vs. replay of recorded LLM tool calls).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from vatic.core.guards import check_domain, check_guard, references_output
from vatic.core.paths import PathError, resolve
from vatic.core.phrasing import RenderError, phrase_with_llm, render
from vatic.core.session import FlowState
from vatic.core.tools import SideEffect, ToolRegistry
from vatic.core.transforms import TRANSFORMS, TransformContext
from vatic.ir.schema import (
    END,
    FALLBACK,
    AskStep,
    Binding,
    BranchStep,
    ConfirmStep,
    EndStep,
    FlowGraph,
    LearnedDomain,
    LLMPhrase,
    LLMStep,
    Say,
    SayStep,
    ToolStep,
)
from vatic.llm.client import LLMClient
from vatic.trace.schema import GuardResult, ToolCallRecord

ToolCaller = Callable[[ToolStep, dict[str, Any]], Awaitable[ToolCallRecord | None]]


class BindingError(Exception):
    pass


def resolve_binding(b: Binding, state: FlowState, ctx: TransformContext) -> Any:
    if b.source == "const":
        return b.value
    if b.source == "slot":
        if b.slot is None or b.slot not in state.slots:
            raise BindingError(f"slot {b.slot!r} not filled")
        return state.slots[b.slot]
    if b.source == "output":
        if b.step is None or b.step not in state.step_outputs:
            raise BindingError(f"no output for step {b.step!r}")
        try:
            return resolve(state.step_outputs[b.step], b.path or "")
        except PathError as exc:
            raise BindingError(f"missing output path {b.step}.{b.path}") from exc
    if b.source == "transform":
        if b.input is None or b.fn not in TRANSFORMS:
            raise BindingError(f"bad transform {b.fn!r}")
        value = TRANSFORMS[b.fn](resolve_binding(b.input, state, ctx), ctx)
        if value is None:
            raise BindingError(f"transform {b.fn} could not handle input")
        return value
    raise BindingError("llm bindings require the LLM; falling back")


def _is_pre(d: LearnedDomain) -> bool:
    return not d.path.startswith("output")


@dataclass
class ExecResult:
    status: Literal["waiting", "ended", "fallback"]
    step_id: str | None = None  # waiting step, or the step that failed
    reason: str | None = None
    text_parts: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    guard_results: list[GuardResult] = field(default_factory=list)
    rendered_values: list[str] = field(default_factory=list)
    executed_steps: list[str] = field(default_factory=list)
    irreversible_called: bool = False
    irreversible_guard_failure: bool = False

    @property
    def text(self) -> str:
        return " ".join(p for p in self.text_parts if p)


class FlowExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        phrasing_llm: LLMClient | None = None,
    ) -> None:
        self.registry = registry
        self.phrasing_llm = phrasing_llm

    def side_effect(self, tool: str) -> SideEffect:
        if tool in self.registry:
            return self.registry.get(tool).side_effect
        return SideEffect.IRREVERSIBLE  # unknown tools are treated as the worst case

    async def _say(
        self, say: Say, state: FlowState, ctx: TransformContext, res: ExecResult
    ) -> bool:
        ns = state.namespace()
        if say.template is not None:
            try:
                r = render(say.template, ns, ctx)
            except RenderError as exc:
                res.reason = f"render:{exc}"
                return False
            res.text_parts.append(r.text)
            res.rendered_values.extend(r.values)
            return True
        if say.llm is not None and self.phrasing_llm is not None:
            try:
                res.text_parts.append(await phrase_with_llm(self.phrasing_llm, say.llm, ns))
                return True
            except Exception as exc:
                res.reason = f"phrasing:{type(exc).__name__}"
                return False
        res.reason = "phrasing_unavailable"
        return False

    async def run(
        self,
        flow: FlowGraph,
        state: FlowState,
        start: str,
        *,
        call_tool: ToolCaller,
        ctx: TransformContext,
        auto_confirm_inserted: bool = False,
    ) -> ExecResult:
        res = ExecResult(status="fallback")
        current = start
        for _ in range(len(flow.steps) + 2):
            if current == END:
                res.status = "ended"
                return res
            if current == FALLBACK:
                res.reason = res.reason or "flow_fallback_target"
                return res
            try:
                step = flow.step(current)
            except KeyError:
                res.reason = f"unknown_step:{current}"
                return res
            res.step_id = step.id
            if isinstance(step, ToolStep):
                nxt = await self._tool(step, state, ctx, res, call_tool)
                if nxt is None:
                    return res
                current = nxt
            elif isinstance(step, (AskStep, ConfirmStep)):
                if isinstance(step, ConfirmStep) and step.inserted and auto_confirm_inserted:
                    res.executed_steps.append(step.id)
                    current = step.on_yes
                    continue
                if not await self._say(step.say, state, ctx, res):
                    return res
                state.step_id = step.id
                state.status = "waiting"
                res.status = "waiting"
                return res
            elif isinstance(step, SayStep):
                if not await self._say(step.say, state, ctx, res):
                    return res
                res.executed_steps.append(step.id)
                current = step.next
            elif isinstance(step, LLMStep):
                say = Say(llm=_phrase_from_step(step))
                if not await self._say(say, state, ctx, res):
                    return res
                res.executed_steps.append(step.id)
                current = step.next
            elif isinstance(step, BranchStep):
                ns = state.namespace()
                target = step.default
                for case in step.cases:
                    g = check_guard(case.when, ns, ctx, step_id=step.id)
                    res.guard_results.append(g)
                    if g.passed:
                        target = case.next
                        break
                if target == FALLBACK:
                    res.reason = "branch_no_case"
                    return res
                res.executed_steps.append(step.id)
                current = target
            elif isinstance(step, EndStep):
                res.status = "ended"
                return res
        res.reason = "step_limit"
        return res

    async def _tool(
        self,
        step: ToolStep,
        state: FlowState,
        ctx: TransformContext,
        res: ExecResult,
        call_tool: ToolCaller,
    ) -> str | None:
        irreversible = (
            step.side_effect == "irreversible"
            or self.side_effect(step.tool) == SideEffect.IRREVERSIBLE
        )
        try:
            args = {k: resolve_binding(b, state, ctx) for k, b in step.args.items()}
        except BindingError as exc:
            res.reason = f"binding:{exc}"
            return None
        ns: dict[str, Any] = {**state.namespace(), "args": args}
        invariants = self.registry.get(step.tool).invariants if step.tool in self.registry else []
        declared = [*invariants, *step.guards]
        pre = [g for g in declared if not references_output(g)]
        post = [g for g in declared if references_output(g)]
        for expr in pre:
            g = check_guard(expr, ns, ctx, step_id=step.id)
            res.guard_results.append(g)
            if not g.passed:
                res.reason = f"guard:{expr}"
                return None
        for d in step.learned_guards:
            if _is_pre(d):
                g = check_domain(d, ns, ctx, step_id=step.id)
                res.guard_results.append(g)
                if not g.passed:
                    res.reason = f"learned_guard:{g.expr}"
                    return None
        record = await call_tool(step, args)
        if record is None:
            res.reason = "no_recorded_output"
            return None
        record.step_id = step.id
        res.tool_calls.append(record)
        res.irreversible_called = res.irreversible_called or irreversible
        state.step_args[step.id] = args
        if record.error is not None or record.output is None:
            res.reason = f"tool_error:{record.error}"
            res.irreversible_guard_failure = irreversible
            return None
        state.step_outputs[step.id] = record.output
        ns = {**state.namespace(), "args": args, "output": record.output}
        for expr in post:
            g = check_guard(expr, ns, ctx, step_id=step.id)
            res.guard_results.append(g)
            if not g.passed:
                res.reason = f"guard:{expr}"
                res.irreversible_guard_failure = irreversible
                return None
        for d in step.learned_guards:
            if not _is_pre(d):
                g = check_domain(d, ns, ctx, step_id=step.id)
                res.guard_results.append(g)
                if not g.passed:
                    res.reason = f"learned_guard:{g.expr}"
                    res.irreversible_guard_failure = irreversible
                    return None
        res.executed_steps.append(step.id)
        return step.next


def _phrase_from_step(step: LLMStep) -> LLMPhrase:
    return LLMPhrase(intent=step.intent, inputs=step.inputs, max_tokens=step.max_tokens)


def check_ask_guards(step: AskStep, state: FlowState, ctx: TransformContext) -> list[GuardResult]:
    """Guards evaluated right after an ask step's slots are extracted."""
    ns = state.namespace()
    results = [check_guard(expr, ns, ctx, step_id=step.id) for expr in step.guards]
    results += [check_domain(d, ns, ctx, step_id=step.id) for d in step.learned_guards]
    return results
