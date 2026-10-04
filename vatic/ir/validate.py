"""Static checks on FlowGraph IR.

Errors make a flow unusable (it is refused by the compiler and never loaded by
the runtime). Rules enforced here:

* unique step ids, valid targets, every step reachable, no cycles, END reachable
* every slot / transform / extractor referenced exists
* every ``output`` binding refers to a tool step that runs on *every* path before
  the consumer (dominates it); every ``slot`` binding's slot is provided by entry
  or by an ask step that dominates the consumer
* tool steps bind every required argument (when a registry is supplied)
* irreversible tool steps: no ``llm`` bindings, and immediately preceded by a
  ``confirm`` step whose ``on_yes`` is the tool step and that is its only predecessor
* guard expressions only use the whitelisted syntax
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Literal

from vatic.core.extract import EXTRACTORS
from vatic.core.formatters import FORMATTERS
from vatic.core.phrasing import placeholders
from vatic.core.tools import ToolCatalog, ToolRegistry
from vatic.core.transforms import TRANSFORMS
from vatic.ir.schema import (
    END,
    FALLBACK,
    AskStep,
    Binding,
    BranchStep,
    ConfirmStep,
    FlowGraph,
    LLMStep,
    Say,
    SayStep,
    ToolStep,
)


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    message: str
    step_id: str | None = None

    def __str__(self) -> str:
        where = f"[{self.step_id}] " if self.step_id else ""
        return f"{self.severity}: {where}{self.message}"


_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Name,
    ast.Load,
    ast.Attribute,
    ast.Subscript,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Call,
)


def guard_syntax_problem(expr: str) -> str | None:
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return f"syntax error: {exc.msg}"
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            return f"disallowed syntax {type(node).__name__}"
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or (
                node.func.id != "len" and node.func.id not in TRANSFORMS
            ):
                return "only len() and registered transforms may be called"
    return None


def _inner_bindings(b: Binding) -> list[Binding]:
    out = [b]
    if b.input is not None:
        out += _inner_bindings(b.input)
    return out


def predecessors(flow: FlowGraph) -> dict[str, list[str]]:
    preds: dict[str, list[str]] = {s.id: [] for s in flow.steps}
    for s in flow.steps:
        for t in flow.successors(s.id):
            if t in preds and s.id not in preds[t]:
                preds[t].append(s.id)
    return preds


def dominators(flow: FlowGraph) -> dict[str, set[str]]:
    """dom[n] = steps on every path from the first step to n (including n)."""
    ids = [s.id for s in flow.steps]
    preds = predecessors(flow)
    dom = {sid: set(ids) for sid in ids}
    dom[flow.first_step] = {flow.first_step}
    changed = True
    while changed:
        changed = False
        for sid in ids:
            if sid == flow.first_step:
                continue
            ps = [dom[p] for p in preds[sid]]
            new = (set.intersection(*ps) if ps else set()) | {sid}
            if new != dom[sid]:
                dom[sid] = new
                changed = True
    return dom


def _say_issues(say: Say, step_id: str) -> list[ValidationIssue]:
    issues = []
    if say.template is None and say.llm is None:
        issues.append(ValidationIssue("error", "say has neither template nor llm", step_id))
    for ph in placeholders(say.template or ""):
        ref, *filters = ph.split("|")
        if not (ref.startswith("slot.") or ref.startswith("steps.")):
            issues.append(ValidationIssue("error", f"bad template ref {ref!r}", step_id))
        for f in filters:
            if f not in TRANSFORMS and f not in FORMATTERS:
                issues.append(ValidationIssue("error", f"unknown filter {f!r}", step_id))
    return issues


def validate(
    flow: FlowGraph, tools: ToolRegistry | ToolCatalog | None = None
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    catalog = tools.catalog() if isinstance(tools, ToolRegistry) else tools

    def err(msg: str, step_id: str | None = None) -> None:
        issues.append(ValidationIssue("error", msg, step_id))

    if not flow.steps:
        err("flow has no steps")
        return issues
    ids = [s.id for s in flow.steps]
    if len(ids) != len(set(ids)):
        err("duplicate step ids")
        return issues
    for sid in ids:
        if sid in (END, FALLBACK):
            err(f"step id {sid!r} is reserved", sid)
    for s in flow.steps:
        for t in flow.successors(s.id):
            if t not in (END, FALLBACK) and t not in ids:
                err(f"unknown target {t!r}", s.id)
    if any(i.severity == "error" for i in issues):
        return issues

    # Reachability and cycles.
    seen: set[str] = set()
    on_stack: set[str] = set()
    reaches_end = False

    def dfs(sid: str) -> None:
        nonlocal reaches_end
        if sid == END:
            reaches_end = True
            return
        if sid == FALLBACK:
            return
        if sid in on_stack:
            err("cycle detected", sid)
            return
        if sid in seen:
            return
        seen.add(sid)
        on_stack.add(sid)
        for t in flow.successors(sid):
            dfs(t)
        on_stack.discard(sid)

    dfs(flow.first_step)
    for sid in ids:
        if sid not in seen:
            err("unreachable step", sid)
    if not reaches_end:
        err("no path reaches end")
    if any(i.severity == "error" for i in issues):
        return issues

    # Slots and extractors.
    for name, sd in flow.slots.items():
        if sd.extractor not in EXTRACTORS:
            err(f"slot {name!r} uses unknown extractor {sd.extractor!r}")
    for name in flow.entry.slots:
        if name not in flow.slots:
            err(f"entry slot {name!r} is not defined")
    for expr in flow.entry.guards:
        if (p := guard_syntax_problem(expr)) is not None:
            err(f"entry guard {expr!r}: {p}")

    dom = dominators(flow)
    preds = predecessors(flow)
    by_id = {s.id: s for s in flow.steps}
    for s in flow.steps:
        if isinstance(s, AskStep):
            for name in s.expects:
                if name not in flow.slots:
                    err(f"expects undefined slot {name!r}", s.id)
            issues += _say_issues(s.say, s.id)
        if isinstance(s, (ConfirmStep, SayStep)):
            issues += _say_issues(s.say, s.id)
        if isinstance(s, LLMStep) and not s.intent:
            err("llm step needs an intent", s.id)
        if isinstance(s, BranchStep):
            for c in s.cases:
                if (p := guard_syntax_problem(c.when)) is not None:
                    err(f"branch condition {c.when!r}: {p}", s.id)
        for expr in getattr(s, "guards", []):
            if (p := guard_syntax_problem(expr)) is not None:
                err(f"guard {expr!r}: {p}", s.id)
        if not isinstance(s, ToolStep):
            continue

        # Tool steps.
        if catalog is not None:
            info = catalog.get(s.tool)
            if info is None:
                err(f"unknown tool {s.tool!r}", s.id)
            else:
                if info.side_effect.value != s.side_effect:
                    err(f"side_effect {s.side_effect} != registry {info.side_effect.value}", s.id)
                for fname in info.required:
                    if fname not in s.args:
                        err(f"required argument {fname!r} is unbound", s.id)
                for arg in s.args:
                    if arg not in info.params:
                        err(f"unknown argument {arg!r}", s.id)
        provided = set(flow.entry.slots)
        for d in dom[s.id]:
            step = by_id[d]
            if isinstance(step, AskStep):
                provided |= set(step.expects)
        for arg, b in s.args.items():
            for inner in _inner_bindings(b):
                if inner.source == "slot":
                    if inner.slot not in flow.slots:
                        err(f"{arg}: undefined slot {inner.slot!r}", s.id)
                    elif inner.slot not in provided:
                        err(f"{arg}: slot {inner.slot!r} not provided on every path", s.id)
                elif inner.source == "output":
                    src = by_id.get(inner.step or "")
                    if not isinstance(src, ToolStep):
                        err(f"{arg}: output binding to non-tool step {inner.step!r}", s.id)
                    elif inner.step not in dom[s.id] or inner.step == s.id:
                        err(f"{arg}: step {inner.step!r} does not run before on every path", s.id)
                elif inner.source == "transform":
                    if inner.fn not in TRANSFORMS:
                        err(f"{arg}: unknown transform {inner.fn!r}", s.id)
                    if inner.input is None:
                        err(f"{arg}: transform without input", s.id)
                elif inner.source == "llm" and s.side_effect == "irreversible":
                    err(f"{arg}: llm binding in irreversible step", s.id)
        if s.side_effect == "irreversible":
            ps = preds[s.id]
            if len(ps) != 1:
                err("irreversible step must have exactly one predecessor (a confirm step)", s.id)
            else:
                pred = by_id[ps[0]]
                if not isinstance(pred, ConfirmStep) or pred.on_yes != s.id:
                    err("irreversible step is not immediately preceded by a confirm step", s.id)
    return issues


def errors(
    flow: FlowGraph, tools: ToolRegistry | ToolCatalog | None = None
) -> list[ValidationIssue]:
    return [i for i in validate(flow, tools) if i.severity == "error"]
