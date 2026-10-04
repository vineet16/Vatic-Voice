"""Flow entry and exit: the ``enter_flow`` / ``resume_flow`` virtual tools.

Routing into a flow is done once, by the LLM (v0.1 design). Slot values the LLM
passes are only accepted if a deterministic extractor recovers them from
something the caller actually said in this session ("grounded").
"""

from __future__ import annotations

from typing import Any

from vatic.core.extract import EXTRACTOR_DESCRIPTIONS, extract, normalize_text
from vatic.core.session import FlowState
from vatic.core.types import FlowContext, ResumableStep
from vatic.ir.schema import END, FALLBACK, AskStep, Binding, ConfirmStep, FlowGraph, ToolStep


def enter_flow_schema(flows: list[FlowGraph]) -> dict[str, Any]:
    lines = []
    for f in flows:
        slot_desc = ", ".join(f"{s} ({_slot_description(f, s)})" for s in f.entry.slots)
        lines.append(f"- {f.flow_id}: {f.description} Entry slots: {slot_desc or 'none'}.")
    return {
        "type": "function",
        "function": {
            "name": "enter_flow",
            "description": (
                "Hand the rest of the caller's task to a verified flow. Call it as soon as the "
                "request clearly matches one of the flows below and the caller has already said "
                "every entry slot. Pass each slot value using the caller's exact words. After "
                "calling it successfully, reply with an empty message: the flow answers the "
                "caller.\nFlows:\n" + "\n".join(lines)
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "enum": [f.flow_id for f in flows]},
                    "slots": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["flow_id", "slots"],
            },
        },
    }


def _slot_description(flow: FlowGraph, slot: str) -> str:
    sd = flow.slots[slot]
    return sd.description or EXTRACTOR_DESCRIPTIONS.get(sd.extractor, "")


def resume_flow_schema(fc: FlowContext) -> dict[str, Any]:
    steps = "\n".join(
        f"- {s.step_id} ({s.kind}"
        + (f"; expects: {', '.join(s.expects)}" if s.expects else "")
        + f"): {s.prompt!r}"
        for s in fc.resumable_steps
    )
    return {
        "type": "function",
        "function": {
            "name": "resume_flow",
            "description": (
                f"The verified flow '{fc.flow_id}' is paused ({fc.reason}). Once the caller's "
                "side request is handled and you are about to ask one of the questions below, "
                "call this with that step id and any slot values the caller gave (exact words), "
                "then ask the question yourself in your reply.\nSteps:\n" + steps
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "flow_id": {"type": "string", "enum": [fc.flow_id]},
                    "step_id": {
                        "type": "string",
                        "enum": [s.step_id for s in fc.resumable_steps],
                    },
                    "slots": {"type": "object", "additionalProperties": {"type": "string"}},
                },
                "required": ["flow_id", "step_id"],
            },
        },
    }


def ground_slot(flow: FlowGraph, slot: str, value: Any, transcripts: list[str]) -> str | None:
    """Return the caller's span for ``value`` if the slot's extractor finds it, else None."""
    if not isinstance(value, str) or slot not in flow.slots:
        return None
    target = normalize_text(value)
    if not target:
        return None
    extractor = flow.slots[slot].extractor
    for text in reversed(transcripts):
        for span in extract(extractor, text):
            if normalize_text(span.text) == target:
                return span.text
    return None


def premature_slots(flow: FlowGraph, transcript: str) -> list[str]:
    """Slots a later ask would request that the caller already gave in ``transcript``.

    Entering such a flow would make it ask again for something just said, so the
    flow is the wrong variant for this caller (another variant, or the LLM, fits).
    """
    later = {n for s in flow.steps if isinstance(s, AskStep) for n in s.expects}
    out = []
    for name in sorted(later - set(flow.entry.slots)):
        if name in flow.slots and extract(flow.slots[name].extractor, transcript):
            out.append(name)
    return out


def _bindings(b: Binding) -> list[Binding]:
    out = [b]
    if b.input is not None:
        out += _bindings(b.input)
    return out


def reachable_from(flow: FlowGraph, start: str) -> list[str]:
    seen: list[str] = []
    stack = [start]
    while stack:
        sid = stack.pop()
        if sid in (END, FALLBACK) or sid in seen or not flow.has_step(sid):
            continue
        seen.append(sid)
        stack.extend(reversed(flow.successors(sid)))
    return seen


def resume_problems(flow: FlowGraph, state: FlowState, step_id: str) -> list[str]:
    """Check every binding downstream of ``step_id`` can be satisfied."""
    downstream = reachable_from(flow, step_id)
    provided_slots = set(state.slots)
    for sid in downstream:
        s = flow.step(sid)
        if isinstance(s, AskStep):
            provided_slots |= set(s.expects)
    problems = []
    for sid in downstream:
        s = flow.step(sid)
        if not isinstance(s, ToolStep):
            continue
        for arg, b in s.args.items():
            for inner in _bindings(b):
                if inner.source == "slot" and inner.slot not in provided_slots:
                    problems.append(f"{sid}.{arg}: slot {inner.slot} unavailable")
                if (
                    inner.source == "output"
                    and inner.step not in state.step_outputs
                    and inner.step not in downstream
                ):
                    problems.append(f"{sid}.{arg}: output of {inner.step} unavailable")
    return problems


def resumable_steps(flow: FlowGraph, state: FlowState, render_prompt: Any) -> list[ResumableStep]:
    out = []
    for s in flow.steps:
        if isinstance(s, (AskStep, ConfirmStep)) and not resume_problems(flow, state, s.id):
            prompt = render_prompt(s)
            out.append(
                ResumableStep(
                    step_id=s.id,
                    kind=s.kind,
                    expects=list(s.expects) if isinstance(s, AskStep) else [],
                    prompt=prompt,
                )
            )
    return out
