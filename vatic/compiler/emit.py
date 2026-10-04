"""Assemble FlowGraph IR from aligned, bound steps and write it to disk.

Output is stable: step ids are positional, every list is sorted or follows
step order, YAML is dumped with fixed key order, and the content hash covers
everything except lifecycle fields (status, version).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from vatic.compiler.align import AlignedStep
from vatic.compiler.bind import BindResult
from vatic.compiler.cluster import Cluster
from vatic.compiler.domains import (
    arg_domains,
    list_containers,
    member_of,
    output_domains,
    slot_domains,
)
from vatic.compiler.examples import Example
from vatic.compiler.templates import induce
from vatic.core.extract import EXTRACTOR_DESCRIPTIONS
from vatic.core.phrasing import placeholders
from vatic.core.step_check import check_ask
from vatic.core.tools import SideEffect
from vatic.ir.schema import (
    END,
    FALLBACK,
    AskStep,
    Binding,
    ConfirmStep,
    Entry,
    FlowGraph,
    LearnedDomain,
    Membership,
    Provenance,
    Say,
    SayStep,
    SlotDef,
    Step,
    ToolStep,
    dump_flow,
    load_flow,
    save_flow,
)


@dataclass
class EmitConfig:
    residual_max_tokens: int = 3
    template_min_coverage: float = 0.9


def _turn_tools(steps: list[AlignedStep], upto_turn: int) -> list[str]:
    return [s.id for s in steps if s.kind == "tool" and s.turn <= upto_turn]


def _slots_upto(br: BindResult, turn: int) -> list[str]:
    return [name for name, src in br.slots.items() if src.turn <= turn]


def _used_output_paths(
    step_id: str, bindings: dict[str, dict[str, Binding]], templates: list[str]
) -> set[str]:
    used: set[str] = set()

    def walk(b: Binding) -> None:
        if b.source == "output" and b.step == step_id and b.path:
            used.add(b.path)
        if b.input is not None:
            walk(b.input)

    for args in bindings.values():
        for b in args.values():
            walk(b)
    prefix = f"steps.{step_id}.output."
    for t in templates:
        for ph in placeholders(t):
            ref = ph.split("|", 1)[0]
            if ref.startswith(prefix):
                used.add(ref[len(prefix) :])
    return used


def _slot_of(b: Binding) -> tuple[str | None, str | None]:
    """(slot name, transform) feeding a binding, if it comes from a slot."""
    if b.source == "slot":
        return b.slot, None
    if b.source == "transform" and b.input is not None and b.input.source == "slot":
        return b.input.slot, b.fn
    return None, None


def _human_filters(fn: str | None) -> list[str]:
    if fn == "normalize_date":
        return ["normalize_date", "human_date"]
    if fn == "normalize_time":
        return ["normalize_time", "human_time"]
    return []


def inserted_confirm_template(step: ToolStep) -> str:
    action = step.tool.replace("_", " ")
    parts = []
    for arg, b in sorted(step.args.items()):
        slot, fn = _slot_of(b)
        if slot is None:
            continue
        parts.append(
            f"{arg.replace('_', ' ')} {{" + "|".join([f"slot.{slot}", *_human_filters(fn)]) + "}"
        )
    details = (" with " + ", ".join(parts)) if parts else ""
    return f"Just to confirm, I'll {action}{details}. Shall I go ahead?"


def build_flow(
    flow_id: str,
    cluster: Cluster,
    steps: list[AlignedStep],
    br: BindResult,
    cfg: EmitConfig,
) -> FlowGraph:
    views = br.views
    texts_at = {
        k: [seg.turns[k].agent_text for seg in cluster.segments] for k in range(len(cluster.shape))
    }
    says: dict[str, Say] = {}
    for st in steps:
        if st.kind in ("ask", "confirm", "say"):
            says[st.id] = induce(
                texts_at[st.turn],
                views,
                _turn_tools(steps, st.turn),
                _slots_upto(br, st.turn),
                intent=f"{st.kind} reply after turn {st.turn}",
                min_coverage=cfg.template_min_coverage,
            ).say
    templates = [s.template for s in says.values() if s.template]

    ir_steps: list[Step] = []
    for st in steps:
        if st.kind == "tool":
            assert st.tool is not None and st.side_effect is not None
            used = _used_output_paths(st.id, br.bindings, templates)
            earlier = [s.id for s in steps[: steps.index(st)] if s.kind == "tool"]
            containers = list_containers(views, earlier)
            learned = arg_domains(st.id, views) + output_domains(st.id, views, used)
            for arg in sorted(br.bindings[st.id]):
                vals = [v.steps[st.id]["args"][arg] for v in views]
                c = member_of(vals, views, containers)
                if c is not None:
                    learned.append(
                        LearnedDomain(
                            path=f"args.{arg}", kind="member_of", container=c, support=len(views)
                        )
                    )
            ir_steps.append(
                ToolStep(
                    id=st.id,
                    tool=st.tool,
                    side_effect=st.side_effect.value,
                    args=br.bindings[st.id],
                    learned_guards=learned,
                )
            )
        elif st.kind == "ask":
            expects = [n for n, src in br.slots.items() if src.turn == st.turn + 1]
            ir_steps.append(AskStep(id=st.id, expects=expects, say=says[st.id]))
        elif st.kind == "confirm":
            ir_steps.append(ConfirmStep(id=st.id, say=says[st.id], on_yes=END))
        else:
            ir_steps.append(SayStep(id=st.id, say=says[st.id]))

    # Mandatory confirmation before irreversible tools.
    final: list[Step] = []
    for s in ir_steps:
        if isinstance(s, ToolStep) and s.side_effect == SideEffect.IRREVERSIBLE.value:
            if not final or not isinstance(final[-1], ConfirmStep):
                final.append(
                    ConfirmStep(
                        id=f"{s.id}c",
                        say=Say(template=inserted_confirm_template(s)),
                        on_yes=s.id,
                        inserted=True,
                    )
                )
        final.append(s)
    # Link steps in order.
    for i, s in enumerate(final):
        nxt = final[i + 1].id if i + 1 < len(final) else END
        if isinstance(s, ConfirmStep):
            s.on_yes = nxt
            s.on_no = FALLBACK
        elif isinstance(s, (ToolStep, AskStep, SayStep)):
            s.next = nxt

    # Ask-step guards: slot shapes, and membership of transformed slots in offered lists.
    tool_turn = {st.id: st.turn for st in steps if st.kind == "tool"}
    for s in final:
        if not isinstance(s, AskStep):
            continue
        ask_turn = next(st.turn for st in steps if st.id == s.id)
        for name in s.expects:
            s.learned_guards.extend(slot_domains(name, views))
        prior = [sid for sid, t in tool_turn.items() if t <= ask_turn]
        containers = list_containers(views, prior)
        for tool in final:
            if not isinstance(tool, ToolStep):
                continue
            for arg in sorted(tool.args):
                slot, fn = _slot_of(tool.args[arg])
                if slot not in s.expects:
                    continue
                vals = [v.steps[tool.id]["args"][arg] for v in views]
                c = member_of(vals, views, containers)
                if c is not None:
                    s.learned_guards.append(
                        LearnedDomain(
                            path=f"slot.{slot}",
                            kind="member_of",
                            container=c,
                            fn=fn,
                            support=len(views),
                        )
                    )

    entry_slots = [n for n, src in br.slots.items() if src.turn == 0]
    entry_guards = [d for n in entry_slots for d in slot_domains(n, views)]
    flow = FlowGraph(
        flow_id=flow_id,
        description=f"Handles calls that run {' -> '.join(cluster.signature)}.",
        entry=Entry(slots=entry_slots, learned_guards=entry_guards),
        slots={
            n: SlotDef(extractor=src.extractor, description=EXTRACTOR_DESCRIPTIONS[src.extractor])
            for n, src in br.slots.items()
        },
        steps=final,
        provenance=Provenance(
            support=cluster.support,
            sessions=[s.session_id for s in cluster.segments],
            tool_signature=list(cluster.signature),
            shape=[list(t) for t in cluster.shape],
            bindings=dict(sorted(br.provenance.items())),
        ),
    )
    _set_residual_limits(flow, cluster, steps, cfg)
    flow.content_hash = flow.compute_hash()
    return flow


def _set_residual_limits(
    flow: FlowGraph, cluster: Cluster, steps: list[AlignedStep], cfg: EmitConfig
) -> None:
    """residual_max_tokens = largest residual seen on-path, capped by config."""
    turn_of = {st.id: st.turn for st in steps}
    for s in flow.steps:
        if not isinstance(s, AskStep):
            continue
        probe = s.model_copy(update={"membership": Membership(residual_max_tokens=10**6)})
        k = turn_of[s.id]
        observed = []
        for seg in cluster.segments:
            m = check_ask(probe, flow, seg.turns[k + 1].user_transcript)
            if m.reason in ("rules", "residual"):
                observed.append(len(m.residual))
        limit = min(cfg.residual_max_tokens, max(observed, default=0))
        s.membership = Membership(residual_max_tokens=limit)


@dataclass
class WriteReport:
    written: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)


def write_flows(
    flows: list[FlowGraph],
    out_dir: str | Path,
    examples: dict[str, dict[str, list[Example]]] | None = None,
) -> WriteReport:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = WriteReport()
    for flow in sorted(flows, key=lambda f: f.flow_id):
        path = out / f"{flow.flow_id}.yaml"
        if path.exists():
            old = load_flow(path)
            if old.content_hash == flow.content_hash:
                report.unchanged.append(flow.flow_id)
                continue
            history = out / ".history"
            history.mkdir(exist_ok=True)
            save_flow(old, history / f"{flow.flow_id}.v{old.version}.yaml")
            flow = flow.model_copy(update={"version": old.version + 1})
        path.write_text(dump_flow(flow), encoding="utf-8")
        report.written.append(flow.flow_id)
        for step_id, exs in sorted((examples or {}).get(flow.flow_id, {}).items()):
            ex_dir = out / "examples" / flow.flow_id
            ex_dir.mkdir(parents=True, exist_ok=True)
            lines = [json.dumps(e.to_json(), sort_keys=True) for e in exs]
            (ex_dir / f"{step_id}.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
