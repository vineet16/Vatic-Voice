"""Branch synthesis (milestone 4).

Two parts:

* ``split_by_behaviour``: traces with the same tool shape can still behave
  differently (e.g. "no openings, another day?" vs. offering times). If a reply
  step's traces fall into distinct placeholder classes, the cluster is split into
  sub-clusters (classes below ``min_support`` are dropped, never guessed).
* ``merge``: two flows that share a step prefix and then diverge are merged
  behind a ``branch`` step when a depth-limited, deterministic decision tree over
  earlier tool outputs and slots separates their traces *perfectly*. Otherwise
  they stay separate flows. A probabilistic split is never used.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from vatic.compiler.align import AlignedStep
from vatic.compiler.bind import BindResult, TraceView
from vatic.compiler.cluster import Cluster
from vatic.compiler.templates import induce
from vatic.core.paths import has_index, leaves
from vatic.core.phrasing import placeholders
from vatic.ir.schema import (
    FALLBACK,
    AskStep,
    Binding,
    BranchCase,
    BranchStep,
    ConfirmStep,
    FlowGraph,
    LearnedDomain,
    LLMStep,
    Membership,
    SayStep,
    Step,
    ToolStep,
)

MAX_DEPTH = 2


def split_by_behaviour(
    cluster: Cluster,
    steps: list[AlignedStep],
    br: BindResult,
    min_support: int,
    min_coverage: float,
) -> list[Cluster] | None:
    """Sub-clusters by reply behaviour at the first step no template explains, or None."""
    for st in steps:
        if st.kind not in ("ask", "confirm", "say"):
            continue
        texts = [seg.turns[st.turn].agent_text for seg in cluster.segments]
        tool_ids = [s.id for s in steps if s.kind == "tool" and s.turn <= st.turn]
        slots = [n for n, src in br.slots.items() if src.turn <= st.turn]
        induced = induce(texts, br.views, tool_ids, slots, intent="", min_coverage=min_coverage)
        if induced.say.template is not None:
            continue
        classes: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for i, t in enumerate(induced.per_trace):
            classes[tuple(sorted(placeholders(t)))].append(i)
        keep = sorted(
            (idx for idx in classes.values() if len(idx) >= min_support),
            key=lambda idx: (-len(idx), idx[0]),
        )
        if not keep or (len(keep) == 1 and len(keep[0]) == cluster.support):
            return None
        return [
            Cluster(cluster.signature, cluster.shape, tuple(cluster.segments[i] for i in idx))
            for idx in keep
        ]
    return None


# -- merging ------------------------------------------------------------------------


def _comparable(step: Step) -> str:
    data = step.model_dump(
        mode="json",
        by_alias=True,
        exclude={"next", "on_yes", "on_no", "learned_guards", "membership"},
    )
    if isinstance(data.get("say"), dict):
        data["say"].pop("coverage", None)
    return json.dumps(data, sort_keys=True)


def common_prefix(a: FlowGraph, b: FlowGraph) -> int:
    k = 0
    while k < min(len(a.steps), len(b.steps)) and _comparable(a.steps[k]) == _comparable(
        b.steps[k]
    ):
        k += 1
    return k


def _features(view: TraceView, tool_ids: list[str], slot_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sid in tool_ids:
        for path, value in leaves(view.steps[sid]["output"] or {}):
            if not has_index(path) and (
                value is None or isinstance(value, (bool, int, float, str))
            ):
                out[f"steps.{sid}.output.{path}"] = value
    for name in slot_names:
        if name in view.slots:
            out[f"slot.{name}"] = view.slots[name]
    return out


def _literal(v: Any) -> str:
    return repr(v)


def _tests(
    feats_a: list[dict[str, Any]], feats_b: list[dict[str, Any]]
) -> list[tuple[str, str, list[bool], list[bool]]]:
    """Candidate single-feature tests: (expr for A, expr for B, holds-on-A, holds-on-B)."""
    common = sorted(set.intersection(*(set(f) for f in feats_a + feats_b)))
    out = []
    for path in common:
        va = [f[path] for f in feats_a]
        vb = [f[path] for f in feats_b]
        values = sorted({json.dumps(v, sort_keys=True) for v in va + vb})
        for raw in values:
            v = json.loads(raw)
            if isinstance(v, bool) or v is None or isinstance(v, str):
                out.append(
                    (
                        f"{path} == {_literal(v)}",
                        f"{path} != {_literal(v)}",
                        [x == v for x in va],
                        [x == v for x in vb],
                    )
                )
        nums = [x for x in va + vb if isinstance(x, (int, float)) and not isinstance(x, bool)]
        if len(nums) == len(va) + len(vb):
            for t in sorted(set(nums)):
                out.append(
                    (f"{path} < {t}", f"{path} >= {t}", [x < t for x in va], [x < t for x in vb])
                )
    return out


def learn_condition(
    feats_a: list[dict[str, Any]], feats_b: list[dict[str, Any]]
) -> tuple[str, str] | None:
    """Conditions (for A, for B) separating the two trace sets perfectly, or None."""
    tests = _tests(feats_a, feats_b)
    # Depth 1, clearest first: one feature with a single value on each side.
    common = sorted(set.intersection(*(set(f) for f in feats_a + feats_b)))
    for path in common:
        va = {json.dumps(f[path], sort_keys=True) for f in feats_a}
        vb = {json.dumps(f[path], sort_keys=True) for f in feats_b}
        if len(va) == 1 and len(vb) == 1 and va != vb:
            a_val, b_val = json.loads(next(iter(va))), json.loads(next(iter(vb)))
            if all(isinstance(v, (bool, str)) or v is None for v in (a_val, b_val)):
                return f"{path} == {_literal(a_val)}", f"{path} == {_literal(b_val)}"
    for ea, eb, ha, hb in tests:  # depth 1
        if all(ha) and not any(hb):
            return ea, eb
        if not any(ha) and all(hb):
            return eb, ea
    if MAX_DEPTH >= 2:  # depth 2: a conjunction selects A, its negation B
        for i, (e1, _, h1a, h1b) in enumerate(tests):
            for e2, _, h2a, h2b in tests[i + 1 :]:
                if all(x and y for x, y in zip(h1a, h2a, strict=True)) and not any(
                    x and y for x, y in zip(h1b, h2b, strict=True)
                ):
                    return f"{e1} and {e2}", f"not ({e1} and {e2})"
    return None


def _union(a: list[LearnedDomain], b: list[LearnedDomain]) -> list[LearnedDomain]:
    """Domains present in both branches, widened to cover both."""
    out = []
    bmap = {(d.path, d.kind, d.container, d.fn): d for d in b}
    for d in a:
        o = bmap.get((d.path, d.kind, d.container, d.fn))
        if o is None:
            continue
        u = d.model_copy(update={"support": d.support + o.support})
        if d.kind in ("enum", "pattern"):
            u.values = sorted({json.dumps(v) for v in (d.values or []) + (o.values or [])})
            u.values = [json.loads(v) for v in u.values]
        elif d.kind in ("range", "length"):
            u.min = None if d.min is None or o.min is None else min(d.min, o.min)
            u.max = None if d.max is None or o.max is None else max(d.max, o.max)
        out.append(u)
    return out


def _rename_refs(text: str, renames: dict[str, str]) -> str:
    for old, new in sorted(renames.items(), key=lambda kv: -len(kv[0])):
        text = text.replace(f"steps.{old}.", f"steps.{new}.")
    return text


def _rename_binding(b: Binding, renames: dict[str, str]) -> Binding:
    upd: dict[str, Any] = {}
    if b.step in renames:
        upd["step"] = renames[b.step]
    if b.input is not None:
        upd["input"] = _rename_binding(b.input, renames)
    return b.model_copy(update=upd)


def _rename_step(step: Step, renames: dict[str, str]) -> Step:
    s = step.model_copy(deep=True)
    s.id = renames.get(s.id, s.id)
    for attr in ("next", "on_yes", "on_no"):
        if hasattr(s, attr) and getattr(s, attr) in renames:
            setattr(s, attr, renames[getattr(s, attr)])
    if isinstance(s, ToolStep):
        s.args = {k: _rename_binding(v, renames) for k, v in s.args.items()}
        s.guards = [_rename_refs(g, renames) for g in s.guards]
    if isinstance(s, (ToolStep, AskStep)):
        for d in s.learned_guards:
            if d.container:
                d.container = _rename_refs(d.container + ".", renames)[:-1]
    if isinstance(s, (AskStep, ConfirmStep, SayStep)) and s.say.template:
        s.say.template = _rename_refs(s.say.template, renames)
    if isinstance(s, LLMStep):
        s.inputs = [_rename_refs(i + ".", renames)[:-1] for i in s.inputs]
    return s


@dataclass
class MergeResult:
    flow: FlowGraph | None
    note: str


def merge(
    a: FlowGraph, views_a: list[TraceView], b: FlowGraph, views_b: list[TraceView]
) -> MergeResult:
    if a.entry.slots != b.entry.slots:
        return MergeResult(None, "different entry slots")
    k = common_prefix(a, b)
    if k == 0 or k >= len(a.steps) or k >= len(b.steps):
        return MergeResult(None, f"no divergence after a shared prefix (prefix={k})")
    prefix = a.steps[:k]
    tool_ids = [s.id for s in prefix if isinstance(s, ToolStep)]
    slot_names = sorted(
        set(a.entry.slots) | {n for s in prefix if isinstance(s, AskStep) for n in s.expects}
    )
    cond = learn_condition(
        [_features(v, tool_ids, slot_names) for v in views_a],
        [_features(v, tool_ids, slot_names) for v in views_b],
    )
    if cond is None:
        return MergeResult(None, f"no deterministic condition separates {a.flow_id} / {b.flow_id}")
    for name, sd in b.slots.items():
        if name in a.slots and a.slots[name] != sd:
            return MergeResult(None, f"slot {name!r} conflicts")

    a_ids = {s.id for s in a.steps}
    renames = {s.id: f"{s.id}b" for s in b.steps[k:]}
    if a_ids & set(renames.values()):
        return MergeResult(None, "step id collision")
    branch_id = f"{prefix[-1].id}x"
    merged_prefix: list[Step] = []
    for sa, sb in zip(prefix, b.steps[:k], strict=True):
        s = sa.model_copy(deep=True)
        if isinstance(s, (ToolStep, AskStep)) and isinstance(sb, (ToolStep, AskStep)):
            s.learned_guards = _union(sa.learned_guards, sb.learned_guards)  # type: ignore[union-attr]
        if isinstance(s, AskStep) and isinstance(sb, AskStep):
            limit = min(s.membership.residual_max_tokens, sb.membership.residual_max_tokens)
            s.membership = Membership(residual_max_tokens=limit)
        merged_prefix.append(s)
    last = merged_prefix[-1]
    if isinstance(last, ConfirmStep):
        last.on_yes = branch_id
    elif isinstance(last, (ToolStep, AskStep, SayStep, LLMStep)):
        last.next = branch_id
    else:
        return MergeResult(None, "cannot branch after this step kind")
    branch = BranchStep(
        id=branch_id,
        cases=[
            BranchCase(when=cond[0], next=a.steps[k].id),
            BranchCase(when=cond[1], next=renames[b.steps[k].id]),
        ],
        default=FALLBACK,
    )
    steps: list[Step] = [
        *merged_prefix,
        branch,
        *a.steps[k:],
        *(_rename_step(s, renames) for s in b.steps[k:]),
    ]
    prov = a.provenance.model_copy(deep=True)
    prov.support = a.provenance.support + b.provenance.support
    prov.sessions = sorted(set(a.provenance.sessions) | set(b.provenance.sessions))
    for key, v in b.provenance.bindings.items():
        sid, rest = key.split(".", 1)
        prov.bindings[f"{renames.get(sid, sid)}.{rest}"] = v
    prov.bindings = dict(sorted(prov.bindings.items()))
    prov.notes = [
        *a.provenance.notes,
        *b.provenance.notes,
        f"branch {branch_id}: merged {b.flow_id} when {cond[1]}",
    ]
    flow = a.model_copy(
        update={
            "steps": steps,
            "slots": {**a.slots, **{n: d for n, d in b.slots.items() if n not in a.slots}},
            "provenance": prov,
        }
    )
    flow.content_hash = flow.compute_hash()
    return MergeResult(flow, f"merged {b.flow_id} into {a.flow_id} on {cond[0]}")
