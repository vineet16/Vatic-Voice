"""Flow lifecycle: candidate -> shadow -> active -> retired.

Every transition is persisted to the flow's YAML and logged as a
``LifecycleEvent`` with its reason and evidence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from vatic.ir.schema import FlowGraph, FlowStatus, load_flow, save_flow
from vatic.ir.validate import validate
from vatic.trace.schema import LifecycleEvent
from vatic.trace.store import TraceStore


class PromotionRules(BaseModel):
    min_sessions: int = 50
    min_match_rate: float = 0.98
    max_irreversible_mismatches: int = 0


@dataclass
class ShadowStats:
    flow_id: str
    version: int
    sessions: int  # sessions whose shadow run reached the flow's end
    touched: int  # sessions with at least one shadow comparison
    turns: int
    compared: int
    matched: int
    irreversible_mismatches: int
    wrong_route: dict[str, tuple[int, int]]  # step -> (wrong, on_path decisions)
    reasons: dict[str, int]

    @property
    def match_rate(self) -> float:
        return self.matched / self.compared if self.compared else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "touched_sessions": self.touched,
            "compared_turns": self.compared,
            "matched_turns": self.matched,
            "match_rate": round(self.match_rate, 4),
            "irreversible_mismatches": self.irreversible_mismatches,
            "wrong_route": {k: list(v) for k, v in sorted(self.wrong_route.items())},
        }


def flow_path(flows_dir: str | Path, flow_id: str) -> Path:
    return Path(flows_dir) / f"{flow_id}.yaml"


def load_flows(flows_dir: str | Path) -> list[FlowGraph]:
    d = Path(flows_dir)
    if not d.exists():
        return []
    return [load_flow(p) for p in sorted(d.glob("*.yaml"))]


def shadow_stats(store: TraceStore, flow: FlowGraph) -> ShadowStats:
    sessions: set[str] = set()
    touched: set[str] = set()
    turns = compared = matched = irr = 0
    wrong: dict[str, list[int]] = {}
    reasons: dict[str, int] = {}
    for t in store.iter_turns(route="shadow", flow_id=flow.flow_id):
        cmp = t.shadow
        if cmp is None or cmp.flow_version != flow.version:
            continue
        touched.add(t.session_id)
        if cmp.run_ended:
            sessions.add(t.session_id)
        turns += 1
        if cmp.decision == "on_path" and cmp.step_id is not None:
            w = wrong.setdefault(cmp.step_id, [0, 0])
            w[1] += 1
            if not cmp.matched:
                w[0] += 1
        if not cmp.compared or cmp.decision not in (None, "on_path"):
            continue
        compared += 1
        if cmp.matched:
            matched += 1
        else:
            if cmp.irreversible:
                irr += 1
            for r in cmp.reasons:
                key = r.split(":", 1)[0]
                reasons[key] = reasons.get(key, 0) + 1
    return ShadowStats(
        flow_id=flow.flow_id,
        version=flow.version,
        sessions=len(sessions),
        touched=len(touched),
        turns=turns,
        compared=compared,
        matched=matched,
        irreversible_mismatches=irr,
        wrong_route={k: (v[0], v[1]) for k, v in wrong.items()},
        reasons=reasons,
    )


def transition(
    flows_dir: str | Path,
    flow_id: str,
    to_status: FlowStatus,
    reason: str,
    *,
    store: TraceStore | None = None,
    evidence: dict[str, Any] | None = None,
    manual: bool = False,
) -> LifecycleEvent:
    path = flow_path(flows_dir, flow_id)
    flow = load_flow(path)
    if to_status in ("shadow", "active"):
        errors = [i for i in validate(flow) if i.severity == "error"]
        if errors:
            raise ValueError(f"cannot move invalid flow to {to_status}: {errors[0].message}")
    event = LifecycleEvent(
        flow_id=flow_id,
        version=flow.version,
        from_status=flow.status,
        to_status=to_status,
        reason=reason,
        evidence=evidence or {},
        at=time.time(),
        manual=manual,
    )
    flow.status = to_status
    save_flow(flow, path)
    if store is not None:
        store.write(event)
    return event


_NEXT: dict[str, FlowStatus] = {"candidate": "shadow", "shadow": "active"}


def promote_one(
    flows_dir: str | Path, flow_id: str, store: TraceStore | None = None
) -> LifecycleEvent:
    """Manual promotion one step forward (logged as manual override)."""
    flow = load_flow(flow_path(flows_dir, flow_id))
    target = _NEXT.get(flow.status)
    if target is None:
        raise ValueError(f"{flow_id} is {flow.status}; nothing to promote to")
    return transition(flows_dir, flow_id, target, "manual promotion", store=store, manual=True)


def demote_one(
    flows_dir: str | Path,
    flow_id: str,
    reason: str = "manual demotion",
    *,
    store: TraceStore | None = None,
    evidence: dict[str, Any] | None = None,
    manual: bool = True,
) -> LifecycleEvent:
    flow = load_flow(flow_path(flows_dir, flow_id))
    target: FlowStatus = "shadow" if flow.status == "active" else "retired"
    return transition(
        flows_dir, flow_id, target, reason, store=store, evidence=evidence, manual=manual
    )


def evaluate(
    flows_dir: str | Path,
    store: TraceStore,
    rules: PromotionRules | None = None,
) -> list[LifecycleEvent]:
    """Apply automatic rules: valid candidates -> shadow; qualifying shadow -> active."""
    rules = rules or PromotionRules()
    events = []
    for flow in load_flows(flows_dir):
        if flow.status == "candidate":
            if not any(i.severity == "error" for i in validate(flow)):
                events.append(
                    transition(
                        flows_dir, flow.flow_id, "shadow", "candidate validated", store=store
                    )
                )
        elif flow.status == "shadow":
            st = shadow_stats(store, flow)
            ok = (
                st.sessions >= rules.min_sessions
                and st.match_rate >= rules.min_match_rate
                and st.irreversible_mismatches <= rules.max_irreversible_mismatches
            )
            if ok:
                events.append(
                    transition(
                        flows_dir,
                        flow.flow_id,
                        "active",
                        f"shadow evidence: {st.sessions} sessions, match {st.match_rate:.1%}",
                        store=store,
                        evidence=st.as_dict(),
                    )
                )
    return events
