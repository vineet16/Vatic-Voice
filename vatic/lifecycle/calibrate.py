"""Conformal calibration of membership thresholds from shadow outcomes.

Calibration data per step: shadow turns where the rules (layers 1-3) said
``on_path``, labelled by what the LLM actually did (``matched`` = on-path,
mismatch = a wrong route if accepted). Each utterance is scored by the step's
classifier.

``accept_threshold`` uses split conformal risk control (Angelopoulos et al.):
with loss L_i(t) = 1[score_i >= t and turn i was off-path] and n calibration
points, choose the smallest t with  n/(n+1) * mean(L(t)) + 1/(n+1) <= alpha.
This bounds the expected wrong-route rate at that step by ``alpha``. It needs
n >= 1/alpha - 1 points; with fewer, the step stays rules-only (no claim made).
``reject_threshold`` is the 5th percentile of calibration on-path scores (capped
at the accept threshold); scores in between are "uncertain" and hedged.
The achieved rate is reported on a held-out split.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vatic.core.classifier import StepClassifier, step_prompt
from vatic.ir.schema import AskStep, FlowGraph, save_flow
from vatic.trace.store import TraceStore


@dataclass
class StepCalibration:
    step_id: str
    feasible: bool
    n_cal: int
    n_holdout: int
    accept_threshold: float | None = None
    reject_threshold: float | None = None
    cal_risk: float | None = None
    holdout_wrong_route: float | None = None
    holdout_accept_rate: float | None = None
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None and k != "step_id"}


@dataclass
class CalibrationReport:
    flow_id: str
    alpha: float
    steps: list[StepCalibration] = field(default_factory=list)


def conformal_threshold(scores: list[float], off: list[bool], alpha: float) -> float | None:
    """Smallest t whose conformal risk bound is <= alpha (None if infeasible)."""
    n = len(scores)
    if n == 0 or 1.0 / (n + 1) > alpha:
        return None
    candidates = sorted(set(scores))
    for t in candidates:
        risk = sum(1 for s, o in zip(scores, off, strict=True) if o and s >= t) / n
        if n / (n + 1) * risk + 1.0 / (n + 1) <= alpha:
            return t
    return None


def _is_holdout(trace_id: str, fraction: float) -> bool:
    h = int(hashlib.sha256(trace_id.encode()).hexdigest(), 16) % 1000
    return h < fraction * 1000


def calibration_data(
    store: TraceStore, flow: FlowGraph, step_id: str
) -> list[tuple[str, str, bool]]:
    """(trace_id, utterance, off_path) for shadow turns the rules accepted at a step."""
    out = []
    for t in store.iter_turns(route="shadow", flow_id=flow.flow_id):
        cmp = t.shadow
        if cmp is None or cmp.flow_version != flow.version or cmp.step_id != step_id:
            continue
        if cmp.decision is None or not cmp.compared or t.membership is None:
            continue
        if (t.membership.rules or t.membership.decision) != "on_path":
            continue
        out.append((t.trace_id, t.user_transcript, not cmp.matched))
    return out


def calibrate_flow(
    flow: FlowGraph,
    store: TraceStore,
    flows_dir: Path,
    *,
    alpha: float = 0.01,
    holdout_fraction: float = 0.3,
    save: bool = True,
) -> CalibrationReport:
    report = CalibrationReport(flow.flow_id, alpha)
    models: dict[str, StepClassifier] = {}
    for step in flow.steps:
        if not isinstance(step, AskStep) or not step.membership.classifier:
            continue
        path = step.membership.classifier
        clf = models.get(path) or models.setdefault(path, StepClassifier(flows_dir / path))
        prompt = step_prompt(step)
        data = calibration_data(store, flow, step.id)
        scored = [(tid, clf.score(prompt, text), off) for tid, text, off in data]
        cal = [(s, o) for tid, s, o in scored if not _is_holdout(tid, holdout_fraction)]
        hold = [(s, o) for tid, s, o in scored if _is_holdout(tid, holdout_fraction)]
        sc = StepCalibration(step.id, False, len(cal), len(hold))
        t = conformal_threshold([s for s, _ in cal], [o for _, o in cal], alpha)
        if t is None:
            sc.note = f"insufficient calibration data (need >= {int(1 / alpha)} points); rules only"
            step.membership.accept_threshold = None
            step.membership.reject_threshold = None
        else:
            on_scores = sorted(s for s, o in cal if not o)
            p5 = on_scores[int(0.05 * (len(on_scores) - 1))] if on_scores else t
            sc.feasible = True
            sc.accept_threshold = round(t, 6)
            sc.reject_threshold = round(min(p5, t), 6)
            sc.cal_risk = round(sum(1 for s, o in cal if o and s >= t) / len(cal), 6)
            if hold:
                wrong = sum(1 for s, o in hold if o and s >= t)
                sc.holdout_wrong_route = round(wrong / len(hold), 6)
                sc.holdout_accept_rate = round(sum(1 for s, _ in hold if s >= t) / len(hold), 6)
            step.membership.accept_threshold = sc.accept_threshold
            step.membership.reject_threshold = sc.reject_threshold
        step.membership.calibration = {"alpha": alpha, **sc.as_dict()}
        report.steps.append(sc)
    if save and report.steps:
        flow.content_hash = flow.compute_hash()
        save_flow(flow, flows_dir / f"{flow.flow_id}.yaml")
    return report
