"""Latency, coverage, fallback and routing-quality metrics computed from traces."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from vatic.core.loop_monitor import percentile
from vatic.trace.store import TraceStore


def _pcts(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "p50": round(percentile(values, 50), 2),
        "p95": round(percentile(values, 95), 2),
        "p99": round(percentile(values, 99), 2),
    }


@dataclass
class Metrics:
    sessions: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    turns: int = 0
    routes: Counter[str] = field(default_factory=Counter)
    llm_calls: int = 0
    decision_ms: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    e2e_ms: list[float] = field(default_factory=list)
    ttfa_ms: list[float] = field(default_factory=list)
    membership_ms: list[float] = field(default_factory=list)
    fallback_reasons: Counter[str] = field(default_factory=Counter)
    hedged: int = 0
    membership_checks: int = 0
    # Shadow: per (flow, step) -> [wrong routes, on_path decisions]; per flow match counts.
    wrong_route: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    shadow_match: dict[str, list[int]] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.outcomes["success"] / self.sessions if self.sessions else 0.0

    @property
    def compiled_share(self) -> float:
        live = self.routes["compiled"] + self.routes["llm"]
        return self.routes["compiled"] / live if live else 0.0

    @property
    def llm_calls_per_session(self) -> float:
        return self.llm_calls / self.sessions if self.sessions else 0.0

    @property
    def hedge_rate(self) -> float:
        return self.hedged / self.membership_checks if self.membership_checks else 0.0

    def wrong_route_rate(self) -> float:
        wrong = sum(w for w, _ in self.wrong_route.values())
        total = sum(n for _, n in self.wrong_route.values())
        return wrong / total if total else 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "success_rate": round(self.success_rate, 4),
            "turns": self.turns,
            "compiled_share": round(self.compiled_share, 4),
            "llm_calls_per_session": round(self.llm_calls_per_session, 2),
            "decision_ms": {k: _pcts(v) for k, v in sorted(self.decision_ms.items())},
            "end_to_end_ms": _pcts(self.e2e_ms),
            "time_to_first_audio_ms": _pcts(self.ttfa_ms),
            "membership_ms": _pcts(self.membership_ms),
            "fallback_reasons": dict(self.fallback_reasons.most_common()),
            "hedge_rate": round(self.hedge_rate, 4),
            "wrong_route_rate": round(self.wrong_route_rate(), 4),
            "wrong_route_by_step": {
                f"{f}.{s}": {"wrong": w, "on_path": n}
                for (f, s), (w, n) in sorted(self.wrong_route.items())
            },
            "shadow_match": {
                f: {"matched": m, "compared": c, "rate": round(m / c, 4) if c else None}
                for f, (m, c) in sorted(self.shadow_match.items())
            },
        }


def _reason_key(reason: str) -> str:
    parts = reason.split(":")
    return ":".join(parts[:2]) if parts[0] == "membership" else parts[0]


def compute(store: TraceStore, session_prefix: str | None = None) -> Metrics:
    m = Metrics()
    for s in store.iter_sessions():
        if session_prefix and not s.session_id.startswith(session_prefix):
            continue
        m.sessions += 1
        m.outcomes[s.outcome or "unknown"] += 1
    for t in store.iter_turns():
        if session_prefix and not t.session_id.startswith(session_prefix):
            continue
        if t.route == "shadow":
            cmp = t.shadow
            if cmp is None:
                continue
            if cmp.decision == "on_path" and cmp.step_id:
                w = m.wrong_route.setdefault((cmp.flow_id, cmp.step_id), [0, 0])
                w[1] += 1
                w[0] += 0 if cmp.matched else 1
            if cmp.compared:
                sm = m.shadow_match.setdefault(cmp.flow_id, [0, 0])
                sm[1] += 1
                sm[0] += 1 if cmp.matched else 0
            continue
        m.turns += 1
        m.routes[t.route] += 1
        if t.route == "llm":
            m.llm_calls += t.llm_calls
        tm = t.timings
        if tm.decision_start and tm.decision_end:
            m.decision_ms[t.route].append((tm.decision_end - tm.decision_start) * 1000)
        if tm.stt_end and tm.turn_end:
            m.e2e_ms.append((tm.turn_end - tm.stt_end) * 1000)
        if tm.stt_end and tm.tts_first_audio:
            m.ttfa_ms.append((tm.tts_first_audio - tm.stt_end) * 1000)
        if t.membership is not None and not t.membership.reason.startswith("confirm:"):
            m.membership_checks += 1
            m.membership_ms.append(t.membership.latency_ms)
            m.hedged += 1 if t.membership.hedged else 0
        if t.fallback_reason:
            m.fallback_reasons[_reason_key(t.fallback_reason)] += 1
    return m
