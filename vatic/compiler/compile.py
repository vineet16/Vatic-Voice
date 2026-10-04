"""Compiler entry point: traces -> FlowGraph IR. No LLM, fully deterministic.

Same traces + same config -> byte-identical output (trace order does not matter).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vatic.compiler.align import AlignedStep, align
from vatic.compiler.bind import BindResult, bind
from vatic.compiler.branch import merge, split_by_behaviour
from vatic.compiler.cluster import Cluster, Shape, Skipped, cluster
from vatic.compiler.emit import EmitConfig, build_flow
from vatic.compiler.examples import Example, step_examples
from vatic.core.tools import ToolCatalog
from vatic.ir.schema import BranchStep, FlowGraph
from vatic.ir.validate import errors
from vatic.trace.schema import SessionTrace, TurnTrace

Corpus = list[tuple[SessionTrace, list[TurnTrace]]]


@dataclass
class CompileConfig:
    min_support: int = 20
    max_inapplicable_rate: float = 0.05
    residual_max_tokens: int = 3
    template_min_coverage: float = 0.9


@dataclass(frozen=True)
class Refusal:
    flow_id: str
    signature: tuple[str, ...]
    shape: Shape
    support: int
    reason: str


@dataclass
class CompileResult:
    accepted: list[FlowGraph] = field(default_factory=list)
    refused: list[Refusal] = field(default_factory=list)
    skipped: list[Skipped] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    examples: dict[str, dict[str, list[Example]]] = field(default_factory=dict)


def _flow_ids(clusters: list[Cluster]) -> list[str]:
    ids: list[str] = []
    for c in clusters:
        base = c.signature[-1]
        fid, n = base, 2
        while fid in ids:
            fid = f"{base}_{n}"
            n += 1
        ids.append(fid)
    return ids


def _canonical_corpus(corpus: Corpus) -> Corpus:
    """Order-independent view of the corpus (determinism under shuffling)."""
    return sorted(
        ((s, sorted(turns, key=lambda t: (t.turn_index, t.trace_id))) for s, turns in corpus),
        key=lambda st: st[0].session_id,
    )


@dataclass
class _Built:
    cluster: Cluster
    steps: list[AlignedStep]
    br: BindResult
    flow: FlowGraph
    dropped: list[str]


def _bind_with_drops(
    c: Cluster, catalog: ToolCatalog, cfg: CompileConfig
) -> tuple[Cluster, list[AlignedStep], BindResult, list[str]]:
    steps = align(c, catalog)
    br = bind(c, steps, cfg.max_inapplicable_rate)
    dropped: list[str] = []
    while br.drop:
        # Traces where a binding is inapplicable (e.g. ASR noise produced two date
        # spans) would fall back at runtime; drop them and bind again.
        dropped += [f"{c.segments[i].session_id} ({br.drop_reason})" for i in sorted(br.drop)]
        keep = tuple(s for i, s in enumerate(c.segments) if i not in br.drop)
        c = Cluster(signature=c.signature, shape=c.shape, segments=keep)
        if c.support < cfg.min_support:
            break
        steps = align(c, catalog)
        br = bind(c, steps, cfg.max_inapplicable_rate)
    return c, steps, br, dropped


def compile_corpus(
    corpus: Corpus, catalog: ToolCatalog, config: CompileConfig | None = None
) -> CompileResult:
    cfg = config or CompileConfig()
    corpus = _canonical_corpus(corpus)
    clusters, skipped = cluster(corpus, cfg.min_support)
    result = CompileResult(skipped=skipped)
    emit_cfg = EmitConfig(
        residual_max_tokens=cfg.residual_max_tokens,
        template_min_coverage=cfg.template_min_coverage,
    )
    queue: list[tuple[Cluster, int]] = [(c, 0) for c in clusters]
    built: list[_Built] = []
    while queue:
        original, depth = queue.pop(0)
        provisional = _flow_ids([original])[0]
        c, steps, br, dropped = _bind_with_drops(original, catalog, cfg)
        if c.support < cfg.min_support:
            reason = f"support fell to {c.support} after dropping inapplicable traces"
            result.refused.append(Refusal(provisional, c.signature, c.shape, c.support, reason))
            continue
        if br.refusal is not None:
            result.refused.append(Refusal(provisional, c.signature, c.shape, c.support, br.refusal))
            continue
        if depth < 2:
            subs = split_by_behaviour(c, steps, br, cfg.min_support, cfg.template_min_coverage)
            if subs is not None:
                kept = sum(s.support for s in subs)
                result.notes.append(
                    f"split {' -> '.join(c.signature)} ({c.support} traces) by reply behaviour "
                    f"into {[s.support for s in subs]}; {c.support - kept} traces dropped"
                )
                queue[0:0] = [(s, depth + 1) for s in subs]
                continue
        flow = build_flow(provisional, c, steps, br, emit_cfg)
        problems = errors(flow, catalog)
        if problems:
            reason = "; ".join(str(p) for p in problems)
            result.refused.append(Refusal(provisional, c.signature, c.shape, c.support, reason))
            continue
        built.append(_Built(c, steps, br, flow, dropped))

    built.sort(
        key=lambda b: (
            -b.cluster.support,
            b.cluster.signature,
            b.cluster.shape,
            b.cluster.segments[0].session_id,
        )
    )
    for b, fid in zip(built, _flow_ids([b.cluster for b in built]), strict=True):
        b.flow = b.flow.model_copy(update={"flow_id": fid})
        if b.dropped:
            b.flow.provenance.notes = [f"dropped (inapplicable): {d}" for d in b.dropped]
        b.flow.content_hash = b.flow.compute_hash()
    examples = {b.flow.flow_id: step_examples(b.cluster, b.steps, corpus) for b in built}

    # Branch synthesis: merge flows with the same goal that share a prefix and diverge
    # on a condition that separates their traces perfectly.
    merged_into: dict[str, str] = {}
    for i, a in enumerate(built):
        if a.flow.flow_id in merged_into:
            continue
        for b in built[i + 1 :]:
            if b.flow.flow_id in merged_into or b.cluster.signature[-1] != a.cluster.signature[-1]:
                continue
            if any(isinstance(s, BranchStep) for s in a.flow.steps):
                break  # one branch per flow in v0.1
            m = merge(a.flow, a.br.views, b.flow, b.br.views)
            result.notes.append(f"branch {a.flow.flow_id} + {b.flow.flow_id}: {m.note}")
            if m.flow is None or errors(m.flow, catalog):
                continue
            a.flow = m.flow
            merged_into[b.flow.flow_id] = a.flow.flow_id
            examples[a.flow.flow_id].update(
                {f"{sid}b": ex for sid, ex in examples.pop(b.flow.flow_id).items()}
            )
    for b in built:
        if b.flow.flow_id not in merged_into:
            result.accepted.append(b.flow)
            result.examples[b.flow.flow_id] = examples[b.flow.flow_id]
    return result
