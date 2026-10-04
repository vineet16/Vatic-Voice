"""Argument provenance: classify every tool argument's binding.

Order (first explanation consistent across *all* traces wins):

1. ``const``     - identical value in every trace
2. ``output``    - equals a field of an earlier tool output (same JSON path everywhere)
3. ``slot``      - the caller's words, recovered by one deterministic extractor from the
                   same aligned turn of every trace
4. ``transform`` - ``fn(x)`` for a registered transform, ``x`` an output field or a slot
5. ``llm``       - otherwise; only allowed in non-irreversible steps, else the flow is refused
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from vatic.compiler.align import AlignedStep
from vatic.compiler.cluster import Cluster, Segment
from vatic.core.extract import EXTRACTORS, normalize_text, unique_span
from vatic.core.paths import has_index, is_missing, leaves, try_resolve
from vatic.core.tools import SideEffect
from vatic.core.transforms import TRANSFORMS
from vatic.ir.schema import Binding, BindingProvenance


@dataclass(frozen=True)
class SlotSource:
    turn: int
    extractor: str


@dataclass
class TraceView:
    """Per-trace values available to bindings, guards and templates."""

    segment: Segment
    slots: dict[str, str] = field(default_factory=dict)
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)  # id -> {args, output}

    def namespace(self) -> dict[str, Any]:
        return {"slot": dict(self.slots), "steps": self.steps}


@dataclass
class BindResult:
    bindings: dict[str, dict[str, Binding]] = field(default_factory=dict)
    provenance: dict[str, BindingProvenance] = field(default_factory=dict)
    slots: dict[str, SlotSource] = field(default_factory=dict)
    views: list[TraceView] = field(default_factory=list)
    refusal: str | None = None
    drop: set[int] = field(default_factory=set)  # trace indices to drop, then re-bind
    drop_reason: str | None = None


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _call(seg: Segment, step: AlignedStep) -> tuple[dict[str, Any], dict[str, Any]]:
    rec = seg.turns[step.turn].tool_calls[step.call]
    return rec.args, rec.output or {}


_NA = object()


class _Binder:
    def __init__(self, cluster: Cluster, steps: list[AlignedStep], max_na: int) -> None:
        self.max_na = max_na
        self.cluster = cluster
        self.steps = steps
        self.segs = cluster.segments
        self.n = len(self.segs)
        # Turns whose user transcript answers a confirm: never a slot source.
        self.confirm_answers = {s.turn + 1 for s in steps if s.kind == "confirm"}
        self.slot_ids: dict[SlotSource, str] = {}
        self.result = BindResult(views=[TraceView(segment=s) for s in self.segs])

    # -- candidate sources ---------------------------------------------------------

    def _unique_span(self, i: int, src: SlotSource) -> str | None:
        seg = self.segs[i]
        texts = seg.entry_transcripts() if src.turn == 0 else [seg.turns[src.turn].user_transcript]
        return unique_span(src.extractor, texts)

    def _slot_sources(self, upto_turn: int) -> list[SlotSource]:
        return [
            SlotSource(j, e)
            for j in range(upto_turn, -1, -1)
            if j not in self.confirm_answers
            for e in EXTRACTORS
        ]

    def _output_paths(self, prior: list[AlignedStep]) -> list[tuple[AlignedStep, str]]:
        """(step, path) for every node of every prior output, preferred first."""
        found: set[tuple[int, str]] = set()
        for pos, st in enumerate(prior):
            for seg in self.segs:
                for path, _value in leaves(_call(seg, st)[1]):
                    found.add((pos, path))
        ordered = sorted(found, key=lambda t: (has_index(t[1]), -t[0], t[1]))
        return [(prior[pos], path) for pos, path in ordered]

    def _slot_binding(self, src: SlotSource, arg: str, na: set[int]) -> Binding:
        name = self.slot_ids.get(src)
        if name is None:
            name = arg
            k = 2
            while name in self.result.slots:
                name = f"{arg}_{k}"
                k += 1
            if not na:  # with drops pending, the binder restarts on a smaller cluster
                self.slot_ids[src] = name
                self.result.slots[name] = src
                for i, view in enumerate(self.result.views):
                    span = self._unique_span(i, src)
                    assert span is not None
                    view.slots[name] = span
        return Binding(source="slot", slot=name)

    # -- classification ----------------------------------------------------------------
    #
    # Each candidate is checked against every trace: "match", "contradict", or
    # "na" (not applicable: no unique span / missing path - the runtime would fall
    # back on such a call). Contradictions always reject a candidate; "na" traces
    # are tolerated only up to ``allow_na`` and are then dropped from the cluster.

    def _check(self, fn: Any, vals: list[Any]) -> tuple[bool, set[int]]:
        na: set[int] = set()
        for i in range(self.n):
            got = fn(i)
            if got is _NA:
                na.add(i)
            elif canonical(got) != canonical(vals[i]):
                return False, na
        return len(na) < self.n, na

    def _span_value(self, src: SlotSource, transform: Any = None) -> Any:
        def get(i: int) -> Any:
            span = self._unique_span(i, src)
            if span is None:
                return _NA
            return transform(span, self.segs[i].ctx) if transform else span

        return get

    def classify(
        self, step: AlignedStep, arg: str, prior: list[AlignedStep], allow_na: int
    ) -> tuple[Binding, set[int]] | None:
        vals = [_call(seg, step)[0].get(arg) for seg in self.segs]

        def accept(fn: Any) -> set[int] | None:
            ok, na = self._check(fn, vals)
            return na if ok and len(na) <= allow_na else None

        # 1. const - unless the caller's own words explain the value in every trace:
        #    a value that merely never varied in the data (e.g. "next Saturday" in a
        #    corpus spanning one week) must still follow what the caller says.
        if all(canonical(v) == canonical(vals[0]) for v in vals):
            spoken = self._from_caller(step, arg, vals)
            return spoken if spoken is not None else (Binding(source="const", value=vals[0]), set())
        all_outputs = self._output_paths(prior)
        outputs = [(st, p) for st, p in all_outputs if not has_index(p)]
        indexed = [(st, p) for st, p in all_outputs if has_index(p)]
        # 2. output (whole fields; list positions are tried last, see below)
        for st, path in outputs:
            na = accept(self._output_value(st, path))
            if na is not None:
                return Binding(source="output", step=st.id, path=path), na
        # 3. slot (compared after text normalisation)
        if all(isinstance(v, str) for v in vals):
            norm = [normalize_text(str(v)) for v in vals]
            for src in self._slot_sources(step.turn):
                get = self._span_value(src)
                ok, na = self._check(
                    lambda i, g=get: x if (x := g(i)) is _NA else normalize_text(x), norm
                )
                if ok and len(na) <= allow_na:
                    return self._slot_binding(src, arg, na), na
        # 4. transform
        for fn_name, fn in TRANSFORMS.items():
            for st, path in outputs:
                get = self._output_value(st, path)
                na = accept(
                    lambda i, g=get, f=fn: x if (x := g(i)) is _NA else f(x, self.segs[i].ctx)
                )
                if na is not None:
                    inner = Binding(source="output", step=st.id, path=path)
                    return Binding(source="transform", fn=fn_name, input=inner), na
            for src in self._slot_sources(step.turn):
                na = accept(self._span_value(src, fn))
                if na is not None:
                    inner = self._slot_binding(src, arg, na)
                    return Binding(source="transform", fn=fn_name, input=inner), na
        # 5. a fixed list position in an earlier output ("the first result"). Ranked
        #    after the caller's own words: a stable position is weaker evidence.
        for st, path in indexed:
            na = accept(self._output_value(st, path))
            if na is not None:
                return Binding(source="output", step=st.id, path=path), na
        return None

    def _from_caller(
        self, step: AlignedStep, arg: str, vals: list[Any]
    ) -> tuple[Binding, set[int]] | None:
        """A slot or transform-of-slot explanation with no inapplicable traces."""
        for src in self._slot_sources(step.turn):
            if all(isinstance(v, str) for v in vals):
                norm = [normalize_text(str(v)) for v in vals]
                get = self._span_value(src)
                ok, na = self._check(
                    lambda i, g=get: x if (x := g(i)) is _NA else normalize_text(x), norm
                )
                if ok and not na:
                    return self._slot_binding(src, arg, na), na
            for fn_name, fn in TRANSFORMS.items():
                ok, na = self._check(self._span_value(src, fn), vals)
                if ok and not na:
                    inner = self._slot_binding(src, arg, na)
                    return Binding(source="transform", fn=fn_name, input=inner), na
        return None

    def _output_value(self, st: AlignedStep, path: str) -> Any:
        def get(i: int) -> Any:
            v = try_resolve(_call(self.segs[i], st)[1], path)
            return _NA if is_missing(v) else v

        return get

    def run(self) -> BindResult:
        tool_steps = [s for s in self.steps if s.kind == "tool"]
        for pos, step in enumerate(tool_steps):
            assert step.tool is not None
            prior = tool_steps[:pos]
            args0 = _call(self.segs[0], step)[0]
            arg_names = sorted(set().union(*(_call(s, step)[0].keys() for s in self.segs)))
            if any(set(_call(s, step)[0]) != set(args0) for s in self.segs):
                self.result.refusal = f"{step.id}: argument names differ across traces"
                return self.result
            bound: dict[str, Binding] = {}
            for arg in arg_names:
                found = self.classify(step, arg, prior, allow_na=0)
                if found is None and self.max_na:
                    found = self.classify(step, arg, prior, allow_na=self.max_na)
                    if found is not None and found[1]:
                        self.result.drop = found[1]
                        self.result.drop_reason = f"{step.id}.{arg}"
                        return self.result
                b = found[0] if found is not None else None
                if b is None:
                    if step.side_effect == SideEffect.IRREVERSIBLE:
                        self.result.refusal = (
                            f"{step.id} ({step.tool}).{arg}: no deterministic binding "
                            "for an irreversible step"
                        )
                        return self.result
                    b = Binding(source="llm", hint=f"{step.tool}.{arg}")
                bound[arg] = b
                self.result.provenance[f"{step.id}.args.{arg}"] = BindingProvenance(
                    source=b.source, support=self.n
                )
            self.result.bindings[step.id] = bound
            for view in self.result.views:
                args, output = _call(view.segment, step)
                view.steps[step.id] = {"args": args, "output": output}
        return self.result


def bind(cluster: Cluster, steps: list[AlignedStep], max_na_rate: float = 0.0) -> BindResult:
    """Bind every tool argument. If the result has ``drop`` set, the caller must
    remove those traces from the cluster and bind again."""
    return _Binder(cluster, steps, int(max_na_rate * cluster.support)).run()


def slots_available_at(result: BindResult, turn: int) -> list[str]:
    """Slots whose source turn is at or before ``turn``."""
    return [name for name, src in result.slots.items() if src.turn <= turn]
