"""Learned value domains (conservative runtime guards).

Only values the flow *depends on* get a domain: tool arguments, output fields
that are bound or templated (and their parents), top-level boolean status
flags, and slots. At runtime a value outside its observed domain falls back to
the LLM.
"""

from __future__ import annotations

from typing import Any

from vatic.compiler.bind import TraceView
from vatic.core.guards import value_shape
from vatic.core.paths import has_index, is_missing, leaves, try_resolve
from vatic.ir.schema import LearnedDomain

MAX_SHAPES = 5
MAX_ENUM = 3
ENUM_SUPPORT_FACTOR = 10


def domain_for(values: list[Any], path: str) -> list[LearnedDomain]:
    """Domains describing ``values`` (one per trace) at namespace ``path``."""
    n = len(values)
    if not values or any(is_missing(v) for v in values):
        return []
    out: list[LearnedDomain] = []
    if all(v is not None for v in values) and any(isinstance(v, (dict, list)) for v in values):
        out.append(LearnedDomain(path=path, kind="not_null", support=n))
    if all(isinstance(v, bool) for v in values):
        observed = sorted(set(values))
        if len(observed) == 1:
            out.append(LearnedDomain(path=path, kind="enum", values=observed, support=n))
    elif all(isinstance(v, str) for v in values):
        distinct = sorted(set(values))
        if len(distinct) <= MAX_ENUM and n >= ENUM_SUPPORT_FACTOR * len(distinct):
            out.append(LearnedDomain(path=path, kind="enum", values=distinct, support=n))
        else:
            shapes = sorted({value_shape(v) for v in values})
            if len(shapes) <= MAX_SHAPES:
                out.append(LearnedDomain(path=path, kind="pattern", values=shapes, support=n))
    elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
        out.append(
            LearnedDomain(path=path, kind="range", min=min(values), max=max(values), support=n)
        )
    elif all(isinstance(v, list) for v in values):
        lengths = [len(v) for v in values]
        out.append(
            LearnedDomain(path=path, kind="length", min=min(lengths), max=max(lengths), support=n)
        )
    return out


def ancestors(path: str) -> list[str]:
    parts = path.split(".")
    return [".".join(parts[: i + 1]) for i in range(len(parts))]


def output_domains(
    step_id: str, views: list[TraceView], used_paths: set[str]
) -> list[LearnedDomain]:
    outputs = [v.steps[step_id]["output"] for v in views]
    paths: set[str] = set()
    for p in used_paths:
        paths.update(a for a in ancestors(p) if not has_index(a))
    for key, value in sorted(outputs[0].items()):
        if isinstance(value, bool):
            paths.add(key)
    out: list[LearnedDomain] = []
    for p in sorted(paths):
        out.extend(domain_for([try_resolve(o, p) for o in outputs], f"output.{p}"))
    return out


def arg_domains(step_id: str, views: list[TraceView]) -> list[LearnedDomain]:
    args = [v.steps[step_id]["args"] for v in views]
    out: list[LearnedDomain] = []
    for name in sorted(args[0]):
        out.extend(domain_for([a.get(name) for a in args], f"args.{name}"))
    return out


def list_containers(views: list[TraceView], step_ids: list[str]) -> list[str]:
    """Namespace paths of non-indexed list fields in the given steps' outputs."""
    found: list[str] = []
    for sid in step_ids:
        output = views[0].steps[sid]["output"]
        for path, value in leaves(output):
            if isinstance(value, list) and not has_index(path):
                found.append(f"steps.{sid}.output.{path}")
    return found


def member_of(values: list[Any], views: list[TraceView], containers: list[str]) -> str | None:
    """First container path whose list contains the value in every trace."""
    for c in containers:
        ok = True
        for v, view in zip(values, views, strict=True):
            lst = try_resolve(view.namespace(), c)
            if is_missing(lst) or not isinstance(lst, list) or v not in lst:
                ok = False
                break
        if ok:
            return c
    return None


def slot_domains(name: str, views: list[TraceView]) -> list[LearnedDomain]:
    values = [v.slots.get(name) for v in views]
    if any(not isinstance(v, str) for v in values):
        return []
    shapes = sorted({value_shape(str(v)) for v in values})
    if len(shapes) > MAX_SHAPES:
        return []
    return [LearnedDomain(path=f"slot.{name}", kind="pattern", values=shapes, support=len(values))]
