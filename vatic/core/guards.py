"""Guard evaluation: a tiny whitelisted expression evaluator plus learned domains.

Expressions are parsed with ``ast`` and interpreted node by node; nothing is
ever passed to ``eval``. Allowed: comparisons, ``and``/``or``/``not``,
attribute or key access, integer subscripts, literals, and calls to ``len``
or a registered deterministic transform. Any error makes the guard fail.
"""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable, Mapping
from functools import lru_cache
from typing import Any

from vatic.core.paths import is_missing, try_resolve
from vatic.core.transforms import TRANSFORMS, TransformContext
from vatic.ir.schema import LearnedDomain
from vatic.trace.schema import GuardResult


class GuardError(Exception):
    pass


_CMP: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not,
}


@lru_cache(maxsize=4096)
def _parse(expr: str) -> ast.expr:
    try:
        return ast.parse(expr, mode="eval").body
    except SyntaxError as exc:
        raise GuardError(f"syntax error: {exc.msg}") from exc


class _Evaluator:
    def __init__(self, namespace: Mapping[str, Any], ctx: TransformContext | None) -> None:
        self.ns = namespace
        self.ctx = ctx

    def eval(self, node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            if node.value is None or isinstance(node.value, (bool, int, float, str)):
                return node.value
            raise GuardError("unsupported constant")
        if isinstance(node, ast.Name):
            if node.id not in self.ns:
                raise GuardError(f"unknown name {node.id!r}")
            return self.ns[node.id]
        if isinstance(node, ast.Attribute):
            base = self.eval(node.value)
            if isinstance(base, Mapping) and node.attr in base:
                return base[node.attr]
            raise GuardError(f"missing key {node.attr!r}")
        if isinstance(node, ast.Subscript):
            base = self.eval(node.value)
            key = self.eval(node.slice)
            try:
                if isinstance(base, list) and isinstance(key, int):
                    return base[key]
                if isinstance(base, Mapping) and isinstance(key, str):
                    return base[key]
            except (IndexError, KeyError) as exc:
                raise GuardError("missing index") from exc
            raise GuardError("bad subscript")
        if isinstance(node, (ast.List, ast.Tuple)):
            return [self.eval(e) for e in node.elts]
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self._truth(v) for v in node.values)
            return any(self._truth(v) for v in node.values)
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self._truth(node.operand)
            if isinstance(node.op, ast.USub):
                val = self.eval(node.operand)
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return -val
            raise GuardError("unsupported unary op")
        if isinstance(node, ast.Compare):
            left = self.eval(node.left)
            for op, comp in zip(node.ops, node.comparators, strict=True):
                fn = _CMP.get(type(op))
                if fn is None:
                    raise GuardError("unsupported comparison")
                right = self.eval(comp)
                try:
                    ok = fn(left, right)
                except TypeError as exc:
                    raise GuardError(f"type error: {exc}") from exc
                if not ok:
                    return False
                left = right
            return True
        if isinstance(node, ast.Call):
            if node.keywords or not isinstance(node.func, ast.Name) or len(node.args) != 1:
                raise GuardError("unsupported call")
            arg = self.eval(node.args[0])
            name = node.func.id
            if name == "len":
                if isinstance(arg, (list, str, dict)):
                    return len(arg)
                raise GuardError("len() of unsupported type")
            if name in TRANSFORMS:
                if self.ctx is None:
                    raise GuardError("transform call without context")
                return TRANSFORMS[name](arg, self.ctx)
            raise GuardError(f"function {name!r} not allowed")
        raise GuardError(f"disallowed syntax: {type(node).__name__}")

    def _truth(self, node: ast.expr) -> bool:
        val = self.eval(node)
        if not isinstance(val, bool):
            raise GuardError("boolean operand required")
        return val


def evaluate(expr: str, namespace: Mapping[str, Any], ctx: TransformContext | None = None) -> bool:
    """Evaluate a guard expression. Raises GuardError on any problem."""
    result = _Evaluator(namespace, ctx).eval(_parse(expr))
    if not isinstance(result, bool):
        raise GuardError("guard must evaluate to a boolean")
    return result


def check_guard(
    expr: str,
    namespace: Mapping[str, Any],
    ctx: TransformContext | None = None,
    *,
    step_id: str | None = None,
) -> GuardResult:
    try:
        ok = evaluate(expr, namespace, ctx)
        return GuardResult(expr=expr, kind="declared", passed=ok, step_id=step_id)
    except GuardError as exc:
        return GuardResult(
            expr=expr, kind="declared", passed=False, step_id=step_id, detail=str(exc)
        )


def references_output(expr: str) -> bool:
    """True if the expression reads the current step's ``output`` (post-call guard)."""
    try:
        tree = _parse(expr)
    except GuardError:
        return False
    return any(isinstance(n, ast.Name) and n.id == "output" for n in ast.walk(tree))


# -- learned domains ------------------------------------------------------------


def value_shape(value: str) -> str:
    """Abstract a string into a shape: digits->9, letter runs->A/a."""
    shape = re.sub(r"[0-9]", "9", value)
    shape = re.sub(r"[A-Z][a-z]+", "Aa", shape)
    shape = re.sub(r"[a-z]+", "a", shape)
    shape = re.sub(r"[A-Z]+", "A", shape)
    return shape


def describe_domain(d: LearnedDomain) -> str:
    if d.kind == "enum":
        return f"{d.path} in {d.values}"
    if d.kind == "range":
        return f"{d.min} <= {d.path} <= {d.max}"
    if d.kind == "pattern":
        return f"shape({d.path}) in {d.values}"
    if d.kind == "length":
        return f"{d.min} <= len({d.path}) <= {d.max}"
    if d.kind == "member_of":
        inner = f"{d.fn}({d.path})" if d.fn else d.path
        return f"{inner} in {d.container}"
    return f"{d.path} is not null"


def check_domain(
    d: LearnedDomain,
    namespace: Mapping[str, Any],
    ctx: TransformContext | None = None,
    *,
    step_id: str | None = None,
) -> GuardResult:
    expr = describe_domain(d)

    def result(ok: bool, detail: str | None = None) -> GuardResult:
        return GuardResult(expr=expr, kind="learned", passed=ok, step_id=step_id, detail=detail)

    value = try_resolve(namespace, d.path)
    if is_missing(value):
        return result(False, "missing value")
    if d.kind == "not_null":
        return result(value is not None)
    if d.kind == "enum":
        return result(value in (d.values or []), None if value in (d.values or []) else repr(value))
    if d.kind == "range":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return result(False, "not numeric")
        lo = d.min if d.min is not None else float("-inf")
        hi = d.max if d.max is not None else float("inf")
        return result(lo <= value <= hi, repr(value))
    if d.kind == "length":
        if not isinstance(value, (list, str)):
            return result(False, "no length")
        lo = d.min if d.min is not None else 0
        hi = d.max if d.max is not None else float("inf")
        return result(lo <= len(value) <= hi, f"len={len(value)}")
    if d.kind == "pattern":
        if not isinstance(value, str):
            return result(False, "not a string")
        shape = value_shape(value)
        return result(shape in (d.values or []), shape)
    if d.kind == "member_of":
        container = try_resolve(namespace, d.container or "")
        if is_missing(container) or not isinstance(container, list):
            return result(False, "missing container")
        if d.fn:
            if ctx is None:
                return result(False, "no transform context")
            value = TRANSFORMS[d.fn](value, ctx)
        return result(value in container, repr(value))
    return result(False, "unknown domain kind")
