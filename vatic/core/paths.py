"""Dotted-path access into JSON-like values. List indices are numeric segments."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

_MISSING = object()


class PathError(KeyError):
    pass


def resolve(obj: Any, path: str) -> Any:
    """Resolve ``a.b.0.c`` against nested dicts/lists. Raises PathError if absent."""
    cur = obj
    if path == "":
        return cur
    for seg in path.split("."):
        if isinstance(cur, dict):
            if seg not in cur:
                raise PathError(path)
            cur = cur[seg]
        elif isinstance(cur, list) and seg.isdigit():
            idx = int(seg)
            if idx >= len(cur):
                raise PathError(path)
            cur = cur[idx]
        else:
            raise PathError(path)
    return cur


def try_resolve(obj: Any, path: str) -> Any:
    try:
        return resolve(obj, path)
    except PathError:
        return _MISSING


def is_missing(value: Any) -> bool:
    return value is _MISSING


def leaves(obj: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    """Yield (path, value) for every node, depth-first, in sorted key order.

    Containers are yielded too (so a whole list can be bound), before children.
    """
    if prefix:
        yield prefix, obj
    if isinstance(obj, dict):
        for key in sorted(obj):
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from leaves(obj[key], child)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            child = f"{prefix}.{i}" if prefix else str(i)
            yield from leaves(item, child)


def has_index(path: str) -> bool:
    return any(seg.isdigit() for seg in path.split("."))
