"""Value formatters for response templates (``{ref|formatter}``)."""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from typing import Any


def human_date(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        d = dt.date.fromisoformat(value)
    except ValueError:
        return None
    return f"{d.strftime('%A')}, {d.strftime('%B')} {d.day}"


def human_time(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    m = re.fullmatch(r"(\d{2}):(\d{2})", value)
    if not m:
        return None
    hour, minute = int(m.group(1)), int(m.group(2))
    if hour > 23 or minute > 59:
        return None
    suffix = "AM" if hour < 12 else "PM"
    h12 = hour % 12 or 12
    return f"{h12}:{minute:02d} {suffix}"


def join_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def human_time_list(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    parts = [human_time(v) for v in value]
    if any(p is None for p in parts):
        return None
    return join_list([p for p in parts if p is not None])


def raw(value: Any) -> str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (str, int, float)):
        return str(value)
    return None


# Order matters: the template inducer tries formatters in this order.
FORMATTERS: dict[str, Callable[[Any], str | None]] = {
    "raw": raw,
    "human_date": human_date,
    "human_time": human_time,
    "human_time_list": human_time_list,
}
