"""Fixed registry of deterministic transforms used by bindings and guards.

Every transform is a pure function of (value, context). Registry order is part
of the compiler's determinism contract: the compiler tries transforms in this
order and picks the first that explains all traces.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
MONTH_ABBR = {m[:3]: i + 1 for i, m in enumerate(MONTHS)} | {"sept": 9}
NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
ORDINAL_WORDS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "twenty-first": 21,
    "twenty-second": 22,
    "twenty-third": 23,
    "twenty-fourth": 24,
    "twenty-fifth": 25,
    "twenty-sixth": 26,
    "twenty-seventh": 27,
    "twenty-eighth": 28,
    "twenty-ninth": 29,
    "thirtieth": 30,
    "thirty-first": 31,
}


@dataclass(frozen=True)
class TransformContext:
    today: dt.date

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> TransformContext:
        raw = metadata.get("today")
        today = dt.date.fromisoformat(raw) if isinstance(raw, str) else dt.date.today()
        return cls(today=today)


Transform = Callable[[Any, TransformContext], Any]


def _month_index(word: str) -> int | None:
    w = word.lower().rstrip(".")
    if w in MONTHS:
        return MONTHS.index(w) + 1
    return MONTH_ABBR.get(w)


def _day_number(token: str) -> int | None:
    t = token.lower()
    if t in ORDINAL_WORDS:
        return ORDINAL_WORDS[t]
    m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?", t)
    return int(m.group(1)) if m else None


def _safe_date(year: int, month: int, day: int) -> dt.date | None:
    try:
        return dt.date(year, month, day)
    except ValueError:
        return None


def _explicit_date(text: str, today: dt.date) -> dt.date | None:
    """Month/day or 'the Nth' forms, resolved to the next such date on/after today."""
    m = re.search(r"([a-z]+)\.?\s+(?:the\s+)?([a-z0-9-]+)", text)
    if m and _month_index(m.group(1)) and _day_number(m.group(2)):
        month, day = _month_index(m.group(1)), _day_number(m.group(2))
        assert month is not None and day is not None
        d = _safe_date(today.year, month, day)
        if d is not None and d < today:
            d = _safe_date(today.year + 1, month, day)
        return d
    m = re.search(r"([a-z0-9-]+)\s+of\s+([a-z]+)", text)
    if m and _month_index(m.group(2)) and _day_number(m.group(1)):
        month, day = _month_index(m.group(2)), _day_number(m.group(1))
        assert month is not None and day is not None
        d = _safe_date(today.year, month, day)
        if d is not None and d < today:
            d = _safe_date(today.year + 1, month, day)
        return d
    m = re.search(r"\bthe\s+([a-z0-9-]+)", text) or re.fullmatch(r"(\d{1,2}(?:st|nd|rd|th))", text)
    if m and _day_number(m.group(1)):
        day = _day_number(m.group(1))
        assert day is not None
        d = _safe_date(today.year, today.month, day)
        if d is None or d < today:
            nm, ny = (1, today.year + 1) if today.month == 12 else (today.month + 1, today.year)
            d = _safe_date(ny, nm, day)
        return d
    return None


def normalize_date(value: Any, ctx: TransformContext) -> str | None:
    """Resolve a spoken date phrase to ISO format relative to ``ctx.today``."""
    if not isinstance(value, str):
        return None
    text = re.sub(r"[,]", " ", value.strip().lower())
    text = re.sub(r"\s+", " ", text)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        try:
            return dt.date.fromisoformat(text).isoformat()
        except ValueError:
            return None
    today = ctx.today
    if "day after tomorrow" in text:
        return (today + dt.timedelta(days=2)).isoformat()
    if text == "tomorrow":
        return (today + dt.timedelta(days=1)).isoformat()
    if text == "today":
        return today.isoformat()
    weekday = next((i for i, w in enumerate(WEEKDAYS) if re.search(rf"\b{w}\b", text)), None)
    rest = text
    if weekday is not None:
        rest = re.sub(rf"\b{WEEKDAYS[weekday]}\b", " ", text).strip()
        rest = re.sub(r"^(this coming|this|next|coming)\b", "", rest).strip()
    explicit = _explicit_date(rest, today) if rest else None
    if weekday is None:
        return explicit.isoformat() if explicit else None
    if explicit is not None:
        # Conservative: a weekday that contradicts the explicit date is unresolvable.
        return explicit.isoformat() if explicit.weekday() == weekday else None
    if re.match(r"^next\b", text):
        next_monday = today + dt.timedelta(days=7 - today.weekday())
        return (next_monday + dt.timedelta(days=weekday)).isoformat()
    ahead = (weekday - today.weekday()) % 7 or 7
    return (today + dt.timedelta(days=ahead)).isoformat()


def normalize_time(value: Any, ctx: TransformContext) -> str | None:
    """Resolve a spoken time phrase to 24h ``HH:MM``. Bare 1-7 means PM (clinic hours)."""
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    if text in ("noon", "midday"):
        return "12:00"
    m = re.fullmatch(
        r"(\d{1,2}|[a-z]+)(?::(\d{2}))?\s*"
        r"(am|pm|a\.m\.?|p\.m\.?|o'?clock)?"
        r"(?:\s+in\s+the\s+(morning|afternoon|evening))?",
        text,
    )
    if not m:
        return None
    raw_h, raw_m, suffix, part = m.groups()
    if raw_h.isdigit():
        hour = int(raw_h)
    elif raw_h in NUMBER_WORDS:
        hour = NUMBER_WORDS[raw_h]
    else:
        return None
    minute = int(raw_m) if raw_m else 0
    if minute > 59 or hour > 23:
        return None
    meridiem = None
    if suffix and suffix.startswith("a"):
        meridiem = "am"
    elif suffix and suffix.startswith("p"):
        meridiem = "pm"
    elif part == "morning":
        meridiem = "am"
    elif part in ("afternoon", "evening"):
        meridiem = "pm"
    if hour > 12:
        if meridiem == "am":
            return None
    elif meridiem == "am":
        hour = hour % 12
    elif meridiem == "pm":
        hour = hour % 12 + 12
    elif 1 <= hour <= 7:
        hour += 12
    return f"{hour:02d}:{minute:02d}"


def _str_fn(fn: Callable[[str], str]) -> Transform:
    def wrapped(value: Any, ctx: TransformContext) -> Any:
        return fn(value) if isinstance(value, str) else None

    return wrapped


def _digits(value: str) -> str:
    return "".join(ch for ch in value if ch.isdigit())


# Order matters (see module docstring).
TRANSFORMS: dict[str, Transform] = {
    "normalize_date": normalize_date,
    "normalize_time": normalize_time,
    "lower": _str_fn(str.lower),
    "upper": _str_fn(str.upper),
    "title": _str_fn(str.title),
    "strip": _str_fn(str.strip),
    "digits": _str_fn(_digits),
}


def apply_transform(name: str, value: Any, ctx: TransformContext) -> Any:
    fn = TRANSFORMS.get(name)
    if fn is None:
        raise KeyError(f"unknown transform {name!r}")
    return fn(value, ctx)
