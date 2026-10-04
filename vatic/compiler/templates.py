"""Response template induction.

For each agent reply, replace values the runtime will know (slots, earlier tool
arguments/outputs, optionally formatted) with placeholders. A template is
accepted if at most ``max_variants`` templates with identical placeholders
cover at least ``min_coverage`` of the traces; the most frequent one is used.
Otherwise the step becomes an ``llm`` phrasing node.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from vatic.compiler.bind import TraceView
from vatic.core.formatters import FORMATTERS
from vatic.core.paths import has_index, leaves
from vatic.core.phrasing import apply_filters, placeholders
from vatic.core.transforms import TransformContext
from vatic.ir.schema import LLMPhrase, Say

_SLOT_CHAINS: tuple[tuple[str, ...], ...] = (
    (),
    ("normalize_date", "human_date"),
    ("normalize_time", "human_time"),
)
_VALUE_CHAINS: tuple[tuple[str, ...], ...] = ((), *((f,) for f in FORMATTERS if f != "raw"))


@dataclass(frozen=True)
class Candidate:
    ref: str
    filters: tuple[str, ...]
    text: str

    @property
    def placeholder(self) -> str:
        return "{" + "|".join([self.ref, *self.filters]) + "}"


def _renderable(text: str) -> bool:
    return len(text) >= 3 or (len(text) == 2 and not text.isdigit())


def candidates(
    view: TraceView, step_ids: list[str], slot_names: list[str], ctx: TransformContext
) -> list[Candidate]:
    """All renderable values, most-preferred first.

    Order: whole fields of the latest steps (outputs before args), then slots, then
    list elements (``times.2``), whose position varies between traces.
    """
    out: list[Candidate] = []
    indexed: list[Candidate] = []
    for sid in reversed(step_ids):
        data = view.steps.get(sid)
        if data is None:
            continue
        for part in ("output", "args"):
            for path, value in leaves(data.get(part) or {}):
                if isinstance(value, dict) or value is None or isinstance(value, bool):
                    continue
                for chain in _VALUE_CHAINS:
                    try:
                        text = apply_filters(value, list(chain), ctx)
                    except Exception:
                        continue
                    if _renderable(text):
                        bucket = indexed if has_index(path) else out
                        bucket.append(Candidate(f"steps.{sid}.{part}.{path}", chain, text))
    for name in slot_names:
        value = view.slots.get(name)
        if value is None:
            continue
        for chain in _SLOT_CHAINS:
            try:
                text = apply_filters(value, list(chain), ctx)
            except Exception:
                continue
            if _renderable(text):
                out.append(Candidate(f"slot.{name}", chain, text))
    return out + indexed


def _pattern(value: str) -> re.Pattern[str]:
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(value) + r"(?![A-Za-z0-9])")


def templatize(text: str, cands: list[Candidate], freq: dict[str, int] | None = None) -> str:
    """Replace candidate values in ``text`` with placeholders.

    Longest values first; among equally long values, the placeholder that explains
    the most traces (``freq``) wins, then the candidate preference order.
    """
    freq = freq or {}
    order = sorted(
        enumerate(cands), key=lambda ic: (-len(ic[1].text), -freq.get(ic[1].placeholder, 0), ic[0])
    )
    taken: list[tuple[int, int, str]] = []
    for _, c in order:
        for m in _pattern(c.text).finditer(text):
            if any(m.start() < e and s < m.end() for s, e, _ in taken):
                continue
            taken.append((m.start(), m.end(), c.placeholder))
    taken.sort()
    pieces: list[str] = []
    pos = 0
    for s, e, ph in taken:
        pieces.append(text[pos:s].replace("{", "{{").replace("}", "}}"))
        pieces.append(ph)
        pos = e
    pieces.append(text[pos:].replace("{", "{{").replace("}", "}}"))
    return "".join(pieces)


@dataclass(frozen=True)
class Induced:
    say: Say
    per_trace: list[str]


def induce(
    texts: list[str],
    views: list[TraceView],
    step_ids: list[str],
    slot_names: list[str],
    *,
    intent: str,
    min_coverage: float = 0.9,
    max_variants: int = 3,
) -> Induced:
    cands = [candidates(v, step_ids, slot_names, v.segment.ctx) for v in views]
    freq: dict[str, int] = {}
    for text, cs in zip(texts, cands, strict=True):
        for key in {c.placeholder for c in cs if _pattern(c.text).search(text)}:
            freq[key] = freq.get(key, 0) + 1
    per_trace = [templatize(t, cs, freq) for t, cs in zip(texts, cands, strict=True)]
    counts = Counter(per_trace)
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    top = ranked[0][0]
    shape = sorted(placeholders(top))
    covered = 0
    for template, n in ranked[:max_variants]:
        if sorted(placeholders(template)) == shape:
            covered += n
    coverage = covered / len(texts)
    if coverage >= min_coverage:
        return Induced(Say(template=top, coverage=round(coverage, 4)), per_trace)
    inputs = sorted({p.split("|", 1)[0] for p in placeholders(top)})
    phrase = LLMPhrase(intent=intent, inputs=inputs, example=texts[0])
    return Induced(Say(llm=phrase, coverage=round(coverage, 4)), per_trace)


def namespace_for(view: TraceView, step_ids: list[str], slot_names: list[str]) -> dict[str, Any]:
    return {
        "slot": {k: v for k, v in view.slots.items() if k in slot_names},
        "steps": {k: v for k, v in view.steps.items() if k in step_ids},
    }
