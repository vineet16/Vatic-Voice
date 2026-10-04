"""Per-step on-path / off-path utterance datasets for membership checks.

On-path: the caller's answer at a step in successful traces of the cluster.
Off-path: answers at the same point in sessions whose turn structure diverged
right there (the LLM had to do something the flow does not), plus seeded
digression / correction / multi-intent templates built from on-path answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vatic.compiler.align import AlignedStep
from vatic.compiler.cluster import Cluster, Shape
from vatic.trace.schema import VIRTUAL_TOOLS, SessionTrace, TurnTrace

SEED_SUFFIXES = (
    ", and also what are your hours?",
    ", but can I also cancel my other appointment?",
    ". Actually no, wait.",
)
SEED_PREFIXES = ("No, actually ", "Wait, sorry, not ", "Hmm, what about ")
SEED_STANDALONE = (
    "What are your hours?",
    "Can I talk to a real person?",
    "Sorry, what was that?",
    "Do you take insurance?",
    "I need to cancel instead.",
    "Never mind.",
)
SEEDS_PER_STEP = 20


@dataclass(frozen=True)
class Example:
    text: str
    label: Literal["on_path", "off_path"]
    source: Literal["trace", "deviation", "seeded"]

    def to_json(self) -> dict[str, str]:
        return {"text": self.text, "label": self.label, "source": self.source}


def _loose_segment(turns: list[TurnTrace]) -> tuple[Shape, list[TurnTrace]] | None:
    ordered = sorted(turns, key=lambda t: t.turn_index)
    if any(t.route != "llm" for t in ordered):
        return None
    with_tools = [i for i, t in enumerate(ordered) if t.tool_calls]
    if not with_tools:
        return None
    seg = ordered[with_tools[0] :]
    shape = tuple(tuple(c.tool for c in t.tool_calls if c.tool not in VIRTUAL_TOOLS) for t in seg)
    return shape, seg


def step_examples(
    cluster: Cluster,
    steps: list[AlignedStep],
    corpus: list[tuple[SessionTrace, list[TurnTrace]]],
) -> dict[str, list[Example]]:
    loose = [x for _, turns in corpus if (x := _loose_segment(turns)) is not None]
    out: dict[str, list[Example]] = {}
    for st in steps:
        if st.kind not in ("ask", "confirm"):
            continue
        k = st.turn
        on = sorted({seg.turns[k + 1].user_transcript for seg in cluster.segments})
        examples = [Example(t, "on_path", "trace") for t in on]
        off: set[str] = set()
        for shape, seg_turns in loose:
            if len(seg_turns) <= k + 1 or shape[: k + 1] != cluster.shape[: k + 1]:
                continue
            if shape[k + 1] != cluster.shape[k + 1]:
                off.add(seg_turns[k + 1].user_transcript)
        off -= set(on)
        examples += [Example(t, "off_path", "deviation") for t in sorted(off)]
        seeded: set[str] = set(SEED_STANDALONE)
        for i, text in enumerate(on[:SEEDS_PER_STEP]):
            base = text.rstrip(".!? ")
            seeded.add(base + SEED_SUFFIXES[i % len(SEED_SUFFIXES)])
            seeded.add(SEED_PREFIXES[i % len(SEED_PREFIXES)] + base[:1].lower() + base[1:])
        examples += [Example(t, "off_path", "seeded") for t in sorted(seeded - set(on))]
        out[st.id] = examples
    return out
