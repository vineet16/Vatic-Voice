"""Group successful LLM-driven task segments into candidate flows.

A segment is the span of a session from the first turn with a tool call to the
last one. Segments are grouped by tool signature (ordered tool names) and then
by *shape* (tool names per turn). Each shape group with enough support becomes
a candidate flow; smaller groups are reported, never merged by guesswork.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from dataclasses import dataclass

from vatic.core.transforms import TransformContext
from vatic.trace.schema import VIRTUAL_TOOLS, SessionTrace, TurnTrace

Shape = tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class Segment:
    session: SessionTrace
    turns: tuple[TurnTrace, ...]
    prefix: tuple[TurnTrace, ...] = ()  # turns before the entry turn

    def entry_transcripts(self) -> list[str]:
        """Entry turn first, then earlier turns, most recent first."""
        return [self.turns[0].user_transcript] + [t.user_transcript for t in reversed(self.prefix)]

    @property
    def session_id(self) -> str:
        return self.session.session_id

    @property
    def shape(self) -> Shape:
        return tuple(tuple(c.tool for c in t.tool_calls) for t in self.turns)

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(name for turn in self.shape for name in turn)

    @property
    def ctx(self) -> TransformContext:
        raw = self.session.metadata.get("today")
        if isinstance(raw, str):
            return TransformContext(dt.date.fromisoformat(raw))
        return TransformContext(dt.date.fromtimestamp(self.session.started_at))


@dataclass(frozen=True)
class Cluster:
    signature: tuple[str, ...]
    shape: Shape
    segments: tuple[Segment, ...]

    @property
    def support(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class Skipped:
    signature: tuple[str, ...]
    shape: Shape
    support: int
    reason: str


def segment(session: SessionTrace, turns: list[TurnTrace]) -> Segment | None:
    """The LLM-driven task span of one session, or None if it is not compilable."""
    ordered = sorted(turns, key=lambda t: t.turn_index)
    for t in ordered:
        if t.route != "llm" or t.flow_id is not None:
            return None  # only purely LLM-driven sessions teach the compiler
        for c in t.tool_calls:
            if c.tool in VIRTUAL_TOOLS or c.error is not None or c.output is None:
                return None
    with_tools = [i for i, t in enumerate(ordered) if t.tool_calls]
    if not with_tools:
        return None
    first, last = with_tools[0], with_tools[-1]
    return Segment(
        session=session,
        turns=tuple(ordered[first : last + 1]),
        prefix=tuple(ordered[:first]),
    )


def cluster(
    corpus: list[tuple[SessionTrace, list[TurnTrace]]], min_support: int
) -> tuple[list[Cluster], list[Skipped]]:
    groups: dict[tuple[tuple[str, ...], Shape], list[Segment]] = defaultdict(list)
    for session, turns in corpus:
        if session.outcome != "success":
            continue
        seg = segment(session, turns)
        if seg is not None:
            groups[(seg.signature, seg.shape)].append(seg)
    clusters: list[Cluster] = []
    skipped: list[Skipped] = []
    for (sig, shape), segs in groups.items():
        segs.sort(key=lambda s: s.session_id)
        if len(segs) >= min_support:
            clusters.append(Cluster(signature=sig, shape=shape, segments=tuple(segs)))
        else:
            skipped.append(
                Skipped(sig, shape, len(segs), f"support {len(segs)} < min_support {min_support}")
            )
    clusters.sort(key=lambda c: (-c.support, c.signature, c.shape))
    skipped.sort(key=lambda s: (-s.support, s.signature, s.shape))
    return clusters, skipped
