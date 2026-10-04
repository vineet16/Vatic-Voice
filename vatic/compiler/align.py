"""Align the turns of a cluster's segments into one step sequence.

All segments in a cluster share the same shape, so alignment is positional:
for every turn, its tool calls become ``tool`` steps and the agent's reply
becomes an ``ask`` (the caller answers in the next turn), a ``confirm`` (the
caller's answer is a clean yes and the next turn runs a side-effecting tool),
or, on the last turn, a ``say``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vatic.compiler.cluster import Cluster
from vatic.core.step_check import classify_confirm
from vatic.core.tools import SideEffect, ToolCatalog

AlignedKind = Literal["tool", "ask", "confirm", "say"]

CONFIRM_YES_RATE = 0.9


@dataclass(frozen=True)
class AlignedStep:
    id: str
    kind: AlignedKind
    turn: int  # segment-relative turn index
    tool: str | None = None
    call: int = 0  # index of the call within the turn
    side_effect: SideEffect | None = None


def _side_effect(catalog: ToolCatalog, tool: str) -> SideEffect:
    info = catalog.get(tool)
    return info.side_effect if info is not None else SideEffect.IRREVERSIBLE


def align(cluster: Cluster, catalog: ToolCatalog) -> list[AlignedStep]:
    steps: list[AlignedStep] = []
    n_turns = len(cluster.shape)

    def next_id() -> str:
        return f"s{len(steps) + 1}"

    for k, tools in enumerate(cluster.shape):
        for j, tool in enumerate(tools):
            steps.append(
                AlignedStep(
                    next_id(), "tool", k, tool=tool, call=j, side_effect=_side_effect(catalog, tool)
                )
            )
        if k == n_turns - 1:
            steps.append(AlignedStep(next_id(), "say", k))
            continue
        kind: AlignedKind = "ask"
        acting = any(_side_effect(catalog, t) != SideEffect.READ_ONLY for t in cluster.shape[k + 1])
        if acting:
            answers = [seg.turns[k + 1].user_transcript for seg in cluster.segments]
            yes = sum(classify_confirm(a) == "yes" for a in answers)
            if yes / len(answers) >= CONFIRM_YES_RATE:
                kind = "confirm"
        steps.append(AlignedStep(next_id(), kind, k))
    return steps
