"""Session and flow-state tracking."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Literal

from vatic.core.transforms import TransformContext
from vatic.core.types import CompiledExchange
from vatic.ir.schema import FlowGraph


@dataclass
class FlowState:
    flow: FlowGraph
    step_id: str  # step currently waiting for the caller (ask/confirm)
    slots: dict[str, str] = field(default_factory=dict)
    step_args: dict[str, dict[str, Any]] = field(default_factory=dict)
    step_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    status: Literal["waiting", "suspended"] = "waiting"
    entered_turn: int = 0
    suspended_turns: int = 0
    reason: str | None = None

    def namespace(self) -> dict[str, Any]:
        steps = {
            sid: {"args": self.step_args.get(sid, {}), "output": self.step_outputs.get(sid)}
            for sid in sorted(set(self.step_args) | set(self.step_outputs))
        }
        return {"slot": dict(self.slots), "steps": steps}


@dataclass
class Session:
    session_id: str
    metadata: dict[str, Any]
    started_at: float
    ctx: TransformContext
    turn_index: int = 0
    transcripts: list[str] = field(default_factory=list)
    flow: FlowState | None = None
    history_delta: list[CompiledExchange] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    shadow: dict[str, Any] = field(default_factory=dict)  # owned by lifecycle.shadow
    pending: Any = None  # the runtime's in-progress PendingTurn, if any
