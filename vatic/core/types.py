"""Runtime-facing types: what the user's pipeline sees."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from vatic.trace.schema import (
    Decision,
    GuardResult,
    MembershipResult,
    ToolCallRecord,
    TurnTimings,
)

__all__ = [
    "CompiledExchange",
    "Decision",
    "FlowContext",
    "GuardResult",
    "LLMFallback",
    "MembershipResult",
    "ResumableStep",
    "ToolCallRecord",
    "TurnContext",
    "TurnResult",
    "TurnTimings",
]


@dataclass
class CompiledExchange:
    """A turn the runtime answered without the LLM, replayed into the agent's history."""

    turn_index: int
    user: str
    agent_text: str
    tool_calls: list[ToolCallRecord]


class ResumableStep(BaseModel):
    step_id: str
    kind: Literal["ask", "confirm"]
    expects: list[str] = Field(default_factory=list)
    prompt: str = ""


class FlowContext(BaseModel):
    """Summary of a suspended flow, given to the LLM on fallback."""

    flow_id: str
    version: int
    description: str
    step_id: str
    reason: str
    slots: dict[str, str] = Field(default_factory=dict)
    resumable_steps: list[ResumableStep] = Field(default_factory=list)


class TurnResult(BaseModel):
    route: Literal["compiled", "llm"]
    text: str
    trace_id: str
    flow_id: str | None = None
    flow_step: str | None = None
    fallback_reason: str | None = None
    membership: MembershipResult | None = None


ToolDispatch = Callable[[str, dict[str, Any], str | None], Awaitable[dict[str, Any]]]


@dataclass
class TurnContext:
    """Passed to the user's ``llm_fallback``.

    The agent must execute every tool through :meth:`call_tool` so the call is
    traced and virtual tools (``enter_flow`` / ``resume_flow``) are handled.
    When :attr:`handed_off` becomes True the flow has taken over the turn: the
    agent should stop and return; its text is ignored in favour of
    :attr:`handoff_text`.
    """

    session_id: str
    transcript: str
    turn_index: int
    tools: list[dict[str, Any]]
    history_delta: list[CompiledExchange]
    flow: FlowContext | None
    metadata: dict[str, Any]
    _dispatch: ToolDispatch
    handed_off: bool = False
    handoff_text: str | None = None
    llm_calls: int = 0
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    async def call_tool(
        self, name: str, args: dict[str, Any], call_id: str | None = None
    ) -> dict[str, Any]:
        return await self._dispatch(name, args, call_id)

    def note_llm_call(self) -> None:
        self.llm_calls += 1


LLMFallback = Callable[[TurnContext], Awaitable[str]]
