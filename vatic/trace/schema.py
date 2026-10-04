"""Framework-neutral trace records. One TurnTrace per user turn."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Route = Literal["llm", "compiled", "shadow"]
Decision = Literal["on_path", "off_path", "uncertain"]
SessionOutcome = Literal["success", "failure", "abandoned"]

# Tools injected by the runtime; never part of a compiled flow.
VIRTUAL_TOOLS = frozenset({"enter_flow", "resume_flow"})


class ToolCallRecord(BaseModel):
    tool: str
    args: dict[str, Any]
    output: dict[str, Any] | None = None
    error: str | None = None
    started_at: float
    ended_at: float
    call_id: str | None = None
    step_id: str | None = None  # set when the call was made by a compiled step


class GuardResult(BaseModel):
    expr: str
    kind: Literal["declared", "learned"]
    passed: bool
    step_id: str | None = None
    detail: str | None = None


class MembershipResult(BaseModel):
    decision: Decision  # final decision (rules + classifier)
    reason: str
    rules: Decision | None = None  # layers 1-3 alone
    slots: dict[str, str] = Field(default_factory=dict)
    residual: list[str] = Field(default_factory=list)
    classifier_score: float | None = None
    hedged: bool = False
    latency_ms: float = 0.0


class TurnTimings(BaseModel):
    stt_end: float | None = None
    decision_start: float | None = None
    decision_end: float | None = None
    first_text: float | None = None
    tts_first_audio: float | None = None
    turn_end: float | None = None


class ShadowComparison(BaseModel):
    flow_id: str
    flow_version: int
    step_id: str | None
    decision: Decision | None  # None on the entry turn
    compared: bool  # False when the compiled flow would have fallen back
    matched: bool
    irreversible: bool
    run_ended: bool = False  # the shadow run reached the flow's end on this turn
    reasons: list[str] = Field(default_factory=list)
    expected_tools: list[dict[str, Any]] = Field(default_factory=list)


class TurnTrace(BaseModel):
    trace_id: str
    session_id: str
    turn_index: int
    user_transcript: str
    route: Route
    flow_id: str | None = None
    flow_version: int | None = None
    flow_step: str | None = None
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    agent_text: str = ""
    guard_results: list[GuardResult] = Field(default_factory=list)
    timings: TurnTimings = Field(default_factory=TurnTimings)
    outcome: Literal["ok", "fallback", "error"] | None = None
    fallback_reason: str | None = None
    membership: MembershipResult | None = None
    llm_calls: int = 0
    shadow: ShadowComparison | None = None


class SessionTrace(BaseModel):
    session_id: str
    outcome: SessionOutcome | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    started_at: float
    ended_at: float | None = None
    n_turns: int = 0


class LifecycleEvent(BaseModel):
    flow_id: str
    version: int
    from_status: str
    to_status: str
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    at: float
    manual: bool = False
