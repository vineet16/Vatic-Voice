"""Pipecat adapter.

Pinned against ``pipecat-ai==1.12.0`` (checked against its source: universal
``LLMContext`` / ``LLMContextFrame``, ``LLMService.register_function``,
``FunctionCallParams``, ``FunctionCallResultProperties(run_llm=...)``).

    vatic = VaticPipecat(runtime, session_id="call-1", session_metadata={"today": ...})
    vatic.register_functions(llm)
    pipeline = Pipeline([transport.input(), stt, pair.user(), vatic.input(), llm,
                         vatic.output(), tts, transport.output(), pair.assistant()])

``input()`` sits between the user context aggregator and the LLM. On a completed
user turn it asks the runtime first: a compiled reply is pushed downstream as LLM
text (TTS speaks it, the assistant aggregator records it) and the LLM is not run;
otherwise the context frame passes through with this turn's tools (including
``enter_flow`` / ``resume_flow``) set on the context. ``output()`` sits after the
LLM and closes the turn when a response ends without tool calls. Tool handlers run
through the runtime, so every call is traced.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    Frame,
    FunctionCallResultProperties,
    FunctionCallsStartedFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import FunctionCallParams, LLMService

from vatic.core.runtime import PendingTurn, VaticRuntime
from vatic.core.types import TurnResult
from vatic.trace.schema import ToolCallRecord

_FLOW_NOTE = "[vatic-flow] "


def _user_text(context: LLMContext) -> str:
    for m in reversed(context.get_messages()):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content or "")
    return ""


def _tool_messages(calls: list[ToolCallRecord], prefix: str) -> list[dict[str, Any]]:
    if not calls:
        return []
    ids = [f"{prefix}_{i}" for i in range(len(calls))]
    fn = [{"name": c.tool, "arguments": json.dumps(c.args)} for c in calls]
    msgs: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": i, "type": "function", "function": f} for i, f in zip(ids, fn, strict=True)
            ],
        },
    ]
    for i, c in zip(ids, calls, strict=True):
        out = c.output if c.error is None else {"error": c.error}
        msgs.append({"role": "tool", "tool_call_id": i, "content": json.dumps(out)})
    return msgs


class VaticPipecat:
    def __init__(
        self,
        runtime: VaticRuntime,
        session_id: str,
        session_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.runtime = runtime
        self.session_id = session_id
        self.pending: PendingTurn | None = None
        self.results: asyncio.Queue[TurnResult] = asyncio.Queue()  # every finished turn
        self._context: LLMContext | None = None
        runtime.start_session(session_id, session_metadata)
        self._input = _VaticInput(self)
        self._output = _VaticOutput(self)

    def input(self) -> FrameProcessor:
        return self._input

    def output(self) -> FrameProcessor:
        return self._output

    def register_functions(self, llm: LLMService) -> None:
        for name in [*self.runtime.tools.names(), "enter_flow", "resume_flow"]:
            llm.register_function(name, self._handle_function)

    async def _handle_function(self, params: FunctionCallParams) -> None:
        pending = self.pending
        if pending is None:
            await params.result_callback({"error": "no active turn"})
            return
        out = await pending.ctx.call_tool(
            params.function_name, dict(params.arguments), params.tool_call_id
        )
        if not pending.ctx.handed_off:
            await params.result_callback(out)
            return
        # A flow took over: speak its reply instead of running the LLM again.
        await params.llm.push_frame(LLMTextFrame(pending.ctx.handoff_text or ""))
        await params.result_callback(out, properties=FunctionCallResultProperties(run_llm=False))
        await self._complete(pending, "")

    async def _begin(self, frame: LLMContextFrame) -> bool:
        """Start a turn. Returns True if the LLM should run."""
        context = frame.context
        self._context = context
        began = await self.runtime.begin_turn(self.session_id, _user_text(context))
        if isinstance(began, TurnResult):
            await self._input.push_frame(LLMFullResponseStartFrame())
            await self._input.push_frame(LLMTextFrame(began.text))
            await self._input.push_frame(LLMFullResponseEndFrame())
            self.results.put_nowait(began)
            return False
        self.pending = began
        ctx = began.ctx
        # History the framework never saw: tool calls made by compiled steps, by a
        # flow before it fell back, and the paused flow's state.
        extra: list[dict[str, Any]] = []
        for ex in ctx.history_delta:
            extra += _tool_messages(ex.tool_calls, f"vatic_{ex.turn_index}")
        extra += _tool_messages(ctx.tool_calls, f"vatic_{ctx.turn_index}_p")
        if ctx.flow is not None:
            extra.append({"role": "system", "content": _FLOW_NOTE + ctx.flow.model_dump_json()})
        if extra:
            messages = list(context.get_messages())
            context.set_messages([*messages[:-1], *extra, messages[-1]])
        tools = [s["function"] for s in ctx.tools]
        context.set_tools(
            ToolsSchema(
                standard_tools=[
                    FunctionSchema(
                        name=t["name"],
                        description=t.get("description", ""),
                        properties=t["parameters"].get("properties", {}),
                        required=t["parameters"].get("required", []),
                    )
                    for t in tools
                ]
            )
        )
        return True

    async def _complete(self, pending: PendingTurn, text: str) -> None:
        if self.pending is not pending:
            return
        self.pending = None
        if self._context is not None:  # the flow note was only for this turn
            kept = [
                m
                for m in self._context.get_messages()
                if not (isinstance(m, dict) and str(m.get("content") or "").startswith(_FLOW_NOTE))
            ]
            self._context.set_messages(kept)
        self.results.put_nowait(await self.runtime.complete_turn(pending, text))


class _VaticInput(FrameProcessor):
    def __init__(self, owner: VaticPipecat) -> None:
        super().__init__(name="VaticInput")
        self._owner = owner

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if (
            isinstance(frame, LLMContextFrame)
            and direction == FrameDirection.DOWNSTREAM
            and not await self._owner._begin(frame)
        ):
            return  # answered by the compiled flow: the LLM does not run
        await self.push_frame(frame, direction)


class _VaticOutput(FrameProcessor):
    def __init__(self, owner: VaticPipecat) -> None:
        super().__init__(name="VaticOutput")
        self._owner = owner
        self._text: list[str] = []
        self._called_tools = False

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, LLMFullResponseStartFrame):
            self._text, self._called_tools = [], False
        elif isinstance(frame, LLMTextFrame):
            self._text.append(frame.text)
        elif isinstance(frame, FunctionCallsStartedFrame):
            self._called_tools = True
        elif isinstance(frame, LLMFullResponseEndFrame):
            pending = self._owner.pending
            if pending is not None and not self._called_tools:
                await self._owner._complete(pending, "".join(self._text))
        await self.push_frame(frame, direction)
