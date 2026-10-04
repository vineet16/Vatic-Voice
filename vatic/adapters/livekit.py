"""LiveKit Agents adapter.

Pinned against ``livekit-agents==1.8.4`` (checked against its source: ``Agent.llm_node``,
``Agent.default.llm_node``, ``llm.function_tool(raw_schema=...)``, ``ChatContext`` items).

``VaticAgent`` overrides the agent's LLM step. Each user turn first goes to
``VaticRuntime.begin_turn``: a compiled reply is yielded straight to TTS; otherwise
the turn is delegated to LiveKit's default LLM node, and LiveKit runs its usual
tool loop. Every tool (domain tools plus ``enter_flow`` / ``resume_flow``) is
registered on the agent and executes through the runtime, so calls are traced and
flow hand-offs are honoured.

    agent = VaticAgent(runtime=runtime, session_id=ctx.room.name,
                       session_metadata={"today": "2026-10-05"}, instructions=PROMPT)
    await AgentSession(stt=..., llm=..., tts=..., vad=...).start(agent, room=ctx.room)
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterable
from typing import Any

from livekit.agents import Agent, ModelSettings, llm

from vatic.core.runtime import PendingTurn, VaticRuntime
from vatic.core.types import TurnResult
from vatic.trace.schema import ToolCallRecord

_OPEN_SCHEMA = {"type": "object", "properties": {}, "additionalProperties": True}


def _raw_tool(name: str, description: str, parameters: dict[str, Any], fn: Any) -> llm.Tool:
    schema = {"name": name, "description": description, "parameters": parameters}
    return llm.function_tool(fn, raw_schema=schema)


def _items(rec: ToolCallRecord, call_id: str) -> list[llm.ChatItem]:
    output = rec.output if rec.error is None else {"error": rec.error}
    return [
        llm.FunctionCall(
            call_id=call_id,
            name=rec.tool,
            arguments=json.dumps(rec.args),
            created_at=rec.started_at,
        ),
        llm.FunctionCallOutput(
            call_id=call_id,
            name=rec.tool,
            output=json.dumps(output),
            is_error=rec.error is not None,
            created_at=rec.ended_at,
        ),
    ]


class VaticAgent(Agent):
    def __init__(
        self,
        *,
        runtime: VaticRuntime,
        session_id: str,
        session_metadata: dict[str, Any] | None = None,
        **agent_kwargs: Any,
    ) -> None:
        tools = [*agent_kwargs.pop("tools", []), *self._registered_tools(runtime)]
        super().__init__(tools=tools, **agent_kwargs)
        self._vatic = runtime
        self._sid = session_id
        self._pending: PendingTurn | None = None
        self._user_at = 0.0  # when the pending turn's user message arrived
        # Tool calls LiveKit's own chat history never saw (made by compiled steps or
        # by a flow before it fell back); merged into every LLM request.
        self._extras: list[llm.ChatItem] = []
        runtime.start_session(session_id, session_metadata)

    def _registered_tools(self, runtime: VaticRuntime) -> list[llm.Tool]:
        def make(name: str) -> llm.Tool:
            async def call(raw_arguments: dict[str, object]) -> str:
                if self._pending is None:
                    return json.dumps({"error": "no active turn"})
                return json.dumps(await self._pending.ctx.call_tool(name, dict(raw_arguments)))

            return _raw_tool(name, f"Vatic tool {name}", _OPEN_SCHEMA, call)

        return [make(n) for n in [*runtime.tools.names(), "enter_flow", "resume_flow"]]

    def _turn_tools(self, pending: PendingTurn) -> list[llm.Tool]:
        """This turn's exact schemas (flow list, resumable steps) for the model to see.

        Execution is by name, through the tools registered on the agent.
        """

        async def unused(raw_arguments: dict[str, object]) -> None:
            return None

        return [
            _raw_tool(
                f["name"], f.get("description", ""), f.get("parameters", _OPEN_SCHEMA), unused
            )
            for f in (s["function"] for s in pending.ctx.tools)
        ]

    def _absorb(self, pending: PendingTurn) -> None:
        for ex in pending.ctx.history_delta:
            for i, rec in enumerate(ex.tool_calls):
                self._extras += _items(rec, f"vatic_{ex.turn_index}_{i}")
        for i, rec in enumerate(pending.ctx.tool_calls):
            self._extras += _items(rec, f"vatic_{pending.ctx.turn_index}_p{i}")

    async def llm_node(
        self, chat_ctx: llm.ChatContext, tools: list[llm.Tool], model_settings: ModelSettings
    ) -> AsyncIterable[llm.ChatChunk | str]:
        last = chat_ctx.items[-1] if chat_ctx.items else None
        continuing = self._pending is not None and isinstance(last, llm.FunctionCallOutput)
        if not continuing:
            user = next(
                (
                    m
                    for m in reversed(chat_ctx.items)
                    if isinstance(m, llm.ChatMessage) and m.role == "user"
                ),
                None,
            )
            began = await self._vatic.begin_turn(self._sid, (user and user.text_content) or "")
            if isinstance(began, TurnResult):
                yield began.text  # compiled: straight to TTS, no LLM call
                return
            self._user_at = user.created_at if user else 0.0
            self._pending = began
            self._absorb(began)
        pending = self._pending
        assert pending is not None
        if pending.ctx.handed_off:  # enter_flow succeeded in the previous step
            self._pending = None
            yield (await self._vatic.complete_turn(pending, "")).text
            return

        request = chat_ctx.copy()
        request.insert(self._extras)
        if pending.ctx.flow is not None:  # paused-flow state, placed just before the user's turn
            note = "[vatic-flow] " + pending.ctx.flow.model_dump_json()
            request.add_message(role="system", content=note, created_at=self._user_at - 1e-6)
        text: list[str] = []
        called_tools = False
        pending.ctx.note_llm_call()
        stream = Agent.default.llm_node(self, request, self._turn_tools(pending), model_settings)
        async for chunk in stream:
            if isinstance(chunk, llm.ChatChunk) and chunk.delta is not None:
                called_tools = called_tools or bool(chunk.delta.tool_calls)
                text.append(chunk.delta.content or "")
            elif isinstance(chunk, str):
                text.append(chunk)
            yield chunk
        if not called_tools:  # final step of the LLM's turn
            self._pending = None
            await self._vatic.complete_turn(pending, "".join(text))
