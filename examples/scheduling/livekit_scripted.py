"""A LiveKit ``llm.LLM`` backed by the offline scripted clinic model (for tests/demos)."""

from __future__ import annotations

import json
import uuid
from typing import Any

from livekit.agents import APIConnectOptions, llm
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr

from examples.scheduling.scripted_llm import ScriptedClinicLLM


def _schema(tool: llm.Tool) -> dict[str, Any]:
    raw = getattr(tool, "info", None)
    raw_schema = getattr(raw, "raw_schema", None)
    if raw_schema is None:
        raise TypeError("ScriptedLiveKitLLM only supports raw-schema tools")
    return {"type": "function", "function": raw_schema}


class _Stream(llm.LLMStream):
    def __init__(self, owner: ScriptedLiveKitLLM, **kw: Any) -> None:
        super().__init__(owner, **kw)
        self._owner = owner

    async def _run(self) -> None:
        messages, _ = self._chat_ctx.to_provider_format("openai")
        tools = [_schema(t) for t in self._tools]
        resp = await self._owner.scripted.complete(messages, tools)
        cid = f"chunk_{uuid.uuid4().hex[:8]}"
        if resp.text:
            delta = llm.ChoiceDelta(role="assistant", content=resp.text)
            self._event_ch.send_nowait(llm.ChatChunk(id=cid, delta=delta))
        if resp.tool_calls:
            calls = [
                llm.FunctionToolCall(name=c.name, arguments=json.dumps(c.arguments), call_id=c.id)
                for c in resp.tool_calls
            ]
            delta = llm.ChoiceDelta(role="assistant", tool_calls=calls)
            self._event_ch.send_nowait(llm.ChatChunk(id=cid, delta=delta))


class ScriptedLiveKitLLM(llm.LLM):
    def __init__(self, model: ScriptedClinicLLM | None = None) -> None:
        super().__init__()
        self.scripted = model or ScriptedClinicLLM()

    @property
    def model(self) -> str:
        return "scripted-clinic"

    def chat(
        self,
        *,
        chat_ctx: llm.ChatContext,
        tools: list[llm.Tool] | None = None,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        parallel_tool_calls: NotGivenOr[bool] = NOT_GIVEN,
        tool_choice: NotGivenOr[llm.ToolChoice] = NOT_GIVEN,
        extra_kwargs: NotGivenOr[dict[str, Any]] = NOT_GIVEN,
    ) -> llm.LLMStream:
        return _Stream(self, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)
