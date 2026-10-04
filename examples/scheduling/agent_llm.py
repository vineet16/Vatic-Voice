"""Baseline clinic agent: a plain tool-calling LLM loop.

Works with any ``LLMClient`` (OpenAI-compatible or the offline scripted model).
Used as Vatic's ``llm_fallback``: every tool goes through ``ctx.call_tool`` so
calls are traced, and compiled turns arrive via ``ctx.history_delta`` so the
conversation history stays complete.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from vatic.core.types import CompiledExchange, TurnContext
from vatic.llm.client import LLMClient
from vatic.trace.schema import ToolCallRecord

SYSTEM_PROMPT = """\
You are the phone receptionist for Riverside Family Clinic. Keep replies short and spoken-style.
You can book, reschedule and cancel appointments, and answer simple questions.
Clinic hours: Monday to Friday, 8 AM to 5 PM. Address: 120 River Road. Most insurance accepted.
Rules:
- Always look up the patient by full name before acting.
- Convert dates to YYYY-MM-DD and times to 24h HH:MM for tools. Today's date is {today}.
- Offer only times returned by check_availability.
- Before book_appointment or cancel_appointment, read the details back and get a clear yes.
- If an enter_flow tool is available, check it BEFORE calling any other tool: when the
  caller's request matches one of its flows and the caller has said every entry slot, call
  enter_flow (with the caller's exact words) instead of looking anything up yourself, then
  reply with an empty message. Only if it is rejected, continue normally.
- If a resume_flow tool is available, call it when you are about to ask one of its steps.
"""


def _tool_messages(calls: list[ToolCallRecord], prefix: str) -> list[dict[str, Any]]:
    if not calls:
        return []
    ids = [c.call_id or f"{prefix}_{i}" for i, c in enumerate(calls)]
    msgs: list[dict[str, Any]] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": cid,
                    "type": "function",
                    "function": {"name": c.tool, "arguments": json.dumps(c.args)},
                }
                for cid, c in zip(ids, calls, strict=True)
            ],
        }
    ]
    for cid, c in zip(ids, calls, strict=True):
        content = c.output if c.error is None else {"error": c.error}
        msgs.append({"role": "tool", "tool_call_id": cid, "content": json.dumps(content)})
    return msgs


def exchange_messages(ex: CompiledExchange) -> list[dict[str, Any]]:
    return [
        {"role": "user", "content": ex.user},
        *_tool_messages(ex.tool_calls, f"c{ex.turn_index}"),
        {"role": "assistant", "content": ex.agent_text},
    ]


class ClinicAgent:
    def __init__(self, llm: LLMClient, max_rounds: int = 6) -> None:
        self.llm = llm
        self.max_rounds = max_rounds
        self._histories: dict[str, list[dict[str, Any]]] = {}

    def end_session(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    async def __call__(self, ctx: TurnContext) -> str:
        msgs = self._histories.get(ctx.session_id)
        if msgs is None:
            today = ctx.metadata.get("today", "unknown")
            msgs = [{"role": "system", "content": SYSTEM_PROMPT.format(today=today)}]
            self._histories[ctx.session_id] = msgs
        start = len(msgs)
        try:
            return await self._turn(ctx, msgs)
        except asyncio.CancelledError:
            # The runtime answered this turn from a compiled flow (hedge won): forget it.
            del msgs[start:]
            raise

    async def _turn(self, ctx: TurnContext, msgs: list[dict[str, Any]]) -> str:
        for ex in ctx.history_delta:
            msgs.extend(exchange_messages(ex))
        if ctx.flow is not None:
            msgs.append({"role": "system", "content": "[vatic-flow] " + ctx.flow.model_dump_json()})
        msgs.append({"role": "user", "content": ctx.transcript})
        # Calls a paused flow already made this turn, before it fell back to us.
        msgs.extend(_tool_messages(list(ctx.tool_calls), f"p{ctx.turn_index}"))

        for _ in range(self.max_rounds):
            ctx.note_llm_call()
            resp = await self.llm.complete(msgs, tools=ctx.tools)
            if not resp.tool_calls:
                msgs.append({"role": "assistant", "content": resp.text})
                return resp.text
            msgs.append(
                {
                    "role": "assistant",
                    "content": resp.text or None,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                        }
                        for tc in resp.tool_calls
                    ],
                }
            )
            for tc in resp.tool_calls:
                out = await ctx.call_tool(tc.name, tc.arguments, tc.id)
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(out)})
            if ctx.handed_off:
                msgs.append({"role": "assistant", "content": ctx.handoff_text or ""})
                return ""
        fallback = "Sorry, could you say that again?"
        msgs.append({"role": "assistant", "content": fallback})
        return fallback
