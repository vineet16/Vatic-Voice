"""A Pipecat ``LLMService`` backed by the offline scripted clinic model (tests/demos)."""

from __future__ import annotations

from typing import Any

from pipecat.frames.frames import (
    Frame,
    FunctionCallFromLLM,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings

from examples.scheduling.scripted_llm import ScriptedClinicLLM


class ScriptedPipecatLLM(LLMService):
    def __init__(self, system_prompt: str, model: ScriptedClinicLLM | None = None, **kw: Any):
        unsupported = dict.fromkeys(
            [
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "filter_incomplete_user_turns",
                "user_turn_completion_config",
            ]
        )
        settings = LLMSettings(
            model="scripted-clinic", system_instruction=system_prompt, **unsupported
        )
        super().__init__(settings=settings, **kw)
        self.system_prompt = system_prompt
        self.scripted = model or ScriptedClinicLLM()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        try:
            params = self.get_llm_adapter().get_llm_invocation_params(
                frame.context, convert_developer_to_user=True
            )
            messages = [{"role": "system", "content": self.system_prompt}, *params["messages"]]
            tools = params.get("tools") or []
            resp = await self.scripted.complete(messages, tools if isinstance(tools, list) else [])
            if resp.text:
                await self.push_frame(LLMTextFrame(resp.text))
            if resp.tool_calls:
                await self.run_function_calls(
                    [
                        FunctionCallFromLLM(
                            context=frame.context,
                            tool_call_id=c.id,
                            function_name=c.name,
                            arguments=c.arguments,
                        )
                        for c in resp.tool_calls
                    ]
                )
        finally:
            await self.push_frame(LLMFullResponseEndFrame())
