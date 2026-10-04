"""Hand-built STT -> LLM -> TTS pipeline helper.

adapter = PipelineAdapter(runtime, llm_fallback=my_agent_turn)
adapter.mark_stt_end(session_id)                      # when STT finalises the turn
async for chunk in adapter.on_transcript(session_id, text):
    await tts.speak(chunk)                            # send each chunk to TTS
adapter.mark_first_audio(session_id)                  # when TTS emits first audio
"""

from __future__ import annotations

import re
import time
from collections.abc import AsyncIterator

from vatic.core.runtime import VaticRuntime
from vatic.core.types import LLMFallback, TurnResult
from vatic.trace.schema import TurnTimings

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_for_tts(text: str) -> list[str]:
    return [s for s in _SENTENCE_END.split(text.strip()) if s]


class PipelineAdapter:
    def __init__(self, runtime: VaticRuntime, llm_fallback: LLMFallback) -> None:
        self.runtime = runtime
        self.llm_fallback = llm_fallback
        self._stt_end: dict[str, float] = {}
        self._last: dict[str, TurnResult] = {}

    def mark_stt_end(self, session_id: str, ts: float | None = None) -> None:
        self._stt_end[session_id] = ts if ts is not None else time.time()

    async def handle(self, session_id: str, text: str) -> TurnResult:
        timings = TurnTimings(stt_end=self._stt_end.pop(session_id, None))
        result = await self.runtime.handle_turn(
            session_id, text, self.llm_fallback, timings=timings
        )
        self._last[session_id] = result
        return result

    async def on_transcript(self, session_id: str, text: str) -> AsyncIterator[str]:
        """Yield text chunks (sentences) for TTS."""
        result = await self.handle(session_id, text)
        for chunk in split_for_tts(result.text):
            yield chunk

    def last_result(self, session_id: str) -> TurnResult | None:
        return self._last.get(session_id)

    def mark_first_audio(self, session_id: str, ts: float | None = None) -> None:
        last = self._last.get(session_id)
        if last is not None:
            self.runtime.annotate_turn(
                last.trace_id, tts_first_audio=ts if ts is not None else time.time()
            )

    def mark_turn_end(self, session_id: str, ts: float | None = None) -> None:
        last = self._last.get(session_id)
        if last is not None:
            self.runtime.annotate_turn(
                last.trace_id, turn_end=ts if ts is not None else time.time()
            )
