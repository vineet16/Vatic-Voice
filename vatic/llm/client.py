"""Provider interface for LLM access.

Used only by the reference agent, the caller simulator and optional phrasing
nodes. The compiler and the membership checks never import this module.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)


@runtime_checkable
class LLMClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse: ...


class LLMDisabledError(RuntimeError):
    pass


class DisabledLLMClient:
    """Raises on any call. Used to prove a code path never touches an LLM."""

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        raise LLMDisabledError("LLM access is disabled in this context")


class OpenAICompatibleClient:
    """Minimal async client for any OpenAI-compatible /chat/completions endpoint."""

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        max_retries: int = 6,
    ) -> None:
        self.max_retries = max_retries
        self.model = model or os.environ.get("VATIC_LLM_MODEL", "gpt-4o-mini")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self._http = httpx.AsyncClient(timeout=timeout)
        # Running totals across calls, for cost reporting.
        self.usage = {"requests": 0, "prompt_tokens": 0, "completion_tokens": 0, "retries": 0}

    async def aclose(self) -> None:
        await self._http.aclose()

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> LLMResponse:
        body: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            body["tools"] = tools
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature
        for attempt in range(self.max_retries + 1):
            resp = await self._http.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
            )
            if resp.status_code not in (429, 500, 502, 503) or attempt == self.max_retries:
                break
            self.usage["retries"] += 1
            await asyncio.sleep(_retry_delay(resp, attempt))
        resp.raise_for_status()
        data = resp.json()
        self.usage["requests"] += 1
        for k in ("prompt_tokens", "completion_tokens"):
            self.usage[k] += int((data.get("usage") or {}).get(k) or 0)
        msg = data["choices"][0]["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            fn = tc["function"]
            raw = fn.get("arguments") or "{}"
            calls.append(ToolCallRequest(id=tc["id"], name=fn["name"], arguments=json.loads(raw)))
        return LLMResponse(text=msg.get("content") or "", tool_calls=calls)


def _retry_delay(resp: httpx.Response, attempt: int) -> float:
    """Honour Retry-After (seconds) when the provider sends it; else back off."""
    raw = resp.headers.get("retry-after")
    try:
        return min(120.0, float(raw)) if raw else min(60.0, 2.0 * 2**attempt)
    except ValueError:
        return min(60.0, 2.0 * 2**attempt)


def default_client() -> LLMClient | None:
    """Return an OpenAI-compatible client if credentials are configured, else None."""
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleClient()
    return None
