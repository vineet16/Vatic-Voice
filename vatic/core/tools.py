"""Typed tool registry with side-effect classes.

Tools are registered once and used by both the user's LLM agent (through
``TurnContext.call_tool``) and compiled flows, so calls are identical on both
paths. Synchronous handlers run on the runtime's bounded executor; async
handlers run on the loop and must not block.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import Executor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, ValidationError

from vatic.trace.schema import ToolCallRecord


class SideEffect(str, Enum):
    READ_ONLY = "read_only"
    REVERSIBLE = "reversible"
    IRREVERSIBLE = "irreversible"


@dataclass(frozen=True)
class ToolContext:
    session_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolInfo:
    """What the compiler and validator need to know about a tool (no handler)."""

    name: str
    side_effect: SideEffect
    params: tuple[str, ...]
    required: tuple[str, ...]
    invariants: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "side_effect": self.side_effect.value,
            "params": list(self.params),
            "required": list(self.required),
            "invariants": list(self.invariants),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> ToolInfo:
        return cls(
            name=str(data["name"]),
            side_effect=SideEffect(data["side_effect"]),
            params=tuple(data.get("params", ())),
            required=tuple(data.get("required", ())),
            invariants=tuple(data.get("invariants", ())),
        )


ToolCatalog = Mapping[str, ToolInfo]


class ToolError(Exception):
    """Raised by a handler for an expected, reportable failure (e.g. slot taken)."""


Handler = Callable[[Any, ToolContext], Any]


class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    side_effect: SideEffect
    invariants: list[str] = []
    handler: Handler | None = None

    def openai_schema(self) -> dict[str, Any]:
        params = self.input_schema.model_json_schema()
        params.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec] | None = None) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._schemas: list[dict[str, Any]] | None = None
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool {spec.name!r} already registered")
        self._specs[spec.name] = spec
        self._schemas = None

    def get(self, name: str) -> ToolSpec:
        return self._specs[name]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def names(self) -> list[str]:
        return list(self._specs)

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def catalog(self) -> dict[str, ToolInfo]:
        out = {}
        for s in self._specs.values():
            fields = s.input_schema.model_fields
            out[s.name] = ToolInfo(
                name=s.name,
                side_effect=s.side_effect,
                params=tuple(fields),
                required=tuple(n for n, f in fields.items() if f.is_required()),
                invariants=tuple(s.invariants),
            )
        return out

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Function-calling schemas (computed once; callers must not mutate them)."""
        if self._schemas is None:
            self._schemas = [s.openai_schema() for s in self._specs.values()]
        return list(self._schemas)

    async def call(
        self,
        name: str,
        args: Mapping[str, Any],
        ctx: ToolContext,
        *,
        executor: Executor | None = None,
        timeout: float = 10.0,
        call_id: str | None = None,
        step_id: str | None = None,
    ) -> ToolCallRecord:
        """Validate, execute and record one tool call. Never raises for tool failures."""
        started = time.time()
        spec = self._specs.get(name)
        output: dict[str, Any] | None = None
        error: str | None = None
        if spec is None or spec.handler is None:
            error = f"unknown tool {name!r}"
        else:
            try:
                parsed = spec.input_schema.model_validate(dict(args))
                result = await asyncio.wait_for(
                    _invoke(spec.handler, parsed, ctx, executor), timeout
                )
                if isinstance(result, BaseModel):
                    output = result.model_dump(mode="json")
                else:
                    output = spec.output_schema.model_validate(result).model_dump(mode="json")
            except ValidationError as exc:
                error = f"invalid arguments: {exc.errors(include_url=False)}"
            except ToolError as exc:
                error = str(exc) or type(exc).__name__
            except TimeoutError:
                error = f"timeout after {timeout}s"
        return ToolCallRecord(
            tool=name,
            args=dict(args),
            output=output,
            error=error,
            started_at=started,
            ended_at=time.time(),
            call_id=call_id,
            step_id=step_id,
        )


async def _invoke(
    handler: Handler, parsed: BaseModel, ctx: ToolContext, executor: Executor | None
) -> Any:
    if inspect.iscoroutinefunction(handler):
        return await handler(parsed, ctx)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(executor, handler, parsed, ctx)
    if isinstance(result, Awaitable):
        return await result
    return result
