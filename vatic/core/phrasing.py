"""Template rendering and the optional small-model phrasing node.

Template syntax: literal text with ``{ref|filter|...}`` placeholders, where
``ref`` is a namespace path (``slot.time``, ``steps.s1.output.patient.first_name``)
and filters are transforms or formatters applied left to right. Literal braces
are written ``{{`` / ``}}``.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from vatic.core.formatters import FORMATTERS
from vatic.core.paths import PathError, resolve
from vatic.core.transforms import TRANSFORMS, TransformContext
from vatic.ir.schema import LLMPhrase
from vatic.llm.client import LLMClient

_TOKEN = re.compile(r"\{\{|\}\}|\{([^{}]+)\}")


class RenderError(Exception):
    pass


@dataclass(frozen=True)
class Rendered:
    text: str
    values: list[str]  # rendered placeholder values, in order


def apply_filters(value: Any, filters: list[str], ctx: TransformContext) -> str:
    for f in filters:
        if f in TRANSFORMS:
            value = TRANSFORMS[f](value, ctx)
        elif f in FORMATTERS:
            value = FORMATTERS[f](value)
        else:
            raise RenderError(f"unknown filter {f!r}")
        if value is None:
            raise RenderError(f"filter {f!r} could not handle the value")
    if not isinstance(value, str):
        formatted = FORMATTERS["raw"](value)
        if formatted is None:
            raise RenderError("value is not renderable")
        return formatted
    return value


def render(template: str, namespace: Mapping[str, Any], ctx: TransformContext) -> Rendered:
    values: list[str] = []

    def sub(m: re.Match[str]) -> str:
        tok = m.group(0)
        if tok == "{{":
            return "{"
        if tok == "}}":
            return "}"
        ref, *filters = m.group(1).split("|")
        try:
            value = resolve(namespace, ref.strip())
        except PathError as exc:
            raise RenderError(f"missing value for {ref!r}") from exc
        out = apply_filters(value, filters, ctx)
        values.append(out)
        return out

    return Rendered(_TOKEN.sub(sub, template), values)


def placeholders(template: str) -> list[str]:
    return [m.group(1) for m in _TOKEN.finditer(template) if m.group(1)]


async def phrase_with_llm(
    client: LLMClient,
    spec: LLMPhrase,
    namespace: Mapping[str, Any],
    *,
    timeout: float = 2.0,
) -> str:
    """Small-model phrasing node: strict inputs, bounded length."""
    inputs = {}
    for ref in spec.inputs:
        try:
            inputs[ref] = resolve(namespace, ref)
        except PathError:
            inputs[ref] = None
    messages = [
        {
            "role": "system",
            "content": (
                "You write one short reply for a phone agent. Use only the given values. "
                f"Intent: {spec.intent}. Example: {spec.example!r}. "
                f"At most {spec.max_tokens} tokens."
            ),
        },
        {"role": "user", "content": json.dumps(inputs, default=str, sort_keys=True)},
    ]
    resp = await asyncio.wait_for(
        client.complete(messages, max_tokens=spec.max_tokens, temperature=0), timeout
    )
    return resp.text.strip()
