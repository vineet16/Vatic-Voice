"""Step membership: does this utterance belong to the current step?

Never calls an LLM. Layers (Section 6.6 of the spec):

1. slot extraction   - each expected slot must be found exactly once
2. residual check    - after removing slot spans and filler, few content tokens remain
3. marker rule       - negation / correction / multi-intent / question markers -> off_path
4. classifier        - optional per-step cross-encoder (see ``classifier.py``), applied
                       by the runtime after layers 1-3 pass.

Layers 1-3 are lexicon/regex work measured in microseconds and run inline.
"""

from __future__ import annotations

import time
from typing import Literal

from vatic.core.extract import Span, Token, extract, tokenize
from vatic.ir.schema import AskStep, FlowGraph
from vatic.trace.schema import MembershipResult

FILLER_PHRASES: tuple[tuple[str, ...], ...] = tuple(
    tuple(p.split())
    for p in [
        "that works",
        "that would be great",
        "would be great",
        "sounds good",
        "thank you",
        "i would like",
        "i'd like",
        "let's do",
        "let's say",
        "how about",
        "my name is",
        "my name's",
        "this is",
        "it is",
        "in the morning",
        "in the afternoon",
        "in the evening",
        "if possible",
        "if that's okay",
        "works for me",
        "for me",
        "all right",
        "please",
        "that's fine",
    ]
)

FILLER_WORDS = frozenset(
    """um uh er erm hmm mm yeah yes yep ok okay alright please thanks thank so well then
    just oh hi hello hey sure great perfect fine good cool lovely awesome that that's this
    it it's its is be would works work sounds i i'd i'm me my we let's lets do say like
    to the a an at on for around about by in of with maybe probably think guess prefer
    want need ideally name name's called here morning afternoon evening o'clock am pm
    you whatever's""".split()
)

DISFLUENCIES = frozenset("um uh er erm hmm mm uhm".split())

NEGATION_WORDS = frozenset(
    """no not don't dont nope never can't cannot won't isn't doesn't actually instead
    wait rather change different wrong sorry nevermind""".split()
)
NEGATION_PHRASES = (("never", "mind"), ("hold", "on"), ("scratch", "that"))
MULTI_INTENT_WORDS = frozenset("also another plus additionally but except".split())
MULTI_INTENT_PHRASES = (("as", "well"), ("and", "then"), ("one", "more"))
QUESTION_WORDS = frozenset("what when where why how which who".split())
QUESTION_PHRASES = (
    ("do", "you"),
    ("can", "you"),
    ("could", "you"),
    ("is", "there"),
    ("are", "you"),
    ("does", "it"),
    ("will", "i"),
    ("should", "i"),
    ("do", "i"),
)

YES_WORDS = frozenset(
    "yes yeah yep yup sure correct right absolutely definitely ok okay alright perfect".split()
)
YES_PHRASES = (
    ("go", "ahead"),
    ("sounds", "good"),
    ("that", "works"),
    ("do", "it"),
    ("book", "it"),
    ("please", "do"),
    ("that's", "right"),
    ("that's", "correct"),
    ("that's", "perfect"),
)
NO_WORDS = frozenset("no nope".split())


def _outside(tokens: list[Token], spans: list[Span]) -> list[Token]:
    return [t for t in tokens if not any(s.start <= t.start and t.end <= s.end for s in spans)]


def _strip_phrases(words: list[str], phrases: tuple[tuple[str, ...], ...]) -> list[str]:
    out: list[str] = []
    i = 0
    ordered = sorted(phrases, key=len, reverse=True)
    while i < len(words):
        for p in ordered:
            if tuple(words[i : i + len(p)]) == p:
                i += len(p)
                break
        else:
            out.append(words[i])
            i += 1
    return out


def _has_phrase(words: list[str], phrases: tuple[tuple[str, ...], ...]) -> tuple[str, ...] | None:
    for p in phrases:
        for i in range(len(words) - len(p) + 1):
            if tuple(words[i : i + len(p)]) == p:
                return p
    return None


def content_residual(words: list[str]) -> list[str]:
    stripped = _strip_phrases(words, FILLER_PHRASES)
    return [w for w in stripped if w not in FILLER_WORDS]


def find_marker(words: list[str]) -> str | None:
    """Return the first negation / multi-intent / question marker, if any."""
    for w in words:
        if w in NEGATION_WORDS:
            return f"negation:{w}"
    if p := _has_phrase(words, NEGATION_PHRASES):
        return "negation:" + " ".join(p)
    for w in words:
        if w in MULTI_INTENT_WORDS:
            return f"multi_intent:{w}"
    if p := _has_phrase(words, MULTI_INTENT_PHRASES):
        return "multi_intent:" + " ".join(p)
    for w in words:
        if w in QUESTION_WORDS:
            return f"question:{w}"
    if p := _has_phrase(words, QUESTION_PHRASES):
        return "question:" + " ".join(p)
    return None


def check_ask(step: AskStep, flow: FlowGraph, transcript: str) -> MembershipResult:
    """Run membership layers 1-3 for an ask step."""
    t0 = time.perf_counter()

    def done(
        decision: Literal["on_path", "off_path"], reason: str, **kw: object
    ) -> MembershipResult:
        return MembershipResult(
            decision=decision,
            reason=reason,
            latency_ms=(time.perf_counter() - t0) * 1000,
            **kw,
        )

    tokens = tokenize(transcript)
    spans: list[Span] = []
    slots: dict[str, str] = {}
    # Layer 1: slot extraction.
    for name in step.expects:
        slot_def = flow.slots.get(name)
        if slot_def is None:
            return done("off_path", f"undefined_slot:{name}")
        found = extract(slot_def.extractor, transcript)
        if not found:
            return done("off_path", f"missing_slot:{name}")
        if len({s.text.lower() for s in found}) > 1:
            return done("off_path", f"ambiguous_slot:{name}", slots=slots)
        spans.append(found[0])
        slots[name] = found[0].text
    # Layer 2: residual content tokens.
    outside = [t.text for t in _outside(tokens, spans) if t.text not in DISFLUENCIES]
    residual = content_residual(outside)
    if len(residual) > step.membership.residual_max_tokens:
        return done("off_path", "residual", slots=slots, residual=residual)
    # Layer 3: negation / correction / multi-intent / question markers.
    marker = find_marker(outside)
    if marker is not None:
        return done("off_path", marker, slots=slots, residual=residual)
    if not step.expects and residual:
        return done("off_path", "residual", slots=slots, residual=residual)
    return done("on_path", "rules", slots=slots, residual=residual)


ConfirmLabel = Literal["yes", "no", "other"]


def classify_confirm(transcript: str) -> ConfirmLabel:
    """Deterministic yes/no/other for confirm steps. Anything unclear is 'other'."""
    words = [t.text for t in tokenize(transcript) if t.text not in DISFLUENCIES]
    if not words:
        return "other"
    has_no = any(w in NO_WORDS for w in words)
    marker = find_marker([w for w in words if w not in NO_WORDS])
    rest = _strip_phrases(words, YES_PHRASES)
    has_yes = any(w in YES_WORDS for w in words) or len(rest) < len(words)
    residual = [w for w in content_residual(rest) if w not in YES_WORDS and w not in NO_WORDS]
    if has_no and not has_yes and marker is None and len(residual) <= 1:
        return "no"
    if has_yes and not has_no and marker is None and not residual:
        return "yes"
    return "other"
