"""Deterministic slot extractors.

Each extractor returns the spans (raw text + offsets) it recognises. Spans keep
the caller's words; normalisation to canonical values happens in transforms.
Extractors are regex/lexicon based and run in microseconds, so the runtime
calls them inline on the event loop.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from vatic.core.transforms import MONTHS, NUMBER_WORDS, ORDINAL_WORDS, WEEKDAYS


@dataclass(frozen=True)
class Span:
    extractor: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


_TOKEN_RE = re.compile(r"\d{1,2}:\d{2}|[a-z0-9]+(?:['’][a-z]+)*", re.IGNORECASE)


def tokenize(text: str) -> list[Token]:
    return [
        Token(m.group(0).lower().replace("’", "'"), m.start(), m.end())
        for m in _TOKEN_RE.finditer(text)
    ]


def normalize_text(text: str) -> str:
    """Casefold and collapse punctuation/whitespace, for value comparison."""
    return " ".join(t.text for t in tokenize(text))


def _select(spans: list[Span]) -> list[Span]:
    """Leftmost-longest, non-overlapping."""
    spans = sorted(spans, key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    for s in spans:
        if not out or s.start >= out[-1].end:
            out.append(s)
    return out


def _alt(words: list[str]) -> str:
    return "|".join(sorted((re.escape(w) for w in words), key=len, reverse=True))


_WD = _alt(WEEKDAYS)
_MON_ABBR = ["jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"]
_MON = _alt([*MONTHS, *_MON_ABBR])
_ORD = r"\d{1,2}(?:st|nd|rd|th)|" + _alt(list(ORDINAL_WORDS))
_DAYNUM = r"\d{1,2}(?:st|nd|rd|th)?|" + _alt(list(ORDINAL_WORDS))
_NUMW = _alt(list(NUMBER_WORDS))

_DATE_PATTERNS = [
    rf"\b(?:{_WD}),?\s+(?:the\s+(?:{_ORD})|(?:{_MON})\.?\s+(?:the\s+)?(?:{_DAYNUM}))\b",
    r"\b(?:the\s+)?day\s+after\s+tomorrow\b",
    r"\b(?:today|tomorrow)\b",
    rf"\b(?:(?:this\s+coming|this|next|coming)\s+)?(?:{_WD})\b",
    rf"\b(?:{_MON})\.?\s+(?:the\s+)?(?:{_DAYNUM})\b",
    rf"\b(?:the\s+)?(?:{_ORD})\s+of\s+(?:{_MON})\b",
    rf"\bthe\s+(?:{_ORD})\b",
    r"\b\d{1,2}(?:st|nd|rd|th)\b",
    r"\b\d{4}-\d{2}-\d{2}\b",
]
_DATE_RES = [re.compile(p, re.IGNORECASE) for p in _DATE_PATTERNS]


def extract_date_phrase(text: str) -> list[Span]:
    spans = [
        Span("date_phrase", m.group(0), m.start(), m.end())
        for rx in _DATE_RES
        for m in rx.finditer(text)
    ]
    return _select(spans)


_MER = r"(?:a\.?m\.?|p\.?m\.?|o'?clock)"
_PART = r"(?:\s+in\s+the\s+(?:morning|afternoon|evening))"
_HOUR = rf"(?:\d{{1,2}}|{_NUMW})"
_TIME_RES = [
    (re.compile(r"\b(?:noon|midday)\b", re.IGNORECASE), 0),
    (re.compile(rf"\b\d{{1,2}}:\d{{2}}(?:\s*(?:a\.?m\.?|p\.?m\.?))?{_PART}?", re.I), 0),
    (re.compile(rf"\b{_HOUR}\s*{_MER}(?![a-z]){_PART}?", re.IGNORECASE), 0),
    (re.compile(rf"\b{_HOUR}{_PART}\b", re.IGNORECASE), 0),
    (re.compile(rf"\b(?:at|around|about|by)\s+({_HOUR})\b(?!\s*(?:st|nd|rd|th|:|/))", re.I), 1),
]


def extract_time_phrase(text: str) -> list[Span]:
    spans: list[Span] = []
    for rx, group in _TIME_RES:
        for m in rx.finditer(text):
            s, e = m.span(group)
            spans.append(
                Span("time_phrase", text[s:e].rstrip("."), s, s + len(text[s:e].rstrip(".")))
            )
    if not spans:
        # A bare hour as the whole answer ("Three." / "2, please").
        content = [t for t in tokenize(text) if t.text not in _BARE_TIME_FILLER]
        if len(content) == 1:
            tok = content[0]
            value = int(tok.text) if tok.text.isdigit() else NUMBER_WORDS.get(tok.text)
            if value is not None and 1 <= value <= 12:
                spans.append(Span("time_phrase", text[tok.start : tok.end], tok.start, tok.end))
    return _select(spans)


_BARE_TIME_FILLER = frozenset(
    "um uh er yeah yes ok okay please let's do say how about maybe then i guess "
    "think that works is fine great".split()
)

_NAMEWORD = r"[A-Z][a-z]*(?:['\-][A-Z]?[a-z]+)*[a-z]"
_NAME_STOP = frozenset(
    {w.capitalize() for w in WEEKDAYS + MONTHS}
    | {
        "Hi",
        "Hello",
        "Hey",
        "Yes",
        "Yeah",
        "No",
        "Thanks",
        "Thank",
        "Okay",
        "Ok",
        "Dr",
        "Doctor",
        "Calling",
        "Just",
        "Sure",
        "Um",
        "Uh",
        "Hmm",
        "Well",
        "So",
        "Sorry",
        "Please",
        "The",
        "And",
        "But",
        "Actually",
        "Wait",
        "Great",
        "Perfect",
        "It",
        "Its",
        "This",
        "That",
        "My",
        "Name",
        "Is",
        "Am",
        "Can",
        "Could",
        "Would",
        "Do",
        "Does",
        "What",
        "When",
        "Where",
        "How",
        "Why",
        "Book",
        "Cancel",
        "Need",
        "Want",
        "Like",
        "Appointment",
        "Tomorrow",
        "Today",
        "Next",
        "Also",
        "Oh",
        "Good",
        "Morning",
        "Afternoon",
        "Evening",
        "Fine",
        "Right",
        "Correct",
    }
)
_NAME_INTRO = re.compile(
    r"(?:\b[Mm]y name is|\b[Mm]y name's|\b[Tt]his is|\bI'm|\bI am|\b[Ii]t's|\b[Ii]t is|"
    r"\b[Nn]ame is|\b[Nn]ame's|\bunder)\s+"
    rf"({_NAMEWORD}(?:\s+{_NAMEWORD}){{0,2}})"
)
_NAME_LEAD_FILLER = frozenset("yeah yes sure um uh oh ok okay it's its it is hi hello".split())


def _trim_name(text: str, start: int) -> Span | None:
    words = list(re.finditer(_NAMEWORD, text))
    kept = []
    for w in words:
        if w.group(0) in _NAME_STOP:
            break
        kept.append(w)
    if not kept:
        return None
    s, e = start + kept[0].start(), start + kept[-1].end()
    return Span("person_name", text[kept[0].start() : kept[-1].end()], s, e)


def extract_person_name(text: str) -> list[Span]:
    spans: list[Span] = []
    for m in _NAME_INTRO.finditer(text):
        span = _trim_name(m.group(1), m.start(1))
        if span is not None:
            spans.append(span)
    if not spans:
        # A sentence that is nothing but a name ("Jane Doe." / "Yeah, it's Jane Doe").
        for sent in re.finditer(r"[^.!?]+", text):
            spans.extend(_bare_name(sent.group(0), sent.start()))
    return _select(spans)


def _bare_name(sentence: str, offset: int) -> list[Span]:
    toks = tokenize(sentence)
    i = 0
    while i < len(toks) and toks[i].text in _NAME_LEAD_FILLER:
        i += 1
    if i >= len(toks):
        return []
    rest = sentence[toks[i].start :].strip().rstrip(".!?, ")
    m = re.fullmatch(rf"({_NAMEWORD}(?:\s+{_NAMEWORD}){{1,2}})(?:,?\s+(?:here|please))?", rest)
    if not m or any(w in _NAME_STOP for w in m.group(1).split()):
        return []
    start = offset + toks[i].start
    return [Span("person_name", m.group(1), start, start + len(m.group(1)))]


_ID_RE = re.compile(r"\b[A-Z]{1,3}-?\d{3,}\b")
_NUM_RE = re.compile(r"\b\d+\b")


def extract_identifier(text: str) -> list[Span]:
    return [Span("identifier", m.group(0), m.start(), m.end()) for m in _ID_RE.finditer(text)]


def extract_number(text: str) -> list[Span]:
    return [Span("number", m.group(0), m.start(), m.end()) for m in _NUM_RE.finditer(text)]


Extractor = Callable[[str], list[Span]]

# Order matters: the compiler tries extractors in this order.
EXTRACTORS: dict[str, Extractor] = {
    "person_name": extract_person_name,
    "date_phrase": extract_date_phrase,
    "time_phrase": extract_time_phrase,
    "identifier": extract_identifier,
    "number": extract_number,
}

EXTRACTOR_DESCRIPTIONS: dict[str, str] = {
    "person_name": "the caller's full name, exactly as spoken",
    "date_phrase": "a date phrase exactly as the caller said it (e.g. 'next Tuesday')",
    "time_phrase": "a time phrase exactly as the caller said it (e.g. '2 pm')",
    "identifier": "an identifier/code exactly as spoken",
    "number": "a number as spoken",
}


def extract(name: str, text: str) -> list[Span]:
    return EXTRACTORS[name](text)


def unique_span(extractor: str, texts: list[str]) -> str | None:
    """The span from the first text that has any; None if that text is ambiguous.

    Entry slots use this over (entry turn, then earlier turns): the caller may have
    given a value before the LLM decided to act on it.
    """
    for text in texts:
        spans = extract(extractor, text)
        if spans:
            if len({normalize_text(s.text) for s in spans}) != 1:
                return None
            return spans[0].text
    return None
