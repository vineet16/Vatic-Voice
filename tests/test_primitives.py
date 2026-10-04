"""Extractors, transforms, formatters, guards, templates."""

from __future__ import annotations

import datetime as dt

import pytest

from vatic.core.extract import (
    extract_date_phrase,
    extract_identifier,
    extract_number,
    extract_person_name,
    extract_time_phrase,
)
from vatic.core.formatters import human_date, human_time, human_time_list
from vatic.core.guards import GuardError, check_domain, evaluate, references_output, value_shape
from vatic.core.phrasing import RenderError, render
from vatic.core.transforms import TransformContext, normalize_date, normalize_time
from vatic.ir.schema import LearnedDomain

CTX = TransformContext(dt.date(2026, 10, 5))  # Monday


def texts(spans: list) -> list[str]:  # type: ignore[type-arg]
    return [s.text for s in spans]


@pytest.mark.parametrize(
    ("utterance", "expected"),
    [
        ("Hi! I'd like to book an appointment. My name is Jane Doe.", ["Jane Doe"]),
        ("Jane Doe.", ["Jane Doe"]),
        ("Yeah, it's Mary O'Brien", ["Mary O'Brien"]),
        ("Cancel my appointment. Anna Taylor.", ["Anna Taylor"]),
        ("This is Tom Smith calling about Tuesday", ["Tom Smith"]),
        ("I'm calling to book", []),
        ("jane doe", []),  # no truecasing -> no name (conservative)
        ("Next Tuesday.", []),
    ],
)
def test_person_name(utterance: str, expected: list[str]) -> None:
    assert texts(extract_person_name(utterance)) == expected


@pytest.mark.parametrize(
    ("utterance", "span", "iso"),
    [
        ("Next Tuesday would be great.", "Next Tuesday", "2026-10-13"),
        ("tuesday", "tuesday", "2026-10-06"),
        ("this Monday", "this Monday", "2026-10-12"),
        ("October 13", "October 13", "2026-10-13"),
        ("the 13th", "the 13th", "2026-10-13"),
        ("Tuesday the 13th", "Tuesday the 13th", "2026-10-13"),
        ("tomorrow please", "tomorrow", "2026-10-06"),
        ("the twelfth of october", "the twelfth of october", "2026-10-12"),
        ("the 2nd", "the 2nd", "2026-11-02"),
        ("2026-10-20", "2026-10-20", "2026-10-20"),
    ],
)
def test_date_phrase(utterance: str, span: str, iso: str) -> None:
    spans = extract_date_phrase(utterance)
    assert texts(spans) == [span]
    assert normalize_date(spans[0].text, CTX) == iso


def test_date_contradiction_is_unresolvable() -> None:
    assert normalize_date("Monday the 13th", CTX) is None  # the 13th is a Tuesday
    assert normalize_date("February 30", CTX) is None


@pytest.mark.parametrize(
    ("utterance", "span", "hhmm"),
    [
        ("Let's do 1 pm.", "1 pm", "13:00"),
        ("2pm", "2pm", "14:00"),
        ("at 2", "2", "14:00"),
        ("around three", "three", "15:00"),
        ("Three.", "Three", "15:00"),
        ("9 in the morning", "9 in the morning", "09:00"),
        ("10:30", "10:30", "10:30"),
        ("noon works", "noon", "12:00"),
        ("two o'clock please", "two o'clock", "14:00"),
        ("14:00", "14:00", "14:00"),
        ("11 am", "11 am", "11:00"),
    ],
)
def test_time_phrase(utterance: str, span: str, hhmm: str) -> None:
    spans = extract_time_phrase(utterance)
    assert texts(spans) == [span]
    assert normalize_time(spans[0].text, CTX) == hhmm


def test_time_not_confused_with_dates() -> None:
    assert extract_time_phrase("the 2nd") == []
    assert normalize_time("13 am", CTX) is None


def test_identifier_and_number() -> None:
    assert texts(extract_identifier("my code is AB-1234 thanks")) == ["AB-1234"]
    assert texts(extract_number("room 12 or 14")) == ["12", "14"]


def test_formatters() -> None:
    assert human_date("2026-10-13") == "Tuesday, October 13"
    assert human_time("14:00") == "2:00 PM"
    assert human_time("00:30") == "12:30 AM"
    assert human_time_list(["09:00", "13:00", "15:00"]) == "9:00 AM, 1:00 PM and 3:00 PM"
    assert human_date("nope") is None


# -- guards ------------------------------------------------------------------------

NS = {
    "output": {"found": True, "patient": {"id": "P1", "tags": ["a", "b"]}, "n": 3},
    "args": {"time": "14:00"},
    "steps": {"s3": {"output": {"times": ["13:00", "14:00"]}}},
    "slot": {"time": "2 pm"},
}


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("output.found == True", True),
        ("output.found == False", False),
        ("output.n >= 3 and output.n < 4", True),
        ("not output.found", False),
        ("output.patient.id in ['P1', 'P2']", True),
        ("len(output.patient.tags) == 2", True),
        ("output.patient.tags[1] == 'b'", True),
        ("args.time in steps.s3.output.times", True),
        ("normalize_time(slot.time) in steps.s3.output.times", True),
        ("output.patient is not None", True),
    ],
)
def test_guard_eval(expr: str, expected: bool) -> None:
    assert evaluate(expr, NS, CTX) is expected


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('true')",
        "output.__class__",
        "(lambda: True)()",
        "[x for x in output.patient.tags]",
        "output.missing == 1",
        "open('/etc/passwd')",
        "output.n",  # not boolean
        "output.found + 1 == 2",
    ],
)
def test_guard_rejects_unsafe_or_invalid(expr: str) -> None:
    with pytest.raises(GuardError):
        evaluate(expr, NS, CTX)


def test_references_output() -> None:
    assert references_output("output.found == True")
    assert not references_output("args.date != ''")


def test_learned_domains() -> None:
    ns = {"output": {"found": True, "times": ["09:00"], "id": "P1001", "n": 5}}
    ok = [
        LearnedDomain(path="output.found", kind="enum", values=[True]),
        LearnedDomain(path="output.times", kind="length", min=1, max=4),
        LearnedDomain(path="output.id", kind="pattern", values=[value_shape("P1234")]),
        LearnedDomain(path="output.n", kind="range", min=1, max=9),
        LearnedDomain(path="output.id", kind="not_null"),
    ]
    for d in ok:
        assert check_domain(d, ns).passed, d
    bad = [
        LearnedDomain(path="output.found", kind="enum", values=[False]),
        LearnedDomain(path="output.times", kind="length", min=2),
        LearnedDomain(path="output.id", kind="pattern", values=["9999"]),
        LearnedDomain(path="output.missing", kind="not_null"),
    ]
    for d in bad:
        assert not check_domain(d, ns).passed, d
    member = LearnedDomain(
        path="slot.time", kind="member_of", container="steps.s3.output.times", fn="normalize_time"
    )
    assert check_domain(member, NS, CTX).passed
    assert not check_domain(member, {**NS, "slot": {"time": "4 pm"}}, CTX).passed


def test_render_templates() -> None:
    ns = {"slot": {"time": "2 pm"}, "steps": {"s1": {"output": {"date": "2026-10-13"}}}}
    r = render(
        "That's {steps.s1.output.date|human_date} at {slot.time|normalize_time|human_time}. {{ok}}",
        ns,
        CTX,
    )
    assert r.text == "That's Tuesday, October 13 at 2:00 PM. {ok}"
    assert r.values == ["Tuesday, October 13", "2:00 PM"]
    with pytest.raises(RenderError):
        render("{slot.missing}", ns, CTX)
