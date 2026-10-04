"""Step membership: rules layers, confirm classifier, no-LLM guarantee, latency."""

from __future__ import annotations

import time

import pytest

from vatic.core.step_check import check_ask, classify_confirm
from vatic.ir.schema import AskStep, FlowGraph, Membership, Say, SayStep, SlotDef


def _flow(expects: str, extractor: str, residual: int = 1) -> tuple[AskStep, FlowGraph]:
    ask = AskStep(
        id="s2",
        expects=[expects],
        say=Say(template="What day?"),
        membership=Membership(residual_max_tokens=residual),
        next="s3",
    )
    flow = FlowGraph(
        flow_id="f",
        slots={expects: SlotDef(extractor=extractor)},
        steps=[ask, SayStep(id="s3", say=Say(template="ok"))],
    )
    return ask, flow


DATE_CASES = [
    # on-path
    ("Next Tuesday would be great.", "on_path"),
    ("Tuesday please.", "on_path"),
    ("Um, the 13th.", "on_path"),
    ("I guess October 14.", "on_path"),
    ("Let's say Friday the 16th.", "on_path"),
    # digressions / missing slot
    ("What are your hours?", "off_path"),
    ("Hmm, I'm not sure. What do you have available?", "off_path"),
    ("Can I talk to a person?", "off_path"),
    # multi-intent
    ("Tuesday at 3, and can I also cancel my other one?", "off_path"),
    ("October 14 please. Also, do I need to bring anything?", "off_path"),
    ("Tuesday, but where do I park?", "off_path"),
    # negation / correction
    ("No, Tuesday.", "off_path"),
    ("Actually, Wednesday instead.", "off_path"),
    ("Not Tuesday.", "off_path"),
    ("Wait, Thursday.", "off_path"),
    # ambiguous
    ("Tuesday or Wednesday?", "off_path"),
    ("How about Tuesday?", "off_path"),
]

TIME_CASES = [
    ("2 pm.", "on_path"),
    ("Let's do 1 pm.", "on_path"),
    ("Oh, three o'clock would be perfect.", "on_path"),
    ("10:00 works for me.", "on_path"),
    ("for pm.", "off_path"),
    ("10:00, and also what are your hours?", "off_path"),
    ("Actually, can we do Wednesday instead?", "off_path"),
    ("No, I said 2 pm.", "off_path"),
]


@pytest.mark.parametrize(("utterance", "expected"), DATE_CASES)
def test_date_step_membership(utterance: str, expected: str) -> None:
    ask, flow = _flow("date", "date_phrase", residual=1)
    assert check_ask(ask, flow, utterance).decision == expected


@pytest.mark.parametrize(("utterance", "expected"), TIME_CASES)
def test_time_step_membership(utterance: str, expected: str) -> None:
    ask, flow = _flow("time", "time_phrase", residual=0)
    assert check_ask(ask, flow, utterance).decision == expected


def test_residual_budget_is_enforced() -> None:
    ask, flow = _flow("date", "date_phrase", residual=0)
    m = check_ask(ask, flow, "Tuesday for my knee checkup")
    assert m.decision == "off_path" and m.reason == "residual"
    assert check_ask(ask, flow, "Um, Tuesday please.").decision == "on_path"


def test_extracted_slots_are_reported() -> None:
    ask, flow = _flow("time", "time_phrase", residual=0)
    m = check_ask(ask, flow, "Let's do 2 pm.")
    assert m.slots == {"time": "2 pm"} and m.latency_ms >= 0


@pytest.mark.parametrize(
    ("utterance", "label"),
    [
        ("Yes.", "yes"),
        ("Yes, please.", "yes"),
        ("Yeah, that's right.", "yes"),
        ("Sounds good.", "yes"),
        ("Sounds um, good.", "yes"),
        ("Yep, go ahead.", "yes"),
        ("Yes, I think so.", "yes"),
        ("No.", "no"),
        ("Nope.", "no"),
        ("No, that's not right. I wanted Tuesday at 2 pm.", "other"),
        ("Yes, and also what are your hours?", "other"),
        ("Yes but make it 3 pm.", "other"),
        ("Before that, what are your hours?", "other"),
        ("Hmm.", "other"),
        ("", "other"),
        ("Yes no.", "other"),
    ],
)
def test_confirm_classifier(utterance: str, label: str) -> None:
    assert classify_confirm(utterance) == label


def test_membership_never_touches_an_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    import vatic.llm.client as client

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("LLM called during membership checks")

    monkeypatch.setattr(client.OpenAICompatibleClient, "complete", boom)
    monkeypatch.setattr(client.DisabledLLMClient, "complete", boom)
    ask, flow = _flow("date", "date_phrase")
    for text, _ in DATE_CASES:
        check_ask(ask, flow, text)
        classify_confirm(text)


def test_membership_latency_budget() -> None:
    ask, flow = _flow("date", "date_phrase")
    samples = []
    for _ in range(20):
        for text, _ in DATE_CASES:
            t0 = time.perf_counter()
            check_ask(ask, flow, text)
            samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    assert samples[int(len(samples) * 0.95)] < 30.0
