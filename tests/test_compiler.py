"""Compiler: binding classes, refusals, confirm insertion, determinism, golden IR."""

from __future__ import annotations

import datetime as dt
import os
import random
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.helpers import GOLDEN_DIR, call, catalog, load_corpus, session, simulated_store
from vatic.compiler.compile import CompileConfig, Corpus, compile_corpus
from vatic.compiler.emit import write_flows
from vatic.core.formatters import human_date, human_time, human_time_list
from vatic.core.tools import SideEffect
from vatic.core.transforms import TransformContext, normalize_date
from vatic.ir.schema import AskStep, ConfirmStep, ToolStep, dump_flow

RO, IRR = SideEffect.READ_ONLY, SideEffect.IRREVERSIBLE
FIRST = ["Jane", "John", "Maria", "David", "Sarah", "Peter", "Emily", "James", "Laura", "Anna"]
LAST = ["Doe", "Smith", "Garcia", "Brown", "Miller", "Davis", "Moore"]
PHRASES = ["Tuesday", "next Wednesday", "October 14", "the 16th", "Thursday"]
TIMES = [("2 pm", "14:00"), ("10 am", "10:00"), ("3pm", "15:00"), ("11 am", "11:00")]


def booking_corpus(
    n: int = 25,
    *,
    confirm: bool = True,
    random_time: bool = False,
    extra_note: bool = False,
    seed: int = 0,
) -> Corpus:
    rng = random.Random(seed)
    today = dt.date(2026, 10, 5)
    ctx = TransformContext(today)
    corpus: Corpus = []
    for i in range(n):
        first, last = FIRST[i % len(FIRST)], LAST[i % len(LAST)]
        name = f"{first} {last}"
        phrase = PHRASES[i % len(PHRASES)]
        iso = normalize_date(phrase, ctx)
        assert iso is not None
        tphrase, hhmm = TIMES[i % len(TIMES)]
        times = sorted({"09:00", hhmm})
        if random_time:
            hhmm = f"{rng.randint(8, 17):02d}:{rng.choice(['05', '25', '45'])}"
        pid = f"P{1000 + i}"
        t1_calls = [call("check", {"date": iso, "site": "main"}, {"date": iso, "times": times})]
        if extra_note:
            t1_calls.append(call("note", {"text": f"free text {rng.random()}"}, {"ok": True}))
        turns = [
            (
                f"Hi, my name is {name}.",
                [
                    call(
                        "lookup",
                        {"name": name},
                        {"found": True, "patient": {"id": pid, "first": first}},
                    )
                ],
                f"Thanks, {first}. What day would you like?",
            ),
            (
                f"{phrase} please.",
                t1_calls,
                f"On {human_date(iso)} I have {human_time_list(times)}. Which time works?",
            ),
        ]
        book = call(
            "book",
            {"pid": pid, "date": iso, "time": hhmm},
            {"id": f"A{i}", "date": iso, "time": hhmm},
        )
        done = f"Booked for {human_date(iso)} at {human_time(hhmm)}."
        if confirm:
            turns.append(
                (
                    f"{tphrase}.",
                    [],
                    f"Just to confirm: {human_date(iso)} at {human_time(hhmm)}. Shall I book it?",
                )
            )
            turns.append(("Yes.", [book], done))
        else:
            turns.append((f"{tphrase}.", [book], done))
        corpus.append(session(f"s{i:03d}", turns))
    return corpus


CAT = catalog(lookup=RO, check=RO, book=IRR, note=RO)
CAT["check"] = CAT["check"].__class__(
    name="check", side_effect=RO, params=("date", "site"), required=("date",)
)


def compile_one(corpus: Corpus, **kw: int) -> object:
    res = compile_corpus(corpus, CAT, CompileConfig(min_support=kw.get("min_support", 20)))
    return res


def test_binding_classes() -> None:
    res = compile_corpus(booking_corpus(), CAT, CompileConfig(min_support=20))
    assert not res.refused, res.refused
    (flow,) = res.accepted
    b = flow.provenance.bindings
    assert b["s1.args.name"].source == "slot"
    assert b["s3.args.date"].source == "transform"
    assert b["s3.args.site"].source == "const"
    assert b["s6.args.pid"].source == "output"
    assert b["s6.args.date"].source == "output"
    assert b["s6.args.time"].source == "transform"
    book = flow.step("s6")
    assert isinstance(book, ToolStep)
    assert book.args["pid"].step == "s1" and book.args["pid"].path == "patient.id"
    assert book.args["time"].fn == "normalize_time"
    assert flow.entry.slots == ["name"]
    ask = flow.step("s4")
    assert isinstance(ask, AskStep) and ask.expects == ["time"]
    assert isinstance(flow.step("s5"), ConfirmStep)
    assert ask.say.template == (
        "On {steps.s3.output.date|human_date} I have {steps.s3.output.times|human_time_list}."
        " Which time works?"
    )


def test_llm_binding_allowed_only_off_irreversible_path() -> None:
    res = compile_corpus(booking_corpus(extra_note=True), CAT, CompileConfig(min_support=20))
    (flow,) = res.accepted
    note = next(s for s in flow.steps if isinstance(s, ToolStep) and s.tool == "note")
    assert note.args["text"].source == "llm"


def test_irreversible_with_unexplained_argument_is_refused() -> None:
    res = compile_corpus(booking_corpus(random_time=True), CAT, CompileConfig(min_support=20))
    assert not res.accepted
    (refusal,) = res.refused
    assert "irreversible" in refusal.reason and "time" in refusal.reason


def test_confirm_inserted_before_irreversible_step() -> None:
    res = compile_corpus(booking_corpus(confirm=False), CAT, CompileConfig(min_support=20))
    (flow,) = res.accepted
    ids = [s.id for s in flow.steps]
    book_idx = next(
        i for i, s in enumerate(flow.steps) if isinstance(s, ToolStep) and s.tool == "book"
    )
    confirm = flow.steps[book_idx - 1]
    assert isinstance(confirm, ConfirmStep) and confirm.inserted and confirm.on_yes == ids[book_idx]
    assert (
        confirm.say.template is not None
        and "{slot.time|normalize_time|human_time}" in confirm.say.template
    )


def test_min_support() -> None:
    res = compile_corpus(booking_corpus(n=10), CAT, CompileConfig(min_support=20))
    assert not res.accepted and res.skipped[0].support == 10


def test_failed_sessions_are_not_compiled() -> None:
    corpus = [(s.model_copy(update={"outcome": "failure"}), t) for s, t in booking_corpus()]
    assert not compile_corpus(corpus, CAT, CompileConfig(min_support=20)).accepted


def _dump(corpus: Corpus, catalog_: dict) -> str:  # type: ignore[type-arg]
    res = compile_corpus(corpus, catalog_, CompileConfig(min_support=20))
    return "\n---\n".join(dump_flow(f) for f in res.accepted) + "\n".join(
        f"{r.flow_id}:{r.reason}" for r in res.refused
    )


def test_compiler_is_deterministic(tmp_path: Path) -> None:
    corpus = booking_corpus()
    a, b = tmp_path / "a", tmp_path / "b"
    for out in (a, b):
        res = compile_corpus(corpus, CAT, CompileConfig(min_support=20))
        write_flows(res.accepted, out, res.examples)
    files_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    files_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    assert files_a == files_b and files_a
    for rel in files_a:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


@settings(max_examples=25, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(st.randoms(use_true_random=False))
def test_output_identical_for_shuffled_traces(rnd: random.Random) -> None:
    corpus = booking_corpus()
    expected = _dump(corpus, CAT)
    shuffled = [(s, rnd.sample(turns, len(turns))) for s, turns in corpus]
    rnd.shuffle(shuffled)
    assert _dump(shuffled, CAT) == expected


# -- golden: simulated traces -> expected IR ---------------------------------------


@pytest.fixture(scope="module")
def sim_corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Corpus, dict]:  # type: ignore[type-arg]
    path = simulated_store(11, 800, str(tmp_path_factory.getbasetemp()))
    return load_corpus(path)


def test_golden_flows(sim_corpus: tuple[Corpus, dict], tmp_path: Path) -> None:  # type: ignore[type-arg]
    corpus, cat = sim_corpus
    res = compile_corpus(corpus, cat, CompileConfig(min_support=20))
    names = {f.flow_id for f in res.accepted}
    assert {"book_appointment", "cancel_appointment"} <= names
    golden = GOLDEN_DIR / "flows"
    if os.environ.get("UPDATE_GOLDEN"):
        golden.mkdir(parents=True, exist_ok=True)
        for p in golden.glob("*.yaml"):
            p.unlink()
        for f in res.accepted:
            (golden / f"{f.flow_id}.yaml").write_text(dump_flow(f))
    expected = {p.stem: p.read_text() for p in golden.glob("*.yaml")}
    assert expected, "golden files missing: run with UPDATE_GOLDEN=1"
    assert {f.flow_id: dump_flow(f) for f in res.accepted} == expected


def test_golden_flows_shuffle_invariant(sim_corpus: tuple[Corpus, dict]) -> None:  # type: ignore[type-arg]
    corpus, cat = sim_corpus
    rnd = random.Random(5)
    shuffled = list(corpus)
    rnd.shuffle(shuffled)
    assert _dump(shuffled, cat) == _dump(corpus, cat)
