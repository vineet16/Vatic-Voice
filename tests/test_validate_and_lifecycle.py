"""IR validation rules and the candidate -> shadow -> active -> retired lifecycle."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from tests.helpers import GOLDEN_DIR
from vatic.ir.schema import (
    AskStep,
    Binding,
    ConfirmStep,
    FlowGraph,
    Say,
    SayStep,
    SlotDef,
    ToolStep,
    dump_flow,
    load_flow,
    parse_flow,
    save_flow,
)
from vatic.ir.validate import errors
from vatic.lifecycle.promote import PromotionRules, demote_one, evaluate, promote_one
from vatic.trace.schema import ShadowComparison, TurnTrace
from vatic.trace.store import TraceStore


def _flow(steps: list, **kw: object) -> FlowGraph:  # type: ignore[type-arg]
    return FlowGraph(
        flow_id="f",
        slots={"name": SlotDef(extractor="person_name"), "time": SlotDef(extractor="time_phrase")},
        entry={"slots": ["name"]},  # type: ignore[arg-type]
        steps=steps,
        **kw,  # type: ignore[arg-type]
    )


def lookup(next_: str = "s2") -> ToolStep:
    return ToolStep(
        id="s1",
        tool="lookup",
        side_effect="read_only",
        args={"name": Binding(source="slot", slot="name")},
        next=next_,
    )


def book(**args: Binding) -> ToolStep:
    return ToolStep(
        id="s3",
        tool="book",
        side_effect="irreversible",
        args=args or {"id": Binding(source="output", step="s1", path="id")},
        next="s4",
    )


def say(id_: str = "s4") -> SayStep:
    return SayStep(id=id_, say=Say(template="done"))


def msgs(flow: FlowGraph) -> list[str]:
    return [i.message for i in errors(flow)]


def test_valid_flow_with_confirm() -> None:
    flow = _flow(
        [lookup(), ConfirmStep(id="s2", say=Say(template="ok?"), on_yes="s3"), book(), say()]
    )
    assert msgs(flow) == []


def test_irreversible_without_confirm_fails() -> None:
    flow = _flow([lookup("s3"), book(), say()])
    assert any("confirm" in m for m in msgs(flow))


def test_irreversible_with_ask_before_fails() -> None:
    ask = AskStep(id="s2", expects=["time"], say=Say(template="when?"), next="s3")
    flow = _flow([lookup(), ask, book(), say()])
    assert any("confirm" in m for m in msgs(flow))


def test_llm_binding_in_irreversible_step_fails() -> None:
    flow = _flow(
        [
            lookup(),
            ConfirmStep(id="s2", say=Say(template="ok?"), on_yes="s3"),
            book(id=Binding(source="llm", hint="?")),
            say(),
        ]
    )
    assert any("llm binding" in m for m in msgs(flow))


def test_output_binding_must_precede_consumer() -> None:
    bad = ToolStep(
        id="s1",
        tool="lookup",
        side_effect="read_only",
        args={"name": Binding(source="output", step="s2", path="x")},
        next="s2",
    )
    second = ToolStep(id="s2", tool="lookup", side_effect="read_only", args={}, next="s3")
    flow = _flow([bad, second, say("s3")])
    assert any("does not run before" in m for m in msgs(flow))


def test_slot_must_be_provided() -> None:
    step = ToolStep(
        id="s1",
        tool="lookup",
        side_effect="read_only",
        args={"name": Binding(source="slot", slot="time")},
        next="s2",
    )
    assert any("not provided" in m for m in msgs(_flow([step, say("s2")])))


def test_unreachable_cycle_and_bad_targets() -> None:
    assert any("unknown target" in m for m in msgs(_flow([lookup("nope")])))
    loop = AskStep(id="s2", say=Say(template="?"), next="s2")
    assert any("cycle" in m for m in msgs(_flow([lookup(), loop])))
    orphan = _flow([lookup("s2"), say("s2"), say("s9")])
    assert any("unreachable" in m for m in msgs(orphan))


def test_guard_syntax_is_whitelisted() -> None:
    step = lookup()
    step.guards = ["__import__('os')"]
    assert any("guard" in m for m in msgs(_flow([step, say("s2")])))


def test_yaml_roundtrip_is_stable() -> None:
    text = (GOLDEN_DIR / "flows" / "book_appointment.yaml").read_text()
    flow = parse_flow(text)
    assert dump_flow(flow) == text
    assert flow.compute_hash() == flow.content_hash


# -- lifecycle -----------------------------------------------------------------------


def _shadow_turn(
    sid: str,
    i: int,
    flow: FlowGraph,
    matched: bool,
    irreversible: bool = False,
    ended: bool = False,
) -> TurnTrace:
    return TurnTrace(
        trace_id=f"{sid}:{i}:shadow:{flow.flow_id}",
        session_id=sid,
        turn_index=i,
        user_transcript="x",
        route="shadow",
        flow_id=flow.flow_id,
        flow_version=flow.version,
        shadow=ShadowComparison(
            flow_id=flow.flow_id,
            flow_version=flow.version,
            step_id="s2",
            decision="on_path",
            compared=True,
            matched=matched,
            irreversible=irreversible,
            run_ended=ended,
        ),
    )


@pytest.fixture
def lifecycle_env(tmp_path: Path) -> tuple[Path, TraceStore, FlowGraph]:
    flows = tmp_path / "flows"
    flows.mkdir()
    flow = load_flow(GOLDEN_DIR / "flows" / "cancel_appointment.yaml")
    save_flow(flow, flows / "cancel_appointment.yaml")
    return flows, TraceStore(tmp_path / "store"), flow


def test_promotion_rules(lifecycle_env: tuple[Path, TraceStore, FlowGraph]) -> None:
    flows, store, flow = lifecycle_env
    rules = PromotionRules(min_sessions=10, min_match_rate=0.98)
    (e,) = evaluate(flows, store, rules)
    assert (e.from_status, e.to_status) == ("candidate", "shadow")
    # 9 complete matching sessions: not enough evidence yet.
    store.write_batch([_shadow_turn(f"a{i}", 1, flow, True, ended=True) for i in range(9)])
    assert evaluate(flows, store, rules) == []
    store.write(_shadow_turn("a9", 1, flow, True, ended=True))
    (e,) = evaluate(flows, store, rules)
    assert e.to_status == "active" and e.evidence["match_rate"] == 1.0
    assert [x.to_status for x in store.iter_events("cancel_appointment")] == ["shadow", "active"]


def test_irreversible_mismatch_blocks_promotion(
    lifecycle_env: tuple[Path, TraceStore, FlowGraph],
) -> None:
    flows, store, flow = lifecycle_env
    evaluate(flows, store)
    store.write_batch([_shadow_turn(f"a{i}", 1, flow, True, ended=True) for i in range(60)])
    store.write(_shadow_turn("bad", 1, flow, False, irreversible=True))
    assert evaluate(flows, store, PromotionRules(min_sessions=50, min_match_rate=0.9)) == []


def test_manual_promote_and_demote_are_logged(
    lifecycle_env: tuple[Path, TraceStore, FlowGraph],
) -> None:
    flows, store, _ = lifecycle_env
    assert promote_one(flows, "cancel_appointment", store).to_status == "shadow"
    assert promote_one(flows, "cancel_appointment", store).to_status == "active"
    e = demote_one(flows, "cancel_appointment", "bad week", store=store)
    assert (e.from_status, e.to_status, e.manual) == ("active", "shadow", True)
    assert load_flow(flows / "cancel_appointment.yaml").status == "shadow"
    assert all(ev.at <= time.time() for ev in store.iter_events())
