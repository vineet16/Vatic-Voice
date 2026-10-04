"""M4: branch synthesis, per-step classifier, conformal calibration, hedged fallback."""

from __future__ import annotations

import asyncio
import random
import time
from pathlib import Path
from typing import Any

import pytest

from examples.scheduling.agent_llm import ClinicAgent
from examples.scheduling.scripted_llm import ScriptedClinicLLM
from tests.helpers import GOLDEN_DIR, call, catalog, session, trained_flows
from tests.test_compiler import CAT, booking_corpus
from vatic.compiler.branch import learn_condition
from vatic.compiler.compile import CompileConfig, compile_corpus
from vatic.core.formatters import human_date
from vatic.core.runtime import RuntimeConfig, VaticRuntime
from vatic.core.tools import SideEffect, ToolContext, ToolRegistry, ToolSpec
from vatic.core.types import TurnContext
from vatic.ir.schema import AskStep, BranchStep, load_flow
from vatic.lifecycle.calibrate import conformal_threshold
from vatic.trace.schema import MembershipResult

# -- branch synthesis --------------------------------------------------------------


def test_learn_condition_depths() -> None:
    a = [{"x": True, "n": 1}, {"x": True, "n": 2}]
    b = [{"x": False, "n": 1}, {"x": False, "n": 9}]
    assert learn_condition(a, b) == ("x == True", "x == False")
    assert learn_condition([{"n": 1}, {"n": 2}], [{"n": 5}, {"n": 7}]) == ("n < 5", "n >= 5")
    # Only a conjunction separates these.
    a2 = [{"p": "a", "q": "y"}]
    b2 = [{"p": "a", "q": "z"}, {"p": "b", "q": "y"}]
    cond = learn_condition(a2, b2)
    assert cond is not None and " and " in cond[0]
    # Overlapping traces never get a probabilistic split.
    assert learn_condition([{"x": 1}, {"x": 2}], [{"x": 2}]) is None


def _unavailable_corpus(n: int) -> list:  # type: ignore[type-arg]
    """Like booking_corpus, but the first day asked for is full."""
    out = []
    for i, (_, turns) in enumerate(booking_corpus(n, seed=3)):
        raw = [(t.user_transcript, t.tool_calls, t.agent_text) for t in turns]
        iso_sat = "2026-10-10"
        miss = call("check", {"date": iso_sat, "site": "main"}, {"date": iso_sat, "times": []})
        sorry = (
            "Saturday please.",
            [miss],
            f"Sorry, no openings on {human_date(iso_sat)}. Which other day?",
        )
        out.append(session(f"u{i:03d}", [raw[0], sorry, *raw[1:]]))
    return out


def test_compiler_synthesises_a_branch() -> None:
    corpus = booking_corpus(25) + _unavailable_corpus(25)
    # Mark availability explicitly so the condition is a real output field.
    for _, turns in corpus:
        for t in turns:
            for c in t.tool_calls:
                if c.tool == "check" and c.output is not None:
                    c.output["available"] = bool(c.output["times"])
    res = compile_corpus(corpus, CAT, CompileConfig(min_support=20))
    assert len(res.accepted) == 1, (res.notes, res.refused)
    flow = res.accepted[0]
    branch = next(s for s in flow.steps if isinstance(s, BranchStep))
    assert [c.when for c in branch.cases] == [
        "steps.s3.output.available == True",
        "steps.s3.output.available == False",
    ]
    assert branch.default == "fallback"
    assert flow.provenance.support == 50


def test_golden_flow_has_branch() -> None:
    flow = load_flow(GOLDEN_DIR / "flows" / "book_appointment.yaml")
    assert any(isinstance(s, BranchStep) for s in flow.steps)


# -- conformal calibration ------------------------------------------------------------


def test_conformal_threshold_infeasible_with_little_data() -> None:
    assert conformal_threshold([0.9] * 50, [False] * 50, alpha=0.01) is None
    assert conformal_threshold([0.9] * 120, [False] * 120, alpha=0.01) == 0.9


def test_conformal_threshold_controls_heldout_wrong_route_rate() -> None:
    rng = random.Random(0)

    def sample(n: int) -> tuple[list[float], list[bool]]:
        off = [rng.random() < 0.15 for _ in range(n)]
        scores = [min(1.0, max(0.0, rng.gauss(0.35 if o else 0.8, 0.15))) for o in off]
        return scores, off

    alpha = 0.02
    risks = []
    for _ in range(30):
        cal_s, cal_o = sample(400)
        t = conformal_threshold(cal_s, cal_o, alpha)
        assert t is not None
        hold_s, hold_o = sample(2000)
        risks.append(sum(1 for s, o in zip(hold_s, hold_o, strict=True) if o and s >= t) / 2000)
    # The guarantee is on the expected rate; the average over draws stays within target.
    assert sum(risks) / len(risks) <= alpha


# -- classifier ------------------------------------------------------------------------


@pytest.fixture(scope="module")
def flows_with_classifier(tmp_path_factory: pytest.TempPathFactory) -> Path:
    pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    return trained_flows(str(tmp_path_factory.getbasetemp()))


def test_classifier_separates_on_and_off_path(flows_with_classifier: Path) -> None:
    from vatic.core.classifier import StepClassifier, step_prompt

    flow = load_flow(flows_with_classifier / "book_appointment.yaml")
    step = next(s for s in flow.steps if isinstance(s, AskStep) and s.membership.classifier)
    clf = StepClassifier(flows_with_classifier / step.membership.classifier)  # type: ignore[operator]
    prompt = step_prompt(step)
    on = {"date": "Next Tuesday please.", "time": "2 pm."}[step.expects[0].split("_")[0]]
    assert clf.score(prompt, on) > clf.score(prompt, "What are your hours?")
    t0 = time.perf_counter()
    for _ in range(50):
        clf.score(prompt, on)
    assert (time.perf_counter() - t0) / 50 < 0.030  # well inside the membership budget


@pytest.mark.allow_slow_callbacks  # ONNX session creation happens at runtime start
async def test_runtime_uses_classifier_thresholds(flows_with_classifier: Path) -> None:
    from examples.scheduling.backend import BackendPool
    from examples.scheduling.tools import build_registry

    flow = load_flow(flows_with_classifier / "book_appointment.yaml")
    step = next(s for s in flow.steps if isinstance(s, AskStep) and s.membership.classifier)
    async with VaticRuntime(build_registry(BackendPool()), flows_with_classifier) as rt:
        utterance = {"date": "Next Tuesday please.", "time": "2 pm."}[step.expects[0].split("_")[0]]
        m = await rt._membership(step, flow, utterance)
        assert m.rules == "on_path" and m.classifier_score is not None
        assert m.decision in ("on_path", "uncertain", "off_path")
        assert m.latency_ms < 30
        # A classifier that cannot answer in time makes the decision "uncertain".
        rt.config.membership_budget_ms = 0.0
        m2 = await rt._membership(step, flow, utterance)
        assert m2.decision == "uncertain" and m2.rules == "on_path"


# -- hedged fallback ---------------------------------------------------------------------


class _Out(__import__("pydantic").BaseModel):
    ok: bool = True


class _In(__import__("pydantic").BaseModel):
    name: str = ""
    date: str = ""


def _hedge_registry(delay: float, booked: list[str]) -> ToolRegistry:
    from examples.scheduling.backend import BackendPool
    from examples.scheduling.tools import build_registry

    reg = build_registry(BackendPool())
    real = reg.get("check_availability").handler

    async def slow_check(args: Any, ctx: ToolContext) -> Any:
        await asyncio.sleep(delay)
        return {"date": args.date, "available": True, "times": ["09:00", "10:00"]}

    reg.get("check_availability").handler = slow_check
    _ = real

    def book(args: Any, ctx: ToolContext) -> Any:
        booked.append(ctx.session_id)
        return {"appointment_id": "A9", "date": args.date, "time": args.time, "provider": "Dr. Kim"}

    reg.get("book_appointment").handler = book
    _ = (SideEffect, ToolSpec, catalog)
    return reg


async def _hedged_session(rt: VaticRuntime, llm: Any) -> Any:
    from vatic.core.session import FlowState

    flow = rt.flows("active")[0]
    session = rt.start_session("h1", {"today": "2026-10-05"})
    session.flow = FlowState(
        flow=flow,
        step_id="s2",
        slots={"name": "Jane Doe"},
        step_args={"s1": {"name": "Jane Doe"}},
        step_outputs={
            "s1": {
                "found": True,
                "patient": {
                    "id": "P1001",
                    "first_name": "Jane",
                    "last_name": "Doe",
                    "next_appointment": None,
                },
            }
        },
    )

    async def uncertain(step: Any, flow: Any, text: str) -> MembershipResult:
        return MembershipResult(
            decision="uncertain",
            reason="classifier:uncertain",
            rules="on_path",
            slots={"date": "Tuesday"},
            classifier_score=0.3,
        )

    rt._membership = uncertain  # type: ignore[method-assign]
    return await rt.handle_turn("h1", "Tuesday.", llm)


@pytest.fixture
async def hedge_flows(tmp_path: Path) -> Path:
    from tests.test_runtime_flows import flows_dir

    return await asyncio.to_thread(flows_dir, tmp_path, {"book_appointment": "active"})


async def test_hedge_compiled_wins_and_llm_is_cancelled(hedge_flows: Path) -> None:
    cancelled = asyncio.Event()

    async def slow_llm(ctx: TurnContext) -> str:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "llm"

    booked: list[str] = []
    cfg = RuntimeConfig(hedge_budget_ms=500)
    async with VaticRuntime(_hedge_registry(0.01, booked), hedge_flows, config=cfg) as rt:
        res = await _hedged_session(rt, slow_llm)
    assert res.route == "compiled" and "Which time" in res.text
    assert res.membership is not None and res.membership.hedged
    assert cancelled.is_set()


async def test_hedge_llm_wins_when_flow_is_slow_and_gates_side_effects(hedge_flows: Path) -> None:
    order: list[str] = []

    async def llm(ctx: TurnContext) -> str:
        t0 = time.perf_counter()
        await ctx.call_tool(
            "book_appointment", {"patient_id": "P1001", "date": "2026-10-06", "time": "09:00"}
        )
        order.append(f"book after {time.perf_counter() - t0:.2f}s")
        return "llm answer"

    booked: list[str] = []
    cfg = RuntimeConfig(hedge_budget_ms=100)
    async with VaticRuntime(_hedge_registry(1.0, booked), hedge_flows, config=cfg) as rt:
        res = await _hedged_session(rt, llm)
    assert res.route == "llm" and res.text == "llm answer"
    assert booked == ["h1"]
    # The side-effecting call waited for the hedge to resolve (~hedge budget).
    assert float(order[0].split()[2][:-1]) >= 0.09


async def test_agent_history_rolls_back_on_cancel() -> None:
    agent = ClinicAgent(ScriptedClinicLLM(latency_s=5))

    async def dispatch(name: str, args: dict[str, Any], cid: str | None) -> dict[str, Any]:
        return {}

    ctx = TurnContext(
        session_id="s",
        transcript="hi",
        turn_index=0,
        tools=[],
        history_delta=[],
        flow=None,
        metadata={},
        _dispatch=dispatch,
    )
    task = asyncio.ensure_future(agent(ctx))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert len(agent._histories["s"]) == 1  # only the system prompt remains
