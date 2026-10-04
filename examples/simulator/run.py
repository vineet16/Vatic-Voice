"""Batch simulation runner.

Runs N simulated calls through VaticRuntime with the clinic agent as the LLM
fallback, stores traces, and judges task success from backend state.

    python -m examples.simulator.run --sessions 500 --store .vatic/traces [--flows flows]
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import typer

from examples.scheduling.agent_llm import ClinicAgent
from examples.scheduling.backend import BackendPool, ClinicBackend
from examples.scheduling.scripted_llm import ScriptedClinicLLM
from examples.scheduling.tools import build_registry
from examples.simulator.asr_noise import AsrNoise
from examples.simulator.caller import LLMCaller, Persona, ScriptedCaller, Utterance, make_persona
from vatic.core.runtime import RuntimeConfig, VaticRuntime
from vatic.llm.client import LLMClient, OpenAICompatibleClient
from vatic.trace.schema import SessionOutcome
from vatic.trace.store import TraceStore


@dataclass
class SimConfig:
    sessions: int = 100
    seed: int = 7
    store: Path = Path(".vatic/traces")
    flows: Path | None = None
    asr_noise: float = 0.1
    llm_latency_s: float = 0.0
    llm_jitter_s: float = 0.0
    concurrency: int = 8
    max_turns: int = 14
    prefix: str = "sim"
    caller: str = "scripted"  # or "llm"
    agent: str = "scripted"  # or "llm"


@dataclass
class SimReport:
    sessions: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    turns: int = 0
    compiled_turns: int = 0
    llm_calls: int = 0
    decision_ms: dict[str, list[float]] = field(default_factory=lambda: {"llm": [], "compiled": []})
    wall_s: float = 0.0
    loop_lag_p99_ms: float = 0.0
    dropped_traces: int = 0
    llm_usage: dict[str, int] = field(default_factory=dict)  # real-LLM agent only

    @property
    def success_rate(self) -> float:
        return self.outcomes["success"] / self.sessions if self.sessions else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "sessions": self.sessions,
            "outcomes": dict(self.outcomes),
            "success_rate": round(self.success_rate, 4),
            "turns": self.turns,
            "compiled_turns": self.compiled_turns,
            "llm_calls": self.llm_calls,
            "wall_s": round(self.wall_s, 2),
            "loop_lag_p99_ms": round(self.loop_lag_p99_ms, 2),
            "dropped_traces": self.dropped_traces,
            **({"llm_usage": self.llm_usage} if self.llm_usage else {}),
        }


def judge(backend: ClinicBackend, persona: Persona, finished: bool) -> SessionOutcome:
    """Task success from backend state only."""
    before = backend.initial_state
    after = backend.snapshot()
    ok = False
    if persona.goal == "book":
        mine = [a for a in backend.appointments_for(persona.patient_id) if a["status"] == "booked"]
        ok = len(after) == len(before) + 1 and any(
            a["date"] == persona.target_date and a["time"] == persona.target_time for a in mine
        )
    elif persona.goal == "cancel":
        appts = {a["id"]: a for a in backend.appointments_for(persona.patient_id)}
        changed = [r for r in after if r not in before]
        ok = (
            appts.get(persona.appointment_id or "", {}).get("status") == "cancelled"
            and len(changed) == 1
        )
    elif persona.goal == "reschedule":
        appts = {a["id"]: a for a in backend.appointments_for(persona.patient_id)}
        a = appts.get(persona.appointment_id or "", {})
        changed = [r for r in after if r not in before]
        ok = (
            a.get("status") == "booked"
            and a.get("date") == persona.target_date
            and a.get("time") == persona.target_time
            and len(changed) == 1
        )
    else:
        ok = after == before and finished
    if ok:
        return "success"
    return "failure" if finished else "abandoned"


async def run_simulation(cfg: SimConfig, runtime_config: RuntimeConfig | None = None) -> SimReport:
    pool = BackendPool()
    registry = build_registry(pool)
    store = TraceStore(cfg.store)
    llm: LLMClient
    if cfg.agent == "llm":
        llm = OpenAICompatibleClient()
    else:
        llm = ScriptedClinicLLM(latency_s=cfg.llm_latency_s, jitter_s=cfg.llm_jitter_s)
    agent = ClinicAgent(llm)
    caller_llm = OpenAICompatibleClient() if cfg.caller == "llm" else None
    noise = AsrNoise(cfg.asr_noise)
    report = SimReport()
    t0 = time.perf_counter()

    # Harness setup (SQLite seeding, personas) happens before the runtime starts so
    # it never competes with the event loop for the GIL during the measured run.
    setups = await asyncio.to_thread(_setup_all, cfg)

    async with VaticRuntime(registry, cfg.flows, store, config=runtime_config) as runtime:

        async def one(i: int) -> None:
            sid = f"{cfg.prefix}-{cfg.seed}-{i:05d}"
            backend, persona, rng = setups[i]
            pool.add(sid, backend)
            labels: list[dict[str, Any]] = []
            runtime.start_session(
                sid,
                {
                    "today": backend.today.isoformat(),
                    "goal": persona.goal,
                    "style": persona.style,
                    "labels": labels,
                },
            )
            caller: ScriptedCaller | LLMCaller
            utt: Utterance | None
            if caller_llm is not None:
                caller = LLMCaller(persona, caller_llm)
                utt = await caller.opening()
            else:
                caller = ScriptedCaller(persona, rng)
                utt = caller.opening()
            finished = False
            for _ in range(cfg.max_turns):
                assert utt is not None
                heard, applied = noise.apply(utt.text, rng)
                labels.append({"label": utt.label, "noise": applied, "said": utt.text})
                res = await runtime.handle_turn(sid, heard, agent)
                report.turns += 1
                if res.route == "compiled":
                    report.compiled_turns += 1
                if isinstance(caller, LLMCaller):
                    utt = await caller.respond(res.text)
                else:
                    utt = caller.respond(res.text)
                if utt is None:
                    finished = True
                    break
            outcome = await asyncio.to_thread(judge, backend, persona, finished)
            report.outcomes[outcome] += 1
            report.sessions += 1
            await runtime.end_session(sid, outcome)
            agent.end_session(sid)
            pool.remove(sid)

        indices = iter(range(cfg.sessions))

        async def worker() -> None:
            for i in indices:
                await one(i)

        await asyncio.gather(*(worker() for _ in range(cfg.concurrency)))
        await runtime.flush()
        report.loop_lag_p99_ms = runtime.loop_monitor.p(99)
        report.dropped_traces = runtime.dropped_traces
    report.wall_s = time.perf_counter() - t0
    if isinstance(llm, OpenAICompatibleClient):
        report.llm_usage = dict(llm.usage)
        await llm.aclose()
    await asyncio.to_thread(_collect, store, f"{cfg.prefix}-{cfg.seed}-", report)
    store.close()
    return report


def _setup_all(cfg: SimConfig) -> list[tuple[ClinicBackend, Persona, random.Random]]:
    out = []
    for i in range(cfg.sessions):
        rng = random.Random(f"{cfg.seed}:{i}")
        backend = ClinicBackend(seed=cfg.seed * 1_000_003 + i)
        out.append((backend, make_persona(backend, rng), rng))
    return out


def _collect(store: TraceStore, prefix: str, report: SimReport) -> None:
    for t in store.iter_turns():
        if t.route == "shadow" or not t.session_id.startswith(prefix):
            continue
        report.llm_calls += t.llm_calls if t.route == "llm" else 0
        if t.timings.decision_start and t.timings.decision_end:
            ms = (t.timings.decision_end - t.timings.decision_start) * 1000
            report.decision_ms["compiled" if t.route == "compiled" else "llm"].append(ms)


def main(
    sessions: int = typer.Option(100),
    seed: int = typer.Option(7),
    store: Path = typer.Option(Path(".vatic/traces")),
    flows: Path | None = typer.Option(None),
    asr_noise: float = typer.Option(0.1),
    llm_latency_ms: float = typer.Option(0.0, help="simulated LLM latency per call"),
    concurrency: int = typer.Option(8),
    prefix: str = typer.Option("sim"),
    caller: str = typer.Option("scripted", help="scripted | llm"),
    agent: str = typer.Option("scripted", help="scripted | llm"),
) -> None:
    cfg = SimConfig(
        sessions=sessions,
        seed=seed,
        store=store,
        flows=flows,
        asr_noise=asr_noise,
        llm_latency_s=llm_latency_ms / 1000,
        concurrency=concurrency,
        prefix=prefix,
        caller=caller,
        agent=agent,
    )
    report = asyncio.run(run_simulation(cfg))
    for k, v in report.summary().items():
        typer.echo(f"{k:>18}: {v}")


if __name__ == "__main__":
    typer.run(main)
