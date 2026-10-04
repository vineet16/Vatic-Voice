"""Baseline vs Vatic benchmark, reproducible with one command:

    python -m benchmarks.run_benchmark --out benchmarks/results

Pipeline: baseline LLM-only calls -> compile -> (train classifiers) -> shadow ->
calibrate -> promote -> calls with Vatic. Also measures compiled-turn share as a
function of how many sessions the compiler has seen. The LLM is the offline
scripted model with simulated network latency (or, with ``--agent llm``, a real
OpenAI-compatible model configured by OPENAI_API_KEY / OPENAI_BASE_URL /
VATIC_LLM_MODEL); callers are scripted with ASR noise.

Exit code is non-zero if a hard gate fails: membership p95 > 30 ms, p99 loop lag
> 5 ms, or wrong-route rate above the calibration target.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from examples.simulator.run import SimConfig, SimReport, run_simulation
from vatic.compiler.compile import CompileConfig, compile_corpus
from vatic.compiler.emit import write_flows
from vatic.core.loop_monitor import percentile
from vatic.ir.schema import save_flow
from vatic.lifecycle.promote import PromotionRules, evaluate, load_flows, shadow_stats
from vatic.trace.store import TraceStore

OFF_PATH_LABELS = {"side_question", "multi_intent", "correction"}


@dataclass
class Phase:
    name: str
    report: SimReport
    wrong_route: float
    membership_p95: float
    llm_calls_per_session: float

    def row(self) -> dict[str, Any]:
        all_ms = self.report.decision_ms["llm"] + self.report.decision_ms["compiled"]
        live = self.report.turns
        return {
            "phase": self.name,
            "sessions": self.report.sessions,
            "turn_latency_p50_ms": round(percentile(all_ms, 50), 1),
            "turn_latency_p95_ms": round(percentile(all_ms, 95), 1),
            "compiled_turn_latency_p50_ms": round(
                percentile(self.report.decision_ms["compiled"], 50), 2
            )
            if self.report.decision_ms["compiled"]
            else None,
            "llm_calls_per_session": round(self.llm_calls_per_session, 2),
            "compiled_share": round(self.report.compiled_turns / live, 4) if live else 0.0,
            "wrong_route_rate": round(self.wrong_route, 4),
            "task_success": round(self.report.success_rate, 4),
            "membership_p95_ms": round(self.membership_p95, 3),
            "loop_lag_p99_ms": round(self.report.loop_lag_p99_ms, 2),
        }


def _phase_stats(store: TraceStore, prefix: str) -> tuple[float, float, float]:
    """(ground-truth wrong-route rate, membership p95 ms, LLM calls per session)."""
    labels: dict[str, list[dict[str, Any]]] = {
        s.session_id: s.metadata.get("labels", [])
        for s in store.iter_sessions()
        if s.session_id.startswith(prefix)
    }
    compiled = wrong = calls = 0
    membership: list[float] = []
    for t in store.iter_turns():
        if not t.session_id.startswith(prefix) or t.route == "shadow":
            continue
        calls += t.llm_calls if t.route == "llm" else 0
        if t.membership is not None and not t.membership.reason.startswith("confirm:"):
            membership.append(t.membership.latency_ms)
        if t.route == "compiled":
            compiled += 1
            lab = labels.get(t.session_id, [])
            if t.turn_index < len(lab) and lab[t.turn_index]["label"] in OFF_PATH_LABELS:
                wrong += 1
    n = max(1, len(labels))
    return (wrong / compiled if compiled else 0.0), percentile(membership, 95), calls / n


async def _run(cfg: SimConfig, store: TraceStore, name: str) -> Phase:
    report = await run_simulation(cfg)
    wrong, mem_p95, calls = await asyncio.to_thread(
        _phase_stats, store, f"{cfg.prefix}-{cfg.seed}-"
    )
    return Phase(name, report, wrong, mem_p95, calls)


def _compile(store_dir: Path, flows: Path, limit: int | None = None) -> list[str]:
    store = TraceStore(store_dir)
    corpus = store.load_corpus(outcome=None)
    corpus = [c for c in corpus if c[0].session_id.startswith("base-")]
    if limit is not None:
        corpus = sorted(corpus, key=lambda c: c[0].session_id)[:limit]
    res = compile_corpus(corpus, store.tool_manifest(), CompileConfig())
    write_flows(res.accepted, flows, res.examples)
    return [f.flow_id for f in res.accepted]


def _activate_all(flows: Path) -> None:
    for f in load_flows(flows):
        f.status = "active"
        save_flow(f, flows / f"{f.flow_id}.yaml")


async def benchmark(
    out: Path,
    sessions: int,
    shadow_sessions: int,
    latency_ms: float,
    train: bool,
    alpha: float,
    seed: int,
    agent: str = "scripted",
    concurrency: int = 32,
    curve: bool = True,
) -> dict[str, Any]:
    await asyncio.to_thread(shutil.rmtree, out, ignore_errors=True)
    store_dir, flows = out / "traces", out / "flows"
    store = TraceStore(store_dir)
    lat = latency_ms / 1000 if agent == "scripted" else 0.0
    common: dict[str, Any] = {
        "store": store_dir,
        "llm_latency_s": lat,
        "llm_jitter_s": lat / 2,
        "concurrency": concurrency,
        "agent": agent,
    }

    print(f"[1/6] baseline: {sessions} LLM-only sessions")
    base = await _run(
        SimConfig(sessions=sessions, seed=seed, prefix="base", **common), store, "baseline"
    )
    print("[2/6] compile")
    compiled = await asyncio.to_thread(_compile, store_dir, flows)
    await asyncio.to_thread(evaluate, flows, store)  # valid candidates -> shadow
    trained: dict[str, Any] = {}
    if train:
        print("[3/6] train per-step classifiers")
        from vatic.lifecycle.train import train_flow

        for f in await asyncio.to_thread(load_flows, flows):
            r = await asyncio.to_thread(train_flow, f, flows)
            trained[f.flow_id] = {"steps": r.steps, "holdout_accuracy": r.holdout_accuracy}
    print(f"[4/6] shadow: {shadow_sessions} sessions")
    await _run(
        SimConfig(sessions=shadow_sessions, seed=seed + 1, prefix="shadow", flows=flows, **common),
        store,
        "shadow",
    )
    calibration: dict[str, Any] = {}
    if train:
        from vatic.lifecycle.calibrate import calibrate_flow

        for f in await asyncio.to_thread(load_flows, flows):
            rep = await asyncio.to_thread(calibrate_flow, f, store, flows, alpha=alpha)
            calibration[f.flow_id] = {s.step_id: s.as_dict() for s in rep.steps}
    print("[5/6] promote")
    events = await asyncio.to_thread(evaluate, flows, store, PromotionRules())
    shadow = {f.flow_id: shadow_stats(store, f).as_dict() for f in load_flows(flows)}
    print(f"[6/6] Vatic: {sessions} sessions")
    vatic = await _run(
        SimConfig(sessions=sessions, seed=seed + 2, prefix="vatic", flows=flows, **common),
        store,
        "vatic",
    )

    learning_curve = []
    points = sorted({k for k in (25, 50, 100, 200, 300, sessions) if k <= sessions})
    if curve:
        print("[+] learning curve: compiled share vs sessions observed")
    for k in points if curve else []:
        cflows = out / f"curve-{k}"
        await asyncio.to_thread(_compile, store_dir, cflows, k)
        await asyncio.to_thread(_activate_all, cflows)
        r = await run_simulation(
            SimConfig(
                sessions=150,
                seed=seed + 100,
                prefix=f"curve{k}",
                store=out / f"curve-{k}-traces",
                flows=cflows,
                concurrency=concurrency,
                agent=agent,
            )
        )
        learning_curve.append(
            {"sessions_observed": k, "compiled_share": round(r.compiled_turns / r.turns, 4)}
        )

    results = {
        "config": {
            "sessions": sessions,
            "shadow_sessions": shadow_sessions,
            "llm_latency_ms": latency_ms,
            "alpha": alpha,
            "seed": seed,
            "classifier": train,
            "agent": agent,
            "model": os.environ.get("VATIC_LLM_MODEL") if agent == "llm" else "scripted",
        },
        "phases": [base.row(), vatic.row()],
        "flows_compiled": compiled,
        "promotions": [{"flow": e.flow_id, "to": e.to_status, "reason": e.reason} for e in events],
        "shadow": shadow,
        "classifiers": trained,
        "calibration": calibration,
        "learning_curve": learning_curve,
        "llm_usage": {
            name: r.llm_usage for name, r in (("baseline", base.report), ("vatic", vatic.report))
        },
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results


def table(results: dict[str, Any]) -> str:
    b, v = results["phases"]
    rows = [
        ("Turn latency p50 (ms)", "turn_latency_p50_ms"),
        ("Turn latency p95 (ms)", "turn_latency_p95_ms"),
        ("LLM calls / session", "llm_calls_per_session"),
        ("Compiled turns", "compiled_share"),
        ("Wrong-route rate", "wrong_route_rate"),
        ("Task success", "task_success"),
    ]
    out = ["| Metric | Baseline | Vatic |", "|---|---:|---:|"]
    for label, key in rows:
        fmt = (
            (lambda x: f"{x:.1%}")
            if key in ("compiled_share", "task_success", "wrong_route_rate")
            else (lambda x: f"{x:g}")
        )
        out.append(f"| {label} | {fmt(b[key])} | {fmt(v[key])} |")
    return "\n".join(out)


BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1-2 (validated reference palette)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e7e6e2", "#fcfcfb"


def chart(results: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    b, v = results["phases"]
    panels = [
        ("Turn latency p50 (ms)", "turn_latency_p50_ms", "{:.0f}"),
        ("Turn latency p95 (ms)", "turn_latency_p95_ms", "{:.0f}"),
        ("LLM calls per session", "llm_calls_per_session", "{:.1f}"),
        ("Compiled turns", "compiled_share", "{:.0%}"),
        ("Wrong-route rate", "wrong_route_rate", "{:.1%}"),
        ("Task success", "task_success", "{:.1%}"),
    ]
    plt.rcParams.update(
        {
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
        }
    )
    fig = plt.figure(figsize=(13, 7.2), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 4, height_ratios=[1, 1.05], hspace=0.55, wspace=0.45)
    for i, (title, key, fmt) in enumerate(panels):
        ax = fig.add_subplot(grid[i // 3, i % 3] if i < 3 else grid[1, i - 3])
        vals = [b[key], v[key]]
        bars = ax.bar([0, 1], vals, width=0.6, color=[BLUE, ORANGE], edgecolor=SURFACE, linewidth=2)
        for bar, val in zip(bars, vals, strict=True):
            ax.annotate(
                fmt.format(val),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                color=INK,
                fontsize=9,
            )
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_xticks([0, 1], ["Baseline", "Vatic"])
        ax.set_facecolor(SURFACE)
        if key in ("compiled_share", "task_success"):
            ax.set_ylim(0, 1.12)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        elif key == "wrong_route_rate":
            target = results["config"]["alpha"]
            ax.axhline(target, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
            ax.annotate(
                f"target {target:.0%}",
                (1.0, target),
                xycoords=("axes fraction", "data"),
                xytext=(0, 3),
                textcoords="offset points",
                ha="right",
                fontsize=8,
                color=INK_2,
            )
            ax.set_ylim(0, max(max(vals), target) * 1.6)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
        else:
            ax.set_ylim(0, max(max(vals) * 1.25, 1e-3))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(axis="y", length=0)
    curve = results["learning_curve"]
    ax = fig.add_subplot(grid[:, 3])
    if not curve:
        ax.set_axis_off()
    xs = [c["sessions_observed"] for c in curve]
    ys = [c["compiled_share"] for c in curve]
    ax.plot(
        xs,
        ys,
        color=ORANGE,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
    )
    if ys:
        ax.annotate(
            f"{ys[-1]:.0%}",
            (xs[-1], ys[-1]),
            xytext=(-4, 8),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color=INK,
        )
    ax.set_title("Compiled-turn share\nvs sessions observed", loc="left", fontsize=10, color=INK)
    ax.set_xlabel("baseline sessions compiled from")
    ax.set_ylim(0, max([*ys, 0.1]) * 1.25)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    cfg = results["config"]
    llm = (
        f"real LLM {cfg['model']}"
        if cfg.get("agent") == "llm"
        else f"simulated LLM latency {cfg['llm_latency_ms']:.0f} ms"
    )
    fig.suptitle(
        f"Vatic vs LLM-only baseline - {cfg['sessions']} simulated clinic calls per arm, {llm}",
        x=0.06,
        ha="left",
        fontsize=12,
        color=INK,
    )
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main(
    out: Path = typer.Option(Path("benchmarks/results")),
    sessions: int = typer.Option(500),
    shadow_sessions: int = typer.Option(1000),
    llm_latency_ms: float = typer.Option(400.0),
    train: bool = typer.Option(True, help="train + calibrate per-step classifiers (needs torch)"),
    alpha: float = typer.Option(0.01),
    seed: int = typer.Option(2026),
    agent: str = typer.Option("scripted", help="scripted | llm (real OpenAI-compatible model)"),
    concurrency: int = typer.Option(32),
    curve: bool = typer.Option(True, help="measure compiled share vs sessions observed"),
) -> None:
    results = asyncio.run(
        benchmark(
            out,
            sessions,
            shadow_sessions,
            llm_latency_ms,
            train,
            alpha,
            seed,
            agent=agent,
            concurrency=concurrency,
            curve=curve,
        )
    )
    md = table(results)
    (out / "results.md").write_text(md + "\n")
    chart(results, out / "benchmark.png")
    print("\n" + md)
    v = results["phases"][1]
    gates = {
        "membership p95 <= 30 ms": v["membership_p95_ms"] <= 30,
        "loop lag p99 <= 5 ms": v["loop_lag_p99_ms"] <= 5,
        f"wrong-route rate <= {alpha:.0%}": v["wrong_route_rate"] <= alpha,
    }
    for name, ok in gates.items():
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"\nwrote {out / 'results.json'}, {out / 'results.md'}, {out / 'benchmark.png'}")
    if not all(gates.values()):
        sys.exit(1)


if __name__ == "__main__":
    typer.run(main)
