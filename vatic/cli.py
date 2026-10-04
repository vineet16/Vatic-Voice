"""``vatic`` command-line interface."""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

import typer

from vatic.compiler.compile import CompileConfig, compile_corpus
from vatic.compiler.emit import write_flows
from vatic.ir.schema import dump_flow, load_flow
from vatic.lifecycle.promote import (
    PromotionRules,
    demote_one,
    evaluate,
    flow_path,
    load_flows,
    promote_one,
    shadow_stats,
)
from vatic.telemetry.metrics import compute
from vatic.telemetry.report import metrics_report, shadow_report, table
from vatic.trace.store import TraceStore

app = typer.Typer(help="Vatic: compile recurring LLM call flows into guarded graphs.")
traces_app = typer.Typer(help="Inspect stored traces.")
flows_app = typer.Typer(help="Inspect compiled flows.")
shadow_app = typer.Typer(help="Shadow-mode results.")
app.add_typer(traces_app, name="traces")
app.add_typer(flows_app, name="flows")
app.add_typer(shadow_app, name="shadow")

StoreOpt = typer.Option(
    Path(os.environ.get("VATIC_STORE", ".vatic/traces")), "--store", help="trace store directory"
)
FlowsOpt = typer.Option(
    Path(os.environ.get("VATIC_FLOWS", "flows")), "--flows", help="flows directory"
)


def _store(path: Path) -> TraceStore:
    if not path.exists():
        raise typer.BadParameter(f"no trace store at {path}")
    return TraceStore(path)


@traces_app.command("stats")
def traces_stats(store: Path = StoreOpt) -> None:
    """Counts, sessions and success rate."""
    s = _store(store)
    sessions = list(s.iter_sessions())
    outcomes: dict[str, int] = {}
    for x in sessions:
        outcomes[x.outcome or "unknown"] = outcomes.get(x.outcome or "unknown", 0) + 1
    routes: dict[str, int] = {}
    for t in s.iter_turns():
        routes[t.route] = routes.get(t.route, 0) + 1
    ok = outcomes.get("success", 0)
    typer.echo(f"sessions: {len(sessions)}  success rate: {ok / max(1, len(sessions)):.1%}")
    typer.echo(f"outcomes: {json.dumps(outcomes, sort_keys=True)}")
    typer.echo(f"turns:    {json.dumps(routes, sort_keys=True)}")


@app.command("compile")
def compile_cmd(
    store: Path = StoreOpt,
    out: Path = typer.Option(Path("flows"), "--out"),
    min_support: int = typer.Option(20, "--min-support"),
) -> None:
    """Compile successful traces into candidate flows (deterministic, no LLM)."""
    s = _store(store)
    catalog = s.tool_manifest()
    if not catalog:
        raise typer.BadParameter(f"{store} has no tools.json (run the runtime with this store)")
    result = compile_corpus(s.load_corpus(outcome=None), catalog, CompileConfig(min_support))
    report = write_flows(result.accepted, out, result.examples)
    for f in result.accepted:
        state = "unchanged" if f.flow_id in report.unchanged else "written"
        notes = f"  ({len(f.provenance.notes)} traces dropped)" if f.provenance.notes else ""
        typer.echo(f"ACCEPTED  {f.flow_id:28} support={f.provenance.support:<4} {state}{notes}")
    for r in result.refused:
        typer.echo(f"REFUSED   {r.flow_id:28} support={r.support:<4} {r.reason}")
    for sk in result.skipped:
        sig = " -> ".join(sk.signature)
        typer.echo(f"SKIPPED   {sig[:60]:60} support={sk.support}")


@flows_app.command("list")
def flows_list(flows: Path = FlowsOpt) -> None:
    rows = [
        [
            f.flow_id,
            str(f.version),
            f.status,
            str(f.provenance.support),
            " -> ".join(f.provenance.tool_signature),
        ]
        for f in load_flows(flows)
    ]
    typer.echo(table(rows, ["flow", "ver", "status", "support", "tools"]))


@flows_app.command("show")
def flows_show(flow_id: str, flows: Path = FlowsOpt) -> None:
    typer.echo(dump_flow(load_flow(flow_path(flows, flow_id))))


@flows_app.command("diff")
def flows_diff(flow_id: str, v1: int, v2: int, flows: Path = FlowsOpt) -> None:
    def text(v: int) -> list[str]:
        current = load_flow(flow_path(flows, flow_id))
        if current.version == v:
            return dump_flow(current).splitlines(keepends=True)
        path = flows / ".history" / f"{flow_id}.v{v}.yaml"
        if not path.exists():
            raise typer.BadParameter(f"version {v} of {flow_id} not found")
        return path.read_text().splitlines(keepends=True)

    typer.echo("".join(difflib.unified_diff(text(v1), text(v2), f"v{v1}", f"v{v2}")))


@shadow_app.command("report")
def shadow_report_cmd(store: Path = StoreOpt, flows: Path = FlowsOpt) -> None:
    s = _store(store)
    typer.echo(shadow_report([shadow_stats(s, f) for f in load_flows(flows)]))


@app.command("promote")
def promote_cmd(
    flow_id: str | None = typer.Argument(None),
    auto: bool = typer.Option(False, "--auto", help="apply promotion rules to all flows"),
    min_sessions: int = typer.Option(50, "--min-sessions"),
    min_match_rate: float = typer.Option(0.98, "--min-match-rate"),
    store: Path = StoreOpt,
    flows: Path = FlowsOpt,
) -> None:
    """Promote one flow a step (manual override, logged) or apply the rules (--auto)."""
    s = _store(store)
    if auto:
        rules = PromotionRules(min_sessions=min_sessions, min_match_rate=min_match_rate)
        events = evaluate(flows, s, rules)
    elif flow_id:
        events = [promote_one(flows, flow_id, s)]
    else:
        raise typer.BadParameter("give a flow id or --auto")
    for e in events:
        typer.echo(f"{e.flow_id}: {e.from_status} -> {e.to_status} ({e.reason})")
    if not events:
        typer.echo("no transitions")


@app.command("demote")
def demote_cmd(
    flow_id: str,
    reason: str = typer.Option("manual demotion", "--reason"),
    store: Path = StoreOpt,
    flows: Path = FlowsOpt,
) -> None:
    """Demote a flow (active -> shadow, otherwise -> retired). Logged."""
    e = demote_one(flows, flow_id, reason, store=_store(store))
    typer.echo(f"{e.flow_id}: {e.from_status} -> {e.to_status} ({e.reason})")


@app.command("train")
def train_cmd(
    flows: Path = FlowsOpt,
    flow_id: str | None = typer.Argument(None, help="train one flow (default: all)"),
    epochs: int = typer.Option(3, "--epochs"),
    base_model: str = typer.Option("cross-encoder/ms-marco-TinyBERT-L2-v2", "--base-model"),
) -> None:
    """Train per-step membership classifiers (needs torch + transformers)."""
    from vatic.lifecycle.train import train_flow

    for flow in load_flows(flows):
        if flow_id and flow.flow_id != flow_id:
            continue
        r = train_flow(flow, flows, epochs=epochs, base_model=base_model)
        acc = f" holdout acc {r.holdout_accuracy:.1%}" if r.holdout_accuracy is not None else ""
        typer.echo(f"{flow.flow_id}: trained {r.steps} on {r.examples} examples{acc}")
        for sid, why in sorted(r.skipped.items()):
            typer.echo(f"  {sid}: no classifier ({why})")


@app.command("calibrate")
def calibrate_cmd(
    store: Path = StoreOpt,
    flows: Path = FlowsOpt,
    flow_id: str | None = typer.Argument(None),
    alpha: float = typer.Option(0.01, "--alpha", help="target maximum wrong-route rate"),
) -> None:
    """Set membership thresholds per step by conformal risk control on shadow data."""
    from vatic.lifecycle.calibrate import calibrate_flow

    s = _store(store)
    for flow in load_flows(flows):
        if flow_id and flow.flow_id != flow_id:
            continue
        report = calibrate_flow(flow, s, flows, alpha=alpha)
        for st in report.steps:
            typer.echo(f"{flow.flow_id}.{st.step_id}: {json.dumps(st.as_dict(), sort_keys=True)}")


@app.command("metrics")
def metrics_cmd(
    store: Path = StoreOpt,
    prefix: str | None = typer.Option(None, "--sessions", help="session id prefix filter"),
    as_json: bool = typer.Option(False, "--json"),
) -> None:
    """Latency and coverage summary."""
    m = compute(_store(store), prefix)
    typer.echo(json.dumps(m.as_dict(), indent=2) if as_json else metrics_report(m))


if __name__ == "__main__":
    app()
