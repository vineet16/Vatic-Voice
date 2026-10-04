"""Plain-text reports for the CLI."""

from __future__ import annotations

from vatic.lifecycle.promote import ShadowStats
from vatic.telemetry.metrics import Metrics, _pcts


def table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True))  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths]), *(line(r) for r in rows)])


def metrics_report(m: Metrics) -> str:
    out = [
        f"sessions            {m.sessions}  (success rate {m.success_rate:.1%})",
        f"turns               {m.turns}  (compiled {m.compiled_share:.1%})",
        f"llm calls/session   {m.llm_calls_per_session:.2f}",
        f"hedge rate          {m.hedge_rate:.1%}",
        f"wrong-route rate    {m.wrong_route_rate():.2%}  (shadow, on-path decisions)",
        "",
    ]
    rows = []
    for route, values in sorted(m.decision_ms.items()):
        p = _pcts(values)
        rows.append(
            [f"decision ({route})", str(p["n"]), str(p.get("p50", "-")), str(p.get("p95", "-"))]
        )
    for name, values in [
        ("membership check", m.membership_ms),
        ("end-to-end", m.e2e_ms),
        ("time to first audio", m.ttfa_ms),
    ]:
        p = _pcts(values)
        rows.append([name, str(p["n"]), str(p.get("p50", "-")), str(p.get("p95", "-"))])
    out.append(table(rows, ["latency (ms)", "n", "p50", "p95"]))
    if m.fallback_reasons:
        out += [
            "",
            table(
                [[k, str(v)] for k, v in m.fallback_reasons.most_common()],
                ["fallback reason", "count"],
            ),
        ]
    return "\n".join(out)


def shadow_report(stats: list[ShadowStats]) -> str:
    rows = []
    for s in stats:
        wrong = sum(w for w, _ in s.wrong_route.values())
        on = sum(n for _, n in s.wrong_route.values())
        rows.append(
            [
                s.flow_id,
                str(s.version),
                str(s.sessions),
                f"{s.matched}/{s.compared}",
                f"{s.match_rate:.1%}" if s.compared else "-",
                str(s.irreversible_mismatches),
                f"{wrong}/{on}",
            ]
        )
    return table(
        rows,
        ["flow", "ver", "sessions", "matched", "match", "irrev. mismatch", "wrong-route"],
    )
