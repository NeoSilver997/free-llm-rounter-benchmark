from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmark.metrics import ModelMetrics, pareto_frontier
from benchmark.router_eval import PolicyMetrics
from catalog.task_types import TASK_TYPES
from routing.constraints import provider_index

CSV_COLUMNS = (
    "model_key",
    "provider",
    "task_id",
    "task_type",
    "outcome",
    "latency_ms",
    "ttft_ms",
    "prompt_tokens",
    "completion_tokens",
    "tokens_per_sec",
    "http_status",
    "error_class",
    "answer_hash",
    "rate_limit",
    "run_at",
)


def write_json(rows, path: Path | str) -> None:
    Path(path).write_text(json.dumps([r.as_dict() for r in rows], indent=2), encoding="utf-8")


def write_csv(rows, path: Path | str) -> None:
    """Long format, one row per attempt (RouterBench-style)."""
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.as_dict()[name] for name in CSV_COLUMNS})


def write_scores(metrics: dict[str, ModelMetrics], path: Path | str) -> None:
    """scores.json is the input routing/scoring.py reads on the next run."""
    payload = {
        key: {
            "provider": m.provider,
            "category": category_of(m.provider),
            "task_types": list(m.task_types),
            "pass_rate": m.pass_rate,
            "scored_n": m.scored_n,
            "p50_ms": m.p50_ms,
            "p95_ms": m.p95_ms,
            "tokens_per_sec": m.tokens_per_sec,
            "error_rate": m.error_rate,
            "coverage": list(m.coverage),
            "quota_cost": m.quota_cost,
            "on_pareto_frontier": key in pareto_frontier(metrics),
        }
        for key, m in sorted(metrics.items())
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_scores(path: Path | str) -> dict[str, ModelMetrics]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        key: ModelMetrics(
            key=key,
            provider=item["provider"],
            model=key.split("/", 1)[1],
            pass_rate=item["pass_rate"],
            scored_n=item["scored_n"],
            p50_ms=item["p50_ms"],
            p95_ms=item["p95_ms"],
            tokens_per_sec=item["tokens_per_sec"],
            error_rate=item["error_rate"],
            coverage=tuple(item["coverage"]),
            quota_cost=item["quota_cost"],
            task_types=tuple(item["task_types"]),
        )
        for key, item in raw.items()
    }


def category_of(provider: str) -> str:
    entry = provider_index().get(provider)
    return entry.category if entry else "unknown"


def write_html(
    metrics: dict[str, ModelMetrics],
    policy_metrics: dict[str, PolicyMetrics],
    path: Path | str,
    min_n: int = 5,
) -> None:
    """Grouped by provider category -> task type, then the policy table."""
    frontier = set(pareto_frontier(metrics))
    grouped: dict[str, dict[str, list[ModelMetrics]]] = {}
    for model in metrics.values():
        for task_type in model.task_types or ("unknown",):
            grouped.setdefault(category_of(model.provider), {}).setdefault(task_type, []).append(model)

    out = ["<!doctype html><meta charset='utf-8'><title>LLM Router benchmark</title>",
          "<style>body{font-family:system-ui;margin:2rem}table{border-collapse:collapse;margin:1rem 0}"
          "th,td{border:1px solid #ddd;padding:6px 10px;text-align:left}</style>"]

    for category in sorted(grouped):
        out.append(f"<h2>{category}</h2>")
        for task_type in [t for t in TASK_TYPES if t in grouped[category]] + ["unknown"]:
            models = grouped[category].get(task_type)
            if not models:
                continue
            out.append(f"<h3>{task_type}</h3>")
            out.append(
                "<table><tr><th>provider</th><th>model</th><th>scored n</th><th>pass rate</th>"
                "<th>p50 ms</th><th>p95 ms</th><th>tokens/s</th><th>429/err rate</th>"
                "<th>error class</th><th>gate</th><th>pareto</th></tr>"
            )
            for model in sorted(models, key=lambda m: m.key):
                outcomes = model.outcomes or {}
                error_class = ", ".join(sorted(outcomes)) or "-"
                gate = "ranked" if model.scored_n >= min_n else "coverage only"
                out.append(
                    "<tr><td>{p}</td><td>{m}</td><td>{n}</td><td>{pr}</td><td>{p50}</td><td>{p95}</td>"
                    "<td>{tps}</td><td>{err}</td><td>{ec}</td><td>{gate}</td><td>{frontier}</td></tr>".format(
                        p=model.provider,
                        m=model.model,
                        n=model.scored_n,
                        pr=_fmt(model.pass_rate),
                        p50=_fmt(model.p50_ms),
                        p95=_fmt(model.p95_ms),
                        tps=_fmt(model.tokens_per_sec),
                        err=_fmt(model.error_rate),
                        ec=error_class,
                        gate=gate,
                        frontier="yes" if model.key in frontier else "",
                    )
                )
            out.append("</table>")

    out.append("<h2>Policies</h2>")
    out.append(
        "<table><tr><th>policy</th><th>quality</th><th>mean cost</th><th>mean latency</th>"
        "<th>escalation</th><th>budget used</th><th>win rate at budget</th></tr>"
    )
    for pm in sorted(policy_metrics.values(), key=lambda p: p.policy):
        wins = ", ".join(f"{t}={v:.2f}" for t, v in pm.win_rate.items()) or "-"
        out.append(
            f"<tr><td>{pm.policy}</td><td>{_fmt(pm.quality)}</td><td>{_fmt(pm.mean_cost)}</td>"
            f"<td>{_fmt(pm.mean_latency_ms)}</td><td>{_fmt(pm.escalation_rate)}</td>"
            f"<td>{pm.budget_used}</td><td>{wins}</td></tr>"
        )
    out.append("</table>")

    Path(path).write_text("\n".join(out), encoding="utf-8")


def _fmt(value) -> str:
    return "unknown" if value is None else f"{value:.3f}"
