from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from benchmark.metrics import ModelMetrics, aggregate
from benchmark.runner import Row
from catalog.loader import load_catalog, model_index
from catalog.rate_limits import parse_rate_limit, quota_cost
from catalog.schema import Catalog
from catalog.task_types import model_task_types
from routing.policies import AbstractRouter, RoutingContext
from routing.request import RoutingRequest
from routing.scoring import DEFAULT_WEIGHTS


@dataclass
class PolicyMetrics:
    policy: str
    quality: float | None = None
    mean_cost: float | None = None
    mean_latency_ms: float | None = None
    escalation_rate: float | None = None
    budget_used: float = 0.0
    win_rate: dict[str, float] = field(default_factory=dict)
    decisions: int = 0
    scored: int = 0

    def as_dict(self) -> dict:
        return {
            "policy": self.policy,
            "quality": self.quality,
            "mean_cost": self.mean_cost,
            "mean_latency_ms": self.mean_latency_ms,
            "escalation_rate": self.escalation_rate,
            "budget_used": self.budget_used,
            "win_rate": self.win_rate,
            "decisions": self.decisions,
            "scored": self.scored,
        }


def replay(
    policies: Sequence[AbstractRouter],
    tasks: list,
    recorded: dict[str, dict[str, Row]],
    weights: dict[str, float] | None = None,
    catalog: Catalog | None = None,
    ctx: RoutingContext | None = None,
) -> dict[str, PolicyMetrics]:
    """Offline policy replay: no extra API calls, only recorded answers are read."""
    catalog = catalog or load_catalog()
    rows = [row for per_task in recorded.values() for row in per_task.values()]
    metrics = aggregate(rows)

    ctx = ctx or RoutingContext(
        keys={row.provider: "recorded" for row in rows},
        metrics=metrics,
        weights=weights or dict(DEFAULT_WEIGHTS),
        pass_rate_by_task=_pass_rate_by_task(metrics),
    )
    index = model_index(catalog)

    out: dict[str, PolicyMetrics] = {}
    for policy in policies:
        decisions: list[tuple[object, object]] = []
        for task in tasks:
            candidates = [
                model
                for key in {row.model_key for row in recorded.get(task.id, {}).values()}
                for model in index.get(key, [])
                if task.task_type in model_task_types(model)
            ]
            if not candidates:
                continue
            req = RoutingRequest(
                task_type=task.task_type,
                messages=({"role": "user", "content": task.prompt},),
                policy=policy.name,
            )
            decisions.append((task, policy.route(req, candidates, ctx)))
        out[policy.name] = policy_metrics(decisions, recorded, ctx)
    return out


def policy_metrics(
    decisions: list[tuple], recorded: dict[str, dict[str, Row]], ctx: RoutingContext
) -> PolicyMetrics:
    """Quality, normalized cost, latency, escalation rate and budget used for one policy."""
    pm = PolicyMetrics(policy=decisions[0][1].policy if decisions else "unknown")
    qualities: list[float] = []
    costs: list[float] = []
    latencies: list[float] = []
    escalated = 0
    by_task: dict[str, dict[str, int]] = {}

    for task, decision in decisions:
        pm.decisions += 1
        row = recorded.get(task.id, {}).get(decision.key)
        if row is None:
            continue
        if row.outcome in ("pass", "fail"):
            pm.scored += 1
            value = 1.0 if row.outcome == "pass" else 0.0
            qualities.append(value)
            stats = by_task.setdefault(task.task_type, {"n": 0, "wins": 0})
            stats["n"] += 1
            stats["wins"] += int(row.outcome == "pass")

        cost = quota_cost(parse_rate_limit(row.rate_limit), 1, row.prompt_tokens + row.completion_tokens)
        if cost is not None:
            costs.append(cost)
            pm.budget_used += cost

        if row.latency_ms is not None:
            latencies.append(float(row.latency_ms))

        cheapest = _cheapest(decision, ctx)
        if cheapest and decision.key != cheapest:
            escalated += 1

    pm.quality = sum(qualities) / len(qualities) if qualities else None
    pm.mean_cost = sum(costs) / len(costs) if costs else None
    pm.mean_latency_ms = sum(latencies) / len(latencies) if latencies else None
    pm.escalation_rate = escalated / pm.decisions if pm.decisions else None
    pm.budget_used = round(pm.budget_used, 6)
    pm.win_rate = {t: stats["wins"] / stats["n"] for t, stats in sorted(by_task.items())}
    return pm


def _cheapest(decision, ctx: RoutingContext) -> str | None:
    keys = [a.key for a in (decision.attempts or [decision])]
    costs = {k: quota_cost(parse_rate_limit(_rate_limit(ctx, k)), 1, 512) for k in keys}
    known = {k: v for k, v in costs.items() if v is not None}
    return min(known, key=lambda k: known[k]) if known else None


def _rate_limit(ctx: RoutingContext, key: str) -> str:
    metrics = ctx.metrics.get(key)
    return metrics.raw_rate_limit if metrics else ""


def _pass_rate_by_task(metrics: dict[str, ModelMetrics]) -> dict[str, dict[str, float]]:
    return {key: {t: m.pass_rate for t in m.task_types} for key, m in metrics.items() if m.pass_rate is not None}


def cost_quality_curve(policy_metrics: dict[str, PolicyMetrics]) -> list[tuple[float, float]]:
    """(normalized cost, quality) per policy, for the RouterBench-style plot."""
    return [
        (pm.mean_cost, pm.quality)
        for pm in sorted(policy_metrics.values(), key=lambda p: (p.mean_cost is None, p.mean_cost or 0.0))
        if pm.mean_cost is not None and pm.quality is not None
    ]


def render_plot(curves: list[tuple[float, float]], width: int = 40, height: int = 12) -> str:
    """ASCII scatter plus an SVG, no external assets."""
    grid = [[" "] * width for _ in range(height)]
    for cost, quality in curves:
        x = min(width - 1, int(round(cost * (width - 1))))
        y = min(height - 1, int(round((1.0 - quality) * (height - 1))))
        grid[y][x] = "*"

    lines = ["quality ^" + " " * (width - 10) + "cost >"]
    for row in grid:
        lines.append("| " + "".join(row) + " |")
    lines.append("+ " + "-" * width + " +")
    for cost, quality in curves:
        lines.append(f"  cost={cost:.4f} quality={quality:.4f}")

    svg = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="300" viewBox="0 0 480 300">',
        '<rect width="480" height="300" fill="#fff"/>',
        '<line x1="40" y1="260" x2="460" y2="260" stroke="#888"/>',
        '<line x1="40" y1="20" x2="40" y2="260" stroke="#888"/>',
    ]
    for cost, quality in curves:
        cx = 40 + cost * 420
        cy = 260 - quality * 240
        svg.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4" fill="#2b6cb0"/>')
    svg.append("</svg>")
    return "\n".join(lines) + "\n\n" + "\n".join(svg)


def load_recorded(path) -> dict[str, dict[str, Row]]:
    """Read results.json back into the replay shape."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    recorded: dict[str, dict[str, Row]] = {}
    for item in raw:
        row = Row(
            provider=item["provider"],
            model=item["model"],
            task_id=item["task_id"],
            task_type=item["task_type"],
            outcome=item["outcome"],
            latency_ms=item.get("latency_ms"),
            ttft_ms=item.get("ttft_ms"),
            prompt_tokens=item.get("prompt_tokens", 0),
            completion_tokens=item.get("completion_tokens", 0),
            tokens_per_sec=item.get("tokens_per_sec"),
            http_status=item.get("http_status"),
            error_class=item.get("error_class", "ok"),
            answer_hash=item.get("answer_hash", ""),
            rate_limit=item.get("rate_limit", ""),
        )
        recorded.setdefault(row.task_id, {})[row.model_key] = row
    return recorded
