from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Sequence

from catalog.rate_limits import parse_rate_limit, quota_cost
from catalog.task_types import TASK_TYPES
from routing.budget import Budget
from routing.health import percentile

if TYPE_CHECKING:
    from benchmark.runner import Row


@dataclass
class ModelMetrics:
    key: str
    provider: str
    model: str
    pass_rate: float | None = None
    scored_n: int = 0
    attempts: int = 0
    p50_ms: float | None = None
    p95_ms: float | None = None
    tokens_per_sec: float | None = None
    error_rate: float | None = None
    coverage: tuple[str, ...] = ()
    quota_cost: float | None = None
    task_types: tuple[str, ...] = ()
    outcomes: dict[str, int] = field(default_factory=dict)
    raw_rate_limit: str = ""


def aggregate(rows: Sequence[Any]) -> dict[str, ModelMetrics]:
    """Group rows by model. Percentiles include failed and interrupted attempts."""
    groups: dict[str, list[Any]] = {}
    for row in rows:
        groups.setdefault(row.model_key, []).append(row)

    metrics: dict[str, ModelMetrics] = {}
    for key, group in groups.items():
        provider, model = key.split("/", 1)
        scored = [r for r in group if r.outcome in ("pass", "fail")]
        errors = [r for r in group if r.outcome == "error"]
        latencies = [r.latency_ms for r in group if r.latency_ms is not None]
        speeds = [r.tokens_per_sec for r in group if r.tokens_per_sec is not None]
        covered = sorted({r.task_type for r in group if r.task_type})

        costs = [
            quota_cost(parse_rate_limit(r.rate_limit), 1, r.prompt_tokens + r.completion_tokens)
            for r in group
        ]
        costs = [c for c in costs if c is not None]

        metrics[key] = ModelMetrics(
            key=key,
            provider=provider,
            model=model,
            pass_rate=(sum(1 for r in scored if r.outcome == "pass") / len(scored)) if scored else None,
            scored_n=len(scored),
            attempts=len(group),
            p50_ms=percentile(latencies, 0.50),
            p95_ms=percentile(latencies, 0.95),
            tokens_per_sec=(sum(speeds) / len(speeds)) if speeds else None,
            error_rate=(len(errors) / len(group)) if group else None,
            coverage=tuple(t for t in TASK_TYPES if t in covered),
            quota_cost=(sum(costs) / len(costs)) if costs else None,
            task_types=tuple(t for t in TASK_TYPES if t in covered),
            outcomes={o: sum(1 for r in group if r.outcome == o) for o in sorted({r.outcome for r in group})},
            raw_rate_limit=group[0].rate_limit,
        )
    return metrics


def pareto_frontier(metrics: dict[str, ModelMetrics]) -> list[str]:
    """Keys on the quality-vs-normalized-quota-cost frontier (higher quality, lower cost)."""
    scored = [m for m in metrics.values() if m.pass_rate is not None and m.quota_cost is not None]
    frontier: list[str] = []
    for candidate in scored:
        dominated = False
        for other in scored:
            if other.key == candidate.key:
                continue
            if other.pass_rate >= candidate.pass_rate and other.quota_cost <= candidate.quota_cost and (
                other.pass_rate > candidate.pass_rate or other.quota_cost < candidate.quota_cost
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate.key)
    return sorted(frontier)


def win_rate_at_budget(metrics: dict[str, ModelMetrics], budget: Budget) -> dict[str, float]:
    """RouteLLM-style headline metric: best achievable pass rate per task type inside the budget.

    A model counts only when its per-request quota cost fits the budget share and it has
    enough scored samples (min_n is applied by the caller through `min_n_gate`).
    """
    budget_share = 1.0 / max(1, budget.max_requests)
    by_task: dict[str, float] = {}
    for model in metrics.values():
        if model.pass_rate is None or model.quota_cost is None:
            continue
        if model.quota_cost > budget_share:
            continue
        for task_type in model.task_types:
            if model.pass_rate > by_task.get(task_type, -1.0):
                by_task[task_type] = model.pass_rate
    return {t: by_task[t] for t in TASK_TYPES if t in by_task}
