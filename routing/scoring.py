from __future__ import annotations

from typing import Callable, Sequence, TYPE_CHECKING

from catalog.rate_limits import parse_rate_limit, quota_cost
from catalog.schema import ModelEntry

if TYPE_CHECKING:
    from benchmark.metrics import ModelMetrics

DEFAULT_WEIGHTS = {"quality": 0.40, "latency": 0.25, "quota_cost": 0.20, "reliability": 0.15}
AXES = ("quality", "latency", "quota_cost", "reliability")
LOWER_IS_BETTER = frozenset({"latency", "quota_cost"})
DEFAULT_MIN_N = 5


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Non-negative weights, auto-normalized so they always sum to 1."""
    clean = {k: max(0.0, float(v)) for k, v in weights.items() if k in AXES}
    total = sum(clean.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {k: v / total for k, v in clean.items()}


def normalize(values: dict[str, float]) -> dict[str, float]:
    """Min-max normalization over the candidate pool (not per model)."""
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    if hi == lo:
        return {k: 1.0 for k in values}
    return {k: (v - lo) / (hi - lo) for k, v in values.items()}


def candidate_metrics(
    model: ModelEntry,
    metrics: dict[str, ModelMetrics],
    reliability: float | None = None,
) -> dict[str, float | None]:
    """Raw axis values for one model. Unmeasured axes stay None, never fabricated."""
    measured = metrics.get(model.key)
    limits = parse_rate_limit(model.rateLimit)
    reliability_value = reliability
    if reliability_value is None and measured is not None and measured.error_rate is not None:
        reliability_value = 1.0 - measured.error_rate
    return {
        "quality": measured.pass_rate if measured else None,
        "latency": measured.p50_ms if measured else None,
        "quota_cost": quota_cost(limits, 1, 512),
        "reliability": reliability_value,
    }


def weighted_score(
    model: ModelEntry,
    metrics: dict[str, ModelMetrics],
    weights: dict[str, float],
    reliability_of: Callable[[str], float | None] | None = None,
    pool: Sequence[ModelEntry] | None = None,
) -> float:
    """Weighted score over the pool; cost and latency are inverted (lower is better).

    Axes with no measurement are dropped and their weight is redistributed.
    """
    weights = normalize_weights(weights)
    models = list(pool or [model])
    axes = {
        m.key: candidate_metrics(m, metrics, reliability_of(m.provider) if reliability_of else None)
        for m in models
    }

    total = 0.0
    used = 0.0
    for axis, weight in weights.items():
        known = {k: v[axis] for k, v in axes.items() if v[axis] is not None}
        if model.key not in known:
            continue
        value = normalize(known)[model.key]
        if axis in LOWER_IS_BETTER:
            value = 1.0 - value
        total += weight * value
        used += weight
    return total / used if used else 0.0


def rank(
    models: Sequence[ModelEntry],
    metrics: dict[str, ModelMetrics],
    weights: dict[str, float],
    reliability_of: Callable[[str], float | None] | None = None,
    min_n: int = DEFAULT_MIN_N,
) -> list[tuple[ModelEntry, float]]:
    """Rank candidates; models below the minimum-N gate are not scored."""
    eligible = [m for m in models if _scored(metrics, m, min_n)]
    scored = [(m, weighted_score(m, metrics, weights, reliability_of, models)) for m in eligible]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0].key))


def _scored(metrics: dict[str, ModelMetrics], model: ModelEntry, min_n: int) -> bool:
    measured = metrics.get(model.key)
    return bool(measured and measured.scored_n >= min_n)
