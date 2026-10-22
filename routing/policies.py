from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from catalog.rate_limits import parse_rate_limit, quota_cost
from catalog.schema import ModelEntry
from catalog.task_types import model_task_types

if TYPE_CHECKING:
    from benchmark.metrics import ModelMetrics

from routing.budget import QuotaLedger
from routing.constraints import apply_constraints
from routing.decision import RouteDecision
from routing.health import HealthTracker
from routing.request import RoutingRequest
from routing.scoring import (
    DEFAULT_MIN_N,
    DEFAULT_WEIGHTS,
    candidate_metrics,
    rank,
    weighted_score,
)

DEFAULT_THRESHOLDS = {"default": 0.60, "text": 0.70, "code": 0.55, "reasoning": 0.50}

# Ordered, first match wins; deterministic and offline (no embedding dependency).
CLASSIFY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("reasoning", ("reason", "why", "prove", "step by step", "analyze", "compare", "math", "solve")),
    ("code", ("code", "function", "refactor", "bug", "regex", "python", "implement", "script")),
    ("embeddings", ("embed", "embedding", "vector")),
    ("rerank", ("rerank", "rank", "relevance")),
    ("vision", ("image", "photo", "ocr", "picture", "screenshot")),
    ("audio", ("audio", "speech", "transcribe", "listen")),
    ("roleplay", ("roleplay", "character", "in character")),
    ("text", ("classify", "label", "sentiment", "summarize", "translate", "reply")),
)


def classify(text: str) -> str:
    """Intent -> task type. First match wins, fallback 'text'."""
    lowered = (text or "").lower()
    for task_type, needles in CLASSIFY_RULES:
        if any(n in lowered for n in needles):
            return task_type
    return "text"


@dataclass
class RoutingContext:
    keys: dict[str, str] = field(default_factory=dict)
    health: HealthTracker = field(default_factory=HealthTracker)
    ledger: QuotaLedger = field(default_factory=QuotaLedger)
    metrics: dict[str, ModelMetrics] = field(default_factory=dict)
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    min_n: int = DEFAULT_MIN_N
    pass_rate_by_task: dict[str, dict[str, float]] = field(default_factory=dict)

    def reliability(self, provider: str) -> float | None:
        return self.health.success_rate(provider)


def decide(
    model: ModelEntry,
    req: RoutingRequest,
    ctx: RoutingContext,
    reason: str,
    score: float,
    excluded: dict[str, str],
) -> RouteDecision:
    return RouteDecision(
        provider=model.provider,
        model_id=model.id or "",
        key=model.key,
        reason=reason,
        score=round(score, 4),
        excluded=excluded,
        policy=req.policy,
    )


class AbstractRouter(ABC):
    """Extension point: hard constraints first, then a policy-specific selection."""

    name = "abstract"

    def route(self, req: RoutingRequest, candidates: list[ModelEntry], ctx: RoutingContext) -> RouteDecision:
        kept, excluded = apply_constraints(candidates, req, ctx.keys, ctx.health, ctx.ledger)
        if not kept:
            return RouteDecision(
                provider="",
                model_id="",
                key="",
                reason="unknown",
                score=0.0,
                excluded=excluded,
                policy=self.name,
            )
        return self.select(req, kept, excluded, ctx)

    @abstractmethod
    def select(
        self, req: RoutingRequest, kept: list[ModelEntry], excluded: dict[str, str], ctx: RoutingContext
    ) -> RouteDecision:
        raise NotImplementedError


class ExplicitRouter(AbstractRouter):
    """Exact provider/model-id already resolved by catalog.resolve_model."""

    name = "explicit"

    def select(self, req, kept, excluded, ctx) -> RouteDecision:
        best = rank(kept, ctx.metrics, ctx.weights, ctx.reliability, ctx.min_n)
        model = best[0][0] if best else kept[0]
        return decide(model, req, ctx, "explicit", best[0][1] if best else 0.0, excluded)


class QualityRouter(AbstractRouter):
    """Best measured quality inside the task-type group, subject to the minimum-N gate."""

    name = "quality"

    def select(self, req, kept, excluded, ctx) -> RouteDecision:
        scored = [m for m in kept if _measured(m, ctx)]
        if not scored:
            return _fallback_weighted(req, kept, excluded, ctx)
        best = max(scored, key=lambda m: (ctx.metrics[m.key].pass_rate, m.key))
        return decide(best, req, ctx, "quality", ctx.metrics[best.key].pass_rate or 0.0, excluded)


class CheapestQuotaRouter(AbstractRouter):
    """Lowest normalized quota cost (free tiers: cost = quota consumption)."""

    name = "cheapest"

    def select(self, req, kept, excluded, ctx) -> RouteDecision:
        costs = {m.key: quota_cost(parse_rate_limit(m.rateLimit), 1, 512) for m in kept}
        known = {k: v for k, v in costs.items() if v is not None}
        if not known:
            return _fallback_weighted(req, kept, excluded, ctx)
        best_key = min(known, key=lambda k: (known[k], k))
        model = next(m for m in kept if m.key == best_key)
        return decide(model, req, ctx, "cost", 1.0 - known[best_key], excluded)


class WeightedPQRRouter(AbstractRouter):
    """Constraints first, then normalized weighted scoring over the candidate pool."""

    name = "weighted"

    def select(self, req, kept, excluded, ctx) -> RouteDecision:
        scored = rank(kept, ctx.metrics, ctx.weights, ctx.reliability, ctx.min_n)
        if not scored:
            return _fallback_weighted(req, kept, excluded, ctx)
        model, score = scored[0]
        return decide(model, req, ctx, _dominant_reason(model, ctx), score, excluded)


class HybridThresholdRouter(AbstractRouter):
    """Weak-first: stay on the cheap model when its measured pass rate clears the threshold."""

    name = "hybrid"

    def select(self, req, kept, excluded, ctx) -> RouteDecision:
        task_type = req.task_type or classify(req.prompt_text())
        threshold = ctx.thresholds.get(task_type, ctx.thresholds.get("default", 0.6))
        cheap = CheapestQuotaRouter().select(req, kept, excluded, ctx)
        weak = ctx.metrics.get(cheap.key)
        if weak is None or weak.scored_n < ctx.min_n:
            return _fallback_weighted(req, kept, excluded, ctx)

        pass_rate = ctx.pass_rate_by_task.get(cheap.key, {}).get(task_type, weak.pass_rate)
        if pass_rate is not None and pass_rate >= threshold:
            return decide(cheap_model(kept, cheap.key), req, ctx, "threshold", pass_rate, excluded)
        return QualityRouter().select(req, kept, excluded, ctx)


class CascadeRouter(AbstractRouter):
    """FrugalGPT: ordered attempts from cheapest to best, stop on success or budget exhaustion."""

    name = "cascade"

    def select(self, req, kept, excluded, ctx) -> RouteDecision:
        order = sorted(
            kept,
            key=lambda m: (
                quota_cost(parse_rate_limit(m.rateLimit), 1, 512) is None,
                quota_cost(parse_rate_limit(m.rateLimit), 1, 512) or 0.0,
                m.key,
            ),
        )
        attempts: list[RouteDecision] = []
        for model in order:
            reason = "cascade"
            attempts.append(decide(model, req, ctx, reason, 0.0, excluded))
            if ctx.ledger.is_exhausted(model.provider):
                break
        decision = attempts[0]
        decision.policy = self.name
        decision.attempts = attempts
        return decision


def _measured(model: ModelEntry, ctx: RoutingContext) -> bool:
    m = ctx.metrics.get(model.key)
    return bool(m and m.scored_n >= ctx.min_n and m.pass_rate is not None)


def _fallback_weighted(req, kept, excluded, ctx) -> RouteDecision:
    """Nothing measurable: fall back to the first constrained candidate and say so."""
    return decide(kept[0], req, ctx, "unknown", 0.0, excluded)


def cheap_model(kept: list[ModelEntry], key: str) -> ModelEntry:
    return next(m for m in kept if m.key == key)


AXIS_REASON = {"quality": "quality", "latency": "unknown", "quota_cost": "cost", "reliability": "reliability"}


def _dominant_reason(model: ModelEntry, ctx: RoutingContext) -> str:
    axes = candidate_metrics(model, ctx.metrics, ctx.reliability(model.provider))
    weights = {k: v for k, v in ctx.weights.items() if axes[k] is not None}
    if not weights:
        return "unknown"
    return AXIS_REASON[max(weights, key=lambda k: weights[k])]


class SemanticRouter(AbstractRouter):
    """Intent classification decides the task-type group, then quality picks inside it."""

    name = "semantic"

    def select(self, req, kept, excluded, ctx) -> RouteDecision:
        task_type = req.task_type or classify(req.prompt_text())
        group = [m for m in kept if task_type in model_task_types(m)] or kept
        decision = QualityRouter().select(req, group, excluded, ctx)
        decision.policy = self.name
        if decision.reason == "unknown":
            decision.reason = "task_type"
        return decision


POLICIES: dict[str, AbstractRouter] = {
    "explicit": ExplicitRouter(),
    "semantic": SemanticRouter(),
    "quality": QualityRouter(),
    "cheapest": CheapestQuotaRouter(),
    "weighted": WeightedPQRRouter(),
    "hybrid": HybridThresholdRouter(),
    "cascade": CascadeRouter(),
}
