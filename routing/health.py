from __future__ import annotations

import time
from collections import deque
from statistics import median

ALLOWED_OUTCOMES = frozenset({"pass", "fail", "error", "skipped", "policy_blocked"})
SUCCESS_OUTCOMES = frozenset({"pass"})
DEFAULT_WINDOW = 50
DEFAULT_COOLDOWN_S = 30.0


def percentile(values: list[int], q: float) -> float | None:
    """Percentile over *all* attempts, failures included (not success-biased)."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo))


class HealthTracker:
    """Per-provider rolling health used as a hard constraint and as a scoring axis."""

    def __init__(self, window: int = DEFAULT_WINDOW):
        self.window = window
        self._events: dict[str, deque[tuple[str, int, float]]] = {}
        self._cooldown_until: dict[str, float] = {}

    def record(
        self,
        provider: str,
        outcome: str,
        latency_ms: int,
        retry_after: float | None = None,
        now: float | None = None,
    ) -> None:
        if outcome not in ALLOWED_OUTCOMES:
            raise ValueError(f"unknown outcome: {outcome!r}")
        events = self._events.setdefault(provider, deque(maxlen=self.window))
        events.append((outcome, int(latency_ms), now if now is not None else time.time()))
        if retry_after:
            self.set_cooldown(provider, retry_after, now=now)

    def set_cooldown(self, provider: str, seconds: float, now: float | None = None) -> None:
        base = now if now is not None else time.time()
        self._cooldown_until[provider] = base + max(0.0, float(seconds))

    def success_rate(self, provider: str, window: int = 50) -> float | None:
        events = list(self._events.get(provider, ()))
        if not events:
            return None
        recent = events[-window:]
        passes = sum(1 for outcome, _, _ in recent if outcome in SUCCESS_OUTCOMES)
        return passes / len(recent)

    def cooldown_until(self, provider: str) -> float | None:
        until = self._cooldown_until.get(provider)
        if until is None or until <= time.time():
            return None
        return until

    def is_available(self, provider: str, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        until = self._cooldown_until.get(provider)
        return until is None or until <= current

    def latencies(self, provider: str) -> list[int]:
        return [latency for _, latency, _ in self._events.get(provider, ())]

    def snapshot(self) -> dict[str, dict]:
        """Shape consumed by GET /health."""
        out: dict[str, dict] = {}
        for provider, events in self._events.items():
            rows = list(events)
            latencies = [latency for _, latency, _ in rows]
            out[provider] = {
                "n": len(rows),
                "success_rate": self.success_rate(provider),
                "p50_ms": percentile(latencies, 0.50),
                "p95_ms": percentile(latencies, 0.95),
                "median_ms": float(median(latencies)) if latencies else None,
                "last_outcome": rows[-1][0] if rows else None,
                "last_latency_ms": rows[-1][1] if rows else None,
                "cooldown_until": self.cooldown_until(provider),
            }
        return out
