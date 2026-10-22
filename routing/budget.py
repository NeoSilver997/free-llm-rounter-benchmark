from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Budget:
    max_requests: int = 15
    max_tokens: int = 200_000
    per_provider_max_requests: int = 15


@dataclass
class QuotaLedger:
    """Tracks what a run has already consumed so routing stays inside published limits."""

    budget: Budget = field(default_factory=Budget)
    used_requests: dict[str, int] = field(default_factory=dict)
    used_tokens: dict[str, int] = field(default_factory=dict)

    def charge(self, provider: str, model: str, requests: int, tokens: int) -> None:
        self.used_requests[provider] = self.used_requests.get(provider, 0) + requests
        self.used_tokens[provider] = self.used_tokens.get(provider, 0) + tokens

    def remaining(self, provider: str) -> dict:
        per_provider = self.budget.per_provider_max_requests
        return {
            "requests": max(0, per_provider - self.used_requests.get(provider, 0)),
            "tokens": max(0, self.budget.max_tokens - self.used_tokens.get(provider, 0)),
        }

    def is_exhausted(self, provider: str) -> bool:
        remaining = self.remaining(provider)
        return remaining["requests"] <= 0 or remaining["tokens"] <= 0
