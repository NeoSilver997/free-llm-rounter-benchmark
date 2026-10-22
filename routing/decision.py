from __future__ import annotations

from dataclasses import dataclass, field

from telemetry import ALLOWED_REASONS


@dataclass
class RouteDecision:
    provider: str
    model_id: str
    key: str
    reason: str  # telemetry allowlist only
    score: float = 0.0
    excluded: dict[str, str] = field(default_factory=dict)  # model key -> exclusion reason
    policy: str = "weighted"
    attempts: list["RouteDecision"] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.reason not in ALLOWED_REASONS:
            raise ValueError(f"unknown reason: {self.reason!r}")

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model": self.model_id,
            "key": self.key,
            "reason": self.reason,
            "score": round(self.score, 4),
            "policy": self.policy,
            "excluded": self.excluded,
            "attempts": [a.as_dict() for a in self.attempts],
        }
