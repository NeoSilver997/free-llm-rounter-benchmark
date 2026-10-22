from __future__ import annotations

from dataclasses import dataclass, replace

from routing.budget import Budget


@dataclass(frozen=True)
class RoutingRequest:
    """Everything a policy needs: what was asked, what is required, what is off-limits."""

    model: str | None = None
    task_type: str | None = None  # inferred by SemanticRouter when absent
    messages: tuple[dict, ...] = ()
    input_tokens: int | None = None
    output_tokens: int | None = None
    require: frozenset[str] = frozenset()  # capability labels: vision, json, tools, code
    local_only: bool = False  # hard constraint: self-hosted-only providers
    unavailable: frozenset[str] = frozenset()
    budget: Budget | None = None
    policy: str = "weighted"

    @classmethod
    def from_messages(cls, messages: list[dict], **kwargs) -> "RoutingRequest":
        return cls(messages=tuple(messages), **kwargs)

    def prompt_text(self) -> str:
        return " ".join(str(m.get("content", "")) for m in self.messages)

    def with_task_type(self, task_type: str) -> "RoutingRequest":
        return replace(self, task_type=task_type)
