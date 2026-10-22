from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol

ERROR_CLASSES = frozenset({"ok", "429", "timeout", "auth", "schema", "unsupported", "server"})
RETRY_STATUS = frozenset({429, 500, 502, 503})


def classify_status(status: int | None) -> str:
    if status is None:
        return "timeout"
    if status == 429:
        return "429"
    if status in (401, 403):
        return "auth"
    if 500 <= status < 600:
        return "server"
    return "schema"


def retry_after(headers) -> float:
    raw = headers.get("Retry-After") or "0"
    try:
        return float(raw)
    except ValueError:
        return 0.0


@dataclass
class ProviderError(Exception):
    message: str
    status: int | None = None
    error_class: str = "error"
    retry_after: float = 0.0

    def __str__(self) -> str:
        return f"{self.error_class}: {self.message}"


class ProviderClient(Protocol):
    provider: str

    def supports_stream(self) -> bool: ...

    async def chat(self, model: str, payload: dict, stream: bool) -> dict | AsyncIterator[dict]: ...

    async def embeddings(self, model: str, payload: dict) -> dict: ...

    async def list_models(self) -> list[str]: ...
