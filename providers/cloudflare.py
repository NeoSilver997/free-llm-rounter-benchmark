from __future__ import annotations

import os
from typing import AsyncIterator

from providers.base import ProviderError
from providers.openai_adapter import OpenAIAdapter, normalize_openai_response


def normalize_cf(raw: dict) -> dict:
    """Workers AI wraps the answer in {"result": {...}} -> OpenAI shape."""
    result = raw.get("result", raw)
    return normalize_openai_response(
        {"id": raw.get("id", ""), "model": result.get("model", ""), "choices": result.get("choices", []), "usage": result.get("usage", {})}
    )


class CloudflareAdapter(OpenAIAdapter):
    def url(self, path: str = "chat/completions") -> str:
        account = os.environ.get("CF_ACCOUNT_ID", "")
        if not account:
            raise ProviderError("CF_ACCOUNT_ID is required", None, "auth")
        base = self.base.replace("{account_id}", account)
        return f"{base}/{path.lstrip('/')}"

    async def chat(self, model: str, payload: dict, stream: bool = False) -> dict | AsyncIterator[dict]:
        if stream:
            return self._stream(model, payload)
        raw = await self._post(model, {"model": model, **payload})
        return normalize_cf(raw)
