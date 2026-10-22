from __future__ import annotations

import asyncio
import json
from typing import AsyncIterator

import httpx

from providers.base import ProviderError, RETRY_STATUS, classify_status, retry_after
from routing.constraints import provider_index


class OpenAIAdapter:
    """OpenAI-SDK-compatible upstream. One shared httpx client, single retry on 429/5xx."""

    def __init__(self, provider: str, key: str, client: httpx.AsyncClient, timeout: float = 60.0):
        entry = provider_index().get(provider)
        self.provider = provider
        self.key = key
        self.client = client
        self.timeout = timeout
        self.base = (entry.baseUrl if entry else "").rstrip("/")

    def url(self, path: str = "chat/completions") -> str:
        return f"{self.base}/{path.lstrip('/')}"

    def headers(self) -> dict:
        return {"Authorization": f"Bearer {self.key}"} if self.key else {}

    def supports_stream(self) -> bool:
        return True

    async def chat(self, model: str, payload: dict, stream: bool = False) -> dict | AsyncIterator[dict]:
        if stream:
            return self._stream(model, payload)
        raw = await self._post("chat/completions", {"model": model, **payload})
        return normalize_openai_response(raw)

    async def _stream(self, model: str, payload: dict) -> AsyncIterator[dict]:
        async with self.client.stream(
            "POST", self.url("chat/completions"), json={"model": model, **payload}, headers=self.headers()
        ) as resp:
            if resp.status_code >= 400:
                await resp.aread()
                raise ProviderError(
                    resp.text[:200], resp.status_code, classify_status(resp.status_code), retry_after(resp.headers)
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                chunk = line[6:].strip()
                if chunk == "[DONE]":
                    break
                yield json.loads(chunk)

    async def embeddings(self, model: str, payload: dict) -> dict:
        raw = await self._post("embeddings", {"model": model, **payload})
        return {"object": "list", "data": raw.get("data", []), "model": raw.get("model", model), "usage": raw.get("usage", {})}

    async def list_models(self) -> list[str]:
        raw = await self._get("models")
        return [item.get("id", "") for item in raw.get("data", [])]

    async def _post(self, path: str, payload: dict) -> dict:
        for attempt in range(2):
            try:
                resp = await self.client.post(self.url(path), json=payload, headers=self.headers(), timeout=self.timeout)
            except httpx.TimeoutException as exc:
                raise ProviderError("timeout", None, "timeout") from exc
            if resp.status_code < 400:
                return resp.json()
            if attempt == 0 and resp.status_code in RETRY_STATUS:
                await asyncio.sleep(retry_after(resp.headers))
                continue
            raise ProviderError(resp.text[:200], resp.status_code, classify_status(resp.status_code), retry_after(resp.headers))
        raise ProviderError("retry exhausted", None, "error")

    async def _get(self, path: str) -> dict:
        resp = await self.client.get(self.url(path), headers=self.headers(), timeout=self.timeout)
        if resp.status_code >= 400:
            raise ProviderError(resp.text[:200], resp.status_code, classify_status(resp.status_code))
        return resp.json()


def normalize_openai_response(raw: dict) -> dict:
    """Guarantees the id/choices/usage shape; some providers omit usage entirely."""
    usage = raw.get("usage") or {}
    choices = []
    for choice in raw.get("choices", []):
        message = choice.get("message") or {"role": "assistant", "content": choice.get("text", "")}
        choices.append(
            {
                "index": choice.get("index", 0),
                "message": {"role": message.get("role", "assistant"), "content": message.get("content", "")},
                "finish_reason": choice.get("finish_reason"),
            }
        )
    prompt = int(usage.get("prompt_tokens", 0) or 0)
    completion = int(usage.get("completion_tokens", 0) or 0)
    return {
        "id": raw.get("id", ""),
        "object": "chat.completion",
        "model": raw.get("model", ""),
        "choices": choices,
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": int(usage.get("total_tokens", prompt + completion) or 0),
        },
    }
