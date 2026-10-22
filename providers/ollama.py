from __future__ import annotations

import json
from typing import AsyncIterator

from providers.openai_adapter import OpenAIAdapter, normalize_openai_response


def to_ollama_payload(payload: dict) -> dict:
    """OpenAI messages -> Ollama native payload."""
    out = {"messages": payload.get("messages", [])}
    if payload.get("max_tokens"):
        out["options"] = {"num_predict": payload["max_tokens"]}
    if payload.get("temperature") is not None:
        out.setdefault("options", {})["temperature"] = payload["temperature"]
    return out


def parse_ollama_body(text: str) -> dict:
    """Ollama replies with newline-delimited chunks even when stream is not requested.

    Content is spread over the chunks and usage only appears in the last one, so a single
    json.loads on the body fails; merge the chunks before normalizing.
    """
    chunks = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not chunks:
        return {}
    last = chunks[-1]
    content = "".join(chunk.get("message", {}).get("content", "") for chunk in chunks)
    return {
        "id": last.get("id", ""),
        "model": last.get("model", ""),
        "message": {"role": last.get("message", {}).get("role", "assistant"), "content": content},
        "done_reason": last.get("done_reason"),
        "prompt_eval_count": last.get("prompt_eval_count", 0),
        "eval_count": last.get("eval_count", 0),
    }


def normalize_ollama_response(raw: dict) -> dict:
    return normalize_openai_response(
        {
            "id": raw.get("id", ""),
            "model": raw.get("model", ""),
            "choices": [{"index": 0, "message": raw.get("message", {}), "finish_reason": raw.get("done_reason")}],
            "usage": {
                "prompt_tokens": raw.get("prompt_eval_count", 0),
                "completion_tokens": raw.get("eval_count", 0),
            },
        }
    )


class OllamaAdapter(OpenAIAdapter):
    """Native /api/chat, no streaming pass-through and no /v1/models route."""

    def supports_stream(self) -> bool:
        return False

    def parse(self, text: str) -> dict:
        return parse_ollama_body(text)

    async def chat(self, model: str, payload: dict, stream: bool = False) -> dict:
        raw = await self._post("chat", {"model": model, **to_ollama_payload(payload)})
        return normalize_ollama_response(raw)

    async def list_models(self) -> list[str]:
        raw = await self._get("tags")
        return [m.get("model", "") for m in raw.get("models", [])]
