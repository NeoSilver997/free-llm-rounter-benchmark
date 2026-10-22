from __future__ import annotations

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

    async def chat(self, model: str, payload: dict, stream: bool = False) -> dict:
        raw = await self._post("chat", {"model": model, **to_ollama_payload(payload)})
        return normalize_ollama_response(raw)

    async def list_models(self) -> list[str]:
        raw = await self._get("tags")
        return [m.get("model", "") for m in raw.get("models", [])]
