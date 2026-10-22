from __future__ import annotations

import httpx

from providers.base import ProviderClient
from providers.cloudflare import CloudflareAdapter
from providers.huggingface import HuggingFaceAdapter
from providers.ollama import OllamaAdapter
from providers.openai_adapter import OpenAIAdapter

ADAPTERS: dict[str, type[OpenAIAdapter]] = {
    "cloudflare_workers_ai": CloudflareAdapter,
    "ollama_cloud": OllamaAdapter,
    "hugging_face": HuggingFaceAdapter,
}


def adapter_for(
    provider: str, key: str, client: httpx.AsyncClient, base: str = "", timeout: float = 60.0
) -> ProviderClient:
    """Registry keyed by provider slug; everything else is OpenAI-compatible.

    `base` overrides the catalog baseUrl (the per-model host override in the UI).
    """
    adapter = ADAPTERS.get(provider, OpenAIAdapter)(provider, key, client, timeout)
    if base:
        adapter.base = base.rstrip("/")
    return adapter
