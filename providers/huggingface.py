from __future__ import annotations

from providers.openai_adapter import OpenAIAdapter

# The OpenAI-compatible host, not api-inference.huggingface.co/models.
HF_BASE = "https://router.huggingface.co/v1"


class HuggingFaceAdapter(OpenAIAdapter):
    def __init__(self, provider: str, key: str, client, timeout: float = 60.0):
        super().__init__(provider, key, client, timeout)
        self.base = HF_BASE
