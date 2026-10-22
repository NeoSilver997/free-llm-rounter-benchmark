from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from catalog.loader import load_catalog, routable_models
from routing.constraints import ANONYMOUS_PROVIDERS, provider_index

CONFIG_PATH = Path("ui_config.json")
DEFAULT_API_HOST = "http://localhost:8000"
ENV_PREFIX = "ROUTER_KEY_"
DEFAULT_TPM = 20  # token budget per provider when the catalog does not publish one


def env_keys() -> dict[str, str]:
    """Keys come from env only: the LLM_ROUTER_KEYS map as the base, ROUTER_KEY_<SLUG> overrides it."""
    keys: dict[str, str] = {}
    raw = os.environ.get("LLM_ROUTER_KEYS", "")
    if raw:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            keys.update({str(k).lower(): str(v) for k, v in parsed.items() if v})
    for name, value in os.environ.items():
        if name.startswith(ENV_PREFIX) and value:
            keys[name[len(ENV_PREFIX) :].lower()] = value
    return keys


@dataclass
class ModelConfig:
    """Per-model entry. Keys are never persisted to disk (see as_dict)."""

    provider: str
    model_key: str
    host: str = ""
    keyword: str = ""
    api_key: str = ""
    selected: bool = False

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "model_key": self.model_key,
            "host": self.host,
            "keyword": self.keyword,
            "selected": self.selected,
        }


@dataclass
class AppConfig:
    api_host: str = DEFAULT_API_HOST
    api_key: str = ""
    models: list[ModelConfig] = field(default_factory=list)
    provider_keys: dict[str, str] = field(default_factory=dict)  # runtime only, never persisted
    provider_tpm: dict[str, int] = field(default_factory=dict)

    def providers(self) -> list[str]:
        return sorted({m.provider for m in self.models})

    def tpm_for(self, provider: str) -> int:
        return self.provider_tpm.get(provider, DEFAULT_TPM)

    def by_key(self, model_key: str) -> ModelConfig | None:
        for model in self.models:
            if model.model_key == model_key:
                return model
        return None

    def selected_models(self) -> list[ModelConfig]:
        return [m for m in self.models if m.selected]

    def keyword_match(self, prompt: str) -> ModelConfig | None:
        """Deterministic keyword routing: first configured keyword present in the prompt."""
        lowered = (prompt or "").lower()
        for model in self.models:
            if model.keyword and model.keyword.lower() in lowered:
                return model
        return None

    def host_for(self, model_key: str) -> str:
        """Per-model host override, falling back to the catalog baseUrl."""
        model = self.by_key(model_key)
        if model and model.host:
            return model.host.rstrip("/")
        provider = provider_index().get(model.provider if model else model_key.split("/", 1)[0])
        return provider.baseUrl if provider else ""

    def key_for(self, model_key: str) -> str:
        """Model key -> provider key -> env key -> global key. Env is the default source."""
        model = self.by_key(model_key)
        provider = model.provider if model else model_key.split("/", 1)[0]
        if model and model.api_key:
            return model.api_key
        return self.provider_keys.get(provider, "") or env_keys().get(provider, "") or self.api_key

    def as_dict(self) -> dict:
        return {
            "api_host": self.api_host,
            "providers": {p: {"tpm": self.tpm_for(p)} for p in self.providers()},
            "models": [m.as_dict() for m in self.models],
        }


def seed_config(catalog=None) -> AppConfig:
    """Config seeded from the catalog; anonymous tiers start selected."""
    catalog = catalog or load_catalog()
    config = AppConfig()
    for model in routable_models(catalog):
        config.models.append(
            ModelConfig(provider=model.provider, model_key=model.key, selected=model.provider in ANONYMOUS_PROVIDERS)
        )
    return config


def load_config(path: Path | str | None = None) -> AppConfig:
    path = Path(path or CONFIG_PATH)
    if not path.exists():
        return seed_config()
    data = json.loads(path.read_text(encoding="utf-8"))
    config = AppConfig(api_host=data.get("api_host") or DEFAULT_API_HOST)
    for provider, entry in (data.get("providers") or {}).items():
        tpm = entry.get("tpm") if isinstance(entry, dict) else None
        if isinstance(tpm, int) and tpm > 0:
            config.provider_tpm[str(provider)] = tpm
    for row in data.get("models", []):
        config.models.append(
            ModelConfig(
                provider=row.get("provider", ""),
                model_key=row.get("model_key", ""),
                host=row.get("host", ""),
                keyword=row.get("keyword", ""),
                selected=bool(row.get("selected")),
            )
        )
    return config


def save_config(config: AppConfig, path: Path | str | None = None) -> None:
    Path(path or CONFIG_PATH).write_text(json.dumps(config.as_dict(), indent=2), encoding="utf-8")
