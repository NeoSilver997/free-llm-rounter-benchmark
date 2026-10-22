from __future__ import annotations

import json
import re
from pathlib import Path

from catalog.schema import Catalog, ModelEntry, Provider

CATALOG_PATH = Path(__file__).parent / "data.json"


def slug(name: str) -> str:
    """Provider name -> stable identifier used for keys, env vars and route ids."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def load_catalog(path: Path | str = CATALOG_PATH) -> Catalog:
    catalog = Catalog.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
    for provider in catalog.providers:
        for model in provider.models:
            model.provider = slug(provider.name)
            if model.id:
                model.key = f"{model.provider}/{model.id}"
    return catalog


def provider_by_name(catalog: Catalog, name: str) -> Provider | None:
    target = slug(name)
    for provider in catalog.providers:
        if slug(provider.name) == target:
            return provider
    return None


def routable_models(catalog: Catalog) -> list[ModelEntry]:
    """Placeholder rows ('+ N more models') have no id and can never be routed."""
    return [m for p in catalog.providers for m in p.models if m.id]


def model_index(catalog: Catalog) -> dict[str, list[ModelEntry]]:
    index: dict[str, list[ModelEntry]] = {}
    for model in routable_models(catalog):
        index.setdefault(model.key, []).append(model)
    return index


def unique_prefix(prefix: str, providers: set[str]) -> str | None:
    """Accept short provider names, e.g. 'ovhcloud' for 'ovhcloud_ai_endpoints'."""
    matches = [p for p in providers if p == prefix or p.startswith(prefix + "_")]
    return matches[0] if len(matches) == 1 else None


def resolve_model(catalog: Catalog, model: str) -> list[ModelEntry]:
    """Accepts 'provider/model-id', a bare model id, 'auto' or 'auto:<task_type>'.

    Returns candidate entries (empty list for 'auto'), so the caller can rank them.
    Model ids may themselves contain slashes, so the provider prefix is only used
    when its first segment matches a known provider slug.
    """
    if not model or model == "auto" or model.startswith("auto:"):
        return []

    providers = {slug(p.name) for p in catalog.providers}
    index = model_index(catalog)

    if "/" in model:
        first, rest = model.split("/", 1)
        if first in providers or (prefix := unique_prefix(first, providers)):
            hit = index.get(f"{prefix or first}/{rest}")
            if hit:
                return hit

    hit = index.get(model)
    if hit:
        return hit

    needle = model.lower()
    return [m for m in routable_models(catalog) if m.name.lower() == needle or m.id.lower() == needle]
