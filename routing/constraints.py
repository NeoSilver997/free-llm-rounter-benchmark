from __future__ import annotations

from functools import lru_cache
from urllib.parse import urlparse

from catalog.loader import load_catalog, slug
from catalog.rate_limits import parse_tokens
from catalog.schema import ModelEntry, Provider
from catalog.task_types import model_task_types
from routing.budget import QuotaLedger
from routing.request import RoutingRequest

# Tiers that work without credentials, so they are never excluded for a missing key.
ANONYMOUS_PROVIDERS = frozenset({"ovhcloud_ai_endpoints", "llm7_io"})

# 'local_only' is a hard availability constraint, not a network attestation:
# a provider counts as local when its base URL points at loopback or a private host.
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "0.0.0.0", "::1"})
PRIVATE_PREFIXES = ("10.", "192.168.", "172.16.", "172.17.", "172.18.", "172.19.", "172.2")


@lru_cache(maxsize=1)
def provider_index() -> dict[str, Provider]:
    return {slug(p.name): p for p in load_catalog().providers}


def is_local(provider: Provider) -> bool:
    host = (urlparse(provider.baseUrl).hostname or "").lower()
    return host in LOCAL_HOSTS or host.startswith(PRIVATE_PREFIXES)


def model_capabilities(model: ModelEntry) -> set[str]:
    """Capability labels a request can require: task types plus json/tools flags."""
    caps = set(model_task_types(model))
    text = f"{model.name} {model.modality or ''}".lower()
    if "json" in text:
        caps.add("json")
    if "tool" in text:
        caps.add("tools")
    return caps


def apply_constraints(
    candidates: list[ModelEntry],
    req: RoutingRequest,
    keys: dict[str, str],
    health,
    ledger: QuotaLedger | None = None,
) -> tuple[list[ModelEntry], dict[str, str]]:
    """Hard filters, applied before any scoring. Each exclusion records a reason."""
    ledger = ledger or QuotaLedger(budget=req.budget or QuotaLedger().budget)
    providers = provider_index()
    kept: list[ModelEntry] = []
    excluded: dict[str, str] = {}

    for model in candidates:
        if model.placeholder:
            excluded[model.key] = "unknown"
            continue

        provider = providers.get(model.provider)
        if provider is None:
            excluded[model.key] = "unknown"
            continue

        if model.provider in req.unavailable:
            excluded[model.key] = "unavailable"
            continue
        if req.local_only and not is_local(provider):
            excluded[model.key] = "unavailable"
            continue
        if not keys.get(model.provider) and model.provider not in ANONYMOUS_PROVIDERS:
            excluded[model.key] = "no_key"
            continue
        if not health.is_available(model.provider):
            excluded[model.key] = "health"
            continue
        if ledger.is_exhausted(model.provider):
            excluded[model.key] = "budget"
            continue

        if req.task_type and req.task_type not in model_task_types(model):
            excluded[model.key] = "task_type"
            continue

        missing = sorted(req.require - model_capabilities(model))
        if missing:
            excluded[model.key] = "capability"
            continue

        context = parse_tokens(model.context)
        if context is not None and req.input_tokens is not None and context < req.input_tokens:
            excluded[model.key] = "context"
            continue

        max_output = parse_tokens(model.maxOutput)
        if max_output is not None and req.output_tokens is not None and max_output < req.output_tokens:
            excluded[model.key] = "capability"
            continue

        kept.append(model)

    return kept, excluded
