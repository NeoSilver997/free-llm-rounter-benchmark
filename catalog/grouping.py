from __future__ import annotations

from catalog.loader import routable_models
from catalog.schema import Catalog, ModelEntry
from catalog.task_types import TASK_TYPES, model_task_types


def _primary(model: ModelEntry, provider_category: str) -> str:
    return provider_category


def group(
    catalog: Catalog,
    primary: str = "category",
    secondary: str = "task_type",
) -> dict[str, dict[str, list[ModelEntry]]]:
    """category -> task_type -> models (or the reverse when the args are swapped).

    A model that supports several task types appears in each matching group.
    """
    if primary not in ("category", "task_type") or secondary not in ("category", "task_type"):
        raise ValueError("primary and secondary must be 'category' or 'task_type'")

    grouped: dict[str, dict[str, list[ModelEntry]]] = {}
    for provider in catalog.providers:
        for model in provider.models:
            if model.placeholder:
                continue
            types = sorted(model_task_types(model))
            for task_type in types:
                key1 = provider.category if primary == "category" else task_type
                key2 = task_type if secondary == "task_type" else provider.category
                grouped.setdefault(key1, {}).setdefault(key2, []).append(model)

    for key1 in grouped:
        for key2 in grouped[key1]:
            grouped[key1][key2].sort(key=lambda m: m.key)
    return dict(sorted(grouped.items()))


def group_counts(grouped: dict[str, dict[str, list[ModelEntry]]]) -> dict[str, dict[str, int]]:
    return {k1: {k2: len(models) for k2, models in sorted(inner.items())} for k1, inner in grouped.items()}


def render_table(grouped: dict[str, dict[str, list[ModelEntry]]]) -> str:
    lines: list[str] = []
    for key1, inner in grouped.items():
        lines.append(f"## {key1}")
        for key2, models in sorted(inner.items()):
            lines.append(f"### {key2} ({len(models)})")
            for model in models:
                lines.append(
                    f"  {model.key} | provider={model.provider} | context={model.context or '-'} "
                    f"| maxOutput={model.maxOutput or '-'} | modality={model.modality or '-'} "
                    f"| limit={model.rateLimit or '-'}"
                )
        lines.append("")
    return "\n".join(lines)


def task_type_order() -> tuple[str, ...]:
    return TASK_TYPES
