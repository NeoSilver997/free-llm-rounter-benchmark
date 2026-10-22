from __future__ import annotations

from catalog.schema import ModelEntry

TASK_TYPES = (
    "text",
    "reasoning",
    "code",
    "vision",
    "audio",
    "embeddings",
    "rerank",
    "roleplay",
)

# Ordered: a model can match several types, so rules are additive, not exclusive.
_RULES = (
    ("reasoning", ("reasoning",)),
    ("code", ("code", "coder")),
    ("vision", ("image", "vision", "multimodal")),
    ("audio", ("audio", "speech")),
    ("embeddings", ("embedding",)),
    ("rerank", ("rerank",)),
    ("roleplay", ("roleplay",)),
)


def normalize_modality(modality: str | None) -> set[str]:
    if not modality:
        return {"text"}
    text = modality.lower()
    types = {t for t, needles in _RULES if any(n in text for n in needles)}
    return types or {"text"}


def model_task_types(model: ModelEntry) -> set[str]:
    return normalize_modality(model.modality)
