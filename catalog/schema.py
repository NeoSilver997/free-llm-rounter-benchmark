from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

Category = Literal["provider_api", "inference_provider"]


class ModelEntry(BaseModel):
    id: str | None = None
    name: str
    context: str | None = None
    maxOutput: str | None = None
    modality: str | None = None
    rateLimit: str | None = None
    # Filled by the loader, not present in data.json.
    provider: str = ""
    key: str = ""

    @property
    def placeholder(self) -> bool:
        return self.id is None


class Provider(BaseModel):
    name: str
    category: Category
    country: str | None = None
    flag: str | None = None
    url: str | None = None
    baseUrl: str
    description: str | None = None
    footnoteRef: int | None = None
    models: list[ModelEntry] = []


class Catalog(BaseModel):
    lastUpdated: str | None = None
    providers: list[Provider] = []
