from __future__ import annotations

from pydantic import BaseModel, Field


class ResourceStatus(BaseModel):
    name: str
    ready: bool
    detail: str


class SystemStatusResponse(BaseModel):
    app_name: str = "InNOChem Intelligence Hub"
    api_version: str = "0.1.0"
    status: str = "ok"
    resources: list[ResourceStatus] = Field(default_factory=list)
    configured_models: dict[str, str] = Field(default_factory=dict)

