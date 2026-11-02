from __future__ import annotations

from pathlib import Path

import requests
from fastapi import APIRouter

from core.config import (
    CHEM_NER_MODEL_NAME,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_INDEX_DIR,
    RAG_SOURCE_DOCS_DIR,
)
from backend.app.schemas.system import ResourceStatus, SystemStatusResponse


router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "innochem-api"}


@router.get("/status", response_model=SystemStatusResponse)
def system_status() -> SystemStatusResponse:
    resources = [
        _path_status("RAG knowledge base", Path(RAG_INDEX_DIR)),
        _path_status("RAG source documents", Path(RAG_SOURCE_DOCS_DIR)),
        _ollama_status(),
    ]
    return SystemStatusResponse(
        resources=resources,
        configured_models={
            "chemical_ner": CHEM_NER_MODEL_NAME,
            "ollama": OLLAMA_MODEL,
            "gemini": GEMINI_MODEL,
        },
    )


def _path_status(name: str, path: Path) -> ResourceStatus:
    if path.exists():
        return ResourceStatus(name=name, ready=True, detail=str(path))
    return ResourceStatus(name=name, ready=False, detail=f"Missing path: {path}")


def _ollama_status() -> ResourceStatus:
    endpoint = f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
    try:
        response = requests.get(endpoint, timeout=2.0)
        response.raise_for_status()
    except Exception as exc:
        return ResourceStatus(
            name="Ollama",
            ready=False,
            detail=f"Not reachable at {OLLAMA_BASE_URL}: {type(exc).__name__}",
        )

    model_names = {
        item.get("name")
        for item in response.json().get("models", [])
        if isinstance(item, dict)
    }
    if OLLAMA_MODEL in model_names:
        return ResourceStatus(
            name="Ollama",
            ready=True,
            detail=f"Connected; configured model '{OLLAMA_MODEL}' is available.",
        )
    return ResourceStatus(
        name="Ollama",
        ready=True,
        detail=f"Connected; configured model '{OLLAMA_MODEL}' was not listed.",
    )

