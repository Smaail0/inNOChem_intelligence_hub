"""Cached local resource loading for heavy InNOChem components.

This module centralizes Streamlit resource caching so model/index objects are
loaded once per app process, while expensive inference and retrieval remain
explicit user actions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import requests
import streamlit as st

from core.config import (
    CHEM_NER_CACHE_DIR,
    CHEM_NER_DOWNLOAD_IF_MISSING,
    CHEM_NER_MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RAG_EMBEDDING_MODEL,
    RAG_INDEX_DIR,
    RAG_RERANKER_MODEL,
    RAG_SOURCE_DOCS_DIR,
)
from services.gemma_extractor import GemmaExtractionConfig, check_ollama_health
from services.rag_evidence import (
    LocalCrossEncoderRerankerBackend,
    LocalSentenceTransformerEmbeddingBackend,
    RagEvidenceIndex,
    RagEvidenceStore,
)


RESOURCE_MANAGER_VERSION = "resource_manager_v1"


@dataclass
class ResourceHandle:
    name: str
    resource: Any
    diagnostics: dict[str, Any]


def warmup_local_resources(
    *,
    include_ner: bool = True,
    include_bge: bool = True,
    include_rag: bool = True,
    include_reranker: bool = False,
    include_ollama_status: bool = False,
) -> list[ResourceHandle]:
    """Load selected local resources without running inference/retrieval."""
    handles = []
    if include_ner:
        handles.append(get_cached_ner_pipeline())
    if include_bge:
        handles.append(get_cached_bge_embedding_backend())
    if include_rag:
        handles.append(get_cached_rag_store(use_dense=include_bge))
    if include_reranker:
        handles.append(get_cached_reranker())
    if include_ollama_status:
        status = get_cached_ollama_status()
        handles.append(ResourceHandle("ollama_status", status, status))
    return handles


@st.cache_resource(show_spinner=False)
def warmup_cached_ollama_explanation_model(
    base_url: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    timeout_seconds: float = 10.0,
    version: str = RESOURCE_MANAGER_VERSION,
) -> ResourceHandle:
    """Send a tiny prompt so Ollama can load the explanation model early."""
    del version
    started = time.perf_counter()
    timeout_seconds = min(float(timeout_seconds), 10.0)
    endpoint = f"{base_url.rstrip('/')}/api/generate"

    try:
        response = requests.post(
            endpoint,
            json={
                "model": model,
                "prompt": 'Return exactly this JSON: {"status":"ok"}',
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 12,
                },
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        diagnostics = {
            "resource": "ollama_explanation_model",
            "status": "warmed",
            "base_url": base_url,
            "model": model,
            "timeout_seconds": timeout_seconds,
            "load_duration_seconds": round(time.perf_counter() - started, 3),
            "memory_warning": _memory_warning(),
        }
    except Exception as exc:
        diagnostics = {
            "resource": "ollama_explanation_model",
            "status": "warmup_failed",
            "base_url": base_url,
            "model": model,
            "timeout_seconds": timeout_seconds,
            "error": f"{type(exc).__name__}: {exc}",
            "load_duration_seconds": round(time.perf_counter() - started, 3),
            "memory_warning": _memory_warning(),
        }

    return ResourceHandle("ollama_explanation_model", diagnostics, diagnostics)


@st.cache_resource(show_spinner=False)
def get_cached_ner_pipeline(
    model_name: str = CHEM_NER_MODEL_NAME,
    cache_dir: str = CHEM_NER_CACHE_DIR,
    download_if_missing: bool = CHEM_NER_DOWNLOAD_IF_MISSING,
    version: str = RESOURCE_MANAGER_VERSION,
) -> ResourceHandle:
    """Load and cache the local Hugging Face token-classification pipeline."""
    del version
    started = time.perf_counter()
    from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
        )
    except Exception:
        if not download_if_missing:
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=False,
        )
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
    )
    duration = round(time.perf_counter() - started, 3)
    diagnostics = {
        "resource": "transformer_ner_pipeline",
        "status": "loaded",
        "model_name": model_name,
        "cache_dir": cache_dir,
        "model_path": _safe_model_path(model),
        "device": _safe_pipeline_device(ner_pipeline),
        "load_duration_seconds": duration,
        "memory_warning": _memory_warning(),
    }
    return ResourceHandle(
        name="transformer_ner_pipeline",
        resource={"pipeline": ner_pipeline, "tokenizer": tokenizer},
        diagnostics=diagnostics,
    )


@st.cache_resource(show_spinner=False)
def get_cached_bge_embedding_backend(
    model_name: str = RAG_EMBEDDING_MODEL,
    version: str = RESOURCE_MANAGER_VERSION,
) -> ResourceHandle:
    """Load and cache the local dense embedding backend."""
    del version
    started = time.perf_counter()
    backend = LocalSentenceTransformerEmbeddingBackend(model_name)
    duration = round(time.perf_counter() - started, 3)
    diagnostics = {
        "resource": "bge_embedding_model",
        "status": "loaded",
        "model_name": model_name,
        "backend": getattr(backend, "backend_name", ""),
        "model_path": _safe_model_path(getattr(backend, "model", None)),
        "load_duration_seconds": duration,
        "memory_warning": _memory_warning(),
    }
    return ResourceHandle("bge_embedding_model", backend, diagnostics)


@st.cache_resource(show_spinner=False)
def get_cached_rag_store(
    use_dense: bool = False,
    source_docs_dir: str = RAG_SOURCE_DOCS_DIR,
    index_dir: str = RAG_INDEX_DIR,
    embedding_model_name: str = RAG_EMBEDDING_MODEL,
    index_modified_timestamp: float | None = None,
    version: str = RESOURCE_MANAGER_VERSION,
) -> ResourceHandle:
    """Load and cache the local RAG index, optionally with BGE enabled."""
    del version, index_modified_timestamp
    started = time.perf_counter()
    embedding_backend = None
    if use_dense:
        embedding_backend = get_cached_bge_embedding_backend(embedding_model_name).resource
    store = RagEvidenceIndex(
        source_docs_dir=source_docs_dir,
        index_dir=index_dir,
        embedding_backend=embedding_backend,
        use_dense=use_dense,
    ).load_or_build()
    duration = round(time.perf_counter() - started, 3)
    diagnostics = store.retrieval_diagnostics()
    diagnostics.update(
        {
            "resource": "rag_evidence_index",
            "status": "loaded",
            "source_docs_dir": source_docs_dir,
            "index_dir": index_dir,
            "index_path": str(Path(index_dir) / "rag_evidence_index.pkl"),
            "load_duration_seconds": duration,
            "memory_warning": _memory_warning(),
        }
    )
    return ResourceHandle("rag_evidence_index", store, diagnostics)


@st.cache_resource(show_spinner=False)
def get_cached_reranker(
    model_name: str = RAG_RERANKER_MODEL,
    version: str = RESOURCE_MANAGER_VERSION,
) -> ResourceHandle:
    """Load and cache the optional local reranker."""
    del version
    started = time.perf_counter()
    reranker = LocalCrossEncoderRerankerBackend(model_name)
    duration = round(time.perf_counter() - started, 3)
    diagnostics = {
        "resource": "rag_reranker",
        "status": "loaded",
        "model_name": model_name,
        "model_path": _safe_model_path(getattr(reranker, "model", None)),
        "load_duration_seconds": duration,
        "memory_warning": _memory_warning(),
    }
    return ResourceHandle("rag_reranker", reranker, diagnostics)


@st.cache_data(show_spinner=False, ttl=60)
def get_cached_ollama_status(
    base_url: str = OLLAMA_BASE_URL,
    model: str = OLLAMA_MODEL,
    version: str = RESOURCE_MANAGER_VERSION,
) -> dict[str, Any]:
    """Cache optional Ollama health/model status for a short interval."""
    del version
    started = time.perf_counter()
    try:
        check_ollama_health(GemmaExtractionConfig(base_url=base_url, model=model))
    except Exception as exc:
        return {
            "resource": "ollama_status",
            "status": "unavailable",
            "base_url": base_url,
            "model": model,
            "error": str(exc),
            "load_duration_seconds": round(time.perf_counter() - started, 3),
        }
    return {
        "resource": "ollama_status",
        "status": "available",
        "base_url": base_url,
        "model": model,
        "load_duration_seconds": round(time.perf_counter() - started, 3),
    }


def rag_index_modified_timestamp(index_dir: str = RAG_INDEX_DIR) -> float:
    """Return the RAG index mtime used as a cache key component."""
    index_path = Path(index_dir) / "rag_evidence_index.pkl"
    if not index_path.exists():
        return 0.0
    return index_path.stat().st_mtime


def resource_diagnostics(handles: list[ResourceHandle]) -> list[dict[str, Any]]:
    return [handle.diagnostics for handle in handles]


def _safe_pipeline_device(ner_pipeline: Any) -> str:
    device = getattr(ner_pipeline, "device", None)
    if device is not None:
        return str(device)
    model = getattr(ner_pipeline, "model", None)
    if model is not None:
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return "unknown"
    return "unknown"


def _safe_model_path(model: Any) -> str:
    if model is None:
        return ""
    for attribute in ("name_or_path", "_name_or_path"):
        value = getattr(model, attribute, "")
        if value:
            return str(value)
    config = getattr(model, "config", None)
    return str(getattr(config, "_name_or_path", "")) if config is not None else ""


def _memory_warning() -> str:
    try:
        import psutil

        memory = psutil.virtual_memory()
        if memory.available < 2 * 1024**3:
            return "Less than 2 GB RAM appears available; local models may feel slow."
    except Exception:
        return "Memory availability not checked; install psutil for diagnostics."
    return ""
