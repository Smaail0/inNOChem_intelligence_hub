"""Build the local hybrid RAG evidence index."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (  # noqa: E402
    RAG_EMBEDDING_MODEL,
    RAG_INDEX_DIR,
    RAG_RERANKER_MODEL,
    RAG_SOURCE_DOCS_DIR,
    RAG_USE_DENSE,
    RAG_USE_RERANKER,
)
from services.rag_evidence import RagEvidenceIndex  # noqa: E402


def main() -> None:
    index = RagEvidenceIndex(RAG_SOURCE_DOCS_DIR, RAG_INDEX_DIR)
    store = index.load_or_build(force_rebuild=True)
    diagnostics = store.retrieval_diagnostics()
    print("RAG index built")
    print("retrieval stages enabled:")
    print("  bm25: true")
    print(f"  dense: {str(RAG_USE_DENSE).lower()} (RAG_USE_DENSE)")
    print(f"  reranker: {str(RAG_USE_RERANKER).lower()} (RAG_USE_RERANKER)")
    print(f"embedding_model: {RAG_EMBEDDING_MODEL}")
    print(f"reranker_model: {RAG_RERANKER_MODEL}")
    print(f"source_docs_dir: {RAG_SOURCE_DOCS_DIR}")
    print(f"index_dir: {RAG_INDEX_DIR}")
    print(f"chunk count: {diagnostics['chunk_count']}")
    print(f"embedding status: {diagnostics['embedding_status']}")
    print(f"embedding backend: {diagnostics['embedding_backend_name'] or 'none'}")
    print(f"embedding count: {diagnostics['embedding_count']}")
    print(f"embedding dimensions: {diagnostics['embedding_dimensions']}")
    if diagnostics["embedding_error"]:
        print(f"embedding error: {diagnostics['embedding_error']}")
    print(f"index_path: {index.index_path}")


if __name__ == "__main__":
    main()
