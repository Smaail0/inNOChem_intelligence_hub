"""Query the local hybrid RAG evidence index."""

from __future__ import annotations

import argparse
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
    RAG_TOP_K,
    RAG_USE_DENSE,
    RAG_USE_RERANKER,
)
from services.resource_manager import (  # noqa: E402
    get_cached_rag_store,
    get_cached_reranker,
    rag_index_modified_timestamp,
)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Query local RAG evidence snippets.")
    parser.add_argument("query", help="Evidence query, e.g. 'acetone GSK health score'")
    parser.add_argument("--top-k", type=int, default=RAG_TOP_K)
    args = parser.parse_args()

    store_handle = get_cached_rag_store(
        use_dense=RAG_USE_DENSE,
        source_docs_dir=RAG_SOURCE_DOCS_DIR,
        index_dir=RAG_INDEX_DIR,
        embedding_model_name=RAG_EMBEDDING_MODEL,
        index_modified_timestamp=rag_index_modified_timestamp(RAG_INDEX_DIR),
    )
    store = store_handle.resource
    if RAG_USE_RERANKER:
        store.reranker_backend = get_cached_reranker(RAG_RERANKER_MODEL).resource
    results, diagnostics = store.query_with_diagnostics(args.query, top_k=args.top_k)
    print(f"query: {args.query}")
    print("retrieval stages enabled:")
    print("  bm25: true")
    print(f"  dense: {str(RAG_USE_DENSE).lower()} (RAG_USE_DENSE)")
    print(f"  reranker: {str(RAG_USE_RERANKER).lower()} (RAG_USE_RERANKER)")
    print(f"embedding_model: {RAG_EMBEDDING_MODEL}")
    print(f"reranker_model: {RAG_RERANKER_MODEL}")
    print(f"rag resource load seconds: {store_handle.diagnostics.get('load_duration_seconds')}")
    print(f"chunk count: {diagnostics['chunk_count']}")
    print(f"embedding status: {diagnostics['embedding_status']}")
    print(f"embedding backend: {diagnostics['embedding_backend_name'] or 'none'}")
    print(f"embedding count: {diagnostics['embedding_count']}")
    print(f"embedding dimensions: {diagnostics['embedding_dimensions']}")
    if diagnostics["embedding_error"]:
        print(f"embedding error: {diagnostics['embedding_error']}")
    print(f"BM25 candidates: {diagnostics['bm25_candidate_count']}")
    print(f"dense candidates: {diagnostics['dense_candidate_count']}")
    print(f"merged candidates: {diagnostics['merged_candidate_count']}")
    print(f"reranked candidates: {diagnostics['reranked_candidate_count']}")
    print(f"results: {len(results)}")
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        print("-" * 80)
        print(f"result {index}")
        print(f"score: {result.score}")
        print(f"method: {result.retrieval_method}")
        print(f"source: {chunk.source_name}")
        print(f"page: {chunk.page_number}")
        print(f"section: {chunk.section_heading or 'n/a'}")
        print(f"source_type: {chunk.source_type}")
        print(f"chars: {chunk.start_char}-{chunk.end_char}")
        print(f"snippet: {chunk.chunk_text[:600]}")


if __name__ == "__main__":
    main()
