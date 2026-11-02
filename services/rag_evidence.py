"""Hybrid local retrieval for source-backed evidence snippets.

This layer retrieves passages for facts that already exist in Chemical
Intelligence. It does not create facts, call LLMs, score chemicals, or make
recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import pickle
from pathlib import Path
import re
from typing import Any, Callable, Protocol

import fitz
import numpy as np

from core.config import (
    RAG_CHUNK_OVERLAP_CHARS,
    RAG_EMBEDDING_MODEL,
    RAG_INDEX_DIR,
    RAG_MAX_EVIDENCE_QUERIES,
    RAG_MAX_CHARS_PER_CHUNK,
    RAG_RERANKER_MODEL,
    RAG_SOURCE_DOCS_DIR,
    RAG_TOP_K,
    RAG_USE_DENSE,
    RAG_USE_RERANKER,
)


INDEX_VERSION = "hybrid_bm25_dense_v1"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".xlsx", ".xls"}


class EmbeddingBackend(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return one vector per input text."""


class RerankerBackend(Protocol):
    model_name: str

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Return one relevance score per query/text pair."""


@dataclass(frozen=True)
class EvidenceChunk:
    document_id: str
    source_name: str
    document_path: str
    page_number: int | None
    section_heading: str
    chunk_text: str
    start_char: int
    end_char: int
    source_type: str
    chunk_id: str

    @property
    def document_name(self) -> str:
        """Backward-compatible alias used by the current Streamlit UI."""
        return self.source_name

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    chunk: EvidenceChunk
    score: float
    retrieval_method: str
    bm25_score: float = 0.0
    dense_score: float = 0.0
    reranker_score: float | None = None


@dataclass(frozen=True)
class RetrievedEvidence:
    chemical_name: str
    source_type: str
    fact_attribute: str
    topic_group: str
    document_name: str
    page_number: int | None
    section_heading: str
    evidence_snippet: str
    similarity_score: float
    provenance: str
    retrieval_method: str = "hybrid"
    contains_chemical_name: bool = False
    contains_cas: bool = False
    contains_topic_keywords: bool = False
    generic_methodology_penalty: float = 0.0
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalSentenceTransformerEmbeddingBackend:
    """Local BGE encoder for dense retrieval.

    SentenceTransformer is preferred. If the installed sentence-transformers
    stack cannot construct the pooling module, fall back to local Transformers
    mean pooling over the same cached model files.
    """

    def __init__(self, model_name: str = RAG_EMBEDDING_MODEL) -> None:
        self.model_name = model_name
        self.backend_name = "sentence_transformers"
        self.model = None
        self.tokenizer = None

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception as sentence_transformer_error:
            self.backend_name = "transformers_mean_pooling"
            try:
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=True)
                self.model.eval()
            except Exception as transformers_error:
                raise RuntimeError(
                    "Could not load local dense embedding model with SentenceTransformer "
                    f"({sentence_transformer_error}) or Transformers mean pooling "
                    f"({transformers_error})."
                ) from transformers_error

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.backend_name == "sentence_transformers":
            vectors = self.model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(vectors, dtype=float)

        import torch

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():
            model_output = self.model(**encoded)
        token_embeddings = model_output.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size())
        masked_embeddings = token_embeddings * attention_mask
        summed = masked_embeddings.sum(dim=1)
        counts = attention_mask.sum(dim=1).clamp(min=1e-9)
        vectors = summed / counts
        vectors = torch.nn.functional.normalize(vectors, p=2, dim=1)
        return vectors.cpu().numpy()


class LocalCrossEncoderRerankerBackend:
    """Optional local cross-encoder reranker."""

    def __init__(self, model_name: str = RAG_RERANKER_MODEL) -> None:
        self.model_name = model_name
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, local_files_only=True)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(score) for score in self.model.predict([(query, text) for text in texts])]


class BM25Index:
    """Small Okapi BM25 implementation to avoid another runtime dependency."""

    def __init__(
        self,
        tokenized_documents: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.tokenized_documents = tokenized_documents
        self.k1 = k1
        self.b = b
        self.document_count = len(tokenized_documents)
        self.document_lengths = [len(document) for document in tokenized_documents]
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count if self.document_count else 0.0
        )
        self.document_frequencies = self._document_frequencies(tokenized_documents)

    def scores(self, query: str) -> list[float]:
        query_terms = _tokenize(query)
        if not query_terms or not self.tokenized_documents:
            return [0.0 for _ in self.tokenized_documents]

        scores = []
        for document, document_length in zip(self.tokenized_documents, self.document_lengths):
            term_counts = {}
            for term in document:
                term_counts[term] = term_counts.get(term, 0) + 1
            score = 0.0
            for term in query_terms:
                frequency = term_counts.get(term, 0)
                if frequency == 0:
                    continue
                document_frequency = self.document_frequencies.get(term, 0)
                idf = math.log(
                    1
                    + (self.document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + self.k1 * (
                    1
                    - self.b
                    + self.b * document_length / max(self.average_document_length, 1.0)
                )
                score += idf * frequency * (self.k1 + 1) / denominator
            scores.append(score)
        return scores

    @staticmethod
    def _document_frequencies(tokenized_documents: list[list[str]]) -> dict[str, int]:
        frequencies: dict[str, int] = {}
        for document in tokenized_documents:
            for term in set(document):
                frequencies[term] = frequencies.get(term, 0) + 1
        return frequencies


class RagEvidenceIndex:
    """Persistent hybrid BM25 + dense evidence index."""

    def __init__(
        self,
        source_docs_dir: str | Path = RAG_SOURCE_DOCS_DIR,
        index_dir: str | Path = RAG_INDEX_DIR,
        max_chars_per_chunk: int = RAG_MAX_CHARS_PER_CHUNK,
        chunk_overlap_chars: int = RAG_CHUNK_OVERLAP_CHARS,
        embedding_backend: EmbeddingBackend | None = None,
        use_dense: bool = RAG_USE_DENSE,
    ) -> None:
        self.source_docs_dir = Path(source_docs_dir)
        self.index_dir = Path(index_dir)
        self.max_chars_per_chunk = max_chars_per_chunk
        self.chunk_overlap_chars = chunk_overlap_chars
        self.embedding_backend = embedding_backend
        self.use_dense = use_dense
        self.index_path = self.index_dir / "rag_evidence_index.pkl"
        self.embedding_status = "not_initialized"
        self.embedding_error = ""

    def load_or_build(self, force_rebuild: bool = False) -> "RagEvidenceStore":
        self.source_docs_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        manifest_hash = self._manifest_hash()
        embedding_backend = self._embedding_backend()
        embedding_model_name = embedding_backend.model_name if embedding_backend else ""
        embedding_status = self.embedding_status
        embedding_error = self.embedding_error
        if not force_rebuild and self.index_path.exists():
            with self.index_path.open("rb") as handle:
                payload = pickle.load(handle)
            if (
                payload.get("version") == INDEX_VERSION
                and payload.get("manifest_hash") == manifest_hash
                and payload.get("embedding_model_name") == embedding_model_name
            ):
                return RagEvidenceStore(
                    chunks=payload["chunks"],
                    tokenized_documents=payload["tokenized_documents"],
                    dense_embeddings=payload.get("dense_embeddings"),
                    embedding_backend=embedding_backend,
                    source_docs_dir=self.source_docs_dir,
                    embedding_status=embedding_status,
                    embedding_error=embedding_error,
                    embedding_model_name=embedding_model_name,
                )

        chunks = chunk_source_documents(
            self.source_docs_dir,
            max_chars_per_chunk=self.max_chars_per_chunk,
            chunk_overlap_chars=self.chunk_overlap_chars,
        )
        tokenized_documents = [_tokenize(chunk.chunk_text) for chunk in chunks]
        dense_embeddings = None
        if chunks and embedding_backend is not None:
            dense_embeddings = _normalize_vectors(
                embedding_backend.encode([chunk.chunk_text for chunk in chunks])
            )
            self.embedding_status = "enabled"
            embedding_status = self.embedding_status
        payload = {
            "version": INDEX_VERSION,
            "manifest_hash": manifest_hash,
            "embedding_model_name": embedding_model_name,
            "chunks": chunks,
            "tokenized_documents": tokenized_documents,
            "dense_embeddings": dense_embeddings,
        }
        with self.index_path.open("wb") as handle:
            pickle.dump(payload, handle)
        return RagEvidenceStore(
            chunks=chunks,
            tokenized_documents=tokenized_documents,
            dense_embeddings=dense_embeddings,
            embedding_backend=embedding_backend,
            source_docs_dir=self.source_docs_dir,
            embedding_status=embedding_status,
            embedding_error=embedding_error,
            embedding_model_name=embedding_model_name,
        )

    def _embedding_backend(self) -> EmbeddingBackend | None:
        if not self.use_dense:
            self.embedding_status = "disabled_by_config"
            self.embedding_error = (
                "Dense retrieval disabled by RAG_USE_DENSE=false "
                "or RagEvidenceIndex(use_dense=False)."
            )
            return None
        if self.embedding_backend is not None:
            self.embedding_status = "enabled"
            self.embedding_error = ""
            return self.embedding_backend
        try:
            backend = LocalSentenceTransformerEmbeddingBackend()
            self.embedding_status = "enabled"
            self.embedding_error = ""
            return backend
        except Exception as exc:
            self.embedding_status = "unavailable"
            self.embedding_error = (
                f"Could not load RAG_EMBEDDING_MODEL={RAG_EMBEDDING_MODEL!r} locally: {exc}"
            )
            return None

    def _manifest_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(INDEX_VERSION.encode("utf-8"))
        digest.update(str(self.max_chars_per_chunk).encode("utf-8"))
        digest.update(str(self.chunk_overlap_chars).encode("utf-8"))
        for path in _source_paths(self.source_docs_dir):
            stat = path.stat()
            digest.update(str(path.relative_to(self.source_docs_dir)).encode("utf-8"))
            digest.update(str(stat.st_size).encode("utf-8"))
            digest.update(str(int(stat.st_mtime)).encode("utf-8"))
        return digest.hexdigest()


class RagEvidenceStore:
    def __init__(
        self,
        chunks: list[EvidenceChunk],
        tokenized_documents: list[list[str]],
        dense_embeddings: np.ndarray | None,
        embedding_backend: EmbeddingBackend | None,
        source_docs_dir: Path,
        reranker_backend: RerankerBackend | None = None,
        embedding_status: str = "not_initialized",
        embedding_error: str = "",
        embedding_model_name: str = "",
    ) -> None:
        self.chunks = chunks
        self.bm25 = BM25Index(tokenized_documents)
        self.dense_embeddings = dense_embeddings
        self.embedding_backend = embedding_backend
        self.source_docs_dir = source_docs_dir
        self.reranker_backend = reranker_backend
        self.embedding_status = embedding_status
        self.embedding_error = embedding_error
        self.embedding_model_name = embedding_model_name

    def query(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
        use_reranker: bool = RAG_USE_RERANKER,
    ) -> list[RetrievalResult]:
        if not self.chunks or not query.strip():
            return []

        bm25_results = self._bm25_results(query, bm25_top_k or top_k)
        dense_results = self._dense_results(query, dense_top_k or top_k)
        merged = merge_retrieval_results(bm25_results, dense_results)
        reranker = self.reranker_backend if use_reranker else None
        if use_reranker and reranker is None:
            reranker = _optional_reranker()
        if reranker is not None and merged:
            merged = _rerank(query, merged, reranker)
        return merged[:top_k]

    def query_with_diagnostics(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        bm25_top_k: int | None = None,
        dense_top_k: int | None = None,
        use_reranker: bool = RAG_USE_RERANKER,
    ) -> tuple[list[RetrievalResult], dict[str, Any]]:
        if not self.chunks or not query.strip():
            diagnostics = self.retrieval_diagnostics()
            diagnostics.update(
                {
                    "bm25_candidate_count": 0,
                    "dense_candidate_count": 0,
                    "merged_candidate_count": 0,
                    "reranked_candidate_count": 0,
                }
            )
            return [], diagnostics

        bm25_results = self._bm25_results(query, bm25_top_k or top_k)
        dense_results = self._dense_results(query, dense_top_k or top_k)
        merged = merge_retrieval_results(bm25_results, dense_results)
        reranked = []
        reranker = self.reranker_backend if use_reranker else None
        if use_reranker and reranker is None:
            reranker = _optional_reranker()
        if reranker is not None and merged:
            reranked = _rerank(query, merged, reranker)
            results = reranked[:top_k]
        else:
            results = merged[:top_k]
        diagnostics = self.retrieval_diagnostics()
        diagnostics.update(
            {
                "bm25_candidate_count": len(bm25_results),
                "dense_candidate_count": len(dense_results),
                "merged_candidate_count": len(merged),
                "reranked_candidate_count": len(reranked),
                "bm25_enabled": True,
                "dense_enabled": self.embedding_backend is not None
                and self.dense_embeddings is not None,
                "reranker_enabled": reranker is not None,
            }
        )
        return results, diagnostics

    def retrieval_diagnostics(self) -> dict[str, Any]:
        embedding_count = 0
        embedding_dimensions = 0
        if self.dense_embeddings is not None:
            embedding_count = int(self.dense_embeddings.shape[0])
            embedding_dimensions = int(self.dense_embeddings.shape[1])
        return {
            "chunk_count": len(self.chunks),
            "embedding_status": self.embedding_status,
            "embedding_error": self.embedding_error,
            "embedding_model_name": self.embedding_model_name,
            "embedding_backend_name": getattr(self.embedding_backend, "backend_name", ""),
            "embedding_count": embedding_count,
            "embedding_dimensions": embedding_dimensions,
            "bm25_enabled": True,
            "dense_enabled": self.embedding_backend is not None
            and self.dense_embeddings is not None,
            "reranker_config_enabled": RAG_USE_RERANKER,
            "reranker_model_name": RAG_RERANKER_MODEL,
        }

    def _bm25_results(self, query: str, top_k: int) -> list[RetrievalResult]:
        scores = self.bm25.scores(query)
        return _top_results(self.chunks, scores, top_k, "bm25")

    def _dense_results(self, query: str, top_k: int) -> list[RetrievalResult]:
        if self.embedding_backend is None or self.dense_embeddings is None:
            return []
        query_vector = _normalize_vectors(self.embedding_backend.encode([query]))[0]
        scores = self.dense_embeddings @ query_vector
        return _top_results(self.chunks, scores.tolist(), top_k, "dense")


def merge_retrieval_results(
    bm25_results: list[RetrievalResult],
    dense_results: list[RetrievalResult],
) -> list[RetrievalResult]:
    merged: dict[str, RetrievalResult] = {}
    for result in [*bm25_results, *dense_results]:
        existing = merged.get(result.chunk.chunk_id)
        if existing is None:
            merged[result.chunk.chunk_id] = result
            continue
        bm25_score = max(existing.bm25_score, result.bm25_score)
        dense_score = max(existing.dense_score, result.dense_score)
        merged[result.chunk.chunk_id] = RetrievalResult(
            chunk=existing.chunk,
            score=round(bm25_score + dense_score, 4),
            retrieval_method="hybrid",
            bm25_score=bm25_score,
            dense_score=dense_score,
        )
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)


def retrieve_evidence_for_intelligence(
    intelligence_rows: list[dict[str, Any]],
    source_docs_dir: str | Path = RAG_SOURCE_DOCS_DIR,
    index_dir: str | Path = RAG_INDEX_DIR,
    top_k: int = RAG_TOP_K,
    embedding_backend: EmbeddingBackend | None = None,
    reranker_backend: RerankerBackend | None = None,
) -> list[RetrievedEvidence]:
    """Retrieve evidence snippets for existing chemical intelligence facts."""
    store = RagEvidenceIndex(
        source_docs_dir,
        index_dir,
        embedding_backend=embedding_backend,
    ).load_or_build()
    store.reranker_backend = reranker_backend
    return retrieve_evidence_for_intelligence_with_store(
        intelligence_rows,
        store,
        top_k=top_k,
    )


def retrieve_evidence_for_intelligence_with_store(
    intelligence_rows: list[dict[str, Any]],
    store: RagEvidenceStore,
    top_k: int = RAG_TOP_K,
    max_queries: int | None = RAG_MAX_EVIDENCE_QUERIES,
    use_reranker: bool = RAG_USE_RERANKER,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[RetrievedEvidence]:
    """Retrieve evidence snippets using an already-loaded local RAG store."""
    evidence: list[RetrievedEvidence] = []
    query_rows = _dedupe_intelligence_rows_by_query(intelligence_rows)
    if max_queries is not None and max_queries > 0:
        query_rows = query_rows[:max_queries]
    total = len(query_rows)
    if progress_callback:
        progress_callback(
            {
                "stage": "start",
                "completed": 0,
                "total": total,
                "message": f"[RAG] Retrieving evidence for {total} fact queries",
            }
        )
    for index, row in enumerate(query_rows, start=1):
        query = build_evidence_query(row)
        if progress_callback:
            progress_callback(
                {
                    "stage": "query_start",
                    "completed": index - 1,
                    "total": total,
                    "current_query": query,
                    "message": f"[RAG] Query {index}/{total}: {query[:120]}",
                }
            )
        candidate_results = store.query(query, top_k=max(top_k * 4, top_k), use_reranker=use_reranker)
        results = _rerank_with_relevance_diagnostics(row, candidate_results)[:top_k]
        if progress_callback:
            progress_callback(
                {
                    "stage": "query_complete",
                    "completed": index,
                    "total": total,
                    "current_query": query,
                    "result_count": len(results),
                    "message": f"[RAG] Completed query {index}/{total}; snippets={len(results)}",
                }
            )
        for result in results:
            chunk = result.chunk
            chemical = row.get("chemical", {})
            diagnostics = evidence_relevance_diagnostics(row, chunk)
            evidence.append(
                RetrievedEvidence(
                    chemical_name=chemical.get("canonical_name")
                    or chemical.get("original_surface_form")
                    or "unknown",
                    source_type=str(row.get("source", "")),
                    fact_attribute=str(row.get("attribute", "")),
                    topic_group=_topic_group(str(row.get("attribute", ""))),
                    document_name=chunk.source_name,
                    page_number=chunk.page_number,
                    section_heading=chunk.section_heading,
                    evidence_snippet=_best_snippet(chunk.chunk_text, query),
                    similarity_score=result.score,
                    provenance=_chunk_provenance(chunk),
                    retrieval_method=result.retrieval_method,
                    contains_chemical_name=diagnostics["contains_chemical_name"],
                    contains_cas=diagnostics["contains_cas"],
                    contains_topic_keywords=diagnostics["contains_topic_keywords"],
                    generic_methodology_penalty=diagnostics["generic_methodology_penalty"],
                    relevance_score=diagnostics["relevance_score"],
                )
            )
    if progress_callback:
        progress_callback(
            {
                "stage": "complete",
                "completed": total,
                "total": total,
                "evidence_count": len(evidence),
                "message": f"[RAG] Evidence retrieval complete; snippets={len(evidence)}",
            }
        )
    return evidence


def build_evidence_query(intelligence_row: dict[str, Any]) -> str:
    chemical = intelligence_row.get("chemical", {})
    query_parts = [
        chemical.get("canonical_name"),
        chemical.get("original_surface_form"),
        *chemical.get("aliases", []),
        *chemical.get("pubchem_synonyms", []),
        *_as_list(chemical.get("cas")),
        chemical.get("pubchem_cid"),
        chemical.get("cid"),
        intelligence_row.get("source"),
        intelligence_row.get("attribute"),
        _topic_group(str(intelligence_row.get("attribute", ""))),
        intelligence_row.get("value"),
    ]
    return " ".join(_unique_strings(query_parts))


def evidence_relevance_diagnostics(
    intelligence_row: dict[str, Any],
    chunk: EvidenceChunk,
) -> dict[str, Any]:
    """Return transparent relevance diagnostics for a retrieved evidence chunk."""
    text = chunk.chunk_text.casefold()
    chemical = intelligence_row.get("chemical", {})
    chemical_terms = [
        chemical.get("canonical_name"),
        chemical.get("original_surface_form"),
        *chemical.get("aliases", []),
        *chemical.get("pubchem_synonyms", []),
    ]
    cas_terms = _as_list(chemical.get("cas"))
    topic_keywords = _topic_keywords(
        str(intelligence_row.get("attribute", "")),
        _topic_group(str(intelligence_row.get("attribute", ""))),
    )
    contains_chemical_name = any(_contains_term(text, term) for term in chemical_terms)
    contains_cas = any(str(term).casefold() in text for term in cas_terms if term)
    contains_topic_keywords = any(_contains_term(text, term) for term in topic_keywords)
    generic_penalty = _generic_methodology_penalty(chunk, contains_chemical_name, contains_cas)
    relevance_score = (
        (1.0 if contains_chemical_name else 0.0)
        + (1.0 if contains_cas else 0.0)
        + (0.5 if contains_topic_keywords else 0.0)
        - generic_penalty
    )
    return {
        "contains_chemical_name": contains_chemical_name,
        "contains_cas": contains_cas,
        "contains_topic_keywords": contains_topic_keywords,
        "generic_methodology_penalty": generic_penalty,
        "relevance_score": round(relevance_score, 3),
    }


def _rerank_with_relevance_diagnostics(
    intelligence_row: dict[str, Any],
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    adjusted_results = []
    for result in results:
        diagnostics = evidence_relevance_diagnostics(intelligence_row, result.chunk)
        adjusted_results.append(
            RetrievalResult(
                chunk=result.chunk,
                score=round(result.score + diagnostics["relevance_score"], 4),
                retrieval_method=result.retrieval_method,
                bm25_score=result.bm25_score,
                dense_score=result.dense_score,
                reranker_score=result.reranker_score,
            )
        )
    return sorted(adjusted_results, key=lambda item: item.score, reverse=True)


def _dedupe_intelligence_rows_by_query(
    intelligence_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Avoid repeated retrieval work for duplicate fact queries."""
    seen: set[str] = set()
    unique_rows = []
    for row in intelligence_rows:
        query = re.sub(r"\s+", " ", build_evidence_query(row).casefold()).strip()
        if not query or query in seen:
            continue
        seen.add(query)
        unique_rows.append(row)
    return unique_rows


def _as_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for value in values:
        if value in (None, ""):
            continue
        text = str(value).strip()
        if not text:
            continue
        marker = text.casefold()
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(text)
    return unique


def _contains_term(text: str, term: Any) -> bool:
    if term in (None, ""):
        return False
    normalized = str(term).casefold().strip()
    if len(normalized) < 2:
        return False
    if re.search(r"\d{2,7}-\d{2}-\d", normalized):
        return normalized in text
    if re.search(r"[^a-z0-9]", normalized):
        return normalized in text
    return re.search(rf"\b{re.escape(normalized)}\b", text) is not None


def _topic_keywords(attribute: str, topic_group: str) -> list[str]:
    tokens = [
        token
        for token in re.split(r"[_\W]+", f"{attribute} {topic_group}".casefold())
        if len(token) >= 3
    ]
    synonyms = {
        "health": ["health", "toxicity", "toxic", "exposure"],
        "safety": ["safety", "hazard", "flammability", "reactivity"],
        "environment": ["environment", "ecotoxicity", "aquatic", "biodegradation"],
        "environmental": ["environment", "ecotoxicity", "aquatic", "biodegradation"],
        "ghs": ["ghs", "classification", "hazard"],
        "nfpa": ["nfpa", "hazard"],
        "carcinogenicity": ["carcinogenicity", "carcinogen", "cancer"],
        "regulatory": ["regulatory", "osha", "tsca", "cercla", "rcra", "dot"],
        "score": ["score", "ranking", "assessment"],
    }
    expanded = []
    for token in tokens:
        expanded.append(token)
        expanded.extend(synonyms.get(token, []))
    return _unique_strings(expanded)


def _generic_methodology_penalty(
    chunk: EvidenceChunk,
    contains_chemical_name: bool,
    contains_cas: bool,
) -> float:
    text = chunk.chunk_text.casefold()
    generic_markers = [
        "methodology",
        "method",
        "calculated without",
        "guide are calculated",
        "solvent selection guide",
        "were gaps in the data",
        "life cycle scores",
        "high-level guide",
    ]
    if contains_chemical_name or contains_cas:
        return 0.0
    marker_hits = sum(marker in text for marker in generic_markers)
    if marker_hits == 0:
        return 0.0
    return min(1.5, 0.5 + 0.25 * marker_hits)


def chunk_source_documents(
    source_docs_dir: str | Path,
    max_chars_per_chunk: int = RAG_MAX_CHARS_PER_CHUNK,
    chunk_overlap_chars: int = RAG_CHUNK_OVERLAP_CHARS,
) -> list[EvidenceChunk]:
    chunks: list[EvidenceChunk] = []
    for path in _source_paths(Path(source_docs_dir)):
        document_id = _document_id(path)
        source_type = _source_type(path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            page_texts = _extract_pdf_page_texts(path)
        elif suffix in {".xlsx", ".xls"}:
            page_texts = _extract_spreadsheet_texts(path)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
            page_texts = [(None, text)]
        for page_number, page_text in page_texts:
            heading = ""
            for chunk_index, chunk in enumerate(
                _chunk_text(page_text, max_chars_per_chunk, chunk_overlap_chars),
                start=1,
            ):
                heading = _detect_section_heading(chunk["text"]) or heading
                chunk_id = f"{document_id}:p{page_number or 1}:c{chunk_index}"
                chunks.append(
                    EvidenceChunk(
                        document_id=document_id,
                        source_name=path.name,
                        document_path=str(path),
                        page_number=page_number,
                        section_heading=heading,
                        chunk_text=chunk["text"],
                        start_char=chunk["start_char"],
                        end_char=chunk["end_char"],
                        source_type=source_type,
                        chunk_id=chunk_id,
                    )
                )
    return chunks


def _top_results(
    chunks: list[EvidenceChunk],
    scores: list[float],
    top_k: int,
    retrieval_method: str,
) -> list[RetrievalResult]:
    ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
    results = []
    for index, score in ranked[:top_k]:
        if score <= 0:
            continue
        results.append(
            RetrievalResult(
                chunk=chunks[index],
                score=round(float(score), 4),
                retrieval_method=retrieval_method,
                bm25_score=round(float(score), 4) if retrieval_method == "bm25" else 0.0,
                dense_score=round(float(score), 4) if retrieval_method == "dense" else 0.0,
            )
        )
    return results


def _rerank(
    query: str,
    results: list[RetrievalResult],
    reranker: RerankerBackend,
) -> list[RetrievalResult]:
    scores = reranker.score(query, [result.chunk.chunk_text for result in results])
    reranked = [
        RetrievalResult(
            chunk=result.chunk,
            score=round(float(score), 4),
            retrieval_method="reranked_hybrid",
            bm25_score=result.bm25_score,
            dense_score=result.dense_score,
            reranker_score=round(float(score), 4),
        )
        for result, score in zip(results, scores)
    ]
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def _optional_reranker() -> RerankerBackend | None:
    if not RAG_USE_RERANKER:
        return None
    try:
        return LocalCrossEncoderRerankerBackend()
    except Exception:
        return None


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _tokenize(text: str) -> list[str]:
    return [
        token.casefold()
        for token in re.findall(r"\d{2,7}-\d{2}-\d|[a-zA-Z]+(?:-[a-zA-Z0-9]+)*|\d+[a-zA-Z0-9-]*", text)
    ]


def _source_paths(source_docs_dir: Path) -> list[Path]:
    if not source_docs_dir.exists():
        return []
    return sorted(
        path
        for path in source_docs_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def _extract_pdf_page_texts(path: Path) -> list[tuple[int, str]]:
    with fitz.open(path) as document:
        return [
            (page_index + 1, page.get_text("text"))
            for page_index, page in enumerate(document)
            if page.get_text("text").strip()
        ]


def _extract_spreadsheet_texts(path: Path) -> list[tuple[int | None, str]]:
    """Convert workbook sheets into text chunks with sheet-level provenance.

    The page number field is reused as a stable one-based sheet index because
    RAG evidence chunks share one provenance schema across PDFs and workbooks.
    """
    try:
        import pandas as pd
    except Exception as exc:
        raise RuntimeError(
            "Spreadsheet RAG indexing requires pandas with an Excel engine such as openpyxl."
        ) from exc

    try:
        workbook = pd.read_excel(path, sheet_name=None, dtype=str)
    except Exception as exc:
        raise RuntimeError(
            f"Could not read spreadsheet source document {path}. Install openpyxl for .xlsx files."
        ) from exc

    sheet_texts: list[tuple[int | None, str]] = []
    for sheet_index, (sheet_name, frame) in enumerate(workbook.items(), start=1):
        frame = frame.fillna("")
        lines = [f"Spreadsheet sheet: {sheet_name}"]
        headers = [str(column).strip() for column in frame.columns]
        if headers:
            lines.append("Columns: " + " | ".join(headers))
        for row_number, row in frame.iterrows():
            cells = [
                f"{header}: {str(value).strip()}"
                for header, value in zip(headers, row.tolist())
                if str(value).strip()
            ]
            if cells:
                lines.append(f"Row {int(row_number) + 2}: " + " | ".join(cells))
        text = "\n".join(lines).strip()
        if text:
            sheet_texts.append((sheet_index, text))
    return sheet_texts


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[dict[str, Any]]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    if len(normalized) <= max_chars:
        return [{"text": normalized, "start_char": 0, "end_char": len(normalized)}]

    chunks = []
    start = 0
    step = max(max_chars - overlap_chars, 1)
    while start < len(normalized):
        end = min(start + max_chars, len(normalized))
        if end < len(normalized):
            boundary = normalized.rfind(". ", start, end)
            if boundary > start + max_chars // 2:
                end = boundary + 1
        chunks.append(
            {
                "text": normalized[start:end].strip(),
                "start_char": start,
                "end_char": end,
            }
        )
        if end >= len(normalized):
            break
        start = max(end - overlap_chars, start + step)
    return chunks


def _detect_section_heading(text: str) -> str:
    for line in text.split(". ")[:4]:
        candidate = line.strip(" :-")
        if 4 <= len(candidate) <= 80 and (
            candidate.isupper() or re.match(r"^\d+(\.\d+)*\s+[A-Z]", candidate)
        ):
            return candidate
    return ""


def _source_type(path: Path) -> str:
    text = str(path).casefold()
    if "regulatory" in text or "hsdb" in text or "osha" in text or "reach" in text:
        return "regulatory"
    if "method" in text or "methodology" in text:
        return "methodology"
    if "c5gc01008j1" in text or "d4gc01962h" in text:
        return "methodology"
    if "guide" in text or "gsk" in text or "chem21" in text or "ssg" in text:
        return "guide"
    return "literature"


def _document_id(path: Path) -> str:
    return hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]


def _topic_group(attribute: str) -> str:
    if "." in attribute:
        return attribute.split(".", 1)[0]
    return ""


def _best_snippet(text: str, query: str, max_chars: int = 420) -> str:
    query_terms = [term.casefold() for term in re.findall(r"[A-Za-z0-9][A-Za-z0-9_-]+", query)]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    best = max(
        sentences,
        key=lambda sentence: sum(term in sentence.casefold() for term in query_terms),
        default=text,
    )
    snippet = best if best else text
    snippet = " ".join(snippet.split())
    return snippet[: max_chars - 3] + "..." if len(snippet) > max_chars else snippet


def _chunk_provenance(chunk: EvidenceChunk) -> str:
    page = f"page={chunk.page_number}" if chunk.page_number is not None else "page=n/a"
    section = f"; section={chunk.section_heading}" if chunk.section_heading else ""
    return (
        f"{chunk.document_path}; document_id={chunk.document_id}; source_type={chunk.source_type}; "
        f"{page}; chars={chunk.start_char}-{chunk.end_char}; chunk={chunk.chunk_id}{section}"
    )
