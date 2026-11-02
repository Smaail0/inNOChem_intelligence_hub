"""Application configuration for the first InNOChem vertical slice."""

import os
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

WARMUP_MODE = os.getenv("WARMUP_MODE", "lazy").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta",
)
GEMINI_EXPLANATION_TIMEOUT_SECONDS = float(
    os.getenv("GEMINI_EXPLANATION_TIMEOUT_SECONDS", "30")
)
LLM_CLEANUP_ENABLED = os.getenv("LLM_CLEANUP_ENABLED", "false").lower() == "true"
LLM_EXPLANATION_WARMUP_ON_START = (
    os.getenv("LLM_EXPLANATION_WARMUP_ON_START", "false").lower() == "true"
)
LLM_EXPLANATION_WARMUP_TIMEOUT_SECONDS = float(
    os.getenv("LLM_EXPLANATION_WARMUP_TIMEOUT_SECONDS", "10")
)
CHEM_NER_MODEL_NAME = os.getenv("CHEM_NER_MODEL_NAME", "teman67/matscibert-chem-ner")
CHEM_NER_MAX_CHARS_PER_CHUNK = int(os.getenv("CHEM_NER_MAX_CHARS_PER_CHUNK", "1200"))
CHEM_NER_MAX_TOKENS_PER_CHUNK = int(os.getenv("CHEM_NER_MAX_TOKENS_PER_CHUNK", "384"))
CHEM_NER_STRIDE_TOKENS = int(os.getenv("CHEM_NER_STRIDE_TOKENS", "64"))
CHEM_NER_MIN_CHARS_PER_CHUNK = int(os.getenv("CHEM_NER_MIN_CHARS_PER_CHUNK", "100"))
CHEM_NER_CACHE_DIR = os.getenv("CHEM_NER_CACHE_DIR", str(Path("models") / "huggingface"))
CHEM_NER_RESULT_CACHE_DIR = os.getenv(
    "CHEM_NER_RESULT_CACHE_DIR",
    str(Path("data") / "cache" / "ner_cache"),
)
CHEM_NER_DOWNLOAD_IF_MISSING = os.getenv("CHEM_NER_DOWNLOAD_IF_MISSING", "true").lower() == "true"
CHEM_NER_SKIP_REFERENCES = os.getenv("CHEM_NER_SKIP_REFERENCES", "true").lower() == "true"
PIPELINE_PROGRESS_INTERVAL = int(os.getenv("PIPELINE_PROGRESS_INTERVAL", "10"))
NER_MAX_CHUNKS_DEBUG = int(os.getenv("NER_MAX_CHUNKS_DEBUG", "5"))
PUBCHEM_CACHE_PATH = os.getenv(
    "PUBCHEM_CACHE_PATH",
    str(Path("data") / "cache" / "pubchem_cache.json"),
)
PUBCHEM_MAX_WORKERS = min(10, max(1, int(os.getenv("PUBCHEM_MAX_WORKERS", "6"))))
PUBCHEM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("PUBCHEM_REQUEST_TIMEOUT_SECONDS", "10"))
PUBCHEM_TIMEOUT_SECONDS = PUBCHEM_REQUEST_TIMEOUT_SECONDS
PUBCHEM_MAX_RETRIES = int(os.getenv("PUBCHEM_MAX_RETRIES", "2"))
PUBCHEM_MAX_NETWORK_LOOKUPS = int(os.getenv("PUBCHEM_MAX_NETWORK_LOOKUPS", "200"))
PUBCHEM_VALIDATION_TIMEOUT_SECONDS = float(
    os.getenv("PUBCHEM_VALIDATION_TIMEOUT_SECONDS", "180")
)
CHEM_NER_BATCH_SIZE = int(os.getenv("CHEM_NER_BATCH_SIZE", "8"))
HSDB_ENABLED = os.getenv("HSDB_ENABLED", "true").lower() == "true"
HSDB_CACHE_PATH = os.getenv(
    "HSDB_CACHE_PATH",
    str(Path("data") / "cache" / "hsdb_cache.json"),
)
HSDB_ENRICHMENT_DEFAULT = os.getenv("HSDB_ENRICHMENT_DEFAULT", "false").lower() == "true"
HSDB_TOPIC_CATALOG_PATH = os.getenv(
    "HSDB_TOPIC_CATALOG_PATH",
    str(Path("data") / "knowledge" / "hsdb_topic_catalog.json"),
)
HSDB_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("HSDB_REQUEST_TIMEOUT_SECONDS", str(PUBCHEM_REQUEST_TIMEOUT_SECONDS))
)
PUBCHEM_SYNONYM_CACHE_PATH = os.getenv(
    "PUBCHEM_SYNONYM_CACHE_PATH",
    str(Path("data") / "cache" / "pubchem_synonyms_cache.json"),
)
CHEMICAL_INTELLIGENCE_MAX_WORKERS = min(
    10,
    max(1, int(os.getenv("CHEMICAL_INTELLIGENCE_MAX_WORKERS", "6"))),
)
RAG_SOURCE_DOCS_DIR = os.getenv(
    "RAG_SOURCE_DOCS_DIR",
    str(Path("data") / "rag" / "source_docs"),
)
RAG_INDEX_DIR = os.getenv(
    "RAG_INDEX_DIR",
    str(Path("data") / "rag" / "index"),
)
RAG_ENABLED = os.getenv("RAG_ENABLED", "true").lower() == "true"
RAG_USE_DENSE = os.getenv("RAG_USE_DENSE", "true").lower() == "true"
RAG_UI_USE_DENSE_DEFAULT = os.getenv("RAG_UI_USE_DENSE_DEFAULT", "false").lower() == "true"
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
RAG_RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
RAG_USE_RERANKER = os.getenv("RAG_USE_RERANKER", "false").lower() == "true"
RAG_MAX_CHARS_PER_CHUNK = int(os.getenv("RAG_MAX_CHARS_PER_CHUNK", "1200"))
RAG_CHUNK_OVERLAP_CHARS = int(os.getenv("RAG_CHUNK_OVERLAP_CHARS", "150"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "3"))
RAG_MAX_EVIDENCE_QUERIES = int(os.getenv("RAG_MAX_EVIDENCE_QUERIES", "25"))
PROVIDER_REGISTRY_PATH = os.getenv(
    "PROVIDER_REGISTRY_PATH",
    str(Path("config") / "provider_registry.yaml"),
)

# Keep the first slice bounded so local inference remains responsive.
MAX_EXTRACTION_CHARS = int(os.getenv("MAX_EXTRACTION_CHARS", "12000"))
