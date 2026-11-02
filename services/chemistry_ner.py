"""Chemistry-specific NER service with pluggable extraction backends.

The LLM is intentionally not the scientific decision-maker in this pipeline.
NER produces candidate entities, while PubChem validation and RDKit descriptors
will provide the scientific validation layer in later pipeline stages.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from core.config import (
    CHEM_NER_BATCH_SIZE,
    CHEM_NER_CACHE_DIR,
    CHEM_NER_DOWNLOAD_IF_MISSING,
    CHEM_NER_MAX_CHARS_PER_CHUNK,
    CHEM_NER_MAX_TOKENS_PER_CHUNK,
    CHEM_NER_MIN_CHARS_PER_CHUNK,
    CHEM_NER_MODEL_NAME,
    CHEM_NER_RESULT_CACHE_DIR,
    CHEM_NER_SKIP_REFERENCES,
    CHEM_NER_STRIDE_TOKENS,
    NER_MAX_CHUNKS_DEBUG,
    PIPELINE_PROGRESS_INTERVAL,
)
from services.document_extractor import ExtractedDocument, ExtractedTextPage, TextSegment
from services.pipeline_logger import PipelineLogger


@dataclass(frozen=True)
class ChemicalEntity:
    surface_form: str
    entity_type: str
    sentence_context: str
    start_char: int | None
    end_char: int | None
    page_number: int | None
    extraction_method: str
    chunk_id: int | None = None
    confidence: float | None = None
    occurrence_count: int = 1
    aliases: tuple[str, ...] = ()
    pages_found: tuple[int, ...] = ()
    representative_contexts: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class NerTextChunk:
    chunk_id: int
    text: str
    page_number: int
    start_char: int
    end_char: int
    token_count: int | None = None


class ChemistryNerBackend(Protocol):
    name: str

    def is_available(self) -> bool:
        """Return whether the backend can run in the current environment."""

    def extract(
        self,
        document: ExtractedDocument,
        logger: PipelineLogger | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        max_chunks_debug: int | None = None,
        skip_references: bool = CHEM_NER_SKIP_REFERENCES,
    ) -> list[ChemicalEntity]:
        """Extract chemistry entities from a structured document."""


class TransformerChemicalNERBackend:
    """Primary transformer-based chemical NER backend.

    Uses a local Hugging Face token-classification model. The full document is
    never sent to an LLM; structured text is processed in page-local,
    token-aware chunks that respect transformer context limits.
    """

    name = "transformer_chemical_ner"

    def __init__(
        self,
        model_name: str = CHEM_NER_MODEL_NAME,
        max_chars_per_chunk: int = CHEM_NER_MAX_CHARS_PER_CHUNK,
        max_tokens_per_chunk: int = CHEM_NER_MAX_TOKENS_PER_CHUNK,
        stride_tokens: int = CHEM_NER_STRIDE_TOKENS,
        min_chars_per_chunk: int = CHEM_NER_MIN_CHARS_PER_CHUNK,
        batch_size: int = CHEM_NER_BATCH_SIZE,
        cache_dir: str = CHEM_NER_CACHE_DIR,
        result_cache_dir: str = CHEM_NER_RESULT_CACHE_DIR,
        enable_result_cache: bool = True,
        download_if_missing: bool = CHEM_NER_DOWNLOAD_IF_MISSING,
        ner_pipeline: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_chars_per_chunk = max_chars_per_chunk
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.stride_tokens = stride_tokens
        self.min_chars_per_chunk = min_chars_per_chunk
        self.batch_size = max(1, batch_size)
        self.cache_dir = cache_dir
        self.result_cache_dir = result_cache_dir
        self.enable_result_cache = enable_result_cache
        self.download_if_missing = download_if_missing
        self._pipeline = ner_pipeline
        self._tokenizer = tokenizer
        self._load_error: Exception | None = None

    def is_available(self) -> bool:
        if self._pipeline is not None:
            return True

        try:
            self._load_pipeline(local_files_only=True)
            return True
        except Exception as local_exc:
            self._load_error = local_exc

        if not self.download_if_missing:
            return False

        try:
            self._load_pipeline(local_files_only=False)
        except Exception as download_exc:
            self._load_error = download_exc
            return False

        return True

    def _load_pipeline(self, local_files_only: bool) -> None:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            local_files_only=local_files_only,
        )
        self._pipeline = pipeline(
            "token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
        self._tokenizer = tokenizer
        _print_ner(
            f"[NER] Loaded pipeline for {self.model_name}; device={_pipeline_device(self._pipeline)}"
        )

    def extract(
        self,
        document: ExtractedDocument,
        logger: PipelineLogger | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        max_chunks_debug: int | None = None,
        skip_references: bool = CHEM_NER_SKIP_REFERENCES,
    ) -> list[ChemicalEntity]:
        stage = "Transformer NER"
        _print_ner(f"[NER] Backend selected: {self.name}")
        _print_ner(f"[NER] Model name: {self.model_name}")
        working_document = _filter_references_from_document(document) if skip_references else document
        _print_ner(f"[NER] Pages: {len(working_document.pages)}")
        _print_ner(f"[NER] Skip references: {skip_references}")
        if logger:
            logger.start_stage(
                stage,
                metadata={
                    "model_name": self.model_name,
                    "sentence_count": len(working_document.sentences),
                    "skip_references": skip_references,
                },
            )
            logger.message(
                stage,
                f"[NER] Backend selected: {self.name}",
                metadata={"backend": self.name, "model_name": self.model_name},
            )
        if progress_callback:
            progress_callback(f"[NER] Loading backend {self.name}", 0, 1)

        try:
            cache_key = _build_ner_cache_key(
                document=working_document,
                model_name=self.model_name,
                max_tokens_per_chunk=self.max_tokens_per_chunk,
                stride_tokens=self.stride_tokens,
                skip_references=skip_references,
                max_chunks_debug=max_chunks_debug,
            )
            cached_entities = (
                _load_ner_cache(cache_key, self.result_cache_dir)
                if self.enable_result_cache
                else None
            )
            if cached_entities is not None:
                _print_ner("[NER] Loaded entities from cache")
                if logger:
                    logger.message(
                        stage,
                        "[NER] Loaded entities from cache",
                        metadata={"cache_key": cache_key, "entity_count": len(cached_entities)},
                    )
                    logger.end_stage(
                        stage,
                        metadata={"status": "cache_hit", "entity_count": len(cached_entities)},
                    )
                if progress_callback:
                    progress_callback("[NER] Loaded entities from cache", 1, 1)
                return cached_entities

            if not self.is_available() or self._pipeline is None:
                if logger:
                    logger.message(stage, "Transformer NER model unavailable; falling back")
                    logger.end_stage(stage, metadata={"status": "unavailable"})
                return []

            if self._tokenizer is not None:
                chunks = _iter_token_aware_page_chunks(
                    pages=working_document.pages,
                    tokenizer=self._tokenizer,
                    max_tokens_per_chunk=self.max_tokens_per_chunk,
                    stride_tokens=self.stride_tokens,
                    min_chars_per_chunk=self.min_chars_per_chunk,
                )
                chunking_strategy = "token_aware_page_chunks"
            else:
                chunks = _iter_safe_chunks(document.sentences, self.max_chars_per_chunk)
                chunking_strategy = "legacy_sentence_char_chunks"
            original_chunk_count = len(chunks)
            if max_chunks_debug is not None:
                chunks = chunks[:max_chunks_debug]
                _print_ner(
                    f"[NER] Debug chunk limit enabled: {len(chunks)}/{original_chunk_count} chunks"
                )
            if logger:
                logger.message(
                    stage,
                    f"{len(chunks)} chunks created",
                    metadata={
                        "chunk_count": len(chunks),
                        "original_chunk_count": original_chunk_count,
                        "debug_chunk_limit": max_chunks_debug,
                        "chunking_strategy": chunking_strategy,
                    },
                )

            entities = []
            total_chunks = len(chunks)
            first_chunk_length = len(chunks[0].text) if chunks else 0
            max_chunk_length = max((len(chunk.text) for chunk in chunks), default=0)
            total_characters = sum(len(page.text) for page in working_document.pages)
            average_chunk_length = (
                sum(len(chunk.text) for chunk in chunks) / total_chunks if total_chunks else 0
            )
            token_counts = [
                chunk.token_count
                for chunk in chunks
                if getattr(chunk, "token_count", None) is not None
            ]
            average_tokens_per_chunk = sum(token_counts) / len(token_counts) if token_counts else None
            device = _pipeline_device(self._pipeline)
            _print_ner(f"[NER] Number of text chunks: {total_chunks}")
            _print_ner(f"[NER] Total pages: {len(document.pages)}")
            _print_ner(f"[NER] Total characters: {total_characters}")
            _print_ner(f"[NER] Average chunk length: {average_chunk_length:.1f} chars")
            if average_tokens_per_chunk is not None:
                _print_ner(f"[NER] Average tokens per chunk: {average_tokens_per_chunk:.1f}")
            _print_ner(f"[NER] First chunk character length: {first_chunk_length}")
            _print_ner(f"[NER] Max chunk length: {max_chunk_length}")
            _print_ner(f"[NER] Device used: {device}")
            if total_chunks > 300:
                _print_ner(
                    "[NER] Chunk count is high; consider increasing chunk size or page batching.",
                    level="WARNING",
                )
            if logger:
                logger.message(
                    stage,
                    "[NER] Transformer execution details",
                    metadata={
                        "page_count": len(working_document.pages),
                        "total_characters": total_characters,
                        "chunk_count": total_chunks,
                        "average_chunk_length": round(average_chunk_length, 1),
                        "average_tokens_per_chunk": (
                            round(average_tokens_per_chunk, 1)
                            if average_tokens_per_chunk is not None
                            else None
                        ),
                        "chunking_strategy": chunking_strategy,
                        "first_chunk_char_length": first_chunk_length,
                        "max_chunk_length": max_chunk_length,
                        "device": device,
                        "batch_size": self.batch_size,
                    },
                )
            ner_start = time.perf_counter()
            batch_durations = []
            if self._tokenizer is not None:
                entities.extend(
                    self._extract_chunk_batches(
                        chunks=chunks,
                        total_chunks=total_chunks,
                        stage=stage,
                        logger=logger,
                        progress_callback=progress_callback,
                        batch_durations=batch_durations,
                    )
                )
            else:
                entities.extend(
                    self._extract_chunks_individually(
                        chunks=chunks,
                        total_chunks=total_chunks,
                        stage=stage,
                        logger=logger,
                        progress_callback=progress_callback,
                    )
                )
            ner_duration = time.perf_counter() - ner_start

            deduplicated = _deduplicate_entities(entities)
            if self.enable_result_cache:
                _save_ner_cache(cache_key, deduplicated, self.result_cache_dir)
        except Exception:
            if logger:
                logger.end_stage(stage, metadata={"status": "error"})
            raise

        if logger:
            logger.end_stage(
                stage,
                metadata={
                    "status": "success",
                    "chunk_count": total_chunks,
                    "entity_count": len(deduplicated),
                    "duration_seconds": round(ner_duration, 3),
                    "average_seconds_per_chunk": (
                        round(ner_duration / total_chunks, 4) if total_chunks else 0
                    ),
                    "average_seconds_per_batch": (
                        round(sum(batch_durations) / len(batch_durations), 4)
                        if batch_durations
                        else None
                    ),
                },
            )

        return deduplicated

    def _extract_chunk_batches(
        self,
        chunks: list[TextSegment | NerTextChunk],
        total_chunks: int,
        stage: str,
        logger: PipelineLogger | None,
        progress_callback: Callable[[str, int, int], None] | None,
        batch_durations: list[float],
    ) -> list[ChemicalEntity]:
        entities = []
        completed = 0
        total_batches = _batch_count(total_chunks, self.batch_size)
        for batch_index, (batch_start, batch_chunks) in enumerate(
            _iter_batches(chunks, self.batch_size),
            start=1,
        ):
            batch_end = batch_start + len(batch_chunks) - 1
            batch_message = (
                f"[NER] Processing batch {batch_index}/{total_batches}, "
                f"chunks {batch_start}-{batch_end}"
            )
            _print_ner(batch_message)
            if logger:
                logger.progress(
                    stage,
                    batch_message,
                    metadata={
                        "batch_index": batch_index,
                        "total_batches": total_batches,
                        "chunk_start": batch_start,
                        "chunk_end": batch_end,
                    },
                )
            if progress_callback:
                progress_callback(batch_message, completed, total_chunks)

            batch_texts = [chunk.text for chunk in batch_chunks]
            batch_start_time = time.perf_counter()
            _print_ner(
                f"[NER] Before transformer pipeline batch call for chunks "
                f"{batch_start}-{batch_start + len(batch_chunks) - 1}/{total_chunks}"
            )
            raw_outputs = self._pipeline(batch_texts, batch_size=self.batch_size)
            batch_duration = time.perf_counter() - batch_start_time
            batch_durations.append(batch_duration)
            outputs_by_chunk = _normalize_pipeline_batch_outputs(raw_outputs, len(batch_chunks))
            if batch_duration > 60:
                _print_ner(
                    f"[NER] Batch starting at chunk {batch_start} is slow; consider reducing chunk size.",
                    level="WARNING",
                )

            for offset, (chunk, outputs) in enumerate(zip(batch_chunks, outputs_by_chunk)):
                index = batch_start + offset
                chunk_entities = self._entities_from_pipeline_outputs(outputs, chunk)
                entities.extend(chunk_entities)
                completed += 1
                finish_message = (
                    f"[NER] Finished chunk {index}/{total_chunks} in {batch_duration:.2f}s batch time, "
                    f"found {len(chunk_entities)} entities"
                )
                _log_ner_chunk_finish(
                    finish_message,
                    stage,
                    index,
                    total_chunks,
                    batch_duration,
                    len(chunk_entities),
                    logger,
                )
                if progress_callback:
                    progress_callback(finish_message, completed, total_chunks)

        return entities

    def _extract_chunks_individually(
        self,
        chunks: list[TextSegment | NerTextChunk],
        total_chunks: int,
        stage: str,
        logger: PipelineLogger | None,
        progress_callback: Callable[[str, int, int], None] | None,
    ) -> list[ChemicalEntity]:
        entities = []
        for index, chunk in enumerate(chunks, start=1):
            start_message = f"[NER] Starting chunk {index}/{total_chunks}"
            _log_ner_chunk_start(start_message, stage, chunk, index, total_chunks, logger)
            if progress_callback:
                progress_callback(start_message, index - 1, total_chunks)

            chunk_start = time.perf_counter()
            _print_ner(f"[NER] Before transformer pipeline call for chunk {index}/{total_chunks}")
            outputs = self._pipeline(chunk.text)
            chunk_duration = time.perf_counter() - chunk_start
            chunk_entities = self._entities_from_pipeline_outputs(outputs, chunk)
            entities.extend(chunk_entities)
            finish_message = (
                f"[NER] Finished chunk {index}/{total_chunks} in {chunk_duration:.2f}s, "
                f"found {len(chunk_entities)} entities"
            )
            if chunk_duration > 60:
                _print_ner(
                    f"[NER] Chunk {index} is slow; consider reducing chunk size.",
                    level="WARNING",
                )
            _log_ner_chunk_finish(
                finish_message,
                stage,
                index,
                total_chunks,
                chunk_duration,
                len(chunk_entities),
                logger,
            )
            if progress_callback:
                progress_callback(finish_message, index, total_chunks)

        return entities

    def _entities_from_pipeline_outputs(
        self,
        outputs: list[dict[str, Any]],
        chunk: TextSegment,
    ) -> list[ChemicalEntity]:
        if not outputs:
            return []

        if all("start" in output and "end" in output and "word" in output for output in outputs):
            return self._entities_from_aggregated_outputs(outputs, chunk)

        return self._entities_from_token_outputs(outputs, chunk)

    def _entities_from_aggregated_outputs(
        self,
        outputs: list[dict[str, Any]],
        chunk: TextSegment,
    ) -> list[ChemicalEntity]:
        entities = []
        for output in _merge_adjacent_outputs(outputs):
            start = output.get("start")
            end = output.get("end")
            if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                continue

            surface = chunk.text[start:end].strip()
            if not surface:
                surface = _clean_token_word(str(output.get("word", "")))
            if not _looks_like_entity_surface(surface):
                continue

            confidence = output.get("score")
            entities.append(
                ChemicalEntity(
                    surface_form=surface,
                    entity_type=str(output.get("entity_group") or output.get("entity") or "CHEMICAL"),
                    sentence_context=chunk.text,
                    start_char=chunk.start_char + start,
                    end_char=chunk.start_char + end,
                    page_number=chunk.page_number,
                    extraction_method=self.name,
                    chunk_id=getattr(chunk, "chunk_id", None),
                    confidence=float(confidence) if confidence is not None else None,
                )
            )
        return entities

    def _entities_from_token_outputs(
        self,
        outputs: list[dict[str, Any]],
        chunk: TextSegment,
    ) -> list[ChemicalEntity]:
        entities = []
        active_tokens = []

        for output in outputs:
            label = str(output.get("entity", output.get("entity_group", "O")))
            is_entity = label != "O"
            starts_new = label.startswith("B-") or (is_entity and not active_tokens)

            if active_tokens and (not is_entity or starts_new):
                entity = _entity_from_tokens(active_tokens, chunk, self.name)
                if entity is not None:
                    entities.append(entity)
                active_tokens = []

            if is_entity:
                active_tokens.append(output)

        if active_tokens:
            entity = _entity_from_tokens(active_tokens, chunk, self.name)
            if entity is not None:
                entities.append(entity)

        return entities


class ChemDataExtractorBackend:
    """Optional ChemDataExtractor backend.

    ChemDataExtractor is preferred when installed because it is chemistry-aware.
    The import stays lazy so the app still runs when the dependency is not
    compatible with the local Python environment.
    """

    name = "chemdataextractor"

    def is_available(self) -> bool:
        try:
            import chemdataextractor  # noqa: F401
        except ImportError:
            return False
        return True

    def extract(
        self,
        document: ExtractedDocument,
        logger: PipelineLogger | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        max_chunks_debug: int | None = None,
        skip_references: bool = CHEM_NER_SKIP_REFERENCES,
    ) -> list[ChemicalEntity]:
        try:
            from chemdataextractor import Document
        except ImportError:
            return []

        entities = []
        for sentence in document.sentences:
            try:
                cde_document = Document(sentence.text)
                mentions = getattr(cde_document, "cems", []) or []
            except Exception:
                continue

            for mention in mentions:
                surface = str(getattr(mention, "text", mention)).strip()
                if not surface:
                    continue

                local_start = getattr(mention, "start", None)
                local_end = getattr(mention, "end", None)
                start_char = sentence.start_char + local_start if isinstance(local_start, int) else None
                end_char = sentence.start_char + local_end if isinstance(local_end, int) else None
                entities.append(
                    ChemicalEntity(
                        surface_form=surface,
                        entity_type="CHEMICAL_NAME",
                        sentence_context=sentence.text,
                        start_char=start_char,
                        end_char=end_char,
                        page_number=sentence.page_number,
                        extraction_method=self.name,
                        confidence=None,
                    )
                )

        return _deduplicate_entities(entities)


class LexiconChemistryBackend:
    """Deterministic non-regex fallback for common chemical names.

    This is not meant to be the final scientific extractor. It keeps the local
    pipeline useful when ChemDataExtractor is unavailable, while PubChem/RDKit
    validation remains the future authority.
    """

    name = "chemistry_lexicon"

    _chemical_terms = {
        "acetone",
        "acetic acid",
        "ammonia",
        "benzene",
        "chloroform",
        "dichloromethane",
        "dimethylformamide",
        "dimethyl sulfoxide",
        "ethanol",
        "ethyl acetate",
        "hexane",
        "hydrochloric acid",
        "methanol",
        "sodium chloride",
        "sodium hydroxide",
        "sulfuric acid",
        "tetrahydrofuran",
        "toluene",
        "water",
    }

    _known_abbreviations = {
        "DCM": "dichloromethane",
        "DMF": "dimethylformamide",
        "DMSO": "dimethyl sulfoxide",
        "EtOH": "ethanol",
        "HCl": "hydrochloric acid",
        "MeOH": "methanol",
        "NaCl": "sodium chloride",
        "NaOH": "sodium hydroxide",
        "THF": "tetrahydrofuran",
    }

    def is_available(self) -> bool:
        return True

    def extract(
        self,
        document: ExtractedDocument,
        logger: PipelineLogger | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        max_chunks_debug: int | None = None,
        skip_references: bool = CHEM_NER_SKIP_REFERENCES,
    ) -> list[ChemicalEntity]:
        entities = []
        for sentence in document.sentences:
            for term in sorted(self._chemical_terms, key=len, reverse=True):
                entities.extend(
                    self._extract_phrase(
                        sentence=sentence,
                        phrase=term,
                        entity_type="CHEMICAL_NAME",
                        method=self.name,
                        confidence=0.65,
                        case_sensitive=False,
                    )
                )

            for abbreviation in self._known_abbreviations:
                entities.extend(
                    self._extract_phrase(
                        sentence=sentence,
                        phrase=abbreviation,
                        entity_type="ABBREVIATION",
                        method="fallback_abbreviation_lexicon",
                        confidence=0.7,
                        case_sensitive=True,
                    )
                )

        return _deduplicate_entities(entities)

    def _extract_phrase(
        self,
        sentence: TextSegment,
        phrase: str,
        entity_type: str,
        method: str,
        confidence: float,
        case_sensitive: bool,
    ) -> list[ChemicalEntity]:
        haystack = sentence.text if case_sensitive else sentence.text.lower()
        needle = phrase if case_sensitive else phrase.lower()
        entities = []
        cursor = 0

        while True:
            index = haystack.find(needle, cursor)
            if index == -1:
                break

            end = index + len(needle)
            if _has_term_boundaries(sentence.text, index, end):
                entities.append(
                    ChemicalEntity(
                        surface_form=sentence.text[index:end],
                        entity_type=entity_type,
                        sentence_context=sentence.text,
                        start_char=sentence.start_char + index,
                        end_char=sentence.start_char + end,
                        page_number=sentence.page_number,
                        extraction_method=method,
                        confidence=confidence,
                    )
                )

            cursor = end

        return entities


class CasNumberFallbackBackend:
    """Regex-only fallback for CAS Registry Numbers."""

    name = "fallback_cas_regex"
    _cas_pattern = re.compile(r"\b\d{2,7}-\d{2}-\d\b")

    def is_available(self) -> bool:
        return True

    def extract(
        self,
        document: ExtractedDocument,
        logger: PipelineLogger | None = None,
        progress_callback: Callable[[str, int, int], None] | None = None,
        max_chunks_debug: int | None = None,
        skip_references: bool = CHEM_NER_SKIP_REFERENCES,
    ) -> list[ChemicalEntity]:
        entities = []
        for sentence in document.sentences:
            for match in self._cas_pattern.finditer(sentence.text):
                entities.append(
                    ChemicalEntity(
                        surface_form=match.group(0),
                        entity_type="CAS_NUMBER",
                        sentence_context=sentence.text,
                        start_char=sentence.start_char + match.start(),
                        end_char=sentence.start_char + match.end(),
                        page_number=sentence.page_number,
                        extraction_method=self.name,
                        confidence=0.9,
                    )
                )
        return entities


def extract_chemical_entities(
    document: ExtractedDocument,
    preferred_backend: ChemistryNerBackend | None = None,
    logger: PipelineLogger | None = None,
    progress_callback: Callable[[str, int, int], None] | None = None,
    limit_chunks_for_debugging: bool = False,
    skip_references: bool = CHEM_NER_SKIP_REFERENCES,
) -> list[ChemicalEntity]:
    """Run chemistry NER without sending full documents to an LLM."""
    backend = preferred_backend or TransformerChemicalNERBackend()
    _print_ner(f"[NER] extract_chemical_entities called with backend={backend.name}")
    if logger:
        logger.message(
            "Transformer NER",
            f"[NER] extract_chemical_entities selected backend={backend.name}",
            metadata={"backend": backend.name},
        )
    entities = []
    max_chunks_debug = NER_MAX_CHUNKS_DEBUG if limit_chunks_for_debugging else None

    if isinstance(backend, TransformerChemicalNERBackend):
        entities = backend.extract(
            document,
            logger=logger,
            progress_callback=progress_callback,
            max_chunks_debug=max_chunks_debug,
            skip_references=skip_references,
        )
    elif backend.is_available():
        entities = backend.extract(
            document,
            logger=logger,
            progress_callback=progress_callback,
            max_chunks_debug=max_chunks_debug,
            skip_references=skip_references,
        )

    if not entities:
        chemdata_backend = ChemDataExtractorBackend()
        if chemdata_backend.is_available():
            entities = chemdata_backend.extract(document, logger=logger)

    if not entities:
        entities = LexiconChemistryBackend().extract(document, logger=logger)

    entities.extend(CasNumberFallbackBackend().extract(document, logger=logger))
    return _deduplicate_entities(entities)


def _iter_safe_chunks(
    sentences: list[TextSegment],
    max_chars_per_chunk: int,
) -> list[TextSegment]:
    chunks = []
    for sentence in sentences:
        if len(sentence.text) <= max_chars_per_chunk:
            chunks.append(sentence)
            continue

        chunk_start = 0
        while chunk_start < len(sentence.text):
            chunk_end = min(chunk_start + max_chars_per_chunk, len(sentence.text))
            if chunk_end < len(sentence.text):
                boundary = sentence.text.rfind(" ", chunk_start, chunk_end)
                if boundary > chunk_start:
                    chunk_end = boundary

            chunk_text = sentence.text[chunk_start:chunk_end].strip()
            if chunk_text:
                leading_space = len(sentence.text[chunk_start:chunk_end]) - len(
                    sentence.text[chunk_start:chunk_end].lstrip()
                )
                absolute_start = sentence.start_char + chunk_start + leading_space
                chunks.append(
                    TextSegment(
                        text=chunk_text,
                        page_number=sentence.page_number,
                        start_char=absolute_start,
                        end_char=absolute_start + len(chunk_text),
                    )
                )
            chunk_start = chunk_end + 1

    return chunks


def _build_ner_cache_key(
    document: ExtractedDocument,
    model_name: str,
    max_tokens_per_chunk: int,
    stride_tokens: int,
    skip_references: bool,
    max_chunks_debug: int | None,
) -> str:
    payload = {
        "document_text_sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        "model_name": model_name,
        "max_tokens_per_chunk": max_tokens_per_chunk,
        "stride_tokens": stride_tokens,
        "skip_references": skip_references,
        "max_chunks_debug": max_chunks_debug,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_ner_cache(cache_key: str, result_cache_dir: str = CHEM_NER_RESULT_CACHE_DIR) -> list[ChemicalEntity] | None:
    cache_path = _ner_cache_path(cache_key, result_cache_dir)
    if not cache_path.exists():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            payload = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        return None

    entities = payload.get("entities")
    if not isinstance(entities, list):
        return None

    return [_chemical_entity_from_cache(entity) for entity in entities]


def _save_ner_cache(
    cache_key: str,
    entities: list[ChemicalEntity],
    result_cache_dir: str = CHEM_NER_RESULT_CACHE_DIR,
) -> None:
    cache_path = _ner_cache_path(cache_key, result_cache_dir)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"entities": [entity.to_dict() for entity in entities]}
    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(payload, cache_file, indent=2, sort_keys=True)


def _ner_cache_path(cache_key: str, result_cache_dir: str = CHEM_NER_RESULT_CACHE_DIR) -> Path:
    return Path(result_cache_dir) / f"{cache_key}.json"


def _chemical_entity_from_cache(payload: dict[str, Any]) -> ChemicalEntity:
    return ChemicalEntity(
        surface_form=payload["surface_form"],
        entity_type=payload["entity_type"],
        sentence_context=payload["sentence_context"],
        start_char=payload.get("start_char"),
        end_char=payload.get("end_char"),
        page_number=payload.get("page_number"),
        extraction_method=payload["extraction_method"],
        chunk_id=payload.get("chunk_id"),
        confidence=payload.get("confidence"),
        occurrence_count=payload.get("occurrence_count", 1),
        aliases=tuple(payload.get("aliases", ())),
        pages_found=tuple(payload.get("pages_found", ())),
        representative_contexts=tuple(payload.get("representative_contexts", ())),
    )


def _filter_references_from_document(document: ExtractedDocument) -> ExtractedDocument:
    filtered_pages = []
    for page in document.pages:
        reference_index = _reference_section_start(page.text)
        if reference_index is None:
            filtered_pages.append(page)
            continue

        retained_text = page.text[:reference_index].rstrip()
        if retained_text:
            filtered_pages.append(
                ExtractedTextPage(
                    page_number=page.page_number,
                    text=retained_text,
                    start_char=page.start_char,
                    end_char=page.start_char + len(retained_text),
                )
            )
        break

    filtered_text = "\n\n".join(page.text for page in filtered_pages)
    return ExtractedDocument(
        filename=document.filename,
        file_type=document.file_type,
        text=filtered_text,
        pages=filtered_pages,
        sentences=_split_segments_from_pages(filtered_pages),
    )


def _reference_section_start(text: str) -> int | None:
    cursor = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip().lower().strip(":")
        if stripped in {"references", "bibliography", "reference"}:
            return cursor
        cursor += len(line)
    return None


def _split_segments_from_pages(pages: list[ExtractedTextPage]) -> list[TextSegment]:
    segments = []
    terminators = ".!?"
    for page in pages:
        segment_start = 0
        for index, character in enumerate(page.text):
            if character in terminators or index == len(page.text) - 1:
                end = index + 1
                segment_text = page.text[segment_start:end].strip()
                if segment_text:
                    leading_space = len(page.text[segment_start:end]) - len(
                        page.text[segment_start:end].lstrip()
                    )
                    start_char = page.start_char + segment_start + leading_space
                    segments.append(
                        TextSegment(
                            text=segment_text,
                            page_number=page.page_number,
                            start_char=start_char,
                            end_char=start_char + len(segment_text),
                        )
                    )
                segment_start = end
    return segments


def _iter_batches(
    chunks: list[TextSegment | NerTextChunk],
    batch_size: int,
) -> list[tuple[int, list[TextSegment | NerTextChunk]]]:
    batches = []
    for start in range(0, len(chunks), max(1, batch_size)):
        batches.append((start + 1, chunks[start : start + batch_size]))
    return batches


def _batch_count(total_items: int, batch_size: int) -> int:
    return (total_items + max(1, batch_size) - 1) // max(1, batch_size)


def _normalize_pipeline_batch_outputs(outputs: Any, batch_count: int) -> list[list[dict[str, Any]]]:
    if batch_count == 1 and (not outputs or isinstance(outputs[0], dict)):
        return [outputs]
    return list(outputs)


def _log_ner_chunk_start(
    message: str,
    stage: str,
    chunk: TextSegment | NerTextChunk,
    index: int,
    total_chunks: int,
    logger: PipelineLogger | None,
) -> None:
    _print_ner(message)
    if logger:
        logger.progress(
            stage,
            message,
            metadata={
                "current_chunk": index,
                "total_chunks": total_chunks,
                "chunk_char_length": len(chunk.text),
                "chunk_token_count": getattr(chunk, "token_count", None),
            },
        )


def _log_ner_chunk_finish(
    message: str,
    stage: str,
    index: int,
    total_chunks: int,
    duration_seconds: float,
    entity_count: int,
    logger: PipelineLogger | None,
) -> None:
    _print_ner(message)
    if logger:
        logger.progress(
            stage,
            message,
            metadata={
                "current_chunk": index,
                "total_chunks": total_chunks,
                "duration_seconds": round(duration_seconds, 3),
                "entity_count": entity_count,
            },
        )


def _iter_token_aware_page_chunks(
    pages: list[ExtractedTextPage],
    tokenizer: Any,
    max_tokens_per_chunk: int,
    stride_tokens: int,
    min_chars_per_chunk: int,
) -> list[NerTextChunk]:
    """Create page-local chunks from tokenizer offset mappings.

    The transformer pipeline re-tokenizes each chunk before inference, so this
    builder keeps source text windows below the tokenizer/model token budget
    while preserving page numbers and document-level character offsets.
    """
    effective_max_tokens = _effective_max_tokens(tokenizer, max_tokens_per_chunk)
    effective_stride = min(max(0, stride_tokens), max(0, effective_max_tokens - 1))
    chunks: list[NerTextChunk] = []

    for page in pages:
        page_chunks = _token_chunks_for_page(
            page=page,
            tokenizer=tokenizer,
            max_tokens=effective_max_tokens,
            stride_tokens=effective_stride,
            min_chars_per_chunk=min_chars_per_chunk,
        )
        chunks.extend(page_chunks)

    return _assign_chunk_ids(chunks)


def _token_chunks_for_page(
    page: ExtractedTextPage,
    tokenizer: Any,
    max_tokens: int,
    stride_tokens: int,
    min_chars_per_chunk: int,
) -> list[NerTextChunk]:
    encoding = tokenizer(
        page.text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    input_ids = _flatten_tokenizer_field(encoding.get("input_ids", []))
    offset_mapping = _flatten_offset_mapping(encoding.get("offset_mapping", []))
    if not input_ids or not offset_mapping:
        return _page_to_char_chunks(page, max_tokens=max_tokens, min_chars_per_chunk=min_chars_per_chunk)

    usable_offsets = [
        (index, int(start), int(end))
        for index, (start, end) in enumerate(offset_mapping)
        if isinstance(start, int) and isinstance(end, int) and end > start
    ]
    if not usable_offsets:
        return _page_to_char_chunks(page, max_tokens=max_tokens, min_chars_per_chunk=min_chars_per_chunk)

    step = max(1, max_tokens - stride_tokens)
    page_chunks: list[NerTextChunk] = []
    token_start = 0
    total_tokens = len(usable_offsets)

    while token_start < total_tokens:
        token_end = min(token_start + max_tokens, total_tokens)
        window_offsets = usable_offsets[token_start:token_end]
        char_start = window_offsets[0][1]
        char_end = window_offsets[-1][2]
        chunk = _build_chunk_from_page_span(
            page=page,
            char_start=char_start,
            char_end=char_end,
            token_count=token_end - token_start,
        )
        if chunk is not None:
            page_chunks.append(chunk)

        if token_end >= total_tokens:
            break
        token_start += step

    return _merge_short_tail_chunk(page_chunks, max_tokens, min_chars_per_chunk)


def _page_to_char_chunks(
    page: ExtractedTextPage,
    max_tokens: int,
    min_chars_per_chunk: int,
) -> list[NerTextChunk]:
    # Rare fallback for tokenizer implementations that cannot return offsets.
    max_chars = max(min_chars_per_chunk, max_tokens * 4)
    pseudo_segment = TextSegment(
        text=page.text,
        page_number=page.page_number,
        start_char=page.start_char,
        end_char=page.end_char,
    )
    return [
        NerTextChunk(
            chunk_id=0,
            text=chunk.text,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            token_count=None,
        )
        for chunk in _iter_safe_chunks([pseudo_segment], max_chars)
    ]


def _build_chunk_from_page_span(
    page: ExtractedTextPage,
    char_start: int,
    char_end: int,
    token_count: int,
) -> NerTextChunk | None:
    raw_text = page.text[char_start:char_end]
    chunk_text = raw_text.strip()
    if not chunk_text:
        return None

    leading_trim = len(raw_text) - len(raw_text.lstrip())
    absolute_start = page.start_char + char_start + leading_trim
    return NerTextChunk(
        chunk_id=0,
        text=chunk_text,
        page_number=page.page_number,
        start_char=absolute_start,
        end_char=absolute_start + len(chunk_text),
        token_count=token_count,
    )


def _merge_short_tail_chunk(
    chunks: list[NerTextChunk],
    max_tokens: int,
    min_chars_per_chunk: int,
) -> list[NerTextChunk]:
    if len(chunks) < 2 or len(chunks[-1].text) >= min_chars_per_chunk:
        return chunks

    previous = chunks[-2]
    tail = chunks[-1]
    previous_tokens = previous.token_count or 0
    tail_tokens = tail.token_count or 0
    if previous_tokens + tail_tokens > max_tokens:
        return chunks

    merged_text = f"{previous.text} {tail.text}".strip()
    merged = NerTextChunk(
        chunk_id=previous.chunk_id,
        text=merged_text,
        page_number=previous.page_number,
        start_char=previous.start_char,
        end_char=tail.end_char,
        token_count=previous_tokens + tail_tokens,
    )
    return [*chunks[:-2], merged]


def _assign_chunk_ids(chunks: list[NerTextChunk]) -> list[NerTextChunk]:
    return [
        NerTextChunk(
            chunk_id=index,
            text=chunk.text,
            page_number=chunk.page_number,
            start_char=chunk.start_char,
            end_char=chunk.end_char,
            token_count=chunk.token_count,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _effective_max_tokens(tokenizer: Any, requested_max_tokens: int) -> int:
    model_max_length = _tokenizer_model_max_length(tokenizer)
    requested = max(1, requested_max_tokens)
    if model_max_length is None:
        return requested
    # Leave room for tokenizer-added special tokens inside the HF pipeline.
    usable_model_tokens = max(1, model_max_length - 2)
    return min(requested, usable_model_tokens)


def _tokenizer_model_max_length(tokenizer: Any) -> int | None:
    value = getattr(tokenizer, "model_max_length", None)
    if not isinstance(value, int) or value <= 0:
        return None
    if value > 100_000:
        return None
    return value


def _flatten_tokenizer_field(value: Any) -> list[Any]:
    if value and isinstance(value, list) and isinstance(value[0], list):
        return value[0]
    return list(value)


def _flatten_offset_mapping(value: Any) -> list[tuple[int, int]]:
    if value and isinstance(value, list) and isinstance(value[0], list):
        return value[0]
    return list(value)


def _has_term_boundaries(text: str, start: int, end: int) -> bool:
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return not _is_token_char(before) and not _is_token_char(after)


def _is_token_char(value: str) -> bool:
    return value.isalnum() or value in {"_", "-"}


def _should_log_progress(index: int, total: int) -> bool:
    return index == 1 or index == total or index % PIPELINE_PROGRESS_INTERVAL == 0


def _pipeline_device(ner_pipeline: Any) -> str:
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


def _print_ner(message: str, level: str = "INFO") -> None:
    print(f"[{level}] {message}", flush=True)


def _entity_from_tokens(
    tokens: list[dict[str, Any]],
    chunk: TextSegment,
    method: str,
) -> ChemicalEntity | None:
    starts = [token.get("start") for token in tokens if isinstance(token.get("start"), int)]
    ends = [token.get("end") for token in tokens if isinstance(token.get("end"), int)]
    if starts and ends:
        start = min(starts)
        end = max(ends)
        surface = chunk.text[start:end].strip()
    else:
        start = None
        end = None
        surface = _clean_token_word(" ".join(str(token.get("word", "")) for token in tokens))

    if not surface:
        return None

    scores = [float(token["score"]) for token in tokens if token.get("score") is not None]
    labels = [
        str(token.get("entity", token.get("entity_group", "CHEMICAL"))).replace("B-", "").replace("I-", "")
        for token in tokens
    ]
    entity_type = labels[0] if labels else "CHEMICAL"

    return ChemicalEntity(
        surface_form=surface,
        entity_type=entity_type,
        sentence_context=chunk.text,
        start_char=chunk.start_char + start if start is not None else None,
        end_char=chunk.start_char + end if end is not None else None,
        page_number=chunk.page_number,
        extraction_method=method,
        chunk_id=getattr(chunk, "chunk_id", None),
        confidence=sum(scores) / len(scores) if scores else None,
    )


def _clean_token_word(value: str) -> str:
    return (
        value.replace(" ##", "")
        .replace("##", "")
        .replace("\u0120", "")
        .replace("\u2581", "")
        .strip()
    )


def _merge_adjacent_outputs(outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_outputs = sorted(
        outputs,
        key=lambda output: output.get("start") if isinstance(output.get("start"), int) else -1,
    )
    merged = []

    for output in sorted_outputs:
        start = output.get("start")
        end = output.get("end")
        label = output.get("entity_group") or output.get("entity")
        if not isinstance(start, int) or not isinstance(end, int) or label is None:
            merged.append(output)
            continue

        if merged:
            previous = merged[-1]
            previous_label = previous.get("entity_group") or previous.get("entity")
            previous_end = previous.get("end")
            if previous_label == label and isinstance(previous_end, int) and start <= previous_end:
                previous["end"] = max(previous_end, end)
                previous["word"] = f"{previous.get('word', '')}{output.get('word', '')}"
                previous_scores = previous.setdefault("_scores", [previous.get("score")])
                previous_scores.append(output.get("score"))
                numeric_scores = [float(score) for score in previous_scores if score is not None]
                if numeric_scores:
                    previous["score"] = sum(numeric_scores) / len(numeric_scores)
                continue

        merged.append(dict(output))

    return merged


def _looks_like_entity_surface(surface: str) -> bool:
    return any(character.isalnum() for character in surface)


def _deduplicate_entities(entities: list[ChemicalEntity]) -> list[ChemicalEntity]:
    seen = set()
    unique_entities = []
    for entity in entities:
        key = (
            entity.surface_form.lower(),
            entity.entity_type,
            entity.start_char,
            entity.end_char,
            entity.page_number,
        )
        if key in seen:
            continue
        seen.add(key)
        unique_entities.append(entity)
    return sorted(
        unique_entities,
        key=lambda entity: (
            entity.page_number if entity.page_number is not None else 0,
            entity.start_char if entity.start_char is not None else 0,
            entity.surface_form.lower(),
        ),
    )
