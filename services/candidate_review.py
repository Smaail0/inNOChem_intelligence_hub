"""Local candidate review before PubChem validation.

This stage is intentionally network-free. It collapses repeated NER mentions,
ranks candidates with transparent local signals, and prepares a human review
table. PubChem/RDKit are only invoked after the reviewer selects candidates.
"""

from __future__ import annotations

import math
import string
from dataclasses import asdict, dataclass
from typing import Any

from services.chemistry_ner import ChemicalEntity
from services.normalizer import normalize_entities
from services.pipeline_logger import PipelineLogger


DEFAULT_PRESELECT_LIMIT = 100
LOW_INFORMATION_TERMS = {
    "acid",
    "base",
    "compound",
    "chemical",
    "chemicals",
    "mixture",
    "product",
    "reaction",
    "reagent",
    "sample",
    "solution",
    "solvent",
    "substrate",
}
ELEMENT_TERMS = {
    "hydrogen",
    "oxygen",
    "carbon",
    "sulfur",
    "sulphur",
    "lithium",
    "sodium",
    "potassium",
    "nitrogen",
    "chlorine",
    "bromine",
    "iodine",
    "fluorine",
}
CHEMICAL_CLASS_TERMS = {
    "ether",
    "ethers",
    "amide",
    "amides",
    "alcohol",
    "alcohols",
    "tertiary alcohol",
    "tertiary alcohols",
    "ketone",
    "ketones",
    "carbonate",
    "carbonates",
    "aryl halide",
    "aryl halides",
    "aryl bromide",
    "aryl bromides",
}
REACTION_FRAGMENT_TERMS = {
    "c-h",
    "ch",
    "snar",
}


@dataclass(frozen=True)
class CandidateReviewRecord:
    normalized_name: str
    aliases: tuple[str, ...]
    occurrence_count: int
    pages_found: tuple[int, ...]
    representative_contexts: tuple[str, ...]
    extraction_methods: tuple[str, ...]
    average_confidence: float
    candidate_priority_score: float
    priority_reason: str
    candidate_type: str
    pubchem_validation_recommended: bool
    pubchem_recommendation_reason: str
    include_for_pubchem_validation: bool
    representative_entity: ChemicalEntity

    def to_row(self) -> dict[str, Any]:
        return {
            "include_for_pubchem_validation": self.include_for_pubchem_validation,
            "normalized_name": self.normalized_name,
            "aliases/surface_forms": ", ".join(self.aliases),
            "occurrence_count": self.occurrence_count,
            "pages_found": ", ".join(str(page) for page in self.pages_found),
            "representative_contexts": " | ".join(self.representative_contexts),
            "extraction_methods": ", ".join(self.extraction_methods),
            "average_confidence": round(self.average_confidence, 3),
            "candidate_priority_score": round(self.candidate_priority_score, 3),
            "priority_reason": self.priority_reason,
            "candidate_type": self.candidate_type,
            "pubchem_validation_recommended": self.pubchem_validation_recommended,
            "pubchem_recommendation_reason": self.pubchem_recommendation_reason,
        }

    def to_entity(self) -> ChemicalEntity:
        entity = self.representative_entity
        return ChemicalEntity(
            surface_form=entity.surface_form,
            entity_type=entity.entity_type,
            sentence_context=self.representative_contexts[0] if self.representative_contexts else entity.sentence_context,
            start_char=entity.start_char,
            end_char=entity.end_char,
            page_number=entity.page_number,
            extraction_method="human_selected_candidate_review",
            confidence=self.average_confidence,
            occurrence_count=self.occurrence_count,
            aliases=self.aliases,
            pages_found=self.pages_found,
            representative_contexts=self.representative_contexts,
        )


def build_candidate_review_records(
    entities: list[ChemicalEntity],
    preselect_limit: int = DEFAULT_PRESELECT_LIMIT,
    logger: PipelineLogger | None = None,
) -> list[CandidateReviewRecord]:
    """Create locally deduplicated, ranked candidate records from raw NER output."""
    stage = "Human verification preparation"
    if logger:
        logger.start_stage(stage, metadata={"raw_entity_count": len(entities)})

    groups: dict[str, list[ChemicalEntity]] = {}
    for entity in entities:
        normalized_name = _normalize_candidate_name(entity.surface_form)
        if not normalized_name:
            continue

        # Keep abbreviation expansion from the normalizer, then normalize for grouping.
        lookup_name = normalize_entities([entity])[0].normalized_name
        key = _normalize_candidate_name(lookup_name)
        if not key:
            continue
        groups.setdefault(key, []).append(entity)

    ranked = [
        _record_from_group(key, group)
        for key, group in groups.items()
    ]
    ranked.sort(
        key=lambda record: (
            record.candidate_priority_score,
            record.occurrence_count,
            record.average_confidence,
            record.normalized_name,
        ),
        reverse=True,
    )
    ranked = [
        _with_preselection(
            record,
            include=index < preselect_limit and record.pubchem_validation_recommended,
        )
        for index, record in enumerate(ranked)
    ]

    if logger:
        for index, record in enumerate(ranked[:20], start=1):
            logger.message(
                stage,
                f"[Candidate Priority] {index}. {record.normalized_name}: "
                f"{record.candidate_priority_score:.3f} - {record.priority_reason}",
                metadata={
                    "rank": index,
                    "normalized_name": record.normalized_name,
                    "candidate_priority_score": round(record.candidate_priority_score, 3),
                    "priority_reason": record.priority_reason,
                },
            )
        logger.end_stage(
            stage,
            metadata={
                "raw_entity_count": len(entities),
                "candidate_count": len(ranked),
                "preselected_count": min(len(ranked), preselect_limit),
            },
        )

    return ranked


def candidate_records_to_rows(records: list[CandidateReviewRecord]) -> list[dict[str, Any]]:
    return [record.to_row() for record in records]


def apply_candidate_review_selection(
    records: list[CandidateReviewRecord],
    edited_rows: list[dict[str, Any]],
) -> list[ChemicalEntity]:
    """Convert checked review rows into PubChem validation entities."""
    selected_names = {
        str(row.get("normalized_name", "")).strip()
        for row in edited_rows
        if bool(row.get("include_for_pubchem_validation", False))
    }
    return [
        record.to_entity()
        for record in records
        if record.normalized_name in selected_names
    ]


def filter_candidate_review_records(
    records: list[CandidateReviewRecord],
    minimum_confidence: float = 0.0,
    minimum_occurrence_count: int = 1,
    hide_low_information_terms: bool = True,
    search_text: str = "",
    extraction_method: str = "All",
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[CandidateReviewRecord]:
    """Apply local UI filters without triggering PubChem."""
    search = search_text.strip().lower()
    filtered = []
    for record in records:
        if record.average_confidence < minimum_confidence:
            continue
        if record.occurrence_count < minimum_occurrence_count:
            continue
        if hide_low_information_terms and _is_low_information(record.normalized_name):
            continue
        if search and search not in record.normalized_name.lower() and not any(
            search in alias.lower() for alias in record.aliases
        ):
            continue
        if extraction_method != "All" and extraction_method not in record.extraction_methods:
            continue
        if page_start is not None or page_end is not None:
            if not _record_in_page_range(record, page_start, page_end):
                continue
        filtered.append(record)

    return filtered


def available_extraction_methods(records: list[CandidateReviewRecord]) -> list[str]:
    methods = sorted({method for record in records for method in record.extraction_methods})
    return ["All", *methods]


def _record_from_group(normalized_name: str, entities: list[ChemicalEntity]) -> CandidateReviewRecord:
    aliases = tuple(_unique_ordered(entity.surface_form for entity in entities))
    pages_found = tuple(
        sorted({entity.page_number for entity in entities if entity.page_number is not None})
    )
    representative_contexts = tuple(
        _unique_ordered(entity.sentence_context for entity in entities if entity.sentence_context)[:3]
    )
    extraction_methods = tuple(
        sorted({entity.extraction_method for entity in entities if entity.extraction_method})
    )
    confidences = [entity.confidence for entity in entities if entity.confidence is not None]
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    priority_score, priority_reason = _candidate_priority_score(
        occurrence_count=len(entities),
        average_confidence=average_confidence,
        contexts=tuple(entity.sentence_context for entity in entities if entity.sentence_context),
        pages_found=pages_found,
    )
    candidate_type = classify_candidate_type(normalized_name, aliases, extraction_methods)
    recommended, recommendation_reason = _pubchem_recommendation(candidate_type)

    return CandidateReviewRecord(
        normalized_name=normalized_name,
        aliases=aliases,
        occurrence_count=len(entities),
        pages_found=pages_found,
        representative_contexts=representative_contexts,
        extraction_methods=extraction_methods,
        average_confidence=average_confidence,
        candidate_priority_score=priority_score,
        priority_reason=priority_reason,
        candidate_type=candidate_type,
        pubchem_validation_recommended=recommended,
        pubchem_recommendation_reason=recommendation_reason,
        include_for_pubchem_validation=False,
        representative_entity=entities[0],
    )


def _with_preselection(record: CandidateReviewRecord, include: bool) -> CandidateReviewRecord:
    return CandidateReviewRecord(
        normalized_name=record.normalized_name,
        aliases=record.aliases,
        occurrence_count=record.occurrence_count,
        pages_found=record.pages_found,
        representative_contexts=record.representative_contexts,
        extraction_methods=record.extraction_methods,
        average_confidence=record.average_confidence,
        candidate_priority_score=record.candidate_priority_score,
        priority_reason=record.priority_reason,
        candidate_type=record.candidate_type,
        pubchem_validation_recommended=record.pubchem_validation_recommended,
        pubchem_recommendation_reason=record.pubchem_recommendation_reason,
        include_for_pubchem_validation=include,
        representative_entity=record.representative_entity,
    )


def classify_candidate_type(
    normalized_name: str,
    aliases: tuple[str, ...] = (),
    extraction_methods: tuple[str, ...] = (),
) -> str:
    name = _normalize_candidate_name(normalized_name)
    alias_names = {_normalize_candidate_name(alias) for alias in aliases}
    all_names = {name, *alias_names}

    if any(value in ELEMENT_TERMS for value in all_names):
        return "element"
    if any(value in CHEMICAL_CLASS_TERMS for value in all_names):
        return "chemical_class"
    if any(value in REACTION_FRAGMENT_TERMS for value in all_names):
        return "reaction_fragment"
    if name in LOW_INFORMATION_TERMS:
        return "generic_term"
    if any(method == "fallback_abbreviation_lexicon" for method in extraction_methods):
        return "abbreviation"
    if _looks_like_abbreviation(normalized_name, aliases):
        return "abbreviation"
    if _looks_like_specific_compound(name):
        return "specific_compound"
    return "unknown"


def _pubchem_recommendation(candidate_type: str) -> tuple[bool, str]:
    if candidate_type == "specific_compound":
        return True, "Recommended: specific compound"
    if candidate_type == "abbreviation":
        return True, "Recommended: abbreviation"
    label = candidate_type.replace("_", " ")
    return False, f"Not recommended for PubChem: {label}"


def _normalize_candidate_name(value: str) -> str:
    greek_map = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "π": "pi",
        "ω": "omega",
        "Α": "alpha",
        "Β": "beta",
        "Γ": "gamma",
        "Δ": "delta",
        "Ε": "epsilon",
        "Κ": "kappa",
        "Λ": "lambda",
        "Μ": "mu",
        "Π": "pi",
        "Ω": "omega",
    }
    normalized = value.strip()
    for symbol, replacement in greek_map.items():
        normalized = normalized.replace(symbol, replacement)
    boundary_punctuation = string.punctuation.replace("-", "").replace(",", "")
    normalized = normalized.strip(boundary_punctuation)
    return " ".join(normalized.lower().split())


def _looks_like_abbreviation(normalized_name: str, aliases: tuple[str, ...]) -> bool:
    values = [normalized_name, *aliases]
    for value in values:
        compact = value.strip().replace("-", "").replace(",", "")
        if (
            2 <= len(compact) <= 8
            and any(character.isalpha() for character in compact)
            and compact.upper() == compact
        ):
            return True
    return False


def _looks_like_specific_compound(name: str) -> bool:
    if not name or name in LOW_INFORMATION_TERMS:
        return False
    if any(character.isdigit() for character in name):
        return True
    if any(marker in name for marker in ("-", ",", "(", ")", "[", "]")):
        return True
    return len(name.split()) <= 4


def _candidate_priority_score(
    occurrence_count: int,
    average_confidence: float,
    contexts: tuple[str, ...],
    pages_found: tuple[int, ...],
) -> tuple[float, str]:
    lower_contexts = tuple(context.lower() for context in contexts)
    unique_page_count = len(set(pages_found))
    occurrence_boost = min(0.45, math.log1p(occurrence_count) / 6)
    confidence_component = average_confidence
    title_boost = 0.3 if _has_context_signal(lower_contexts, ("title",)) else 0.0
    abstract_boost = 0.25 if _has_context_signal(lower_contexts, ("abstract",)) else 0.0
    heading_boost = 0.2 if _has_section_heading_signal(contexts) else 0.0
    introduction_boost = 0.15 if _has_context_signal(lower_contexts, ("introduction",)) else 0.0
    page_boost = min(0.25, unique_page_count * 0.03)
    score = (
        confidence_component
        + occurrence_boost
        + title_boost
        + abstract_boost
        + heading_boost
        + introduction_boost
        + page_boost
    )
    return score, _priority_reason(
        occurrence_count=occurrence_count,
        average_confidence=average_confidence,
        unique_page_count=unique_page_count,
        title_boost=title_boost,
        abstract_boost=abstract_boost,
        heading_boost=heading_boost,
        introduction_boost=introduction_boost,
    )


def _priority_reason(
    occurrence_count: int,
    average_confidence: float,
    unique_page_count: int,
    title_boost: float,
    abstract_boost: float,
    heading_boost: float,
    introduction_boost: float,
) -> str:
    signals = []
    if heading_boost:
        signals.append("found in heading")
    if title_boost:
        signals.append("found in title")
    if abstract_boost:
        signals.append("found in abstract")
    if introduction_boost:
        signals.append("found in introduction")
    signals.append(f"appears {occurrence_count} time{'s' if occurrence_count != 1 else ''}")
    if unique_page_count:
        signals.append(f"across {unique_page_count} page{'s' if unique_page_count != 1 else ''}")
    signals.append(f"average confidence {average_confidence:.2f}")
    return "Preselected because " + " and ".join(signals)


def _has_context_signal(contexts: tuple[str, ...], terms: tuple[str, ...]) -> bool:
    return any(any(term in context for term in terms) for context in contexts)


def _has_section_heading_signal(contexts: tuple[str, ...]) -> bool:
    heading_terms = (
        "screening",
        "materials",
        "methods",
        "results",
        "discussion",
        "experimental",
        "synthesis",
        "solvent",
    )
    for context in contexts:
        stripped = context.strip()
        if not stripped:
            continue
        words = stripped.split()
        looks_like_heading = len(words) <= 12 and stripped[:1].isupper()
        if looks_like_heading and any(term in stripped.lower() for term in heading_terms):
            return True
    return False


def _is_low_information(normalized_name: str) -> bool:
    return normalized_name in LOW_INFORMATION_TERMS or len(normalized_name) <= 1


def _record_in_page_range(
    record: CandidateReviewRecord,
    page_start: int | None,
    page_end: int | None,
) -> bool:
    if not record.pages_found:
        return page_start is None and page_end is None
    lower = page_start if page_start is not None else min(record.pages_found)
    upper = page_end if page_end is not None else max(record.pages_found)
    return any(lower <= page <= upper for page in record.pages_found)


def _unique_ordered(values) -> list:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique
