"""Human-in-the-loop verification before scientific validation.

The system must not automatically trust AI or NER extraction. A human reviewer
can exclude false positives and add missed chemicals before PubChem/RDKit
validation receives any candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from services.chemistry_ner import ChemicalEntity
from services.pipeline_logger import PipelineLogger


@dataclass(frozen=True)
class HumanVerificationResult:
    included_entities: list[ChemicalEntity]
    excluded_entities: list[ChemicalEntity]
    manually_added_entities: list[ChemicalEntity]
    audit_trail: dict[str, Any]


def prepare_verification_rows(
    entities: list[ChemicalEntity],
    logger: PipelineLogger | None = None,
) -> list[dict[str, Any]]:
    """Create editable rows with default inclusion enabled."""
    stage = "Human verification preparation"
    if logger:
        logger.start_stage(stage, metadata={"entity_count": len(entities)})

    rows = []
    for entity in entities:
        rows.append(
            {
                "include_for_validation": True,
                "surface_form": entity.surface_form,
                "entity_type": entity.entity_type,
                "sentence_context": entity.sentence_context,
                "page_number": entity.page_number,
                "extraction_method": entity.extraction_method,
                "confidence": entity.confidence,
                "occurrence_count": entity.occurrence_count,
                "aliases": ", ".join(entity.aliases) if entity.aliases else entity.surface_form,
                "pages_found": ", ".join(str(page) for page in entity.pages_found),
                "representative_contexts": " | ".join(entity.representative_contexts),
            }
        )

    if logger:
        logger.end_stage(stage, metadata={"row_count": len(rows)})

    return rows


def apply_human_verification(
    extracted_entities: list[ChemicalEntity],
    verification_rows: list[dict[str, Any]],
    manual_input: str = "",
    timestamp: datetime | None = None,
) -> HumanVerificationResult:
    """Apply reviewer selections and manual additions."""
    included_entities = []
    excluded_entities = []

    for index, entity in enumerate(extracted_entities):
        include = True
        if index < len(verification_rows):
            include = bool(verification_rows[index].get("include_for_validation", True))

        if include:
            included_entities.append(entity)
        else:
            excluded_entities.append(entity)

    manually_added_entities = parse_manual_entities(manual_input)
    validation_entities = included_entities + manually_added_entities
    audit_trail = build_audit_trail(
        extracted_entities=extracted_entities,
        excluded_entities=excluded_entities,
        manually_added_entities=manually_added_entities,
        timestamp=timestamp,
    )

    return HumanVerificationResult(
        included_entities=validation_entities,
        excluded_entities=excluded_entities,
        manually_added_entities=manually_added_entities,
        audit_trail=audit_trail,
    )


def parse_manual_entities(manual_input: str) -> list[ChemicalEntity]:
    """Convert comma-separated reviewer additions into entity records."""
    entities = []
    seen = set()
    for raw_name in manual_input.split(","):
        surface_form = " ".join(raw_name.strip().split())
        if not surface_form:
            continue

        key = surface_form.lower()
        if key in seen:
            continue
        seen.add(key)

        entities.append(
            ChemicalEntity(
                surface_form=surface_form,
                entity_type="CHEMICAL_NAME",
                sentence_context="Manually added by human reviewer",
                start_char=None,
                end_char=None,
                page_number=None,
                extraction_method="human_added",
                confidence=1.0,
            )
        )

    return entities


def build_audit_trail(
    extracted_entities: list[ChemicalEntity],
    excluded_entities: list[ChemicalEntity],
    manually_added_entities: list[ChemicalEntity],
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Create an audit record for the verification decision."""
    timestamp = timestamp or datetime.now(timezone.utc)
    return {
        "timestamp": timestamp.isoformat(),
        "extracted_entities": [asdict(entity) for entity in extracted_entities],
        "excluded_entities": [asdict(entity) for entity in excluded_entities],
        "manually_added_entities": [asdict(entity) for entity in manually_added_entities],
    }
