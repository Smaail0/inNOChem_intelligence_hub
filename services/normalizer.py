"""Entity normalization skeleton for PubChem/RDKit validation.

This layer prepares NER output for future scientific validation. It does not
score, recommend, or claim chemical validity yet.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import string

from services.chemistry_ner import ChemicalEntity


@dataclass(frozen=True)
class NormalizedChemicalCandidate:
    surface_form: str
    normalized_name: str
    entity_type: str
    source_page: int | None
    source_span: tuple[int | None, int | None]
    validation_status: str
    validation_method: str

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_entities(entities: list[ChemicalEntity]) -> list[NormalizedChemicalCandidate]:
    """Prepare extracted entities for later PubChem validation."""
    normalized = []
    for entity in entities:
        normalized.append(
            NormalizedChemicalCandidate(
                surface_form=entity.surface_form,
                normalized_name=_normalize_surface_form(entity.surface_form),
                entity_type=entity.entity_type,
                source_page=entity.page_number,
                source_span=(entity.start_char, entity.end_char),
                validation_status="pending_pubchem_validation",
                validation_method="not_implemented",
            )
        )
    return normalized


def _normalize_surface_form(surface_form: str) -> str:
    cleaned = _normalize_text_for_lookup(surface_form)
    abbreviation_map = {
        "DCM": "dichloromethane",
        "DMF": "N,N-dimethylformamide",
        "DMSO": "dimethyl sulfoxide",
        "EtOH": "ethanol",
        "MeOH": "methanol",
        "THF": "tetrahydrofuran",
    }
    return abbreviation_map.get(cleaned, cleaned)


def _normalize_text_for_lookup(value: str) -> str:
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

    # Strip punctuation at boundaries while preserving chemical punctuation
    # inside names, such as N,N-dimethylformamide and 1,4-dioxane.
    boundary_punctuation = string.punctuation.replace("-", "").replace(",", "")
    normalized = normalized.strip(boundary_punctuation)
    return " ".join(normalized.split())
