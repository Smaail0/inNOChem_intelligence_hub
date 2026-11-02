"""Common contracts for structured chemical knowledge providers.

Providers are intentionally small adapters. A provider may read local files,
call APIs, query databases, or later wrap a graph store, but it must return the
same fact schema so downstream layers can stay provider-agnostic.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Protocol


CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")


@dataclass(frozen=True)
class KnowledgeFact:
    source: str
    attribute: str
    value: Any
    confidence: float
    evidence: str
    provenance: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class KnowledgeProvider(Protocol):
    name: str

    def get_facts(self, chemical_identity) -> list[KnowledgeFact]:
        """Return provider-specific facts for a chemical-like object."""


def normalize_identity(value: str | None) -> str:
    """Normalize names for matching without changing the source record."""
    if not value:
        return ""
    text = str(value).casefold()
    text = text.replace("γ", "gamma").replace("β", "beta").replace("α", "alpha")
    text = text.replace("–", "-").replace("—", "-").replace("_", " ")
    text = re.sub(r"[^\w\s,\-]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;-")
    return text


def normalize_cas(value: str | None) -> str:
    if not value:
        return ""
    match = CAS_PATTERN.search(str(value))
    return match.group(0) if match else ""


def chemical_name(chemical_identity) -> str:
    """Resolve a display name from dicts, dataclasses, or simple strings."""
    if isinstance(chemical_identity, str):
        return chemical_identity
    if isinstance(chemical_identity, dict):
        for key in (
            "canonical_name",
            "pubchem_canonical_name",
            "original_surface_form",
            "surface_form",
            "normalized_name",
            "name",
        ):
            value = chemical_identity.get(key)
            if value:
                return str(value)
    for attr in (
        "canonical_name",
        "pubchem_canonical_name",
        "original_surface_form",
        "surface_form",
        "normalized_name",
        "name",
    ):
        value = getattr(chemical_identity, attr, None)
        if value:
            return str(value)
    return "unknown"


def identity_terms(chemical_identity) -> set[str]:
    """Collect normalized names, aliases, CAS numbers, and PubChem names."""
    terms: set[str] = set()

    def add(value: Any) -> None:
        if value is None:
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                add(item)
            return
        text = str(value).strip()
        if not text:
            return
        normalized = normalize_identity(text)
        cas = normalize_cas(text)
        if normalized:
            terms.add(normalized)
        if cas:
            terms.add(cas)

    if isinstance(chemical_identity, str):
        add(chemical_identity)
        return terms

    if isinstance(chemical_identity, dict):
        for key in (
            "canonical_name",
            "pubchem_canonical_name",
            "original_surface_form",
            "surface_form",
            "normalized_name",
            "name",
            "cas",
            "aliases",
            "surface_forms",
        ):
            add(chemical_identity.get(key))
        return terms

    for attr in (
        "canonical_name",
        "pubchem_canonical_name",
        "original_surface_form",
        "surface_form",
        "normalized_name",
        "name",
        "cas",
        "aliases",
        "surface_forms",
    ):
        add(getattr(chemical_identity, attr, None))
    return terms
