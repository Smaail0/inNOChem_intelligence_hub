"""Compatibility exports for the structured provider interface."""

from knowledge.providers.base import (
    KnowledgeFact,
    KnowledgeProvider,
    chemical_name,
    identity_terms,
    normalize_cas,
    normalize_identity,
)

__all__ = [
    "KnowledgeFact",
    "KnowledgeProvider",
    "chemical_name",
    "identity_terms",
    "normalize_cas",
    "normalize_identity",
]
