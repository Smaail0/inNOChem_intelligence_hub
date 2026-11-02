"""Compatibility export for the HSDB annotation provider."""

from knowledge.providers.hsdb_provider import (
    HsdbProvider,
    HsdbTopicCatalog,
    PubChemHsdbAnnotationClient,
)

__all__ = ["HsdbProvider", "HsdbTopicCatalog", "PubChemHsdbAnnotationClient"]
