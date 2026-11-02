"""Compatibility exports for the structured provider aggregator."""

from knowledge.providers.aggregator import (
    KnowledgeAggregator,
    KnowledgeProviderRegistry,
    default_provider_registry,
)

__all__ = [
    "KnowledgeAggregator",
    "KnowledgeProviderRegistry",
    "default_provider_registry",
]
