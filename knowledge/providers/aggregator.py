"""Provider registry and fact aggregation for chemical knowledge sources."""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from core.config import HSDB_ENABLED
from knowledge.providers.base import KnowledgeFact, KnowledgeProvider
from knowledge.providers.chem21_provider import Chem21Provider
from knowledge.providers.gsk_provider import GskProvider
from knowledge.providers.hsdb_provider import HsdbProvider
from services.pipeline_logger import PipelineLogger


@dataclass
class KnowledgeProviderRegistry:
    providers: dict[str, KnowledgeProvider] = field(default_factory=dict)

    def register(self, provider: KnowledgeProvider, active: bool = True) -> None:
        if active:
            self.providers[provider.name] = provider

    def unregister(self, provider_name: str) -> None:
        self.providers.pop(provider_name, None)

    def active_providers(self) -> list[KnowledgeProvider]:
        return list(self.providers.values())


class KnowledgeAggregator:
    def __init__(self, registry: KnowledgeProviderRegistry | None = None) -> None:
        self.registry = registry or default_provider_registry()

    def collect_facts(
        self,
        chemical_identity,
        logger: PipelineLogger | None = None,
    ) -> list[KnowledgeFact]:
        facts = []
        for provider in self.registry.active_providers():
            stage = _provider_stage_name(provider.name)
            started = time.perf_counter()
            if logger:
                logger.start_stage(stage, metadata={"provider": provider.name})
            try:
                provider_facts = provider.get_facts(chemical_identity)
                facts.extend(provider_facts)
                if logger:
                    logger.end_stage(
                        stage,
                        metadata={
                            "provider": provider.name,
                            "fact_count": len(provider_facts),
                        },
                    )
            except Exception as exc:
                if logger:
                    logger.end_stage(
                        stage,
                        metadata={
                            "provider": provider.name,
                            "status": "error",
                            "error": str(exc),
                            "duration_seconds": round(time.perf_counter() - started, 3),
                        },
                    )
                raise
        return facts

    def get_facts(
        self,
        chemical_identity,
        logger: PipelineLogger | None = None,
    ) -> list[KnowledgeFact]:
        """Alias used by pipeline integration code."""
        return self.collect_facts(chemical_identity, logger=logger)


def default_provider_registry() -> KnowledgeProviderRegistry:
    registry = KnowledgeProviderRegistry()
    registry.register(Chem21Provider())
    registry.register(GskProvider())
    registry.register(HsdbProvider(), active=HSDB_ENABLED)
    return registry


def _provider_stage_name(provider_name: str) -> str:
    mapping = {
        "chem21": "CHEM21 provider",
        "gsk": "GSK provider",
        "hsdb": "HSDB provider",
    }
    return mapping.get(provider_name, f"{provider_name} provider")
