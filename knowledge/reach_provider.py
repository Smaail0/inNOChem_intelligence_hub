"""Knowledge provider adapter for future REACH/regulatory data."""

from __future__ import annotations

from knowledge.provider_base import KnowledgeFact, chemical_name


class ReachProvider:
    name = "reach"

    def get_facts(self, chemical) -> list[KnowledgeFact]:
        name = chemical_name(chemical)
        return [
            KnowledgeFact(
                source=self.name,
                attribute="regulatory_record_status",
                value="not_loaded",
                confidence=0.0,
                evidence=f"Mock REACH provider has no regulatory record loaded for '{name}'.",
            )
        ]

