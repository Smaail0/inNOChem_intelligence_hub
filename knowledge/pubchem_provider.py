"""Knowledge provider adapter for PubChem-derived facts.

This first implementation returns mock/local facts. The adapter boundary is
ready for a future PubChem API-backed implementation without changing the
aggregator contract.
"""

from __future__ import annotations

from knowledge.provider_base import KnowledgeFact, chemical_name


class PubChemProvider:
    name = "pubchem"

    def get_facts(self, chemical) -> list[KnowledgeFact]:
        name = chemical_name(chemical)
        return [
            KnowledgeFact(
                source=self.name,
                attribute="identity_lookup_available",
                value=bool(name and name != "unknown"),
                confidence=0.5,
                evidence=f"Mock PubChem provider can receive candidate '{name}'.",
            )
        ]

