"""Data-backed CHEM21 solvent guide provider."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from knowledge.providers.base import (
    KnowledgeFact,
    identity_terms,
    normalize_cas,
    normalize_identity,
)


DEFAULT_CHEM21_DATASET = Path("data/knowledge/chem21_solvents.csv")

# Identity aliases only. These do not encode any guide ranking or decision.
CHEMICAL_IDENTITY_ALIASES = {
    "dmf": {"n,n-dimethylformamide", "dimethylformamide"},
    "nmp": {
        "n-methyl-2-pyrrolidone",
        "1-methyl-2-pyrrolidone",
        "1-methylpyrrolidin-2-one",
    },
    "2-methf": {"2-methyl-thf", "2-methyltetrahydrofuran"},
    "mtbe": {"tbme", "tert-butyl methyl ether"},
    "tbme": {"mtbe", "tert-butyl methyl ether"},
    "gvl": {"gamma-valerolactone", "γ-valerolactone"},
}


class Chem21Provider:
    name = "chem21"

    def __init__(self, dataset_path: str | Path = DEFAULT_CHEM21_DATASET) -> None:
        self.dataset_path = Path(dataset_path)
        self._rows = self._load_rows()
        self._index = self._build_index(self._rows)

    def get_facts(self, chemical_identity) -> list[KnowledgeFact]:
        row = self._match_row(chemical_identity)
        if not row:
            return []

        provenance = row.get("provenance", "")
        evidence = (
            f"CHEM21 solvent guide record for {row.get('solvent_name', 'unknown')} "
            f"from {row.get('source_reference', 'source dataset')}."
        )

        attributes = [
            "cas",
            "category",
            "ranking_by_default",
            "final_ranking",
            "safety_score",
            "health_score",
            "environment_score",
            "family",
            "boiling_point_c",
            "flash_point_c",
            "worst_h3xx",
            "h4xx",
            "notes",
        ]
        facts: list[KnowledgeFact] = []
        for attribute in attributes:
            value = row.get("family", "") if attribute == "category" else row.get(attribute, "")
            if value == "":
                continue
            facts.append(
                KnowledgeFact(
                    source=self.name,
                    attribute=attribute,
                    value=value,
                    confidence=1.0,
                    evidence=evidence,
                    provenance=provenance,
                )
            )
        return facts

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.dataset_path.exists():
            return []
        with self.dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def _build_index(self, rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        for row in rows:
            keys = {
                normalize_identity(row.get("solvent_name")),
                normalize_identity(row.get("normalized_name")),
                normalize_cas(row.get("cas")),
            }
            aliases = row.get("aliases", "")
            for alias in aliases.replace(";", "|").split("|"):
                keys.add(normalize_identity(alias))
            for key in sorted(k for k in keys if k):
                index.setdefault(key, row)
        return index

    def _match_row(self, chemical_identity: Any) -> dict[str, str] | None:
        terms = set(identity_terms(chemical_identity))
        for term in list(terms):
            terms.update(CHEMICAL_IDENTITY_ALIASES.get(term, set()))
            normalized_alias = normalize_identity(term)
            terms.update(CHEMICAL_IDENTITY_ALIASES.get(normalized_alias, set()))

        for term in terms:
            normalized = normalize_cas(term) or normalize_identity(term)
            if normalized in self._index:
                return self._index[normalized]
        return None
