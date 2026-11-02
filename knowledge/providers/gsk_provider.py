"""Data-backed GSK solvent guide provider.

The GSK PDF source contains image-embedded tables in this project. The provider
therefore only trusts rows that were manually reviewed and marked approved in
the extraction review CSV.
"""

from __future__ import annotations

import csv
from pathlib import Path

from knowledge.providers.base import KnowledgeFact, identity_terms, normalize_cas, normalize_identity


DEFAULT_GSK_REVIEW_DATASET = Path("data/knowledge/gsk_extraction_review.csv")


class GskProvider:
    name = "gsk"

    def __init__(self, dataset_path: str | Path = DEFAULT_GSK_REVIEW_DATASET) -> None:
        self.dataset_path = Path(dataset_path)
        self._rows = self._load_approved_rows()
        self._index = self._build_index(self._rows)

    def get_facts(self, chemical_identity) -> list[KnowledgeFact]:
        row = self._match_row(chemical_identity)
        if not row:
            return []

        evidence = (
            f"GSK solvent guide reviewed row for {row.get('solvent_name', 'unknown')} "
            f"from page {row.get('source_page', '')}, figure {row.get('source_figure', '')}."
        )
        provenance = (
            f"{self.dataset_path}::page={row.get('source_page', '')};"
            f"figure={row.get('source_figure', '')};review_status=approved"
        )
        attributes = [
            "cas",
            "melting_point_c",
            "boiling_point_c",
            "waste_score",
            "environmental_impact_score",
            "health_score",
            "flammability_explosion_score",
            "reactivity_stability_score",
            "life_cycle_score",
            "legislation_flag",
            "ehs_red_flag",
        ]
        facts: list[KnowledgeFact] = []
        confidence = _safe_float(row.get("extraction_confidence"), default=1.0)
        for attribute in attributes:
            value = row.get(attribute, "")
            if value == "":
                continue
            facts.append(
                KnowledgeFact(
                    source=self.name,
                    attribute=attribute,
                    value=value,
                    confidence=confidence,
                    evidence=evidence,
                    provenance=provenance,
                )
            )
        return facts

    def _load_approved_rows(self) -> list[dict[str, str]]:
        if not self.dataset_path.exists():
            return []
        with self.dataset_path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [
                row
                for row in csv.DictReader(handle)
                if row.get("review_status", "").strip().casefold() == "approved"
            ]

    def _build_index(self, rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
        index: dict[str, dict[str, str]] = {}
        for row in rows:
            keys = {
                normalize_identity(row.get("solvent_name")),
                normalize_cas(row.get("cas")),
            }
            for key in sorted(k for k in keys if k):
                index.setdefault(key, row)
        return index

    def _match_row(self, chemical_identity) -> dict[str, str] | None:
        for term in identity_terms(chemical_identity):
            normalized = normalize_cas(term) or normalize_identity(term)
            if normalized in self._index:
                return self._index[normalized]
        return None


def _safe_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value not in (None, "") else default
    except ValueError:
        return default
