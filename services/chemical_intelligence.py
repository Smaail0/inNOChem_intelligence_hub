"""Factual knowledge enrichment for validated chemicals.

This layer attaches provider facts after PubChem/RDKit validation and before
descriptor scoring. It does not use LLMs, does not rank alternatives, and does
not make recommendations.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import threading
from typing import Any

from knowledge.providers.aggregator import KnowledgeAggregator
from knowledge.providers.base import KnowledgeFact
import requests

from core.config import (
    CHEMICAL_INTELLIGENCE_MAX_WORKERS,
    PUBCHEM_REQUEST_TIMEOUT_SECONDS,
    PUBCHEM_SYNONYM_CACHE_PATH,
)
from services.pipeline_logger import PipelineLogger
from services.validator import ValidatedChemical


CAS_PATTERN = re.compile(r"\b\d{2,7}-\d{2}-\d\b")
CHEMICAL_INTELLIGENCE_VERSION = "pubchem_synonyms_v1"


@dataclass(frozen=True)
class ChemicalIntelligenceRecord:
    chemical: dict[str, Any]
    source: str
    attribute: str
    value: Any
    confidence: float
    evidence: str
    provenance: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collect_chemical_intelligence(
    validated_chemicals: list[ValidatedChemical],
    aggregator: KnowledgeAggregator | None = None,
    synonym_resolver=None,
    logger: PipelineLogger | None = None,
    max_workers: int = CHEMICAL_INTELLIGENCE_MAX_WORKERS,
) -> list[ChemicalIntelligenceRecord]:
    """Collect source facts for each validated chemical."""
    knowledge_aggregator = aggregator or KnowledgeAggregator()
    resolver = synonym_resolver or PubChemSynonymResolver()
    intelligence: list[ChemicalIntelligenceRecord] = []
    if logger:
        logger.start_stage(
            "Knowledge Aggregator",
            metadata={"validated_chemical_count": len(validated_chemicals)},
        )
    identities = _resolve_identities_parallel(validated_chemicals, resolver, max_workers, logger)
    for identity in identities:
        if logger:
            logger.progress(
                "Knowledge Aggregator",
                f"[Knowledge] Collecting facts for {identity.get('canonical_name') or identity.get('original_surface_form')}",
                metadata={
                    "chemical": identity.get("canonical_name")
                    or identity.get("original_surface_form"),
                    "cid": identity.get("pubchem_cid"),
                },
            )
        try:
            facts = knowledge_aggregator.get_facts(identity, logger=logger)
        except TypeError:
            facts = knowledge_aggregator.get_facts(identity)
        intelligence.extend(
            _record_from_fact(identity, fact)
            for fact in facts
        )
    if logger:
        logger.end_stage(
            "Knowledge Aggregator",
            metadata={"fact_count": len(intelligence)},
        )
    return intelligence


def chemical_intelligence_to_json(
    intelligence: list[ChemicalIntelligenceRecord],
) -> list[dict[str, Any]]:
    """Return the unified chemical_intelligence.json structure."""
    return [record.to_dict() for record in intelligence]


def _record_from_fact(
    identity: dict[str, Any],
    fact: KnowledgeFact,
) -> ChemicalIntelligenceRecord:
    return ChemicalIntelligenceRecord(
        chemical=identity,
        source=fact.source,
        attribute=fact.attribute,
        value=fact.value,
        confidence=fact.confidence,
        evidence=fact.evidence,
        provenance=fact.provenance,
    )


class PubChemSynonymResolver:
    """Fetch PubChem synonyms for identity matching, not for decision-making."""

    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

    def __init__(
        self,
        timeout_seconds: float = PUBCHEM_REQUEST_TIMEOUT_SECONDS,
        cache_path: str | Path = PUBCHEM_SYNONYM_CACHE_PATH,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[str]] | None = None
        self._lock = threading.Lock()

    def get_synonyms(self, chemical: ValidatedChemical) -> list[str]:
        if chemical.pubchem_cid is None:
            return []
        cache_key = str(chemical.pubchem_cid)
        with self._lock:
            cache = self._load_cache()
            if cache_key in cache:
                return cache[cache_key]
        url = f"{self.base_url}/compound/cid/{chemical.pubchem_cid}/synonyms/JSON"
        try:
            response = requests.get(url, timeout=self.timeout_seconds)
            if response.status_code == 404:
                with self._lock:
                    cache[cache_key] = []
                    self._save_cache(cache)
                return []
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return []

        information = payload.get("InformationList", {}).get("Information", [])
        if not information:
            with self._lock:
                cache[cache_key] = []
                self._save_cache(cache)
            return []
        synonyms = information[0].get("Synonym", [])
        cached_synonyms = [str(synonym) for synonym in synonyms if synonym]
        with self._lock:
            cache[cache_key] = cached_synonyms
            self._save_cache(cache)
        return cached_synonyms

    def _load_cache(self) -> dict[str, list[str]]:
        if self._cache is not None:
            return self._cache
        if not self.cache_path.exists():
            self._cache = {}
            return self._cache
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except (OSError, json.JSONDecodeError):
            loaded = {}
        self._cache = loaded if isinstance(loaded, dict) else {}
        return self._cache

    def _save_cache(self, cache: dict[str, list[str]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)


class NoOpSynonymResolver:
    """Fast resolver that avoids PubChem synonym network calls."""

    def get_synonyms(self, chemical: ValidatedChemical) -> list[str]:
        return []


def _chemical_identity(
    chemical: ValidatedChemical,
    synonym_resolver: PubChemSynonymResolver,
) -> dict[str, Any]:
    pubchem_synonyms = _unique_ordered(synonym_resolver.get_synonyms(chemical))
    aliases = _unique_ordered(
        [
            chemical.original_surface_form,
            chemical.canonical_name,
            *pubchem_synonyms,
        ]
    )
    cas_values = _extract_cas_values(aliases)
    return {
        "canonical_name": chemical.canonical_name,
        "original_surface_form": chemical.original_surface_form,
        "pubchem_cid": chemical.pubchem_cid,
        "cid": chemical.pubchem_cid,
        "cas": cas_values,
        "aliases": aliases,
        "pubchem_synonyms": pubchem_synonyms,
        "molecular_formula": chemical.molecular_formula,
        "canonical_smiles": chemical.canonical_smiles,
        "validation_status": chemical.validation_status,
    }


def _resolve_identities_parallel(
    validated_chemicals: list[ValidatedChemical],
    resolver: PubChemSynonymResolver,
    max_workers: int,
    logger: PipelineLogger | None = None,
) -> list[dict[str, Any]]:
    if not validated_chemicals:
        return []
    if logger:
        logger.start_stage(
            "PubChem synonym resolution",
            metadata={"chemical_count": len(validated_chemicals), "max_workers": max_workers},
        )
    identities_by_index: dict[int, dict[str, Any]] = {}
    worker_count = max(1, min(max_workers, len(validated_chemicals)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(_chemical_identity, chemical, resolver): index
            for index, chemical in enumerate(validated_chemicals)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            identities_by_index[index] = future.result()
            if logger:
                logger.progress(
                    "PubChem synonym resolution",
                    f"[Knowledge] Synonyms resolved {completed}/{len(validated_chemicals)}",
                    metadata={"completed": completed, "total": len(validated_chemicals)},
                )
    if logger:
        logger.end_stage(
            "PubChem synonym resolution",
            metadata={"chemical_count": len(validated_chemicals)},
        )
    return [identities_by_index[index] for index in range(len(validated_chemicals))]


def _extract_cas_values(values: list[str | None]) -> list[str]:
    cas_values = []
    for value in values:
        if not value:
            continue
        cas_values.extend(CAS_PATTERN.findall(str(value)))
    return _unique_ordered(cas_values)


def _unique_ordered(values) -> list:
    seen = set()
    unique = []
    for value in values:
        if value in (None, ""):
            continue
        marker = str(value)
        if marker in seen:
            continue
        seen.add(marker)
        unique.append(marker)
    return unique
