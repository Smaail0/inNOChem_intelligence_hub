"""HSDB annotation provider architecture.

The HSDB JSON catalog bundled with the project is a topic catalog, not a
compound-level fact database. This provider uses that catalog only as an
allowlist and priority map, then retrieves compound annotations by PubChem CID.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import requests

from core.config import HSDB_CACHE_PATH, HSDB_REQUEST_TIMEOUT_SECONDS, HSDB_TOPIC_CATALOG_PATH
from knowledge.providers.base import KnowledgeFact


SAFETY_TOPICS = [
    "GHS Classification",
    "NFPA Hazard Classification",
    "Evidence for Carcinogenicity",
    "Human Toxicity Values",
    "IDLH",
    "TLV",
    "NIOSH Recommendations",
]
REGULATORY_TOPICS = [
    "OSHA Standards",
    "TSCA Requirements",
    "CERCLA Reportable Quantities",
    "RCRA Requirements",
    "Shipment Methods and Regulations",
    "DOT Emergency Guidelines",
]
ENVIRONMENTAL_TOPICS = [
    "Ecotoxicity Values",
    "Environmental Fate",
    "Environmental Biodegradation",
    "Environmental Bioconcentration",
    "Soil Adsorption/Mobility",
]

TOPIC_GROUPS = {
    "safety_intelligence": SAFETY_TOPICS,
    "regulatory_intelligence": REGULATORY_TOPICS,
    "environmental_intelligence": ENVIRONMENTAL_TOPICS,
}

TOPIC_ALIASES = {
    "IDLH": "Immediately Dangerous to Life or Health (IDLH)",
    "TLV": "Threshold Limit Values (TLV)",
}


@dataclass(frozen=True)
class HsdbTopic:
    heading: str
    group: str
    priority: int
    catalog_count: int | None = None
    description: str = ""


class HsdbTopicCatalog:
    """Loads and filters the HSDB topic catalog."""

    def __init__(self, catalog_path: str | Path = HSDB_TOPIC_CATALOG_PATH) -> None:
        self.catalog_path = Path(catalog_path)
        self._catalog_rows = self._load_catalog_rows()
        self._catalog_by_heading = {
            row.get("Heading", "").strip(): row for row in self._catalog_rows if row.get("Heading")
        }
        self._topics = self._build_supported_topics()

    def allowed_topics(self) -> list[HsdbTopic]:
        return list(self._topics.values())

    def allowed_headings(self) -> set[str]:
        return set(self._topics)

    def is_allowed(self, topic: str) -> bool:
        return self.resolve_heading(topic) in self._topics

    def resolve_heading(self, topic: str) -> str:
        return TOPIC_ALIASES.get(topic, topic)

    def topic_for_heading(self, topic: str) -> HsdbTopic | None:
        return self._topics.get(self.resolve_heading(topic))

    def filter_allowed(self, topics: list[str] | tuple[str, ...] | set[str]) -> list[HsdbTopic]:
        allowed = []
        for topic in topics:
            resolved = self.topic_for_heading(topic)
            if resolved:
                allowed.append(resolved)
        return allowed

    def _load_catalog_rows(self) -> list[dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        with self.catalog_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("InformationList", {}).get("Annotation", [])

    def _build_supported_topics(self) -> dict[str, HsdbTopic]:
        topics: dict[str, HsdbTopic] = {}
        priority = 1
        for group, headings in TOPIC_GROUPS.items():
            for heading in headings:
                catalog_heading = self.resolve_heading(heading)
                row = self._catalog_by_heading.get(catalog_heading)
                if not row:
                    priority += 1
                    continue
                topics[catalog_heading] = HsdbTopic(
                    heading=catalog_heading,
                    group=group,
                    priority=priority,
                    catalog_count=_safe_int(row.get("Count")),
                    description=str(row.get("Description", "")),
                )
                priority += 1
        return topics


class PubChemHsdbAnnotationClient:
    """Retrieves HSDB compound-level annotations from PubChem PUG-View."""

    def __init__(
        self,
        timeout_seconds: float = HSDB_REQUEST_TIMEOUT_SECONDS,
        cache_path: str | Path = HSDB_CACHE_PATH,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.cache_path = Path(cache_path)
        self._cache: dict[str, list[dict[str, Any]]] | None = None

    def get_annotations(self, cid: int, topics: list[str]) -> list[dict[str, Any]]:
        cache_key = _cache_key(cid, topics)
        cache = self._load_cache()
        if cache_key in cache:
            return cache[cache_key]

        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON"
        response = requests.get(url, timeout=self.timeout_seconds)
        if response.status_code == 404:
            cache[cache_key] = []
            self._save_cache(cache)
            return []
        response.raise_for_status()
        data = response.json()
        annotations = extract_hsdb_annotations_from_pug_view(data, topics)
        cache[cache_key] = annotations
        self._save_cache(cache)
        return annotations

    def _load_cache(self) -> dict[str, list[dict[str, Any]]]:
        if self._cache is not None:
            return self._cache
        if not self.cache_path.exists():
            self._cache = {}
            return self._cache
        try:
            with self.cache_path.open("r", encoding="utf-8") as handle:
                cache = json.load(handle)
        except (OSError, json.JSONDecodeError):
            cache = {}
        self._cache = cache if isinstance(cache, dict) else {}
        return self._cache

    def _save_cache(self, cache: dict[str, list[dict[str, Any]]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("w", encoding="utf-8") as handle:
            json.dump(cache, handle, indent=2, sort_keys=True)


class HsdbProvider:
    name = "hsdb"

    def __init__(
        self,
        topic_catalog_path: str | Path = HSDB_TOPIC_CATALOG_PATH,
        annotation_client: PubChemHsdbAnnotationClient | None = None,
    ) -> None:
        self.topic_catalog = HsdbTopicCatalog(topic_catalog_path)
        self.annotation_client = annotation_client or PubChemHsdbAnnotationClient()

    def get_facts(self, chemical_identity) -> list[KnowledgeFact]:
        return self.get_facts_for_topics(
            chemical_identity,
            [topic.heading for topic in self.topic_catalog.allowed_topics()],
        )

    def get_facts_for_topics(self, chemical_identity, topics: list[str]) -> list[KnowledgeFact]:
        cid = _extract_cid(chemical_identity)
        if cid is None:
            return []

        allowed_topics = self.topic_catalog.filter_allowed(topics)
        if not allowed_topics:
            return []

        topic_by_heading = {topic.heading: topic for topic in allowed_topics}
        try:
            annotations = self.annotation_client.get_annotations(
                cid,
                [topic.heading for topic in allowed_topics],
            )
        except (requests.RequestException, ValueError, TypeError):
            return []
        facts: list[KnowledgeFact] = []
        for annotation in annotations:
            heading = self.topic_catalog.resolve_heading(str(annotation.get("topic", "")))
            topic = topic_by_heading.get(heading) or self.topic_catalog.topic_for_heading(heading)
            if not topic:
                continue
            facts.append(_annotation_to_fact(cid, topic, annotation))
        return facts


def extract_hsdb_annotations_from_pug_view(
    pug_view_payload: dict[str, Any],
    topics: list[str] | set[str],
) -> list[dict[str, Any]]:
    """Extract requested topic sections from a PubChem PUG-View response."""
    topic_set = set(topics)
    record = pug_view_payload.get("Record", pug_view_payload)
    annotations: list[dict[str, Any]] = []

    def visit(section: dict[str, Any], path: list[str]) -> None:
        heading = str(section.get("TOCHeading", "")).strip()
        current_path = [*path, heading] if heading else path
        if heading in topic_set:
            for value in _section_values(section):
                annotations.append(
                    {
                        "topic": heading,
                        "value": value,
                        "evidence": f"PubChem PUG-View section: {' > '.join(current_path)}",
                        "provenance": "PubChem PUG-View HSDB annotation",
                    }
                )
        for child in section.get("Section", []) or []:
            visit(child, current_path)

    for section in record.get("Section", []) or []:
        visit(section, [])
    return annotations


def _annotation_to_fact(cid: int, topic: HsdbTopic, annotation: dict[str, Any]) -> KnowledgeFact:
    value = annotation.get("value")
    evidence = str(
        annotation.get("evidence")
        or f"HSDB annotation topic '{topic.heading}' for PubChem CID {cid}."
    )
    source_provenance = str(annotation.get("provenance", "PubChem PUG-View HSDB annotation"))
    provenance = (
        f"{source_provenance}; cid={cid}; topic={topic.heading}; group={topic.group}; "
        f"catalog={HSDB_TOPIC_CATALOG_PATH}; catalog_count={topic.catalog_count}"
    )
    return KnowledgeFact(
        source=HsdbProvider.name,
        attribute=f"{topic.group}.{_slug(topic.heading)}",
        value=value,
        confidence=1.0,
        evidence=evidence,
        provenance=provenance,
    )


def _section_values(section: dict[str, Any]) -> list[Any]:
    values = []
    for info in section.get("Information", []) or []:
        value = _extract_info_value(info)
        if value not in (None, "", []):
            values.append(value)
    return values


def _extract_info_value(info: dict[str, Any]) -> Any:
    value = info.get("Value", {})
    if "StringWithMarkup" in value:
        strings = [item.get("String", "") for item in value.get("StringWithMarkup", [])]
        return "\n".join(text for text in strings if text)
    if "StringValue" in value:
        return value.get("StringValue")
    if "Number" in value:
        return value.get("Number")
    if "ExternalDataURL" in value:
        return value.get("ExternalDataURL")
    return value or None


def _extract_cid(chemical_identity) -> int | None:
    if isinstance(chemical_identity, dict):
        return _safe_int(
            chemical_identity.get("cid")
            or chemical_identity.get("pubchem_cid")
            or chemical_identity.get("PubChem CID")
        )
    for attr in ("cid", "pubchem_cid"):
        value = getattr(chemical_identity, attr, None)
        cid = _safe_int(value)
        if cid is not None:
            return cid
    return None


def _safe_int(value: Any) -> int | None:
    try:
        if value in (None, ""):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _slug(value: str) -> str:
    text = value.casefold()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def _cache_key(cid: int, topics: list[str]) -> str:
    normalized_topics = "|".join(sorted(str(topic) for topic in topics))
    return f"{cid}:{normalized_topics}"
