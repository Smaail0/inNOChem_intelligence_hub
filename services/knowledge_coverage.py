from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from services.provider_registry import (
    get_provider_registry,
    get_providers_by_category,
    normalize_provider_name,
)


@dataclass(frozen=True)
class CoverageWeights:
    """
    Coverage scoring weights.

    The score is intentionally grouped by decision value rather than by raw provider count:
    - PubChem validates identity.
    - HSDB covers safety/tox/environment facts.
    - CHEM21/GSK cover sustainability intelligence.
    - REACH/SVHC/CLP/DOT cover regulatory intelligence.
    - RAG/evidence supports explainability.
    """

    pubchem_identity: float = 0.20
    hsdb_safety: float = 0.20
    sustainability: float = 0.25
    regulatory: float = 0.20
    evidence: float = 0.15

    def total(self) -> float:
        return (
            self.pubchem_identity
            + self.hsdb_safety
            + self.sustainability
            + self.regulatory
            + self.evidence
        )


@dataclass(frozen=True)
class ChemicalCoverageRow:
    chemical: str
    canonical_name: str | None
    pubchem_cid: int | None

    pubchem_validated: bool
    hsdb: bool
    chem21: bool
    gsk: bool
    reach: bool
    svhc: bool
    clp: bool
    dot: bool
    rag_evidence: bool

    provider_counts: dict[str, int] = field(default_factory=dict)
    coverage_score: float = 0.0
    coverage_percent: int = 0
    confidence: str = "low"
    recommendation_readiness: str = "weak"
    missing_groups: list[str] = field(default_factory=list)
    missing_providers: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CoverageSummary:
    chemical_count: int
    average_coverage_score: float
    average_coverage_percent: int
    high_confidence_count: int
    medium_confidence_count: int
    low_confidence_count: int
    weak_recommendation_count: int
    provider_presence: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_knowledge_coverage(
    validated_chemicals: Iterable[Any],
    intelligence_records: Iterable[Any] | None = None,
    evidence_records: Iterable[Any] | None = None,
    weights: CoverageWeights | None = None,
) -> list[ChemicalCoverageRow]:
    """
    Build one coverage row per validated chemical.

    This function is intentionally framework-independent. It accepts:
    - ValidatedChemical dataclasses/objects from services.validator
    - dict records exported from chemical intelligence
    - future provider aggregation records
    - optional RAG evidence rows

    The function does not call external APIs and does not trigger RAG retrieval.
    """

    weights = weights or CoverageWeights()

    intelligence_index = _index_records_by_identity(intelligence_records or [])
    evidence_index = _index_records_by_identity(evidence_records or [])

    rows: list[ChemicalCoverageRow] = []

    for chemical in validated_chemicals:
        identity = _chemical_identity(chemical)
        lookup_keys = _identity_lookup_keys(identity)

        intelligence_matches = _lookup_many(intelligence_index, lookup_keys)
        evidence_matches = _lookup_many(evidence_index, lookup_keys)

        provider_counts = _provider_counts_from_records(intelligence_matches)
        provider_counts["rag_evidence"] = max(
            provider_counts.get("rag_evidence", 0),
            _count_evidence(evidence_matches),
        )

        pubchem_validated = _is_pubchem_validated(chemical, intelligence_matches)
        hsdb = provider_counts.get("hsdb", 0) > 0
        chem21 = provider_counts.get("chem21", 0) > 0
        gsk = provider_counts.get("gsk", 0) > 0
        regulatory_topic_count = _regulatory_topic_count(intelligence_matches)
        reach = provider_counts.get("reach", 0) > 0 or regulatory_topic_count > 0
        svhc = provider_counts.get("svhc", 0) > 0
        clp = provider_counts.get("clp", 0) > 0
        dot = provider_counts.get("dot", 0) > 0
        rag_evidence = provider_counts.get("rag_evidence", 0) > 0

        coverage_score = _coverage_score(
            pubchem_validated=pubchem_validated,
            hsdb=hsdb,
            chem21=chem21,
            gsk=gsk,
            reach=reach,
            svhc=svhc,
            clp=clp,
            dot=dot,
            rag_evidence=rag_evidence,
            weights=weights,
        )

        missing_groups = _missing_groups(
            pubchem_validated=pubchem_validated,
            hsdb=hsdb,
            chem21=chem21,
            gsk=gsk,
            reach=reach,
            svhc=svhc,
            clp=clp,
            dot=dot,
            rag_evidence=rag_evidence,
        )

        missing_providers = _missing_providers(
            hsdb=hsdb,
            chem21=chem21,
            gsk=gsk,
            reach=reach,
            svhc=svhc,
            clp=clp,
            dot=dot,
            rag_evidence=rag_evidence,
        )

        confidence = _confidence_label(coverage_score)
        recommendation_readiness = _recommendation_readiness(
            coverage_score=coverage_score,
            pubchem_validated=pubchem_validated,
            hsdb=hsdb,
            has_sustainability=(chem21 or gsk),
            has_regulatory=(reach or svhc or clp or dot),
            rag_evidence=rag_evidence,
        )

        notes = _coverage_notes(
            pubchem_validated=pubchem_validated,
            hsdb=hsdb,
            chem21=chem21,
            gsk=gsk,
            reach=reach,
            svhc=svhc,
            clp=clp,
            dot=dot,
            rag_evidence=rag_evidence,
        )

        rows.append(
            ChemicalCoverageRow(
                chemical=identity["chemical"],
                canonical_name=identity["canonical_name"],
                pubchem_cid=identity["pubchem_cid"],
                pubchem_validated=pubchem_validated,
                hsdb=hsdb,
                chem21=chem21,
                gsk=gsk,
                reach=reach,
                svhc=svhc,
                clp=clp,
                dot=dot,
                rag_evidence=rag_evidence,
                provider_counts=provider_counts,
                coverage_score=round(coverage_score, 4),
                coverage_percent=int(round(coverage_score * 100)),
                confidence=confidence,
                recommendation_readiness=recommendation_readiness,
                missing_groups=missing_groups,
                missing_providers=missing_providers,
                notes=notes,
            )
        )

    return rows


def summarize_coverage(rows: Iterable[ChemicalCoverageRow]) -> CoverageSummary:
    rows = list(rows)
    if not rows:
        return CoverageSummary(
            chemical_count=0,
            average_coverage_score=0.0,
            average_coverage_percent=0,
            high_confidence_count=0,
            medium_confidence_count=0,
            low_confidence_count=0,
            weak_recommendation_count=0,
            provider_presence={},
        )

    provider_names = _tracked_provider_names(include_identity=True)
    provider_presence = {name: 0 for name in provider_names}

    high = medium = low = weak = 0
    total_score = 0.0

    for row in rows:
        total_score += row.coverage_score

        if row.confidence in {"high", "strong"}:
            high += 1
        elif row.confidence == "medium":
            medium += 1
        else:
            low += 1

        if row.recommendation_readiness in {"weak", "identity_only"}:
            weak += 1

        provider_presence["pubchem"] = provider_presence.get("pubchem", 0) + int(row.pubchem_validated)
        provider_presence["hsdb"] = provider_presence.get("hsdb", 0) + int(row.hsdb)
        provider_presence["chem21"] = provider_presence.get("chem21", 0) + int(row.chem21)
        provider_presence["gsk"] = provider_presence.get("gsk", 0) + int(row.gsk)
        provider_presence["reach"] = provider_presence.get("reach", 0) + int(row.reach)
        provider_presence["svhc"] = provider_presence.get("svhc", 0) + int(row.svhc)
        provider_presence["clp"] = provider_presence.get("clp", 0) + int(row.clp)
        provider_presence["dot"] = provider_presence.get("dot", 0) + int(row.dot)
        provider_presence["rag_evidence"] = provider_presence.get("rag_evidence", 0) + int(row.rag_evidence)

    avg = total_score / len(rows)

    return CoverageSummary(
        chemical_count=len(rows),
        average_coverage_score=round(avg, 4),
        average_coverage_percent=int(round(avg * 100)),
        high_confidence_count=high,
        medium_confidence_count=medium,
        low_confidence_count=low,
        weak_recommendation_count=weak,
        provider_presence=provider_presence,
    )


def coverage_rows_to_dicts(rows: Iterable[ChemicalCoverageRow]) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]


def build_coverage_table(
    validated_chemicals: Iterable[Any],
    intelligence_records: Iterable[Any] | None = None,
    evidence_records: Iterable[Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Convenience wrapper for Streamlit tables / CSV export.
    """
    rows = build_knowledge_coverage(
        validated_chemicals=validated_chemicals,
        intelligence_records=intelligence_records,
        evidence_records=evidence_records,
    )
    return coverage_rows_to_dicts(rows)


def _coverage_score(
    *,
    pubchem_validated: bool,
    hsdb: bool,
    chem21: bool,
    gsk: bool,
    reach: bool,
    svhc: bool,
    clp: bool,
    dot: bool,
    rag_evidence: bool,
    weights: CoverageWeights,
) -> float:
    provider_counts = _provider_counts_from_flags(
        pubchem_validated=pubchem_validated,
        hsdb=hsdb,
        chem21=chem21,
        gsk=gsk,
        reach=reach,
        svhc=svhc,
        clp=clp,
        dot=dot,
        rag_evidence=rag_evidence,
    )
    score = 0.0

    if pubchem_validated:
        score += weights.pubchem_identity

    if _category_available(provider_counts, "safety"):
        score += weights.hsdb_safety

    if _category_available(provider_counts, "sustainability"):
        score += weights.sustainability

    if _category_available(provider_counts, "regulatory"):
        score += weights.regulatory

    if _category_available(provider_counts, "evidence"):
        score += weights.evidence

    total = weights.total()
    if total <= 0:
        return 0.0

    return max(0.0, min(1.0, score / total))


def _confidence_label(score: float) -> str:
    if score >= 0.85:
        return "strong"
    if score >= 0.70:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def _recommendation_readiness(
    *,
    coverage_score: float,
    pubchem_validated: bool,
    hsdb: bool,
    has_sustainability: bool,
    has_regulatory: bool,
    rag_evidence: bool,
) -> str:
    if not pubchem_validated:
        return "not_ready_identity_unverified"

    if coverage_score >= 0.85 and hsdb and has_sustainability and has_regulatory and rag_evidence:
        return "strong"

    if coverage_score >= 0.70 and hsdb and has_sustainability:
        return "usable_with_caveats"

    if hsdb or has_sustainability:
        return "weak"

    return "identity_only"


def _missing_groups(
    *,
    pubchem_validated: bool,
    hsdb: bool,
    chem21: bool,
    gsk: bool,
    reach: bool,
    svhc: bool,
    clp: bool,
    dot: bool,
    rag_evidence: bool,
) -> list[str]:
    provider_counts = _provider_counts_from_flags(
        pubchem_validated=pubchem_validated,
        hsdb=hsdb,
        chem21=chem21,
        gsk=gsk,
        reach=reach,
        svhc=svhc,
        clp=clp,
        dot=dot,
        rag_evidence=rag_evidence,
    )
    missing: list[str] = []

    if not pubchem_validated:
        missing.append("identity_validation")

    if not _category_available(provider_counts, "safety"):
        missing.append("safety_toxicology")

    if not _category_available(provider_counts, "sustainability"):
        missing.append("sustainability")

    if not _category_available(provider_counts, "regulatory"):
        missing.append("regulatory")

    if not _category_available(provider_counts, "evidence"):
        missing.append("scientific_evidence")

    return missing


def _missing_providers(
    *,
    hsdb: bool,
    chem21: bool,
    gsk: bool,
    reach: bool,
    svhc: bool,
    clp: bool,
    dot: bool,
    rag_evidence: bool,
) -> list[str]:
    provider_counts = _provider_counts_from_flags(
        pubchem_validated=False,
        hsdb=hsdb,
        chem21=chem21,
        gsk=gsk,
        reach=reach,
        svhc=svhc,
        clp=clp,
        dot=dot,
        rag_evidence=rag_evidence,
    )
    missing: list[str] = []

    for provider_name in _tracked_provider_names(include_identity=False):
        if provider_counts.get(provider_name, 0) > 0:
            continue
        provider = get_provider_registry().get_provider(provider_name)
        missing.append(provider.label if provider else provider_name.upper())

    return missing


def _coverage_notes(
    *,
    pubchem_validated: bool,
    hsdb: bool,
    chem21: bool,
    gsk: bool,
    reach: bool,
    svhc: bool,
    clp: bool,
    dot: bool,
    rag_evidence: bool,
) -> list[str]:
    provider_counts = _provider_counts_from_flags(
        pubchem_validated=pubchem_validated,
        hsdb=hsdb,
        chem21=chem21,
        gsk=gsk,
        reach=reach,
        svhc=svhc,
        clp=clp,
        dot=dot,
        rag_evidence=rag_evidence,
    )
    notes: list[str] = []

    if not pubchem_validated:
        notes.append("Identity is not PubChem/RDKit validated; recommendation should not be generated.")

    if not _category_available(provider_counts, "safety"):
        notes.append("No HSDB safety/toxicology facts available.")

    if not _category_available(provider_counts, "sustainability"):
        notes.append("No sustainability provider coverage available.")

    if not _category_available(provider_counts, "regulatory"):
        notes.append("No regulatory provider coverage available.")

    if not _category_available(provider_counts, "evidence"):
        notes.append("No retrieved evidence attached; explanation should be marked weak.")

    if gsk is False:
        notes.append("GSK absence may reflect missing approved GSK extraction rows, not chemical safety.")

    return notes


def _chemical_identity(chemical: Any) -> dict[str, Any]:
    chemical_name = (
        _get_value(chemical, "original_surface_form")
        or _get_value(chemical, "surface_form")
        or _get_value(chemical, "chemical")
        or _get_value(chemical, "name")
        or _get_value(chemical, "normalized_name")
        or "unknown"
    )

    canonical_name = (
        _get_value(chemical, "canonical_name")
        or _get_value(chemical, "normalized_name")
        or _get_value(chemical, "preferred_name")
    )

    pubchem_cid = (
        _get_value(chemical, "pubchem_cid")
        or _get_value(chemical, "cid")
        or _get_value(chemical, "pubchem_cid")
    )

    try:
        pubchem_cid = int(pubchem_cid) if pubchem_cid is not None else None
    except (TypeError, ValueError):
        pubchem_cid = None

    return {
        "chemical": str(chemical_name),
        "canonical_name": str(canonical_name) if canonical_name is not None else None,
        "pubchem_cid": pubchem_cid,
    }


def _identity_lookup_keys(identity: Mapping[str, Any]) -> list[str]:
    keys: list[str] = []

    cid = identity.get("pubchem_cid")
    if cid is not None:
        keys.append(f"cid:{cid}")

    for field_name in ("chemical", "canonical_name"):
        value = identity.get(field_name)
        if value:
            keys.append(f"name:{_normalize_name(str(value))}")

    return _unique(keys)


def _index_records_by_identity(records: Iterable[Any]) -> dict[str, list[Any]]:
    index: dict[str, list[Any]] = {}

    for record in records:
        chemical_obj = _get_value(record, "chemical")

        cid = None
        if chemical_obj:
            cid = _get_value(chemical_obj, "pubchem_cid") or _get_value(chemical_obj, "cid")
        if cid is None:
            cid = _get_value(record, "pubchem_cid") or _get_value(record, "cid")

        if cid is not None:
            try:
                key = f"cid:{int(cid)}"
                index.setdefault(key, []).append(record)
            except (TypeError, ValueError):
                pass

        name_candidates = [
            _get_value(record, "chemical"),
            _get_value(record, "chemical_name"),
            _get_value(record, "name"),
            _get_value(record, "canonical_name"),
            _get_value(record, "original_surface_form"),
            _get_value(record, "normalized_name"),
        ]

        if chemical_obj:
            name_candidates.extend(
                [
                    _get_value(chemical_obj, "canonical_name"),
                    _get_value(chemical_obj, "original_surface_form"),
                    _get_value(chemical_obj, "name"),
                    _get_value(chemical_obj, "normalized_name"),
                ]
            )

        for name in name_candidates:
            if name:
                key = f"name:{_normalize_name(str(name))}"
                index.setdefault(key, []).append(record)

    return index


def _lookup_many(index: Mapping[str, list[Any]], keys: Iterable[str]) -> list[Any]:
    results: list[Any] = []

    for key in keys:
        results.extend(index.get(key, []))

    return _unique_by_object_id(results)


def _provider_counts_from_records(records: Iterable[Any]) -> dict[str, int]:
    counts = {name: 0 for name in _tracked_provider_names(include_identity=False)}

    field_aliases = _provider_field_aliases()

    for record in records:
        source = (
            _get_value(record, "source")
            or _get_value(record, "provider")
            or _get_value(record, "provider_name")
        )

        if source:
            provider = _normalize_provider_name(str(source))
            if provider in counts:
                counts[provider] += 1

        for provider, aliases in field_aliases.items():
            for alias in aliases:
                value = _get_value(record, alias)
                counts[provider] += _count_value(value)

        generic_facts = (
            _get_value(record, "facts")
            or _get_value(record, "provider_facts")
            or _get_value(record, "knowledge_facts")
        )

        for fact in _as_list(generic_facts):
            fact_source = (
                _get_value(fact, "source")
                or _get_value(fact, "provider")
                or _get_value(fact, "provider_name")
                or ""
            )
            provider = _normalize_provider_name(str(fact_source))
            if provider in counts:
                counts[provider] += 1

    return counts


def _regulatory_topic_count(records: Iterable[Any]) -> int:
    count = 0
    for record in records:
        attribute = str(_get_value(record, "attribute") or "").lower()
        if attribute.startswith("regulatory_intelligence."):
            count += 1
    return count


def _count_evidence(records: Iterable[Any]) -> int:
    count = 0

    for record in records:
        count += _count_value(record)

        for field_name in ("rag_evidence", "evidence", "evidence_rows", "retrieved_evidence"):
            count += _count_value(_get_value(record, field_name))

    return count


def _is_pubchem_validated(chemical: Any, intelligence_matches: Iterable[Any]) -> bool:
    status = str(_get_value(chemical, "validation_status") or "").lower()
    cid = _get_value(chemical, "pubchem_cid") or _get_value(chemical, "cid")

    if cid is not None and "validated" in status:
        return True

    if cid is not None and status in {"resolved", "resolved_pubchem", "validated_pubchem_rdkit"}:
        return True

    for record in intelligence_matches:
        rec_cid = _get_value(record, "pubchem_cid") or _get_value(record, "cid")
        rec_status = str(_get_value(record, "validation_status") or "").lower()
        if rec_cid is not None and ("validated" in rec_status or rec_status == "resolved"):
            return True

    return False


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _count_value(value: Any) -> int:
    if value is None:
        return 0

    if isinstance(value, bool):
        return int(value)

    if isinstance(value, (str, bytes)):
        return 1 if value else 0

    if isinstance(value, Mapping):
        return 1 if value else 0

    if isinstance(value, Iterable):
        try:
            return len(list(value))
        except TypeError:
            return 1

    return 1


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _normalize_name(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _normalize_provider_name(value: str) -> str:
    return normalize_provider_name(value)


def _tracked_provider_names(include_identity: bool) -> list[str]:
    registry_names = sorted(get_provider_registry().providers)
    stable_names = ["pubchem", "hsdb", "chem21", "gsk", "reach", "svhc", "clp", "dot", "rag_evidence"]
    names = _unique([*stable_names, *registry_names])
    if include_identity:
        return names
    return [name for name in names if get_provider_registry().get_provider_category(name) != "identity"]


def _category_available(provider_counts: Mapping[str, int], category: str) -> bool:
    for provider in get_providers_by_category(category):
        if provider_counts.get(provider.name, 0) > 0:
            return True
    return False


def _provider_counts_from_flags(
    *,
    pubchem_validated: bool,
    hsdb: bool,
    chem21: bool,
    gsk: bool,
    reach: bool,
    svhc: bool,
    clp: bool,
    dot: bool,
    rag_evidence: bool,
) -> dict[str, int]:
    return {
        "pubchem": int(pubchem_validated),
        "hsdb": int(hsdb),
        "chem21": int(chem21),
        "gsk": int(gsk),
        "reach": int(reach),
        "svhc": int(svhc),
        "clp": int(clp),
        "dot": int(dot),
        "rag_evidence": int(rag_evidence),
    }


def _provider_field_aliases() -> dict[str, list[str]]:
    field_aliases: dict[str, list[str]] = {}
    registry = get_provider_registry()

    for provider_name in _tracked_provider_names(include_identity=False):
        provider = registry.get_provider(provider_name)
        aliases = [provider_name, f"{provider_name}_facts"]

        if provider:
            aliases.extend(provider.aliases)
            aliases.extend(f"{alias}_facts" for alias in provider.aliases)

        field_aliases[provider_name] = _unique(aliases)

    legacy_category_fields = {
        "hsdb": ["safety_facts", "toxicology_facts"],
        "chem21": ["sustainability_facts"],
        "rag_evidence": ["evidence", "evidence_rows", "retrieved_evidence"],
    }

    for provider_name, aliases in legacy_category_fields.items():
        field_aliases.setdefault(provider_name, [])
        field_aliases[provider_name] = _unique([*field_aliases[provider_name], *aliases])

    return field_aliases


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result


def _unique_by_object_id(values: Iterable[Any]) -> list[Any]:
    seen = set()
    result: list[Any] = []

    for value in values:
        marker = id(value)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(value)

    return result
