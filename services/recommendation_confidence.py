from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from services.provider_registry import get_providers_by_category


@dataclass(frozen=True)
class RecommendationConfidence:
    chemical: str
    canonical_name: str | None
    pubchem_cid: int | None

    confidence_label: str
    confidence_percent: int
    recommendation_status: str

    identity_validated: bool
    safety_available: bool
    sustainability_available: bool
    regulatory_available: bool
    evidence_available: bool

    trust_explanation: list[str] = field(default_factory=list)
    missing_for_strong_recommendation: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecommendationConfidenceSummary:
    chemical_count: int
    strong_count: int
    high_count: int
    medium_count: int
    low_count: int
    not_ready_count: int
    average_confidence_percent: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_recommendation_confidence(
    coverage_rows: Iterable[Any],
) -> list[RecommendationConfidence]:
    """
    Convert knowledge coverage into recommendation confidence.

    This service does not create recommendations by itself.
    It answers: "How trustworthy would a recommendation be for this chemical
    given the current validated knowledge coverage?"

    Expected input:
    - ChemicalCoverageRow objects from services.knowledge_coverage
    - or dictionaries with equivalent fields
    """
    results: list[RecommendationConfidence] = []

    for row in coverage_rows:
        chemical = str(_get_value(row, "chemical", "unknown"))
        canonical_name = _get_value(row, "canonical_name")
        pubchem_cid = _safe_int(_get_value(row, "pubchem_cid"))

        identity_validated = bool(_get_value(row, "pubchem_validated", False))
        safety_available = _category_available(row, "safety")
        sustainability_available = _category_available(row, "sustainability")
        regulatory_available = _category_available(row, "regulatory")
        evidence_available = _category_available(row, "evidence")

        coverage_percent = int(_get_value(row, "coverage_percent", 0) or 0)
        confidence_label = _confidence_label(
            coverage_percent=coverage_percent,
            identity_validated=identity_validated,
            safety_available=safety_available,
            sustainability_available=sustainability_available,
            regulatory_available=regulatory_available,
            evidence_available=evidence_available,
        )

        recommendation_status = _recommendation_status(
            confidence_label=confidence_label,
            identity_validated=identity_validated,
            safety_available=safety_available,
            sustainability_available=sustainability_available,
            regulatory_available=regulatory_available,
            evidence_available=evidence_available,
        )

        trust_explanation = _trust_explanation(
            identity_validated=identity_validated,
            safety_available=safety_available,
            sustainability_available=sustainability_available,
            regulatory_available=regulatory_available,
            evidence_available=evidence_available,
        )

        missing = _missing_for_strong_recommendation(
            identity_validated=identity_validated,
            safety_available=safety_available,
            sustainability_available=sustainability_available,
            regulatory_available=regulatory_available,
            evidence_available=evidence_available,
            row=row,
        )

        warnings = _warnings(
            identity_validated=identity_validated,
            safety_available=safety_available,
            sustainability_available=sustainability_available,
            regulatory_available=regulatory_available,
            evidence_available=evidence_available,
        )

        results.append(
            RecommendationConfidence(
                chemical=chemical,
                canonical_name=str(canonical_name) if canonical_name else None,
                pubchem_cid=pubchem_cid,
                confidence_label=confidence_label,
                confidence_percent=coverage_percent,
                recommendation_status=recommendation_status,
                identity_validated=identity_validated,
                safety_available=safety_available,
                sustainability_available=sustainability_available,
                regulatory_available=regulatory_available,
                evidence_available=evidence_available,
                trust_explanation=trust_explanation,
                missing_for_strong_recommendation=missing,
                warnings=warnings,
            )
        )

    return results


def summarize_recommendation_confidence(
    confidence_rows: Iterable[RecommendationConfidence],
) -> RecommendationConfidenceSummary:
    rows = list(confidence_rows)

    if not rows:
        return RecommendationConfidenceSummary(
            chemical_count=0,
            strong_count=0,
            high_count=0,
            medium_count=0,
            low_count=0,
            not_ready_count=0,
            average_confidence_percent=0,
        )

    strong = high = medium = low = not_ready = 0
    total = 0

    for row in rows:
        total += row.confidence_percent

        if row.recommendation_status == "not_ready":
            not_ready += 1
        elif row.confidence_label == "strong":
            strong += 1
        elif row.confidence_label == "high":
            high += 1
        elif row.confidence_label == "medium":
            medium += 1
        else:
            low += 1

    return RecommendationConfidenceSummary(
        chemical_count=len(rows),
        strong_count=strong,
        high_count=high,
        medium_count=medium,
        low_count=low,
        not_ready_count=not_ready,
        average_confidence_percent=int(round(total / len(rows))),
    )


def confidence_rows_to_dicts(
    rows: Iterable[RecommendationConfidence],
) -> list[dict[str, Any]]:
    return [row.to_dict() for row in rows]


def build_recommendation_confidence_table(
    coverage_rows: Iterable[Any],
) -> list[dict[str, Any]]:
    return confidence_rows_to_dicts(build_recommendation_confidence(coverage_rows))


def _confidence_label(
    *,
    coverage_percent: int,
    identity_validated: bool,
    safety_available: bool,
    sustainability_available: bool,
    regulatory_available: bool,
    evidence_available: bool,
) -> str:
    if not identity_validated:
        return "not_ready"

    if (
        coverage_percent >= 85
        and safety_available
        and sustainability_available
        and regulatory_available
        and evidence_available
    ):
        return "strong"

    if coverage_percent >= 70 and safety_available and sustainability_available:
        return "high"

    if coverage_percent >= 45 and (safety_available or sustainability_available):
        return "medium"

    return "low"


def _recommendation_status(
    *,
    confidence_label: str,
    identity_validated: bool,
    safety_available: bool,
    sustainability_available: bool,
    regulatory_available: bool,
    evidence_available: bool,
) -> str:
    if not identity_validated:
        return "not_ready"

    if confidence_label == "strong":
        return "recommendation_ready"

    if confidence_label == "high":
        return "usable_with_caveats"

    if confidence_label == "medium":
        return "limited_support"

    return "identity_only_or_weak_support"


def _trust_explanation(
    *,
    identity_validated: bool,
    safety_available: bool,
    sustainability_available: bool,
    regulatory_available: bool,
    evidence_available: bool,
) -> list[str]:
    explanation: list[str] = []

    if identity_validated:
        explanation.append("Identity is validated with PubChem/RDKit.")
    else:
        explanation.append("Identity is not validated; no recommendation should be trusted yet.")

    if safety_available:
        explanation.append("Safety/toxicology knowledge is available.")
    else:
        explanation.append("Safety/toxicology knowledge is missing.")

    if sustainability_available:
        explanation.append("Sustainability knowledge is available.")
    else:
        explanation.append("Sustainability knowledge is missing.")

    if regulatory_available:
        explanation.append("Regulatory knowledge is available.")
    else:
        explanation.append("Regulatory knowledge is missing.")

    if evidence_available:
        explanation.append("Retrieved scientific evidence is available.")
    else:
        explanation.append("Retrieved scientific evidence is missing.")

    return explanation


def _missing_for_strong_recommendation(
    *,
    identity_validated: bool,
    safety_available: bool,
    sustainability_available: bool,
    regulatory_available: bool,
    evidence_available: bool,
    row: Any,
) -> list[str]:
    missing: list[str] = []

    if not identity_validated:
        missing.append("identity validation")

    if not safety_available:
        missing.append("safety/toxicology provider")

    if not sustainability_available:
        missing.append("sustainability provider")

    if not regulatory_available:
        missing.append("regulatory provider")

    if not evidence_available:
        missing.append("retrieved evidence")

    # Add known provider-specific gaps when available.
    missing_providers = _get_value(row, "missing_providers", []) or []
    for provider in _as_list(missing_providers):
        provider_label = str(provider)
        if provider_label and provider_label not in missing:
            missing.append(provider_label)

    return missing


def _warnings(
    *,
    identity_validated: bool,
    safety_available: bool,
    sustainability_available: bool,
    regulatory_available: bool,
    evidence_available: bool,
) -> list[str]:
    warnings: list[str] = []

    if not identity_validated:
        warnings.append("Do not generate recommendations for unvalidated chemical identity.")

    if not safety_available:
        warnings.append("Recommendation should be marked weak because safety data is missing.")

    if not sustainability_available:
        warnings.append("Green/safer alternative recommendation is weak without sustainability data.")

    if not regulatory_available:
        warnings.append("Regulatory risk may be incomplete.")

    if not evidence_available:
        warnings.append("Explanation should be marked as unsupported by retrieved evidence.")

    return warnings


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    return [value]


def _category_available(row: Any, category: str) -> bool:
    provider_counts = _get_value(row, "provider_counts", {}) or {}

    for provider in get_providers_by_category(category):
        if _truthy_provider_value(row, provider.name, provider_counts):
            return True

    return False


def _truthy_provider_value(row: Any, provider_name: str, provider_counts: Mapping[str, Any]) -> bool:
    direct = _get_value(row, provider_name, None)
    if direct is not None:
        return bool(direct)

    count = provider_counts.get(provider_name, 0) if isinstance(provider_counts, Mapping) else 0
    try:
        return int(count) > 0
    except (TypeError, ValueError):
        return bool(count)
