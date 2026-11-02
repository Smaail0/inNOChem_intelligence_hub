"""Deterministic descriptor-based preliminary scoring.

This module is intentionally rule-based and explainable. It does not use LLMs,
does not perform final toxicity/safety/regulatory classification, and does not
make recommendations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from services.validator import ValidatedChemical


DISCLAIMER = (
    "Preliminary descriptor-based screening only. This is not a final toxicity, "
    "safety, or regulatory classification."
)


@dataclass(frozen=True)
class ScoreRuleDetail:
    metric: str
    input_value: float | int | str | None
    rule_applied: str
    score_contribution: int | None
    explanation: str
    limitation: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChemicalScore:
    chemical_name: str
    pubchem_cid: int | None
    logp: float | None
    molecular_weight: float | None
    tpsa: float | None
    bioaccumulation_proxy_score: int | None
    size_complexity_proxy_score: int | None
    polarity_proxy_score: int | None
    preliminary_descriptor_risk_score: int | None
    risk_level: str
    score_status: str
    rule_details: list[ScoreRuleDetail]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rule_details"] = [detail.to_dict() for detail in self.rule_details]
        return payload

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "chemical_name": self.chemical_name,
            "pubchem_cid": self.pubchem_cid,
            "logp": self.logp,
            "molecular_weight": self.molecular_weight,
            "tpsa": self.tpsa,
            "preliminary_descriptor_risk_score": self.preliminary_descriptor_risk_score,
            "risk_level": self.risk_level,
            "score_status": self.score_status,
        }


def score_validated_chemicals(chemicals: list[ValidatedChemical]) -> list[ChemicalScore]:
    """Score validated chemicals with transparent descriptor proxy rules."""
    return [score_validated_chemical(chemical) for chemical in chemicals]


def score_validated_chemical(chemical: ValidatedChemical) -> ChemicalScore:
    """Score one validated chemical, or mark it as not scoreable."""
    chemical_name = chemical.canonical_name or chemical.original_surface_form

    if chemical.validation_status != "validated_pubchem_rdkit" or not chemical.canonical_smiles:
        return _cannot_score(
            chemical_name=chemical_name,
            pubchem_cid=chemical.pubchem_cid,
            logp=chemical.logp,
            molecular_weight=chemical.molecular_weight,
            tpsa=chemical.tpsa,
            reason=(
                "Missing validated PubChem/RDKit result or canonical SMILES; descriptor "
                "proxy scoring requires a resolved structure."
            ),
        )

    if chemical.logp is None or chemical.molecular_weight is None or chemical.tpsa is None:
        return _cannot_score(
            chemical_name=chemical_name,
            pubchem_cid=chemical.pubchem_cid,
            logp=chemical.logp,
            molecular_weight=chemical.molecular_weight,
            tpsa=chemical.tpsa,
            reason="One or more required RDKit descriptors are missing.",
        )

    bioaccumulation_detail = _score_logp(chemical.logp)
    size_detail = _score_molecular_weight(chemical.molecular_weight)
    polarity_detail = _score_tpsa(chemical.tpsa)
    total_score = (
        bioaccumulation_detail.score_contribution
        + size_detail.score_contribution
        + polarity_detail.score_contribution
    )

    return ChemicalScore(
        chemical_name=chemical_name,
        pubchem_cid=chemical.pubchem_cid,
        logp=chemical.logp,
        molecular_weight=chemical.molecular_weight,
        tpsa=chemical.tpsa,
        bioaccumulation_proxy_score=bioaccumulation_detail.score_contribution,
        size_complexity_proxy_score=size_detail.score_contribution,
        polarity_proxy_score=polarity_detail.score_contribution,
        preliminary_descriptor_risk_score=total_score,
        risk_level=_risk_level(total_score),
        score_status="scored",
        rule_details=[bioaccumulation_detail, size_detail, polarity_detail],
    )


def _score_logp(logp: float) -> ScoreRuleDetail:
    if logp > 3:
        contribution = 2
        rule = "LogP > 3"
        explanation = "Higher hydrophobicity proxy; may indicate higher bioaccumulation concern."
    elif 1 <= logp <= 3:
        contribution = 1
        rule = "1 <= LogP <= 3"
        explanation = "Moderate hydrophobicity proxy; moderate bioaccumulation concern."
    else:
        contribution = 0
        rule = "LogP < 1"
        explanation = "Lower hydrophobicity proxy; lower bioaccumulation concern by this rule."

    return ScoreRuleDetail(
        metric="bioaccumulation_proxy_score",
        input_value=logp,
        rule_applied=rule,
        score_contribution=contribution,
        explanation=explanation,
        limitation="LogP is only a proxy and does not replace measured bioaccumulation data.",
    )


def _score_molecular_weight(molecular_weight: float) -> ScoreRuleDetail:
    if molecular_weight > 500:
        contribution = 2
        rule = "molecular_weight > 500"
        explanation = "High molecular size/complexity proxy."
    elif molecular_weight >= 300:
        contribution = 1
        rule = "300 <= molecular_weight <= 500"
        explanation = "Moderate molecular size/complexity proxy."
    else:
        contribution = 0
        rule = "molecular_weight < 300"
        explanation = "Lower molecular size/complexity proxy by this rule."

    return ScoreRuleDetail(
        metric="size_complexity_proxy_score",
        input_value=molecular_weight,
        rule_applied=rule,
        score_contribution=contribution,
        explanation=explanation,
        limitation="Molecular weight is a coarse proxy and does not determine hazard or persistence alone.",
    )


def _score_tpsa(tpsa: float) -> ScoreRuleDetail:
    if tpsa < 40:
        contribution = 1
        rule = "TPSA < 40"
        explanation = "Lower polarity proxy; may indicate membrane permeability concern."
    elif tpsa > 140:
        contribution = 1
        rule = "TPSA > 140"
        explanation = "High polarity proxy; may indicate different mobility/handling considerations."
    else:
        contribution = 0
        rule = "40 <= TPSA <= 140"
        explanation = "Moderate polarity proxy by this rule."

    return ScoreRuleDetail(
        metric="polarity_proxy_score",
        input_value=tpsa,
        rule_applied=rule,
        score_contribution=contribution,
        explanation=explanation,
        limitation="TPSA is a structural proxy and not a final permeability, exposure, or toxicity measure.",
    )


def _risk_level(total_score: int) -> str:
    if total_score >= 4:
        return "high"
    if total_score >= 2:
        return "moderate"
    return "low"


def _cannot_score(
    chemical_name: str,
    pubchem_cid: int | None,
    logp: float | None,
    molecular_weight: float | None,
    tpsa: float | None,
    reason: str,
) -> ChemicalScore:
    detail = ScoreRuleDetail(
        metric="preliminary_descriptor_risk_score",
        input_value=None,
        rule_applied="Missing validated structure or descriptors",
        score_contribution=None,
        explanation=reason,
        limitation=DISCLAIMER,
    )
    return ChemicalScore(
        chemical_name=chemical_name,
        pubchem_cid=pubchem_cid,
        logp=logp,
        molecular_weight=molecular_weight,
        tpsa=tpsa,
        bioaccumulation_proxy_score=None,
        size_complexity_proxy_score=None,
        polarity_proxy_score=None,
        preliminary_descriptor_risk_score=None,
        risk_level="not_available",
        score_status="cannot_score",
        rule_details=[detail],
    )
