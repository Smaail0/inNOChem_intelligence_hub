"""Diagnose CHEM21 provider matching for common solvent identities."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from knowledge.providers.base import identity_terms, normalize_cas, normalize_identity
from knowledge.providers.chem21_provider import CHEMICAL_IDENTITY_ALIASES, Chem21Provider
from services.chemical_intelligence import PubChemSynonymResolver
from services.validator import PubChemPyClient, ValidatedChemical


DIAGNOSTIC_INPUTS = [
    "THF",
    "oxolane",
    "tetrahydrofuran",
    "DMF",
    "N,N-dimethylformamide",
    "NMP",
    "1-methylpyrrolidin-2-one",
    "1,4-dioxane",
    "acetone",
    "ethanol",
]


def main() -> None:
    provider = Chem21Provider()
    pubchem_client = PubChemPyClient()
    synonym_resolver = PubChemSynonymResolver()
    print(f"CHEM21 dataset: {provider.dataset_path}", flush=True)
    print(f"Rows loaded: {len(provider._rows)}", flush=True)
    print(f"Index keys: {len(provider._index)}", flush=True)
    print("", flush=True)

    for value in DIAGNOSTIC_INPUTS:
        identity = _app_like_identity(value, pubchem_client, synonym_resolver)
        _print_report(diagnose_match(provider, value, identity))


def diagnose_match(provider: Chem21Provider, value: str, chemical_identity: dict | None = None) -> dict:
    chemical_identity = chemical_identity or {"original_surface_form": value}
    input_terms = sorted(identity_terms(chemical_identity))
    attempts = _matching_attempts(provider, chemical_identity)
    matched_attempt = next((attempt for attempt in attempts if attempt["matched"]), None)
    facts = provider.get_facts(chemical_identity)
    matched_row = matched_attempt["row"] if matched_attempt else None
    aliases = chemical_identity.get("aliases") or []
    cas_values = chemical_identity.get("cas") or []
    return {
        "input": value,
        "normalized_input": normalize_identity(value),
        "candidate_aliases_used": aliases,
        "cas_values": cas_values,
        "app_identity_keys": sorted(chemical_identity.keys()),
        "app_identity": chemical_identity,
        "matched_row": _row_summary(matched_row),
        "matched_strategy": matched_attempt["strategy"] if matched_attempt else "no_match",
        "matched_key": matched_attempt["normalized_key"] if matched_attempt else None,
        "reason_for_no_match": None if matched_attempt else _no_match_reason(provider, attempts),
        "provider_facts_found": len(facts),
        "fact_attributes": [fact.attribute for fact in facts],
        "uses_chem21_aliases": any(attempt["strategy"] == "alias" for attempt in attempts),
        "uses_pubchem_synonyms": bool(chemical_identity.get("pubchem_synonyms")),
        "uses_cas": any(attempt["strategy"] == "CAS" for attempt in attempts),
        "uses_normalized_names": any(attempt["strategy"] == "name" for attempt in attempts),
        "attempts": [{key: value for key, value in attempt.items() if key != "row"} for attempt in attempts],
    }


def _matching_attempts(provider: Chem21Provider, chemical_identity: dict) -> list[dict]:
    attempts: list[dict] = []
    seen = set()
    alias_terms = _indexed_alias_terms(provider)
    name_terms = {
        normalize_identity(row.get("solvent_name"))
        for row in provider._rows
        if row.get("solvent_name")
    }
    cas_terms = {
        normalize_cas(row.get("cas"))
        for row in provider._rows
        if normalize_cas(row.get("cas"))
    }

    def add(term: str, strategy: str) -> None:
        normalized = normalize_cas(term) or normalize_identity(term)
        if strategy == "name" and normalized in alias_terms and normalized not in name_terms:
            strategy = "alias"
        if normalized in cas_terms:
            strategy = "CAS"
        marker = (normalized, strategy)
        if not normalized or marker in seen:
            return
        seen.add(marker)
        row = provider._index.get(normalized)
        attempts.append(
            {
                "strategy": strategy,
                "term": term,
                "normalized_key": normalized,
                "matched": bool(row),
                "row": row,
            }
        )

    for term in _identity_values(chemical_identity, "original_surface_form", "canonical_name"):
        add(term, "CAS" if normalize_cas(term) else "name")
        normalized = normalize_identity(term)
        for synonym in CHEMICAL_IDENTITY_ALIASES.get(term, set()):
            add(synonym, "synonym")
        for synonym in CHEMICAL_IDENTITY_ALIASES.get(normalized, set()):
            add(synonym, "synonym")

    for term in _identity_values(chemical_identity, "aliases", "pubchem_synonyms"):
        normalized = normalize_identity(term)
        if normalized in alias_terms:
            add(term, "alias")
        else:
            add(term, "PubChem synonym")

    for term in _identity_values(chemical_identity, "cas"):
        add(term, "CAS")

    return attempts


def _identity_values(identity: dict, *keys: str) -> list[str]:
    values = []
    for key in keys:
        value = identity.get(key)
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item) for item in value if item)
        elif value:
            values.append(str(value))
    return _unique_ordered(values)


def _app_like_identity(
    value: str,
    pubchem_client: PubChemPyClient,
    synonym_resolver: PubChemSynonymResolver,
) -> dict:
    identity = {
        "original_surface_form": value,
        "canonical_name": None,
        "pubchem_cid": None,
        "cid": None,
        "cas": [],
        "aliases": [value],
        "pubchem_synonyms": [],
    }
    try:
        record = pubchem_client.lookup_by_name(value)
    except Exception as exc:
        identity["pubchem_lookup_error"] = str(exc)
        return identity

    if record is None:
        identity["pubchem_lookup_status"] = "unresolved"
        return identity

    chemical = ValidatedChemical(
        original_surface_form=value,
        pubchem_cid=record.cid,
        canonical_name=record.canonical_name,
        molecular_formula=record.molecular_formula,
        canonical_smiles=record.canonical_smiles,
        validation_status="validated_pubchem_rdkit",
        molecular_weight=None,
        logp=None,
        tpsa=None,
        h_bond_donors=None,
        h_bond_acceptors=None,
    )
    synonyms = synonym_resolver.get_synonyms(chemical)
    aliases = _unique_ordered([value, record.canonical_name, *synonyms])
    return {
        "original_surface_form": value,
        "canonical_name": record.canonical_name,
        "pubchem_cid": record.cid,
        "cid": record.cid,
        "cas": _cas_values(aliases),
        "aliases": aliases,
        "pubchem_synonyms": synonyms,
        "molecular_formula": record.molecular_formula,
        "canonical_smiles": record.canonical_smiles,
        "validation_status": "validated_pubchem_rdkit",
    }


def _cas_values(values: list[str | None]) -> list[str]:
    cas_values = []
    for value in values:
        if not value:
            continue
        normalized = normalize_cas(value)
        if normalized:
            cas_values.append(normalized)
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


def _no_match_reason(provider: Chem21Provider, attempts: list[dict]) -> str:
    attempted_keys = [attempt["normalized_key"] for attempt in attempts]
    if not provider._rows:
        return "CHEM21 dataset is empty or not loaded."
    if not attempted_keys:
        return "No normalized names, aliases, synonyms, or CAS values were available for matching."
    return (
        "No CHEM21 row matched any normalized input, alias, synonym, or CAS key. "
        "This usually means the current CHEM21 dataset does not contain that compound."
    )


def _indexed_alias_terms(provider: Chem21Provider) -> set[str]:
    aliases = set()
    for row in provider._rows:
        for alias in row.get("aliases", "").replace(";", "|").split("|"):
            normalized = normalize_identity(alias)
            if normalized:
                aliases.add(normalized)
    return aliases


def _row_summary(row: dict | None) -> dict | None:
    if not row:
        return None
    return {
        "solvent_name": row.get("solvent_name"),
        "normalized_name": row.get("normalized_name"),
        "aliases": row.get("aliases"),
        "cas": row.get("cas"),
        "family": row.get("family"),
        "ranking_by_default": row.get("ranking_by_default"),
        "provenance": row.get("provenance"),
    }


def _print_report(report: dict) -> None:
    print(f"Input: {report['input']}", flush=True)
    print(f"  Normalized input: {report['normalized_input']}", flush=True)
    print(f"  App identity keys: {', '.join(report['app_identity_keys'])}", flush=True)
    print(
        f"  Candidate aliases used: {_preview(report['candidate_aliases_used'])}",
        flush=True,
    )
    print(f"  CAS if available: {', '.join(report['cas_values']) or 'none'}", flush=True)
    print(f"  Matched row: {report['matched_row']}", flush=True)
    print(f"  Matching strategy used: {report['matched_strategy']}", flush=True)
    print(f"  Matched key: {report['matched_key']}", flush=True)
    print(f"  Reason for no match: {report['reason_for_no_match'] or 'matched'}", flush=True)
    print(f"  Provider facts found: {report['provider_facts_found']}", flush=True)
    print(f"  Fact attributes: {', '.join(report['fact_attributes']) or 'none'}", flush=True)
    print(
        "  Matching sources checked: "
        f"CHEM21 aliases={report['uses_chem21_aliases']}, "
        f"PubChem synonyms={report['uses_pubchem_synonyms']}, "
        f"CAS={report['uses_cas']}, "
        f"normalized names={report['uses_normalized_names']}",
        flush=True,
    )
    attempts = report["attempts"]
    print(f"  Attempts: {len(attempts)} total", flush=True)
    for attempt in attempts[:30]:
        print(
            f"    - {attempt['strategy']}: {attempt['term']} -> "
            f"{attempt['normalized_key']} | matched={attempt['matched']}",
            flush=True,
        )
    if len(attempts) > 30:
        print(f"    ... {len(attempts) - 30} additional attempts omitted", flush=True)
    print("", flush=True)


def _preview(values: list[str], limit: int = 8) -> str:
    if not values:
        return "none"
    visible = [str(value) for value in values[:limit]]
    suffix = f" ... (+{len(values) - limit} more)" if len(values) > limit else ""
    return ", ".join(visible) + suffix


if __name__ == "__main__":
    main()
