"""PubChem validation and RDKit descriptor service.

NER finds candidate entities. PubChem and RDKit are the scientific validation
path for this stage; no scoring or recommendations are performed here.
"""

from __future__ import annotations

import json
import string
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Protocol
from urllib.parse import quote

from services.alias_resolver import resolve_alias

import requests
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

from core.config import (
    PIPELINE_PROGRESS_INTERVAL,
    PUBCHEM_CACHE_PATH,
    PUBCHEM_MAX_NETWORK_LOOKUPS,
    PUBCHEM_MAX_RETRIES,
    PUBCHEM_MAX_WORKERS,
    PUBCHEM_REQUEST_TIMEOUT_SECONDS,
    PUBCHEM_TIMEOUT_SECONDS,
    PUBCHEM_VALIDATION_TIMEOUT_SECONDS,
)
from services.chemistry_ner import ChemicalEntity
from services.normalizer import normalize_entities
from services.normalizer import NormalizedChemicalCandidate
from services.pipeline_logger import PipelineLogger


@dataclass(frozen=True)
class PubChemCompoundRecord:
    cid: int | None
    canonical_name: str | None
    molecular_formula: str | None
    canonical_smiles: str | None


@dataclass(frozen=True)
class PubChemLookupPlan:
    unique_candidates: int
    cache_hits: int
    network_lookups_needed: int


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class LocalCandidateGroup:
    normalized_key: str
    lookup_name: str
    entities: tuple[ChemicalEntity, ...]
    aliases: tuple[str, ...]
    pages_found: tuple[int, ...]
    representative_contexts: tuple[str, ...]

    @property
    def occurrence_count(self) -> int:
        return len(self.entities)

    @property
    def representative_entity(self) -> ChemicalEntity:
        return self.entities[0]


@dataclass(frozen=True)
class ValidatedChemical:
    original_surface_form: str
    pubchem_cid: int | None
    canonical_name: str | None
    molecular_formula: str | None
    canonical_smiles: str | None
    validation_status: str
    molecular_weight: float | None
    logp: float | None
    tpsa: float | None
    h_bond_donors: int | None
    h_bond_acceptors: int | None
    source_page: int | None = None
    source_span: tuple[int | None, int | None] | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class PubChemClient(Protocol):
    def lookup_by_name(self, name: str) -> PubChemCompoundRecord | None:
        """Return the best PubChem match for a normalized candidate name."""


class PubChemLookupLimitExceeded(RuntimeError):
    """Raised before network calls when a document has too many unique candidates."""

    def __init__(self, network_lookups_needed: int, max_network_lookups: int) -> None:
        self.network_lookups_needed = network_lookups_needed
        self.max_network_lookups = max_network_lookups
        super().__init__(
            "PubChem lookup limit exceeded: "
            f"{network_lookups_needed} network lookups needed, "
            f"limit is {max_network_lookups}."
        )


class PubChemRateLimitError(RuntimeError):
    """Raised when PubChem returns a rate-limit response."""


class PubChemTimeoutError(TimeoutError):
    """Raised when a PubChem request times out after retries."""


class PubChemPyClient:
    """Production PubChem client using PubChem PUG REST with timeouts."""

    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    
    def lookup_by_cid(self, cid: int) -> PubChemCompoundRecord | None:
        url = (
            f"{self.base_url}/compound/cid/{cid}/property/"
            "IUPACName,MolecularFormula,CanonicalSMILES,ConnectivitySMILES/JSON"
        )

        response_json = self._get_json_with_retry(url)

        if response_json is None:
            return None

        properties = response_json.get("PropertyTable", {}).get("Properties", [])

        if not properties:
            return None

        compound = properties[0]

        return PubChemCompoundRecord(
            cid=compound.get("CID"),
            canonical_name=compound.get("IUPACName"),
            molecular_formula=compound.get("MolecularFormula"),
            canonical_smiles=compound.get("ConnectivitySMILES")
            or compound.get("CanonicalSMILES"),
    )

    def __init__(
        self,
        timeout_seconds: float = PUBCHEM_REQUEST_TIMEOUT_SECONDS,
        max_retries: int = PUBCHEM_MAX_RETRIES,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def lookup_by_name(self, name: str) -> PubChemCompoundRecord | None:
        encoded_name = quote(name, safe="")
        
        alias_match = resolve_alias(name)

        if alias_match is not None:
            try:
                record = self.lookup_by_cid(alias_match.cid)
                print(
                    f"[AliasResolver] '{name}' -> CID {alias_match.cid}",
                    flush=True,
                )

                if record is not None:
                    return record

            except Exception:
                pass
            
        url = (
            f"{self.base_url}/compound/name/{encoded_name}/property/"
            "IUPACName,MolecularFormula,CanonicalSMILES,ConnectivitySMILES/JSON"
        )
        response_json = self._get_json_with_retry(url)
        if response_json is None:
            return None

        properties = response_json.get("PropertyTable", {}).get("Properties", [])
        if not properties:
            return None

        compound = properties[0]
        return PubChemCompoundRecord(
            cid=compound.get("CID"),
            canonical_name=compound.get("IUPACName") or name,
            molecular_formula=compound.get("MolecularFormula"),
            canonical_smiles=compound.get("ConnectivitySMILES") or compound.get("CanonicalSMILES"),
        )

    def _get_json_with_retry(self, url: str) -> dict[str, Any] | None:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.get(url, timeout=self.timeout_seconds)
                if response.status_code == 404:
                    return None
                if response.status_code == 429:
                    last_error = PubChemRateLimitError(
                        "PubChem rate limited the request. Reduce selected candidates or retry later."
                    )
                    raise last_error
                response.raise_for_status()
                return response.json()
            except requests.Timeout:
                last_error = PubChemTimeoutError(
                    f"PubChem request timed out after {self.timeout_seconds}s."
                )
            except requests.RequestException as exc:
                last_error = exc

            if attempt < self.max_retries:
                time.sleep(0.5 * (2**attempt))

        if last_error is not None:
            raise last_error
        return None


_PUBCHEM_CACHE: dict[str, dict[str, Any]] | None = None


def validate_entities_with_pubchem_rdkit(
    entities: list[ChemicalEntity],
    pubchem_client: PubChemClient | None = None,
    logger: PipelineLogger | None = None,
    allow_large_network_lookup: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> list[ValidatedChemical]:
    """Normalize extracted entities, then validate them with PubChem and RDKit."""
    return validate_candidates_with_pubchem_rdkit(
        normalize_entities(entities),
        pubchem_client=pubchem_client,
        logger=logger,
        allow_large_network_lookup=allow_large_network_lookup,
        progress_callback=progress_callback,
    )


def deduplicate_entities_by_pubchem_cid(
    entities: list[ChemicalEntity],
    pubchem_client: PubChemClient | None = None,
    logger: PipelineLogger | None = None,
    allow_large_network_lookup: bool = False,
) -> list[ChemicalEntity]:
    """Deduplicate extracted entities by PubChem CID when PubChem can resolve them."""
    stage = "Deduplication"
    if logger:
        logger.start_stage(stage, metadata={"input_entity_count": len(entities)})

    local_groups = _group_entities_locally(entities)
    if logger:
        logger.message(
            stage,
            f"[PubChem] Unique candidates before lookup: {len(local_groups)}",
            metadata={
                "input_entity_count": len(entities),
                "unique_candidate_count": len(local_groups),
            },
        )

    candidates = [_candidate_from_group(group) for group in local_groups]
    validated_groups = validate_candidates_with_pubchem_rdkit(
        candidates,
        pubchem_client=pubchem_client,
        logger=logger,
        allow_large_network_lookup=allow_large_network_lookup,
    )
    seen = set()
    key_to_index = {}
    deduplicated = []

    for group, validated in zip(local_groups, validated_groups):
        if validated.pubchem_cid is not None:
            key = ("pubchem_cid", validated.pubchem_cid)
        else:
            key = ("surface_form", group.normalized_key)

        if key in seen:
            _merge_candidate_metadata(deduplicated, key_to_index[key], group)
            continue
        seen.add(key)
        key_to_index[key] = len(deduplicated)
        deduplicated.append(_entity_with_group_metadata(group))

    if logger:
        logger.end_stage(
            stage,
            metadata={
                "input_entity_count": len(entities),
                "local_unique_candidate_count": len(local_groups),
                "deduplicated_entity_count": len(deduplicated),
            },
        )

    return deduplicated


def validate_candidates_with_pubchem_rdkit(
    candidates: list[NormalizedChemicalCandidate],
    pubchem_client: PubChemClient | None = None,
    logger: PipelineLogger | None = None,
    allow_large_network_lookup: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> list[ValidatedChemical]:
    """Validate normalized candidates with PubChem and compute RDKit descriptors."""
    client = pubchem_client or PubChemPyClient()
    use_persistent_cache = pubchem_client is None
    candidate_groups = _group_candidates_locally(candidates)
    cache = _load_pubchem_cache() if use_persistent_cache or _PUBCHEM_CACHE is not None else {}
    cached_records: dict[str, tuple[PubChemCompoundRecord | None, str | None]] = {}
    lookup_results: list[tuple[NormalizedChemicalCandidate, PubChemCompoundRecord | None, str | None]] = []

    pubchem_stage = "PubChem validation"
    cache_hits = sum(1 for group in candidate_groups if group.normalized_key in cache)
    network_groups = [group for group in candidate_groups if group.normalized_key not in cache]
    _print_pubchem(f"[PubChem] Unique candidates before lookup: {len(candidate_groups)}")
    _print_pubchem(f"[PubChem] Cache hits: {cache_hits}")
    _print_pubchem(f"[PubChem] Network lookups: {len(network_groups)}")
    _notify_progress(
        progress_callback,
        stage="preflight",
        selected_candidates=len(candidates),
        unique_candidates=len(candidate_groups),
        cache_hits=cache_hits,
        network_lookups=len(network_groups),
        completed=cache_hits,
        total=len(candidate_groups),
        failed=0,
        current_candidate="",
        message="[PubChem] Preflight complete",
    )

    if len(network_groups) > PUBCHEM_MAX_NETWORK_LOOKUPS and not allow_large_network_lookup:
        if logger:
            logger.message(
                pubchem_stage,
                "[PubChem] Network lookup limit exceeded before validation",
                metadata={
                    "network_lookups_needed": len(network_groups),
                    "max_network_lookups": PUBCHEM_MAX_NETWORK_LOOKUPS,
                },
            )
        raise PubChemLookupLimitExceeded(len(network_groups), PUBCHEM_MAX_NETWORK_LOOKUPS)

    if logger:
        logger.start_stage(
            pubchem_stage,
            metadata={
                "candidate_count": len(candidates),
                "unique_candidate_count": len(candidate_groups),
                "cache_hits": cache_hits,
                "network_lookups_needed": len(network_groups),
            },
        )
        logger.message(pubchem_stage, f"[PubChem] Unique candidates before lookup: {len(candidate_groups)}")
        logger.message(pubchem_stage, f"[PubChem] Cache hits: {cache_hits}")
        logger.message(pubchem_stage, f"[PubChem] Network lookups: {len(network_groups)}")
        logger.message(
            pubchem_stage,
            "[PubChem] Preflight complete",
            metadata={
                "selected_candidates": len(candidates),
                "unique_candidate_count": len(candidate_groups),
                "cache_hits": cache_hits,
                "network_lookups_needed": len(network_groups),
            },
        )

    cache_changed = False
    for group in candidate_groups:
        cached_entry = cache.get(group.normalized_key)
        if cached_entry is not None:
            cached_records[group.normalized_key] = _record_from_cache_entry(cached_entry)

    if cache_hits == len(candidate_groups) and logger:
        logger.message(
            pubchem_stage,
            "[PubChem] All selected unique candidates resolved from cache",
            metadata={
                "unique_candidate_count": len(candidate_groups),
                "cache_hits": cache_hits,
            },
        )
    if cache_hits == len(candidate_groups):
        _notify_progress(
            progress_callback,
            stage="cache_complete",
            selected_candidates=len(candidates),
            unique_candidates=len(candidate_groups),
            cache_hits=cache_hits,
            network_lookups=0,
            completed=cache_hits,
            total=len(candidate_groups),
            failed=0,
            current_candidate="",
            message="[PubChem] Validation complete from cache",
        )

    network_results = _lookup_pubchem_network_records(
        network_groups=network_groups,
        client=client,
        logger=logger,
        pubchem_stage=pubchem_stage,
        max_workers=PUBCHEM_MAX_WORKERS,
        progress_callback=progress_callback,
        cache_hits=cache_hits,
        selected_candidate_count=len(candidates),
        unique_candidate_count=len(candidate_groups),
    )
    for group in network_groups:
        record, status = network_results.get(group.normalized_key, (None, "pubchem_lookup_error"))
        cached_records[group.normalized_key] = (record, status)
        if status == "unresolved_pubchem":
            cache[group.normalized_key] = _cache_entry(group.lookup_name, None, "unresolved")
            cache_changed = True
        elif record is not None and status is None:
            cache[group.normalized_key] = _cache_entry(group.lookup_name, record, "resolved")
            cache_changed = True

    if use_persistent_cache and cache_changed:
        _save_pubchem_cache(cache)

    for candidate in candidates:
        normalized_key = _local_candidate_key(candidate.normalized_name)
        record, status = cached_records.get(normalized_key, (None, "unresolved_pubchem"))
        lookup_results.append((candidate, record, status))

    if logger:
        resolved_count = sum(1 for _, record, _ in lookup_results if record is not None)
        logger.end_stage(
            pubchem_stage,
            metadata={
                "candidate_count": len(candidates),
                "unique_candidate_count": len(candidate_groups),
                "resolved_count": resolved_count,
                "cache_hits": cache_hits,
                "network_lookups_needed": len(network_groups),
            },
        )

    rdkit_stage = "RDKit descriptor generation"
    if logger:
        logger.start_stage(rdkit_stage, metadata={"candidate_count": len(candidates)})

    validated = []
    descriptor_count = 0
    for candidate, record, pubchem_status in lookup_results:
        if pubchem_status is not None:
            validated.append(_unresolved_candidate(candidate, pubchem_status))
            continue

        if record is None:
            validated.append(_unresolved_candidate(candidate, "unresolved_pubchem"))
            continue

        if not record.canonical_smiles:
            validated.append(_from_record(candidate, record, "missing_pubchem_smiles"))
            continue

        descriptors = _compute_rdkit_descriptors(record.canonical_smiles)
        if descriptors is None:
            validated.append(_from_record(candidate, record, "invalid_rdkit_smiles"))
            continue

        descriptor_count += 1
        validated.append(
            ValidatedChemical(
                original_surface_form=candidate.surface_form,
                pubchem_cid=record.cid,
                canonical_name=record.canonical_name,
                molecular_formula=record.molecular_formula,
                canonical_smiles=record.canonical_smiles,
                validation_status="validated_pubchem_rdkit",
                molecular_weight=descriptors["molecular_weight"],
                logp=descriptors["logp"],
                tpsa=descriptors["tpsa"],
                h_bond_donors=descriptors["h_bond_donors"],
                h_bond_acceptors=descriptors["h_bond_acceptors"],
                source_page=candidate.source_page,
                source_span=candidate.source_span,
            )
        )

    if logger:
        logger.end_stage(
            rdkit_stage,
            metadata={
                "candidate_count": len(candidates),
                "descriptor_count": descriptor_count,
            },
        )

    return validated


def plan_pubchem_lookup_for_entities(entities: list[ChemicalEntity]) -> PubChemLookupPlan:
    """Estimate PubChem work after local normalization, without network calls."""
    groups = _group_entities_locally(entities)
    cache = _load_pubchem_cache()
    cache_hits = sum(1 for group in groups if group.normalized_key in cache)
    return PubChemLookupPlan(
        unique_candidates=len(groups),
        cache_hits=cache_hits,
        network_lookups_needed=len(groups) - cache_hits,
    )


def _lookup_pubchem_network_records(
    network_groups: list[LocalCandidateGroup],
    client: PubChemClient,
    logger: PipelineLogger | None,
    pubchem_stage: str,
    max_workers: int,
    progress_callback: ProgressCallback | None = None,
    cache_hits: int = 0,
    selected_candidate_count: int | None = None,
    unique_candidate_count: int | None = None,
) -> dict[str, tuple[PubChemCompoundRecord | None, str | None]]:
    if not network_groups:
        return {}

    worker_count = _capped_pubchem_workers(max_workers)
    _print_pubchem(
        f"[PubChem] Running {len(network_groups)} network lookups with {worker_count} workers"
    )
    results: dict[str, tuple[PubChemCompoundRecord | None, str | None]] = {}
    completed = 0
    failed = 0
    _notify_progress(
        progress_callback,
        stage="network_start",
        selected_candidates=selected_candidate_count or len(network_groups),
        unique_candidates=unique_candidate_count or len(network_groups),
        cache_hits=cache_hits,
        network_lookups=len(network_groups),
        completed=cache_hits,
        total=unique_candidate_count or len(network_groups),
        failed=0,
        current_candidate=network_groups[0].lookup_name if network_groups else "",
        message=f"[PubChem] Running {len(network_groups)} network lookups",
    )

    executor = ThreadPoolExecutor(max_workers=worker_count)
    futures = {
        executor.submit(client.lookup_by_name, group.lookup_name): group
        for group in network_groups
    }
    try:
        for future in as_completed(futures, timeout=PUBCHEM_VALIDATION_TIMEOUT_SECONDS):
            group = futures[future]
            completed += 1
            try:
                record = future.result()
            except PubChemRateLimitError as exc:
                _print_pubchem(f"[PubChem] Rate limit for {group.lookup_name}: {exc}")
                results[group.normalized_key] = (None, "pubchem_rate_limited")
                failed += 1
            except PubChemTimeoutError as exc:
                _print_pubchem(f"[PubChem] Timeout for {group.lookup_name}: {exc}")
                results[group.normalized_key] = (None, "pubchem_timeout")
                failed += 1
            except Exception as exc:
                _print_pubchem(f"[PubChem] Lookup error for {group.lookup_name}: {exc}")
                results[group.normalized_key] = (None, "pubchem_lookup_error")
                failed += 1
            else:
                if record is None:
                    results[group.normalized_key] = (None, "unresolved_pubchem")
                else:
                    results[group.normalized_key] = (record, None)

            total_unique = unique_candidate_count or len(network_groups)
            completed_total = cache_hits + completed
            progress_message = f"[PubChem] Completed {completed_total}/{total_unique}"
            _print_pubchem(progress_message)
            if logger and _should_log_progress(completed, len(network_groups)):
                logger.progress(
                    pubchem_stage,
                    progress_message,
                    metadata={
                        "completed": completed,
                        "completed_including_cache": completed_total,
                        "total_unique_candidates": total_unique,
                        "total_network_lookups": len(network_groups),
                        "candidate_name": group.lookup_name,
                        "failed": failed,
                    },
                )
            _notify_progress(
                progress_callback,
                stage="network_progress",
                selected_candidates=selected_candidate_count or len(network_groups),
                unique_candidates=unique_candidate_count or len(network_groups),
                cache_hits=cache_hits,
                network_lookups=len(network_groups),
                completed=completed_total,
                total=total_unique,
                failed=failed,
                current_candidate=group.lookup_name,
                message=progress_message,
            )
    except FutureTimeoutError:
        unfinished = [future for future in futures if not future.done()]
        failed += len(unfinished)
        for future in unfinished:
            group = futures[future]
            future.cancel()
            results[group.normalized_key] = (None, "pubchem_global_timeout")
        timeout_message = (
            "[PubChem] Global validation timeout reached after "
            f"{PUBCHEM_VALIDATION_TIMEOUT_SECONDS}s; "
            f"{len(unfinished)} candidates marked as timed out."
        )
        _print_pubchem(timeout_message)
        if logger:
            logger.message(
                pubchem_stage,
                timeout_message,
                metadata={
                    "timeout_seconds": PUBCHEM_VALIDATION_TIMEOUT_SECONDS,
                    "unfinished_candidates": len(unfinished),
                    "failed": failed,
                },
            )
        _notify_progress(
            progress_callback,
            stage="global_timeout",
            selected_candidates=selected_candidate_count or len(network_groups),
            unique_candidates=unique_candidate_count or len(network_groups),
            cache_hits=cache_hits,
            network_lookups=len(network_groups),
            completed=cache_hits + completed,
            total=unique_candidate_count or len(network_groups),
            failed=failed,
            current_candidate="",
            message=timeout_message,
        )
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    return results


def _notify_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(payload)
    except Exception:
        # Progress reporting must never break scientific validation.
        return


def _capped_pubchem_workers(max_workers: int) -> int:
    return min(10, max(1, int(max_workers)))


def _group_entities_locally(entities: list[ChemicalEntity]) -> list[LocalCandidateGroup]:
    grouped: dict[str, list[ChemicalEntity]] = {}
    lookup_names: dict[str, str] = {}
    for entity in entities:
        candidate = normalize_entities([entity])[0]
        key = _local_candidate_key(candidate.normalized_name)
        if not key:
            continue
        grouped.setdefault(key, []).append(entity)
        lookup_names.setdefault(key, candidate.normalized_name)

    return [
        _build_entity_group(key, lookup_names[key], grouped[key])
        for key in grouped
    ]


def _group_candidates_locally(
    candidates: list[NormalizedChemicalCandidate],
) -> list[LocalCandidateGroup]:
    grouped: dict[str, list[ChemicalEntity]] = {}
    lookup_names: dict[str, str] = {}
    for candidate in candidates:
        key = _local_candidate_key(candidate.normalized_name)
        if not key:
            continue
        entity = ChemicalEntity(
            surface_form=candidate.surface_form,
            entity_type=candidate.entity_type,
            sentence_context="",
            start_char=candidate.source_span[0],
            end_char=candidate.source_span[1],
            page_number=candidate.source_page,
            extraction_method="normalized_candidate",
            confidence=None,
        )
        grouped.setdefault(key, []).append(entity)
        lookup_names.setdefault(key, candidate.normalized_name)

    return [
        _build_entity_group(key, lookup_names[key], grouped[key])
        for key in grouped
    ]


def _build_entity_group(
    normalized_key: str,
    lookup_name: str,
    entities: list[ChemicalEntity],
) -> LocalCandidateGroup:
    aliases = tuple(_unique_ordered(entity.surface_form for entity in entities))
    pages_found = tuple(
        sorted(
            {
                entity.page_number
                for entity in entities
                if entity.page_number is not None
            }
        )
    )
    representative_contexts = tuple(
        _unique_ordered(
            entity.sentence_context
            for entity in entities
            if entity.sentence_context
        )[:3]
    )
    return LocalCandidateGroup(
        normalized_key=normalized_key,
        lookup_name=lookup_name,
        entities=tuple(entities),
        aliases=aliases,
        pages_found=pages_found,
        representative_contexts=representative_contexts,
    )


def _candidate_from_group(group: LocalCandidateGroup) -> NormalizedChemicalCandidate:
    entity = group.representative_entity
    return NormalizedChemicalCandidate(
        surface_form=entity.surface_form,
        normalized_name=group.lookup_name,
        entity_type=entity.entity_type,
        source_page=entity.page_number,
        source_span=(entity.start_char, entity.end_char),
        validation_status="pending_pubchem_validation",
        validation_method="local_pre_pubchem_dedup",
    )


def _entity_with_group_metadata(group: LocalCandidateGroup) -> ChemicalEntity:
    entity = group.representative_entity
    return ChemicalEntity(
        surface_form=entity.surface_form,
        entity_type=entity.entity_type,
        sentence_context=entity.sentence_context,
        start_char=entity.start_char,
        end_char=entity.end_char,
        page_number=entity.page_number,
        extraction_method=entity.extraction_method,
        confidence=entity.confidence,
        occurrence_count=group.occurrence_count,
        aliases=group.aliases,
        pages_found=group.pages_found,
        representative_contexts=group.representative_contexts,
    )


def _merge_candidate_metadata(
    deduplicated: list[ChemicalEntity],
    existing_index: int,
    group: LocalCandidateGroup,
) -> None:
    existing = deduplicated[existing_index]
    aliases = tuple(_unique_ordered([*existing.aliases, *group.aliases]))
    pages_found = tuple(sorted(set(existing.pages_found).union(group.pages_found)))
    representative_contexts = tuple(
        _unique_ordered([*existing.representative_contexts, *group.representative_contexts])[:3]
    )
    deduplicated[existing_index] = ChemicalEntity(
        surface_form=existing.surface_form,
        entity_type=existing.entity_type,
        sentence_context=existing.sentence_context,
        start_char=existing.start_char,
        end_char=existing.end_char,
        page_number=existing.page_number,
        extraction_method=existing.extraction_method,
        confidence=existing.confidence,
        occurrence_count=existing.occurrence_count + group.occurrence_count,
        aliases=aliases,
        pages_found=pages_found,
        representative_contexts=representative_contexts,
    )


def _local_candidate_key(value: str) -> str:
    greek_map = {
        "α": "alpha",
        "β": "beta",
        "γ": "gamma",
        "δ": "delta",
        "ε": "epsilon",
        "κ": "kappa",
        "λ": "lambda",
        "μ": "mu",
        "π": "pi",
        "ω": "omega",
    }
    normalized = value.lower().strip()
    for symbol, replacement in greek_map.items():
        normalized = normalized.replace(symbol, replacement)
    normalized = normalized.translate(str.maketrans({char: " " for char in string.punctuation}))
    return " ".join(normalized.split())


def _unique_ordered(values) -> list:
    seen = set()
    unique = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return unique


def _cache_entry(
    lookup_name: str,
    record: PubChemCompoundRecord | None,
    status: str,
) -> dict[str, Any]:
    return {
        "lookup_name": lookup_name,
        "status": status,
        "record": asdict(record) if record is not None else None,
    }


def _record_from_cache_entry(
    entry: dict[str, Any],
) -> tuple[PubChemCompoundRecord | None, str | None]:
    if entry.get("status") != "resolved" or not entry.get("record"):
        return None, "unresolved_pubchem"

    record = entry["record"]
    return (
        PubChemCompoundRecord(
            cid=record.get("cid"),
            canonical_name=record.get("canonical_name"),
            molecular_formula=record.get("molecular_formula"),
            canonical_smiles=record.get("canonical_smiles"),
        ),
        None,
    )


def _load_pubchem_cache() -> dict[str, dict[str, Any]]:
    global _PUBCHEM_CACHE
    if _PUBCHEM_CACHE is not None:
        return _PUBCHEM_CACHE

    cache_path = Path(PUBCHEM_CACHE_PATH)
    if not cache_path.exists():
        _PUBCHEM_CACHE = {}
        return _PUBCHEM_CACHE

    try:
        with cache_path.open("r", encoding="utf-8") as cache_file:
            loaded = json.load(cache_file)
    except (OSError, json.JSONDecodeError):
        loaded = {}

    _PUBCHEM_CACHE = loaded if isinstance(loaded, dict) else {}
    return _PUBCHEM_CACHE


def _save_pubchem_cache(cache: dict[str, dict[str, Any]]) -> None:
    cache_path = Path(PUBCHEM_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as cache_file:
        json.dump(cache, cache_file, indent=2, sort_keys=True)


def _print_pubchem(message: str) -> None:
    print(message, flush=True)


def _should_log_progress(index: int, total: int) -> bool:
    return total > 0 and (
        index == 1 or index == total or index % PIPELINE_PROGRESS_INTERVAL == 0
    )


def _compute_rdkit_descriptors(smiles: str) -> dict[str, float | int] | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return {
        "molecular_weight": round(float(Descriptors.MolWt(mol)), 4),
        "logp": round(float(Crippen.MolLogP(mol)), 4),
        "tpsa": round(float(rdMolDescriptors.CalcTPSA(mol)), 4),
        "h_bond_donors": int(rdMolDescriptors.CalcNumHBD(mol)),
        "h_bond_acceptors": int(rdMolDescriptors.CalcNumHBA(mol)),
    }


def _unresolved_candidate(
    candidate: NormalizedChemicalCandidate,
    status: str,
) -> ValidatedChemical:
    return ValidatedChemical(
        original_surface_form=candidate.surface_form,
        pubchem_cid=None,
        canonical_name=None,
        molecular_formula=None,
        canonical_smiles=None,
        validation_status=status,
        molecular_weight=None,
        logp=None,
        tpsa=None,
        h_bond_donors=None,
        h_bond_acceptors=None,
        source_page=candidate.source_page,
        source_span=candidate.source_span,
    )


def _from_record(
    candidate: NormalizedChemicalCandidate,
    record: PubChemCompoundRecord,
    status: str,
) -> ValidatedChemical:
    return ValidatedChemical(
        original_surface_form=candidate.surface_form,
        pubchem_cid=record.cid,
        canonical_name=record.canonical_name,
        molecular_formula=record.molecular_formula,
        canonical_smiles=record.canonical_smiles,
        validation_status=status,
        molecular_weight=None,
        logp=None,
        tpsa=None,
        h_bond_donors=None,
        h_bond_acceptors=None,
        source_page=candidate.source_page,
        source_span=candidate.source_span,
    )
