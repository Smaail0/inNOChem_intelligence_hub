"""Smoke test PubChem validation progress events for known candidates."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from services.chemistry_ner import ChemicalEntity  # noqa: E402
from services.pipeline_logger import PipelineEvent, PipelineLogger  # noqa: E402
from services.validator import validate_entities_with_pubchem_rdkit  # noqa: E402


SMOKE_CANDIDATES = [
    "benzene",
    "ethanol",
    "acetone",
    "DMF",
    "nonsense-input",
]


def main() -> None:
    entities = [
        ChemicalEntity(
            surface_form=name,
            entity_type="CHEMICAL_NAME",
            sentence_context="PubChem smoke test candidate.",
            start_char=None,
            end_char=None,
            page_number=None,
            extraction_method="smoke_test",
            confidence=1.0,
        )
        for name in SMOKE_CANDIDATES
    ]

    def log_sink(event: PipelineEvent) -> None:
        print(f"[LOG] {event.event_type.upper()} {event.stage}: {event.message}")

    def progress_callback(payload: dict) -> None:
        print(
            "[PROGRESS] "
            f"stage={payload.get('stage')} "
            f"completed={payload.get('completed')}/{payload.get('total')} "
            f"current={payload.get('current_candidate') or '-'} "
            f"cache_hits={payload.get('cache_hits')} "
            f"network={payload.get('network_lookups')} "
            f"failed={payload.get('failed')} "
            f"message={payload.get('message')}"
        )

    print(f"Validating {len(entities)} smoke candidates with PubChem...")
    validated = validate_entities_with_pubchem_rdkit(
        entities,
        logger=PipelineLogger(sink=log_sink),
        progress_callback=progress_callback,
    )
    print("Results")
    for chemical in validated:
        print(
            f"- {chemical.original_surface_form}: "
            f"status={chemical.validation_status}; "
            f"cid={chemical.pubchem_cid}; "
            f"name={chemical.canonical_name}"
        )


if __name__ == "__main__":
    main()
