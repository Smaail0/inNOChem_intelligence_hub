from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sys
import time
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    CHEM_NER_SKIP_REFERENCES,
    GEMINI_API_KEY,
    GEMINI_EXPLANATION_TIMEOUT_SECONDS,
    HSDB_ENRICHMENT_DEFAULT,
    LLM_EXPLANATION_WARMUP_ON_START,
    LLM_EXPLANATION_WARMUP_TIMEOUT_SECONDS,
    LLM_CLEANUP_ENABLED,
    PUBCHEM_MAX_NETWORK_LOOKUPS,
    PUBCHEM_REQUEST_TIMEOUT_SECONDS,
    PUBCHEM_VALIDATION_TIMEOUT_SECONDS,
    RAG_ENABLED,
    RAG_MAX_EVIDENCE_QUERIES,
    RAG_TOP_K,
    RAG_UI_USE_DENSE_DEFAULT,
    RAG_USE_RERANKER,
    WARMUP_MODE,
    OLLAMA_MODEL,
    RAG_INDEX_DIR,
    RAG_SOURCE_DOCS_DIR,
)
from knowledge.providers.aggregator import KnowledgeAggregator, KnowledgeProviderRegistry
from knowledge.providers.chem21_provider import Chem21Provider
from knowledge.providers.gsk_provider import GskProvider
from knowledge.providers.hsdb_provider import HsdbProvider
from services.chemistry_ner import extract_chemical_entities
from services.chemistry_ner import TransformerChemicalNERBackend
from services.chemical_intelligence import (
    CHEMICAL_INTELLIGENCE_VERSION,
    NoOpSynonymResolver,
    chemical_intelligence_to_json,
    collect_chemical_intelligence,
)
from services.candidate_review import (
    DEFAULT_PRESELECT_LIMIT,
    apply_candidate_review_selection,
    available_extraction_methods,
    build_candidate_review_records,
    candidate_records_to_rows,
    filter_candidate_review_records,
)

from services.knowledge_coverage import (
    build_knowledge_coverage,
    summarize_coverage,
)

from services.document_extractor import (
    DocumentExtractionError,
    extract_text_from_upload,
    get_text_statistics,
)
from services.human_verification import build_audit_trail, parse_manual_entities
from services.llm_explanation import (
    build_explanation_input,
    build_fallback_explanation,
    generate_chemical_explanation,
    generate_gemini_chemical_explanation,
)
from services.normalizer import normalize_entities
from services.pipeline_logger import PipelineEvent, PipelineLogger
from services.provider_registry import get_provider_registry
from services.rag_evidence import (
    retrieve_evidence_for_intelligence_with_store,
)
from services.resource_manager import (
    get_cached_ner_pipeline,
    get_cached_ollama_status,
    get_cached_rag_store,
    get_cached_reranker,
    rag_index_modified_timestamp,
    warmup_cached_ollama_explanation_model,
)
from services.scorer import DISCLAIMER, score_validated_chemicals
from services.validator import (
    PubChemLookupLimitExceeded,
    plan_pubchem_lookup_for_entities,
    validate_entities_with_pubchem_rdkit,
)

from services.recommendation_confidence import (
    build_recommendation_confidence,
    summarize_recommendation_confidence,
)


st.set_page_config(
    page_title="InNOChem Intelligence Hub",
    layout="wide",
)


ASYNC_GEMINI_MODE = "Fast deterministic + async Gemini polish"


def main() -> None:
    _print_app("[APP] Starting upload pipeline")
    get_provider_registry()
    _apply_workflow_styling()
    st.title("InNOChem Intelligence Hub")
    st.caption(
        "A guided workflow for chemical extraction, human review, validation, intelligence, "
        "evidence, explanation, and preliminary screening."
    )

    _render_ai_system_panel()
    _render_workflow_sidebar()

    _stage_header(
        "1. Upload & Extract Text",
        "Start with a PDF or TXT file. The app extracts structured text locally before any chemical recognition runs.",
    )
    uploaded_file = st.file_uploader(
        "Upload a PDF or TXT document",
        type=["pdf", "txt"],
        help="The app extracts structured text, runs chemistry NER, and prepares entities for validation.",
    )

    if uploaded_file is None:
        _show_sample_output()
        return

    document_key = f"{uploaded_file.name}:{uploaded_file.size}"
    if st.session_state.get("document_key") != document_key:
        for key in (
            "extracted_entities",
            "raw_extracted_entities",
            "candidate_review_records",
            "verification_rows",
            "validated_chemicals",
            "chemical_intelligence",
            "chemical_intelligence_json",
            "chemical_intelligence_version",
            "retrieved_evidence",
            "knowledge_coverage_rows",
            "knowledge_coverage_summary",
            "recommendation_confidence_rows",
            "recommendation_confidence_summary",
            "ai_explanations",
            "ai_explanation_cache",
            "gemini_explanation_job",
            "knowledge_enrichment_status",
            "chemical_scores",
            "human_verification_audit",
            "llm_cleanup_result",
            "llm_cleanup_error",
            "processing_events",
            "timing_summary",
            "pubchem_validation_running",
            "pubchem_validation_status",
            "pubchem_validation_state",
            "pending_pubchem_validation_entities",
        ):
            st.session_state.pop(key, None)
        st.session_state["document_key"] = document_key

    st.session_state["processing_events"] = []
    processing_expander = st.expander("Developer / Processing Log", expanded=False)
    with processing_expander:
        processing_placeholder = st.empty()
        timing_placeholder = st.empty()

    def log_sink(event: PipelineEvent) -> None:
        st.session_state["processing_events"].append(event)
        _render_processing_log(processing_placeholder, st.session_state["processing_events"])

    pipeline_logger = PipelineLogger(sink=log_sink)

    try:
        _print_app("[APP] Before PDF extraction")
        extracted_document = extract_text_from_upload(uploaded_file, logger=pipeline_logger)
    except DocumentExtractionError as exc:
        st.error(str(exc))
        return

    _stage_header(
        "2. Chemical Recognition Setup",
        "Confirm document statistics, choose practical extraction options, then run chemical recognition.",
    )
    stats = get_text_statistics(extracted_document)
    st.write(f"Loaded `{extracted_document.filename}` as `{extracted_document.file_type}`.")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Characters", f"{stats.characters:,}")
    metric_cols[1].metric("Words", f"{stats.words:,}")
    metric_cols[2].metric("Pages", stats.pages)
    metric_cols[3].metric("Sentences", stats.sentences)
    with st.expander("Text preview", expanded=False):
        st.text_area(
            "Preview",
            extracted_document.text[:3000],
            height=220,
            disabled=True,
        )

    with st.expander("Extraction options", expanded=True):
        st.caption(
            "Recommended defaults are selected for normal use. Debug and slower enrichment "
            "options are grouped here so the main workflow stays clean."
        )
        option_col_1, option_col_2 = st.columns(2)
        with option_col_1:
            limit_ner_chunks_for_debugging = st.checkbox(
                "Limit NER chunks for debugging",
                value=stats.pages > 20,
                help="Limits transformer NER to the first configured debug chunks so large PDFs do not appear stuck.",
            )
            skip_references_during_ner = st.checkbox(
                "Skip references section during NER",
                value=CHEM_NER_SKIP_REFERENCES,
                help="Skips References/Bibliography content before transformer NER to reduce noisy citations.",
            )
            allow_large_pubchem_lookup = st.checkbox(
                f"Allow more than {PUBCHEM_MAX_NETWORK_LOOKUPS} PubChem network lookups",
                value=False,
                help="Emergency override for very large candidate sets. Prefer reducing NER scope first.",
            )
        with option_col_2:
            skip_knowledge_enrichment = st.checkbox(
                "Skip knowledge enrichment after PubChem",
                value=False,
                help="Debug toggle: validate with PubChem/RDKit only, then enrich later if needed.",
            )
            include_hsdb_enrichment = st.checkbox(
                "Include HSDB during knowledge enrichment (slower)",
                value=HSDB_ENRICHMENT_DEFAULT,
                help="HSDB requires PubChem PUG-View annotation lookups. Leave off for fast CHEM21/GSK enrichment.",
            )
            fetch_pubchem_synonyms = st.checkbox(
                "Fetch PubChem synonyms during knowledge enrichment (slower)",
                value=False,
                help="Improves provider matching for some chemicals but adds PubChem synonym requests.",
            )
            use_llm_cleanup = st.checkbox(
                "Use optional LLM cleanup on extracted entities",
                value=LLM_CLEANUP_ENABLED,
                help="Disabled by default. If enabled, the LLM receives only extracted candidate entities, never the full document.",
            )
    ner_status_placeholder = st.empty()
    ner_progress_bar = st.progress(0)

    def ner_progress_callback(message: str, current: int, total: int) -> None:
        total = max(total, 1)
        current = min(max(current, 0), total)
        ner_status_placeholder.write(message)
        ner_progress_bar.progress(current / total)

    if st.button("Extract Chemical Entities", type="primary"):
        _print_app("[APP] Before chunk creation and uploaded-document NER")
        try:
            with st.spinner("Loading chemical recognition model..."):
                ner_handle = get_cached_ner_pipeline()
                _record_resource_diagnostics(ner_handle.diagnostics)
                ner_resource = ner_handle.resource
                ner_backend = TransformerChemicalNERBackend(
                    ner_pipeline=ner_resource["pipeline"],
                    tokenizer=ner_resource["tokenizer"],
                )
        except Exception as exc:
            st.error(
                "Could not initialize the chemical recognition model. "
                "Confirm the configured Hugging Face model is available locally or that model download is allowed."
            )
            st.exception(exc)
            return

        with st.spinner("Running chemical recognition over structured document text..."):
            raw_entities = extract_chemical_entities(
                extracted_document,
                preferred_backend=ner_backend,
                logger=pipeline_logger,
                progress_callback=ner_progress_callback,
                limit_chunks_for_debugging=limit_ner_chunks_for_debugging,
                skip_references=skip_references_during_ner,
            )

        if not raw_entities:
            st.warning("No chemical entities were detected.")
            _render_timing_summary(timing_placeholder, pipeline_logger.timing_summary())
            return

        candidate_records = build_candidate_review_records(raw_entities, logger=pipeline_logger)
        st.session_state["raw_extracted_entities"] = raw_entities
        st.session_state["candidate_review_records"] = candidate_records
        st.session_state["verification_rows"] = candidate_records_to_rows(candidate_records)
        st.session_state["extracted_entities"] = [record.to_entity() for record in candidate_records]
        st.session_state.pop("validated_chemicals", None)
        st.session_state.pop("chemical_intelligence", None)
        st.session_state.pop("chemical_intelligence_json", None)
        st.session_state.pop("chemical_intelligence_version", None)
        st.session_state.pop("retrieved_evidence", None)
        st.session_state.pop("knowledge_coverage_rows", None)
        st.session_state.pop("knowledge_coverage_summary", None)
        st.session_state.pop("recommendation_confidence_rows", None)
        st.session_state.pop("recommendation_confidence_summary", None)
        st.session_state.pop("ai_explanations", None)
        st.session_state.pop("ai_explanation_cache", None)
        st.session_state.pop("gemini_explanation_job", None)
        st.session_state.pop("chemical_scores", None)
        st.session_state.pop("human_verification_audit", None)
        st.session_state.pop("llm_cleanup_result", None)
        st.session_state.pop("llm_cleanup_error", None)

        if use_llm_cleanup and candidate_records:
            st.session_state["llm_cleanup_error"] = (
                "LLM cleanup remains disabled for extraction. Use `Generate AI Explanation` "
                "after validation and knowledge coverage when you want a grounded explanation."
            )

        st.session_state["timing_summary"] = pipeline_logger.timing_summary()
        _render_timing_summary(timing_placeholder, st.session_state["timing_summary"])
        ner_progress_bar.progress(1.0)

    candidate_records = st.session_state.get("candidate_review_records", [])
    if not candidate_records:
        return

    _stage_header(
        "3. Human Review",
        "Review locally deduplicated candidates. Uncheck false positives and add missed chemicals before validation.",
    )
    raw_count = len(st.session_state.get("raw_extracted_entities", []))
    candidate_count = len(candidate_records)
    summary_cols = st.columns(3)
    summary_cols[0].metric("Raw mentions", f"{raw_count:,}")
    summary_cols[1].metric("Deduplicated candidates", f"{candidate_count:,}")
    summary_cols[2].metric("Default preselected", min(candidate_count, DEFAULT_PRESELECT_LIMIT))
    if candidate_count > PUBCHEM_MAX_NETWORK_LOOKUPS:
        st.warning(
            f"{candidate_count} candidates detected. PubChem validation is network-bound. "
            "Select the most relevant candidates first."
        )
    if "llm_cleanup_error" in st.session_state:
        st.warning(st.session_state["llm_cleanup_error"])
    if "llm_cleanup_result" in st.session_state:
        with st.expander("Optional LLM Cleanup Output", expanded=False):
            st.json(st.session_state["llm_cleanup_result"])

    with st.expander("Review filters", expanded=True):
        filter_col_1, filter_col_2 = st.columns(2)
        with filter_col_1:
            top_n_candidates = st.number_input(
                "Show top N candidates",
                min_value=1,
                max_value=max(candidate_count, 1),
                value=min(candidate_count, DEFAULT_PRESELECT_LIMIT),
                help="Limits the displayed review table by priority score. All candidates remain available through filters/search.",
            )
            minimum_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05)
            minimum_occurrence_count = st.number_input("Minimum occurrence count", min_value=1, value=1)
            hide_low_information_terms = st.checkbox("Hide low-information terms", value=True)
        with filter_col_2:
            search_text = st.text_input("Search/filter by name", "")
            extraction_method = st.selectbox(
                "Filter by extraction method",
                available_extraction_methods(candidate_records),
            )
            max_page = max(stats.pages, 1)
            if max_page == 1:
                page_start, page_end = 1, 1
                st.caption("Page range filter: page 1 only")
            else:
                page_start, page_end = st.slider(
                    "Filter by page range",
                    min_value=1,
                    max_value=max_page,
                    value=(1, max_page),
                )

    filtered_records = filter_candidate_review_records(
        candidate_records,
        minimum_confidence=minimum_confidence,
        minimum_occurrence_count=int(minimum_occurrence_count),
        hide_low_information_terms=hide_low_information_terms,
        search_text=search_text,
        extraction_method=extraction_method,
        page_start=page_start,
        page_end=page_end,
    )
    filtered_records = filtered_records[: int(top_n_candidates)]
    st.write(f"Showing `{len(filtered_records)}` candidates after local filters.")

    verification_df = pd.DataFrame(candidate_records_to_rows(filtered_records))
    edited_df = st.data_editor(
        verification_df,
        hide_index=True,
        width="stretch",
        disabled=[
            "normalized_name",
            "aliases/surface_forms",
            "occurrence_count",
            "pages_found",
            "representative_contexts",
            "extraction_methods",
            "average_confidence",
            "candidate_priority_score",
            "priority_reason",
            "candidate_type",
            "pubchem_validation_recommended",
            "pubchem_recommendation_reason",
        ],
        column_config={
            "include_for_pubchem_validation": st.column_config.CheckboxColumn(
                "include_for_pubchem_validation",
                default=True,
            )
        },
        key="human_verification_editor",
    )

    manual_input = st.text_input(
        "Add missed chemicals",
        placeholder="benzene, ethanol, dimethylformamide",
        help="Enter comma-separated chemical names missed by extraction.",
    )

    pubchem_status_placeholder = st.empty()
    pubchem_metrics_placeholder = st.empty()
    pubchem_progress_bar = st.progress(0)

    def pubchem_progress_callback(payload: dict) -> None:
        st.session_state["pubchem_validation_state"] = payload
        if payload.get("stage") == "preflight":
            st.session_state["pubchem_validation_status"] = "preflight"
        elif payload.get("stage") in {"network_start", "network_progress"}:
            st.session_state["pubchem_validation_status"] = "running"
        elif payload.get("stage") == "cache_complete":
            st.session_state["pubchem_validation_status"] = "complete"
        elif payload.get("stage") == "global_timeout":
            st.session_state["pubchem_validation_status"] = "failed"
        total = max(int(payload.get("total") or 0), 1)
        completed = min(int(payload.get("completed") or 0), total)
        pubchem_status_placeholder.info(
            payload.get("message")
            or f"Validating selected candidates with PubChem... {completed}/{total}"
        )
        pubchem_progress_bar.progress(completed / total)
        pubchem_metrics_placeholder.dataframe(
            pd.DataFrame(
                [
                    {
                        "selected_candidates": payload.get("selected_candidates"),
                        "unique_candidates": payload.get("unique_candidates"),
                        "current_candidate": payload.get("current_candidate"),
                        "completed": completed,
                        "total": total,
                        "cache_hits": payload.get("cache_hits"),
                        "network_lookups": payload.get("network_lookups"),
                        "failed_lookups": payload.get("failed"),
                        "request_timeout_s": PUBCHEM_REQUEST_TIMEOUT_SECONDS,
                        "global_timeout_s": PUBCHEM_VALIDATION_TIMEOUT_SECONDS,
                    }
                ]
            ),
            hide_index=True,
            width="stretch",
        )

    if st.session_state.get("pubchem_validation_status") == "running":
        pubchem_status_placeholder.info("Validating selected candidates with PubChem...")

    _stage_header(
        "4. Identity Validation",
        "Validate only reviewer-selected candidates with PubChem, then compute RDKit descriptors.",
    )
    if st.button(
        "Validate selected candidates with PubChem",
        type="primary",
        disabled=st.session_state.get("pubchem_validation_status") == "running",
    ):
        st.session_state["pubchem_validation_running"] = True
        st.session_state["pubchem_validation_status"] = "running"
        _print_app("[PubChem] Validate selected candidates button clicked")
        pipeline_logger.message("PubChem validation", "[PubChem] Button clicked")
        selected_entities = apply_candidate_review_selection(
            records=filtered_records,
            edited_rows=edited_df.to_dict("records"),
        )
        manual_entities = parse_manual_entities(manual_input)
        validation_entities = selected_entities + manual_entities
        st.session_state["pending_pubchem_validation_entities"] = validation_entities
        selected_rows_count = int(
            sum(
                1
                for row in edited_df.to_dict("records")
                if bool(row.get("include_for_pubchem_validation", False))
            )
        )
        _print_app(f"[PubChem] Selected rows count: {selected_rows_count}")
        _print_app(f"[PubChem] Candidates passed to validator: {len(validation_entities)}")
        pipeline_logger.message(
            "PubChem validation",
            "[PubChem] Selected candidates prepared",
            metadata={
                "selected_rows_count": selected_rows_count,
                "manual_entity_count": len(manual_entities),
                "candidates_passed_to_validator": len(validation_entities),
            },
        )
        excluded_entities = [
            record.to_entity()
            for record in candidate_records
            if record.normalized_name
            not in {entity.surface_form for entity in selected_entities}
        ]
        st.session_state["human_verification_audit"] = build_audit_trail(
            extracted_entities=[record.to_entity() for record in candidate_records],
            excluded_entities=excluded_entities,
            manually_added_entities=manual_entities,
        )

        if not validation_entities:
            st.session_state["pubchem_validation_running"] = False
            st.session_state["pubchem_validation_status"] = "idle"
            st.warning("No chemicals were selected or manually added for validation.")
            return

        pubchem_status_placeholder.info("Validating selected candidates with PubChem...")
        lookup_plan = plan_pubchem_lookup_for_entities(validation_entities)
        st.session_state["pubchem_validation_status"] = "preflight"
        pubchem_progress_callback(
            {
                "stage": "preflight",
                "selected_candidates": len(validation_entities),
                "unique_candidates": lookup_plan.unique_candidates,
                "current_candidate": "",
                "completed": lookup_plan.cache_hits,
                "total": max(lookup_plan.unique_candidates, 1),
                "cache_hits": lookup_plan.cache_hits,
                "network_lookups": lookup_plan.network_lookups_needed,
                "failed": 0,
                "message": "[PubChem] Preflight complete",
            }
        )
        _print_app(
            "[PubChem] Preflight complete: "
            f"cache_hits={lookup_plan.cache_hits}, "
            f"network_lookups={lookup_plan.network_lookups_needed}"
        )
        pipeline_logger.message(
            "PubChem validation",
            f"[PubChem] Selected candidates before lookup: {lookup_plan.unique_candidates}",
            metadata=lookup_plan.__dict__,
        )
        pipeline_logger.message(
            "PubChem validation",
            "[PubChem] Preflight complete",
            metadata={
                "selected_candidates": len(validation_entities),
                "unique_candidates": lookup_plan.unique_candidates,
                "cache_hits": lookup_plan.cache_hits,
                "network_lookup_count": lookup_plan.network_lookups_needed,
            },
        )
        if len(validation_entities) > PUBCHEM_MAX_NETWORK_LOOKUPS and not allow_large_pubchem_lookup:
            st.session_state["pubchem_validation_running"] = False
            st.session_state["pubchem_validation_status"] = "failed"
            st.warning(
                f"{len(validation_entities)} candidates were selected, which exceeds the "
                f"configured visible safety limit of {PUBCHEM_MAX_NETWORK_LOOKUPS}. "
                "Select fewer candidates or enable the explicit override checkbox."
            )
            _render_timing_summary(timing_placeholder, pipeline_logger.timing_summary())
            return
        if (
            lookup_plan.network_lookups_needed > PUBCHEM_MAX_NETWORK_LOOKUPS
            and not allow_large_pubchem_lookup
        ):
            st.session_state["pubchem_validation_running"] = False
            st.session_state["pubchem_validation_status"] = "failed"
            st.warning(
                "PubChem lookup stopped before network calls. "
                f"{lookup_plan.network_lookups_needed} selected candidates need lookup, "
                f"which exceeds the limit of {PUBCHEM_MAX_NETWORK_LOOKUPS}. "
                "Select fewer candidates or enable the explicit override checkbox."
            )
            _render_timing_summary(timing_placeholder, pipeline_logger.timing_summary())
            return

        normalized_entities = normalize_entities(validation_entities)
        try:
            _print_app("[PubChem] PubChem lookup start")
            _print_app("[PubChem] Entering validation run")
            st.session_state["pubchem_validation_status"] = "running"
            pipeline_logger.message(
                "PubChem validation",
                "[PubChem] Lookup start",
                metadata={
                    "selected_candidates": len(validation_entities),
                    "unique_candidates": lookup_plan.unique_candidates,
                    "cache_hits": lookup_plan.cache_hits,
                    "network_lookups_needed": lookup_plan.network_lookups_needed,
                    "request_timeout_seconds": PUBCHEM_REQUEST_TIMEOUT_SECONDS,
                    "global_timeout_seconds": PUBCHEM_VALIDATION_TIMEOUT_SECONDS,
                },
            )
            pipeline_logger.message(
                "PubChem validation",
                "[PubChem] Entering validation run",
                metadata={
                    "cache_hits": lookup_plan.cache_hits,
                    "network_lookup_count": lookup_plan.network_lookups_needed,
                },
            )
            with st.spinner("Validating entities with PubChem and computing RDKit descriptors..."):
                validated_chemicals = validate_entities_with_pubchem_rdkit(
                    validation_entities,
                    logger=pipeline_logger,
                    allow_large_network_lookup=allow_large_pubchem_lookup,
                    progress_callback=pubchem_progress_callback,
                )
            _print_app("[PubChem] PubChem lookup end")
            _print_app(f"[PubChem] Validation complete; result_count={len(validated_chemicals)}")
            pipeline_logger.message(
                "PubChem validation",
                "[PubChem] Lookup end",
                metadata={"validated_count": len(validated_chemicals)},
            )
            pipeline_logger.message(
                "PubChem validation",
                "[PubChem] Validation complete",
                metadata={"result_count": len(validated_chemicals)},
            )
        except PubChemLookupLimitExceeded as exc:
            st.session_state["pubchem_validation_running"] = False
            st.session_state["pubchem_validation_status"] = "failed"
            st.warning(str(exc))
            _render_timing_summary(timing_placeholder, pipeline_logger.timing_summary())
            return
        except Exception as exc:
            st.session_state["pubchem_validation_running"] = False
            st.session_state["pubchem_validation_status"] = "failed"
            stack_trace = traceback.format_exc()
            _print_app(stack_trace)
            pipeline_logger.message(
                "PubChem validation",
                f"[PubChem] Exception during validation: {exc}",
                metadata={"stack_trace": stack_trace},
            )
            st.error("PubChem validation failed. See the Processing Log and terminal for details.")
            st.exception(exc)
            _render_timing_summary(timing_placeholder, pipeline_logger.timing_summary())
            return
        validated_chemicals = _run_timed_stage(
            "CID deduplication",
            pipeline_logger,
            pubchem_status_placeholder,
            "Checking validated chemicals for CID deduplication...",
            lambda: validated_chemicals,
        )
        st.session_state["validated_chemicals"] = validated_chemicals
        st.session_state["pubchem_validation_running"] = False
        st.session_state["pubchem_validation_status"] = "complete"
        st.session_state.pop("chemical_intelligence", None)
        st.session_state.pop("chemical_intelligence_json", None)
        st.session_state.pop("chemical_intelligence_version", None)
        st.session_state.pop("retrieved_evidence", None)
        st.session_state.pop("knowledge_coverage_rows", None)
        st.session_state.pop("knowledge_coverage_summary", None)
        st.session_state.pop("recommendation_confidence_rows", None)
        st.session_state.pop("recommendation_confidence_summary", None)
        st.session_state.pop("ai_explanations", None)
        st.session_state.pop("ai_explanation_cache", None)
        st.session_state.pop("gemini_explanation_job", None)
        st.session_state["knowledge_enrichment_status"] = (
            "skipped" if skip_knowledge_enrichment else "pending"
        )
        st.session_state["chemical_scores"] = _run_timed_stage(
            "scoring/export/UI rendering",
            pipeline_logger,
            pubchem_status_placeholder,
            "Computing preliminary descriptor screening...",
            lambda: score_validated_chemicals(validated_chemicals),
        )
        st.session_state["timing_summary"] = pipeline_logger.timing_summary()
        _render_timing_summary(timing_placeholder, st.session_state["timing_summary"])
        failed_count = sum(
            1
            for chemical in validated_chemicals
            if chemical.validation_status
            in {
                "pubchem_rate_limited",
                "pubchem_timeout",
                "pubchem_lookup_error",
                "pubchem_global_timeout",
            }
        )
        pubchem_status_placeholder.success(
            (
                "Validation complete from cache: "
                if lookup_plan.network_lookups_needed == 0
                else "PubChem validation finished: "
            )
            + f"{len(validated_chemicals)} candidates processed, "
            f"{failed_count} failed or timed out."
        )
        st.session_state["pubchem_validation_running"] = False
        st.session_state["pubchem_validation_status"] = "complete"

        with st.expander("Prepared candidates sent to PubChem", expanded=False):
            st.dataframe(
                pd.DataFrame(candidate.to_dict() for candidate in normalized_entities),
                hide_index=True,
                width="stretch",
            )

    if "validated_chemicals" in st.session_state:
        st.subheader("Validated Chemicals")
        st.caption("Unresolved entities are kept in the table with an explicit validation status.")
        st.dataframe(
            pd.DataFrame(chemical.to_dict() for chemical in st.session_state["validated_chemicals"]),
            hide_index=True,
            width="stretch",
        )

        _stage_header(
            "5. Chemical Intelligence",
            "Attach provider-backed facts, coverage, and confidence. These are evidence signals, not recommendations.",
        )

        if st.session_state.get("knowledge_enrichment_status") == "skipped":
            st.info("Knowledge enrichment was skipped for this run. Use the button below when ready.")
        if st.button(
            "Run Knowledge Enrichment",
            disabled=st.session_state.get("knowledge_enrichment_status") == "running",
        ):
            st.session_state["knowledge_enrichment_status"] = "running"
            try:
                status = st.empty()
                chemical_intelligence = _run_timed_stage(
                    "Knowledge enrichment orchestration",
                    pipeline_logger,
                    status,
                    "Collecting CHEM21/GSK/HSDB facts...",
                    lambda: collect_chemical_intelligence(
                        st.session_state["validated_chemicals"],
                        aggregator=_get_cached_knowledge_aggregator(include_hsdb_enrichment),
                        synonym_resolver=None if fetch_pubchem_synonyms else NoOpSynonymResolver(),
                        logger=pipeline_logger,
                    ),
                )
                st.session_state["chemical_intelligence"] = chemical_intelligence
                st.session_state["chemical_intelligence_json"] = chemical_intelligence_to_json(
                    chemical_intelligence
                )
                st.session_state["chemical_intelligence_version"] = CHEMICAL_INTELLIGENCE_VERSION
                st.session_state.pop("ai_explanations", None)
                st.session_state.pop("ai_explanation_cache", None)
                st.session_state.pop("gemini_explanation_job", None)
                st.session_state["knowledge_enrichment_status"] = "complete"
                status.success(f"Knowledge enrichment complete: {len(chemical_intelligence)} facts.")
                
                _refresh_coverage_and_confidence()
                
            except Exception as exc:
                st.session_state["knowledge_enrichment_status"] = "failed"
                stack_trace = traceback.format_exc()
                _print_app(stack_trace)
                pipeline_logger.message(
                    "Knowledge Aggregator",
                    f"[Knowledge] Exception during enrichment: {exc}",
                    metadata={"stack_trace": stack_trace},
                )
                st.error("Knowledge enrichment failed. See the Processing Log and terminal.")
                st.exception(exc)

    if "chemical_intelligence_json" in st.session_state:
        _render_chemical_intelligence(
            st.session_state["validated_chemicals"],
            st.session_state["chemical_intelligence_json"],
        )
        
        if "knowledge_coverage_rows" in st.session_state:
            st.subheader("Knowledge Coverage")
            st.caption(
                "Measures how much validated knowledge exists for each chemical across "
                "identity, safety, sustainability, regulatory, and evidence sources."
            )

            summary = st.session_state["knowledge_coverage_summary"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Average Coverage",
                    f"{summary.average_coverage_percent}%"
                )

            with col2:
                st.metric(
                    "High Confidence",
                    summary.high_confidence_count
                )

            with col3:
                st.metric(
                    "Low Confidence",
                    summary.low_confidence_count
                )

            with col4:
                st.metric(
                    "Weak / Identity-only",
                    summary.weak_recommendation_count
                )

            coverage_df = pd.DataFrame(
                [row.to_dict() for row in st.session_state["knowledge_coverage_rows"]]
            )

            display_columns = [
                "chemical",
                "canonical_name",
                "pubchem_cid",
                "coverage_percent",
                "confidence",
                "recommendation_readiness",
                "pubchem_validated",
                "hsdb",
                "chem21",
                "gsk",
                "reach",
                "dot",
                "rag_evidence",
                "missing_groups",
            ]

            available_columns = [
                column
                for column in display_columns
                if column in coverage_df.columns
            ]

            st.dataframe(
                coverage_df[available_columns],
                hide_index=True,
                width="stretch",
            )

            with st.expander("Coverage Details", expanded=False):
                st.dataframe(
                    coverage_df,
                    hide_index=True,
                    width="stretch",
                )
        
        if "recommendation_confidence_rows" in st.session_state:
            st.subheader("Recommendation Confidence")
            st.caption(
                "Estimates how trustworthy a recommendation would be based on identity validation, "
                "safety data, sustainability data, regulatory coverage, and retrieved evidence."
            )

            summary = st.session_state["recommendation_confidence_summary"]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Average Confidence", f"{summary.average_confidence_percent}%")

            with col2:
                st.metric("High", summary.high_count)

            with col3:
                st.metric("Medium", summary.medium_count)

            with col4:
                st.metric("Not Ready", summary.not_ready_count)

            confidence_df = pd.DataFrame(
                [row.to_dict() for row in st.session_state["recommendation_confidence_rows"]]
            )

            display_columns = [
                "chemical",
                "canonical_name",
                "pubchem_cid",
                "confidence_percent",
                "confidence_label",
                "recommendation_status",
                "identity_validated",
                "safety_available",
                "sustainability_available",
                "regulatory_available",
                "evidence_available",
                "missing_for_strong_recommendation",
                "warnings",
            ]

            available_columns = [
                column for column in display_columns
                if column in confidence_df.columns
            ]

            st.dataframe(
                confidence_df[available_columns],
                hide_index=True,
                width="stretch",
            )

        if RAG_ENABLED:
            _stage_header(
                "6. Evidence Retrieval",
                "Retrieve source-document snippets for existing facts. RAG supports explanation; it does not create facts.",
            )
            st.caption(
                "Fast mode uses local BM25 only. Dense mode loads local BGE and is slower, "
                "but can improve semantic matches."
            )
            evidence_col_1, evidence_col_2 = st.columns(2)
            with evidence_col_1:
                rag_use_dense = st.checkbox(
                    "Use dense evidence retrieval with local BGE (slower)",
                    value=RAG_UI_USE_DENSE_DEFAULT,
                    help="When off, evidence retrieval avoids loading the BGE embedding model.",
                )
                rag_use_reranker = st.checkbox(
                    "Use local evidence reranker (slowest)",
                    value=RAG_USE_RERANKER,
                    help="Optional local cross-encoder reranking. Leave off for responsive UI.",
                )
            with evidence_col_2:
                max_evidence_queries = st.number_input(
                    "Max fact queries for this evidence run",
                    min_value=1,
                    max_value=max(len(st.session_state["chemical_intelligence_json"]), 1),
                    value=min(
                        max(len(st.session_state["chemical_intelligence_json"]), 1),
                        RAG_MAX_EVIDENCE_QUERIES,
                    ),
                    help="Bounds the interactive evidence run so large intelligence payloads do not feel stuck.",
                )
                evidence_top_k = st.number_input(
                    "Evidence snippets per fact query",
                    min_value=1,
                    max_value=10,
                    value=RAG_TOP_K,
                )
            evidence_status_placeholder = st.empty()
            evidence_progress_bar = st.progress(0)

            def rag_progress_callback(payload: dict) -> None:
                total = max(int(payload.get("total") or 0), 1)
                completed = min(int(payload.get("completed") or 0), total)
                evidence_status_placeholder.info(
                    payload.get("message")
                    or f"Retrieving evidence snippets... {completed}/{total}"
                )
                evidence_progress_bar.progress(completed / total)
                pipeline_logger.message(
                    "RAG evidence retrieval",
                    payload.get("message") or "[RAG] progress",
                    metadata=payload,
                )

            if st.button("Retrieve Evidence"):
                status = st.empty()
                try:
                    rag_handle = _run_timed_stage(
                        "BGE / embedding model loading"
                        if rag_use_dense
                        else "RAG BM25 index loading",
                        pipeline_logger,
                        status,
                        "Loading local RAG index and embedding model..."
                        if rag_use_dense
                        else "Loading local RAG index without BGE...",
                        lambda: get_cached_rag_store(
                            use_dense=rag_use_dense,
                            index_modified_timestamp=rag_index_modified_timestamp(),
                        ),
                    )
                    _record_resource_diagnostics(rag_handle.diagnostics)
                    rag_store = rag_handle.resource
                    if not rag_store.chunks:
                        st.warning(
                            "RAG index has no source chunks. Add documents under "
                            "`data/rag/source_docs/` and run "
                            "`venv310\\Scripts\\python.exe scripts\\build_rag_index.py`."
                        )
                        return
                    if rag_use_reranker:
                        reranker_handle = _run_timed_stage(
                            "RAG reranker loading",
                            pipeline_logger,
                            status,
                            "Loading local RAG reranker...",
                            get_cached_reranker,
                        )
                        _record_resource_diagnostics(reranker_handle.diagnostics)
                        rag_store.reranker_backend = reranker_handle.resource
                    diagnostics = rag_store.retrieval_diagnostics()
                    pipeline_logger.message(
                        "RAG evidence retrieval",
                        "[RAG] Store loaded",
                        metadata=diagnostics,
                    )
                    evidence_status_placeholder.info(
                        "RAG store loaded: "
                        f"{diagnostics.get('chunk_count')} chunks, "
                        f"dense={'on' if diagnostics.get('dense_enabled') else 'off'}"
                    )
                    st.session_state["retrieved_evidence"] = [
                        evidence.to_dict()
                        for evidence in _run_timed_stage(
                            "RAG evidence retrieval",
                            pipeline_logger,
                            status,
                            "Retrieving local evidence snippets...",
                            lambda: retrieve_evidence_for_intelligence_with_store(
                                st.session_state["chemical_intelligence_json"],
                                rag_store,
                                top_k=int(evidence_top_k),
                                max_queries=int(max_evidence_queries),
                                use_reranker=rag_use_reranker,
                                progress_callback=rag_progress_callback,
                            ),
                        )
                    ]
                    st.session_state.pop("ai_explanations", None)
                    st.session_state.pop("ai_explanation_cache", None)
                    st.session_state.pop("gemini_explanation_job", None)
                    _refresh_coverage_and_confidence()
                    status.success(
                        f"Retrieved {len(st.session_state['retrieved_evidence'])} evidence snippets."
                    )
                except Exception as exc:
                    stack_trace = traceback.format_exc()
                    _print_app(stack_trace)
                    pipeline_logger.message(
                        "RAG evidence retrieval",
                        f"[RAG] Exception during evidence retrieval: {exc}",
                        metadata={"stack_trace": stack_trace},
                    )
                    st.error(
                        "Evidence retrieval failed while preparing AI resources. "
                        "If the RAG index is missing, run "
                        "`venv310\\Scripts\\python.exe scripts\\build_rag_index.py`. "
                        "If dense retrieval is enabled, confirm the embedding model is cached locally."
                    )
                    st.exception(exc)
        else:
            st.info("RAG evidence retrieval is disabled with RAG_ENABLED=false.")

    if st.session_state.get("retrieved_evidence"):
        _render_retrieved_evidence(st.session_state["retrieved_evidence"])

    if "validated_chemicals" in st.session_state:
        _stage_header(
            "7. AI Explanation",
            "Generate a grounded explanation from validated facts and evidence. The explanation layer does not create facts.",
        )
        _render_ai_explanation_controls()

    if "chemical_scores" in st.session_state:
        _stage_header(
            "8. Preliminary Screening",
            "Review deterministic descriptor-based screening. This is not a final safety or regulatory decision.",
        )
        st.warning(DISCLAIMER)
        chemical_scores = st.session_state["chemical_scores"]
        st.dataframe(
            pd.DataFrame(score.to_summary_dict() for score in chemical_scores),
            hide_index=True,
            width="stretch",
        )

        for score in chemical_scores:
            label = (
                f"{score.chemical_name} | CID: {score.pubchem_cid or 'unresolved'} | "
                f"{score.score_status}"
            )
            with st.expander(label):
                st.json([detail.to_dict() for detail in score.rule_details])

    if "human_verification_audit" in st.session_state:
        with st.expander("Developer / Human Verification Audit Trail", expanded=False):
            st.json(st.session_state["human_verification_audit"])

    if "timing_summary" in st.session_state:
        _render_timing_summary(timing_placeholder, st.session_state["timing_summary"])


def _show_sample_output() -> None:
    st.info("Upload a PDF or TXT document to begin.")
    with st.expander("Developer / sample entity JSON", expanded=False):
        st.json(
            [
                {
                    "surface_form": "benzene",
                    "entity_type": "CHEMICAL_NAME",
                    "sentence_context": "Benzene and ethanol were used as solvents.",
                    "start_char": 0,
                    "end_char": 7,
                    "page_number": 1,
                    "extraction_method": "chemistry_lexicon",
                    "confidence": 0.65,
                }
            ]
        )


def _apply_workflow_styling() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1400px;
        }

        /* ── Metric cards: adapt to light & dark themes ── */
        div[data-testid="stMetric"] {
            background: rgba(47, 111, 94, 0.12);
            border: 1px solid rgba(47, 111, 94, 0.35);
            border-radius: 0.75rem;
            padding: 0.75rem 1rem;
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
            color: rgba(180, 215, 200, 0.9) !important;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {
            color: #d4f0e4 !important;
            font-size: 1.6rem;
            font-weight: 700;
        }

        /* ── Stage header banner ── */
        .innochem-stage {
            margin-top: 1.8rem;
            margin-bottom: 0.6rem;
            padding: 0.9rem 1.1rem;
            border-left: 4px solid #3ecf8e;
            background: linear-gradient(
                90deg,
                rgba(47, 111, 94, 0.22) 0%,
                rgba(47, 111, 94, 0.04) 100%
            );
            border-radius: 0 0.75rem 0.75rem 0;
        }
        .innochem-stage h3 {
            margin: 0 0 0.25rem 0;
            color: #3ecf8e;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .innochem-stage p {
            margin: 0;
            color: rgba(200, 230, 215, 0.75);
            font-size: 0.875rem;
            line-height: 1.45;
        }

        /* ── Buttons: primary accent ── */
        div[data-testid="stButton"] > button[kind="primary"] {
            background: #2f6f5e;
            border: 1px solid #3ecf8e;
            color: #d4f0e4;
            font-weight: 600;
            border-radius: 0.5rem;
            transition: background 0.15s, box-shadow 0.15s;
        }
        div[data-testid="stButton"] > button[kind="primary"]:hover {
            background: #3a8a72;
            box-shadow: 0 0 0 3px rgba(62, 207, 142, 0.25);
        }

        /* ── Expanders: subtle tint ── */
        div[data-testid="stExpander"] > details {
            border: 1px solid rgba(62, 207, 142, 0.2);
            border-radius: 0.5rem;
        }

        /* ── Dataframes: minimal border ── */
        div[data-testid="stDataFrame"] {
            border-radius: 0.5rem;
            overflow: hidden;
        }

        /* ── Sidebar progress items ── */
        .wf-stage-row {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.28rem 0;
            font-size: 0.875rem;
        }
        .wf-stage-row .wf-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }
        .wf-dot.done   { background: #3ecf8e; box-shadow: 0 0 5px #3ecf8e88; }
        .wf-dot.pending { background: rgba(200, 200, 200, 0.3); border: 1px solid rgba(200,200,200,0.4); }
        .wf-label.done   { color: #d4f0e4; }
        .wf-label.pending { color: rgba(180, 180, 180, 0.6); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _stage_header(title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="innochem-stage">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_workflow_sidebar() -> None:
    stages = [
        ("Upload", bool(st.session_state.get("document_key"))),
        ("Extract candidates", bool(st.session_state.get("candidate_review_records"))),
        ("Human review", bool(st.session_state.get("human_verification_audit"))),
        ("PubChem/RDKit", bool(st.session_state.get("validated_chemicals"))),
        ("Knowledge", bool(st.session_state.get("chemical_intelligence_json"))),
        ("Evidence", bool(st.session_state.get("retrieved_evidence"))),
        ("AI explanation", bool(st.session_state.get("ai_explanations"))),
        ("Screening", bool(st.session_state.get("chemical_scores"))),
    ]
    with st.sidebar:
        st.subheader("Workflow Progress")
        rows_html = []
        for label, complete in stages:
            state_cls = "done" if complete else "pending"
            rows_html.append(
                f'<div class="wf-stage-row">'
                f'<span class="wf-dot {state_cls}"></span>'
                f'<span class="wf-label {state_cls}">{label}</span>'
                f'</div>'
            )
        st.markdown("\n".join(rows_html), unsafe_allow_html=True)
        st.caption("Complete each stage from top to bottom. Advanced diagnostics are hidden in Developer expanders.")


def _refresh_coverage_and_confidence() -> None:
    if "validated_chemicals" not in st.session_state:
        return
    coverage_rows = build_knowledge_coverage(
        validated_chemicals=st.session_state["validated_chemicals"],
        intelligence_records=st.session_state.get("chemical_intelligence", [])
        or st.session_state.get("chemical_intelligence_json", []),
        evidence_records=st.session_state.get("retrieved_evidence", []),
    )
    confidence_rows = build_recommendation_confidence(coverage_rows)
    st.session_state["knowledge_coverage_rows"] = coverage_rows
    st.session_state["knowledge_coverage_summary"] = summarize_coverage(coverage_rows)
    st.session_state["recommendation_confidence_rows"] = confidence_rows
    st.session_state["recommendation_confidence_summary"] = summarize_recommendation_confidence(
        confidence_rows
    )


def _matching_row(rows, chemical):
    chemical_dict = chemical.to_dict() if hasattr(chemical, "to_dict") else dict(chemical)
    target_cid = chemical_dict.get("pubchem_cid") or chemical_dict.get("cid")
    target_names = {
        _normalize_display_key(chemical_dict.get("canonical_name")),
        _normalize_display_key(chemical_dict.get("original_surface_form")),
        _normalize_display_key(chemical_dict.get("chemical")),
    }
    target_names.discard("")
    for row in rows:
        row_dict = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        row_cid = row_dict.get("pubchem_cid") or row_dict.get("cid")
        if target_cid is not None and row_cid is not None and str(target_cid) == str(row_cid):
            return row
        row_names = {
            _normalize_display_key(row_dict.get("canonical_name")),
            _normalize_display_key(row_dict.get("chemical")),
            _normalize_display_key(row_dict.get("original_surface_form")),
        }
        if target_names.intersection(row_names):
            return row
    return None


def _normalize_display_key(value) -> str:
    if value in (None, ""):
        return ""
    return " ".join(str(value).casefold().split())


def _render_ai_explanations(explanations: list[dict]) -> None:
    for explanation in explanations:
        label = (
            f"{explanation.get('chemical', 'Unknown chemical')} | "
            f"CID: {explanation.get('pubchem_cid') or 'unresolved'} | "
            f"{explanation.get('generated_by', 'unknown')}"
        )
        with st.expander(label):
            diag_col_1, diag_col_2, diag_col_3 = st.columns(3)
            diag_col_1.metric("Mode", explanation.get("mode_used", "unknown"))
            diag_col_2.metric("Generated by", explanation.get("generated_by", "unknown"))
            timeout = explanation.get("timeout_seconds")
            diag_col_3.metric("Timeout", f"{timeout}s" if timeout else "not used")
            st.caption(
                " | ".join(
                    [
                        f"Intelligence rows sent: {explanation.get('intelligence_rows_sent', 0)}",
                        f"Evidence rows sent: {explanation.get('evidence_rows_sent', 0)}",
                        f"Fallback used: {bool(explanation.get('fallback_used', False))}",
                    ]
                )
            )
            st.markdown("**Summary**")
            st.write(explanation.get("summary", ""))
            _render_explanation_list("Key risks", explanation.get("key_risks", []))
            _render_explanation_list(
                "Sustainability notes",
                explanation.get("sustainability_notes", []),
            )
            _render_explanation_list(
                "Regulatory notes",
                explanation.get("regulatory_notes", []),
            )
            _render_explanation_list("Evidence used", explanation.get("evidence_used", []))
            _render_explanation_list(
                "Missing information",
                explanation.get("missing_information", []),
            )
            st.markdown("**Confidence statement**")
            st.write(explanation.get("confidence_statement", ""))
            if explanation.get("warnings"):
                st.warning(" ".join(str(item) for item in explanation["warnings"]))


def _render_explanation_list(title: str, values: list) -> None:
    st.markdown(f"**{title}**")
    if not values:
        st.write("Not available.")
        return
    for value in values:
        st.write(f"- {value}")


def _render_ai_explanation_controls() -> None:
    confidence_rows = st.session_state.get("recommendation_confidence_rows")
    cache = _explanation_cache()
    gemini_job = st.session_state.get("gemini_explanation_job")
    st.caption(
        "Generates grounded explanations from validated data only. Explanations do not create "
        "facts, scores, regulatory status, alternatives, or recommendations."
    )
    with st.expander("Developer / AI Explanation diagnostics", expanded=False):
        st.write(
            {
                "recommendation_confidence_rows_exists": confidence_rows is not None,
                "confidence_row_count": len(confidence_rows or []),
                "cached_explanation_count": len(cache),
                "gemini_api_key_configured": bool(GEMINI_API_KEY),
                "async_gemini_job_running": bool(gemini_job),
            }
        )

    st.write(f"Cached explanations: `{len(cache)}`")
    _render_async_gemini_status()

    if not confidence_rows:
        st.info(
            "Run Knowledge Enrichment and Recommendation Confidence before generating explanations."
        )
        return

    validated_chemicals = st.session_state.get("validated_chemicals", [])
    selectable_chemicals = _chemicals_with_confidence_rows(validated_chemicals)
    if not selectable_chemicals:
        st.info(
            "Recommendation confidence rows exist, but none match the current validated chemicals. "
            "Refresh knowledge enrichment or re-run validation so identities line up."
        )
        return

    labels = [_chemical_option_label(chemical) for chemical in selectable_chemicals]
    selected_label = st.selectbox(
        "Select Chemical",
        labels,
        help="Default mode explains only one selected chemical to keep local Ollama/Gemma responsive.",
    )
    selected_chemical = selectable_chemicals[labels.index(selected_label)]
    cache_key = f"{_explanation_cache_key(selected_chemical)}|mode:{_explanation_mode_key(ASYNC_GEMINI_MODE)}"
    cached = cache.get(cache_key)
    if cached:
        st.success("Existing explanation is cached and will be reused unless you regenerate it.")

    explanation_mode = st.radio(
        "Explanation mode",
        [
            ASYNC_GEMINI_MODE,
            "Fast deterministic explanation",
            "Local LLM polished explanation",
        ],
        index=0 if GEMINI_API_KEY else 1,
        help=(
            "Async Gemini mode shows a deterministic explanation immediately, then polishes it "
            "in the background using only a compact validated evidence packet. Fast deterministic "
            "mode never calls an LLM. Local LLM mode uses Ollama and falls back after 30 seconds."
        ),
    )
    if explanation_mode == ASYNC_GEMINI_MODE and not GEMINI_API_KEY:
        st.warning(
            "Set `GEMINI_API_KEY` before using async Gemini polish. The deterministic explanation "
            "will still work immediately."
        )
    timeout_seconds = 30.0
    regenerate = st.checkbox("Regenerate existing explanation", value=False)
    generate_disabled = bool(
        st.session_state.get("gemini_explanation_job")
        and not st.session_state["gemini_explanation_job"]["future"].done()
    )
    if st.button(
        "Generate AI Explanation",
        key="generate_single_ai_explanation",
        disabled=generate_disabled,
    ):
        if explanation_mode == ASYNC_GEMINI_MODE:
            explanation = _start_async_gemini_explanation(
                selected_chemical,
                timeout_seconds=timeout_seconds,
                regenerate=regenerate,
            )
            st.session_state["ai_explanations"] = _upsert_explanation(
                st.session_state.get("ai_explanations", []),
                explanation,
            )
            if GEMINI_API_KEY:
                st.success(
                    "Deterministic explanation is ready. Gemini is polishing it in the background."
                )
            else:
                st.info("Deterministic explanation is ready. Gemini polish was skipped.")
        else:
            explanation = _get_or_generate_explanation(
                selected_chemical,
                mode=explanation_mode,
                timeout_seconds=timeout_seconds,
                regenerate=regenerate,
            )
            st.session_state["ai_explanations"] = _upsert_explanation(
                st.session_state.get("ai_explanations", []),
                explanation,
            )
            st.success("AI explanation ready.")

    with st.expander("Batch Generate Explanations", expanded=False):
        st.warning(
            "Batch mode can be slow on local Ollama/Gemma. It is capped at 10 chemicals and "
            "only includes validated, evidence-supported candidates. Gemini polishing is single-chemical "
            "and asynchronous; batch mode uses deterministic explanations unless you choose local LLM."
        )
        eligible_chemicals = _eligible_explanation_chemicals(validated_chemicals)
        batch_limit = st.selectbox("Batch limit", ["Top 3", "Top 5", "Top 10"], index=0)
        batch_limit_number = int(str(batch_limit).split()[-1])
        batch_regenerate = st.checkbox("Regenerate cached batch explanations", value=False)
        batch_candidates = eligible_chemicals[:batch_limit_number]
        st.write(f"Eligible batch chemicals: `{len(batch_candidates)}`")
        if st.button("Generate Batch Explanations", key="generate_batch_ai_explanations"):
            status = st.empty()
            explanations = st.session_state.get("ai_explanations", [])
            for index, chemical in enumerate(batch_candidates, start=1):
                status.info(
                    f"Generating explanation {index}/{len(batch_candidates)}: "
                    f"{chemical.canonical_name or chemical.original_surface_form}"
                )
                explanation = _get_or_generate_explanation(
                    chemical,
                    mode=(
                        "Fast deterministic explanation"
                        if explanation_mode == ASYNC_GEMINI_MODE
                        else explanation_mode
                    ),
                    timeout_seconds=timeout_seconds,
                    regenerate=batch_regenerate,
                )
                explanations = _upsert_explanation(explanations, explanation)
            st.session_state["ai_explanations"] = explanations
            status.success(f"Generated/reused {len(batch_candidates)} batch explanations.")

    if st.session_state.get("ai_explanations"):
        st.markdown("### Generated Explanation Output")
        _render_ai_explanations(st.session_state["ai_explanations"])


def _chemicals_with_confidence_rows(validated_chemicals) -> list:
    return [
        chemical
        for chemical in validated_chemicals
        if _matching_row(st.session_state.get("recommendation_confidence_rows", []), chemical)
        is not None
    ]


def _eligible_explanation_chemicals(validated_chemicals) -> list:
    eligible = []
    for chemical in validated_chemicals:
        if chemical.validation_status != "validated_pubchem_rdkit" or chemical.pubchem_cid is None:
            continue
        confidence = _matching_row(
            st.session_state.get("recommendation_confidence_rows", []),
            chemical,
        )
        confidence_dict = confidence.to_dict() if hasattr(confidence, "to_dict") else (confidence or {})
        if confidence_dict.get("confidence_label") == "not_ready":
            continue
        if confidence_dict.get("recommendation_status") == "identity_only_or_weak_support":
            continue
        if not _has_provider_fact_or_evidence(chemical):
            continue
        eligible.append(chemical)
    return eligible


def _has_provider_fact_or_evidence(chemical) -> bool:
    return any(
        _record_matches_validated_chemical(row, chemical)
        for row in st.session_state.get("chemical_intelligence_json", [])
    ) or any(
        _record_matches_validated_chemical(row, chemical)
        for row in st.session_state.get("retrieved_evidence", [])
    )


@st.cache_resource(show_spinner=False)
def _gemini_explanation_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2, thread_name_prefix="gemini_explanation")


def _render_async_gemini_status() -> None:
    job = st.session_state.get("gemini_explanation_job")
    if not job:
        return

    future = job.get("future")
    if future is None:
        st.session_state.pop("gemini_explanation_job", None)
        return

    chemical_name = job.get("chemical_name", "selected chemical")
    elapsed = max(0.0, time.time() - float(job.get("started_at", time.time())))

    if not future.done():
        st.info(
            f"Gemini is polishing the explanation for {chemical_name} in the background "
            f"({elapsed:.0f}s elapsed). The deterministic explanation below is already usable."
        )
        if st.button("Check Gemini result", key="check_async_gemini_result"):
            st.rerun()
        return

    try:
        explanation = future.result().to_dict()
        cache = _explanation_cache()
        cache[job["cache_key"]] = explanation
        st.session_state["ai_explanation_cache"] = cache
        st.session_state["ai_explanations"] = _upsert_explanation(
            st.session_state.get("ai_explanations", []),
            explanation,
        )
        if explanation.get("fallback_used"):
            st.warning(
                "Gemini polish completed with fallback. The deterministic explanation remains available."
            )
            for warning in explanation.get("warnings", []):
                st.error(str(warning))
        else:
            st.success(f"Gemini polished explanation is ready for {chemical_name}.")
    except Exception as exc:
        st.warning(
            "Gemini polish failed in the background. The deterministic explanation remains available."
        )
        st.error(f"{type(exc).__name__}: {exc}")
        _print_app(traceback.format_exc())
    finally:
        st.session_state.pop("gemini_explanation_job", None)


def _start_async_gemini_explanation(
    chemical,
    timeout_seconds: float,
    regenerate: bool = False,
) -> dict:
    cache = _explanation_cache()
    cache_key = f"{_explanation_cache_key(chemical)}|mode:{_explanation_mode_key(ASYNC_GEMINI_MODE)}"
    if not regenerate and cache_key in cache:
        return cache[cache_key]

    explanation_input = _build_explanation_input_for_chemical(chemical)
    deterministic = build_fallback_explanation(explanation_input).to_dict()
    deterministic["mode_used"] = "fast_deterministic_pending_async_gemini"
    deterministic["generated_by"] = "deterministic_fallback"
    deterministic["timeout_seconds"] = None
    deterministic["fallback_used"] = True
    deterministic["warnings"] = [
        *deterministic.get("warnings", []),
        "Gemini polish is asynchronous; this deterministic explanation is shown immediately.",
    ]

    cache[cache_key] = deterministic
    st.session_state["ai_explanation_cache"] = cache

    if GEMINI_API_KEY:
        future = _gemini_explanation_executor().submit(
            generate_gemini_chemical_explanation,
            explanation_input,
            api_key=GEMINI_API_KEY,
            timeout_seconds=min(float(timeout_seconds), GEMINI_EXPLANATION_TIMEOUT_SECONDS),
        )
        st.session_state["gemini_explanation_job"] = {
            "future": future,
            "cache_key": cache_key,
            "chemical_name": deterministic.get("chemical", "selected chemical"),
            "started_at": time.time(),
        }

    return deterministic


def _build_explanation_input_for_chemical(chemical):
    return build_explanation_input(
        validated_chemical=chemical,
        intelligence_records=st.session_state.get("chemical_intelligence_json", []),
        coverage_row=_matching_row(st.session_state.get("knowledge_coverage_rows", []), chemical),
        confidence_row=_matching_row(
            st.session_state.get("recommendation_confidence_rows", []),
            chemical,
        ),
        evidence_records=st.session_state.get("retrieved_evidence", []),
    )


def _get_or_generate_explanation(
    chemical,
    mode: str,
    timeout_seconds: float,
    regenerate: bool = False,
) -> dict:
    cache = _explanation_cache()
    cache_key = f"{_explanation_cache_key(chemical)}|mode:{_explanation_mode_key(mode)}"
    if not regenerate and cache_key in cache:
        return cache[cache_key]

    explanation_input = _build_explanation_input_for_chemical(chemical)
    if mode == "Fast deterministic explanation":
        explanation = build_fallback_explanation(explanation_input).to_dict()
        explanation["mode_used"] = "fast_deterministic_explanation"
        explanation["timeout_seconds"] = None
        explanation["fallback_used"] = True
    elif mode == "Gemini 2.5 Flash polished explanation":
        explanation = generate_gemini_chemical_explanation(
            explanation_input,
            timeout_seconds=min(float(timeout_seconds), GEMINI_EXPLANATION_TIMEOUT_SECONDS),
        ).to_dict()
    else:
        explanation = generate_chemical_explanation(
            explanation_input,
            timeout_seconds=timeout_seconds,
        ).to_dict()
    cache[cache_key] = explanation
    st.session_state["ai_explanation_cache"] = cache
    return explanation


def _explanation_cache() -> dict:
    return st.session_state.setdefault("ai_explanation_cache", {})


def _upsert_explanation(existing: list[dict], explanation: dict) -> list[dict]:
    key = _explanation_dict_key(explanation)
    rows = [row for row in existing if _explanation_dict_key(row) != key]
    rows.append(explanation)
    return rows


def _explanation_cache_key(chemical) -> str:
    cid = getattr(chemical, "pubchem_cid", None)
    if cid is not None:
        return f"cid:{cid}"
    return f"name:{_normalize_display_key(getattr(chemical, 'canonical_name', None) or getattr(chemical, 'original_surface_form', ''))}"


def _explanation_mode_key(mode: str) -> str:
    return _normalize_display_key(mode).replace(" ", "_")


def _explanation_dict_key(explanation: dict) -> str:
    cid = explanation.get("pubchem_cid")
    if cid is not None:
        return f"cid:{cid}"
    return f"name:{_normalize_display_key(explanation.get('chemical'))}"


def _chemical_option_label(chemical) -> str:
    name = chemical.canonical_name or chemical.original_surface_form
    confidence = _matching_row(st.session_state.get("recommendation_confidence_rows", []), chemical)
    confidence_dict = confidence.to_dict() if hasattr(confidence, "to_dict") else (confidence or {})
    return (
        f"{name} | CID {chemical.pubchem_cid} | "
        f"{confidence_dict.get('confidence_label', 'unknown')} | "
        f"{confidence_dict.get('recommendation_status', 'unknown')}"
    )


def _record_matches_validated_chemical(record: dict, chemical) -> bool:
    return _matching_row([record], chemical) is not None


def _record_resource_diagnostics(diagnostics: dict) -> None:
    rows = st.session_state.get("resource_diagnostics", [])
    resource_name = diagnostics.get("resource")
    rows = [row for row in rows if row.get("resource") != resource_name]
    rows.append(diagnostics)
    st.session_state["resource_diagnostics"] = rows


def _print_app(message: str) -> None:
    print(message, flush=True)


def _run_timed_stage(stage: str, logger: PipelineLogger, status_placeholder, message: str, func):
    status_placeholder.info(message)
    _print_app(f"[{stage}] start")
    logger.start_stage(stage, message=f"{stage} started")
    started = time.perf_counter()
    try:
        result = func()
    except Exception:
        duration = round(time.perf_counter() - started, 3)
        _print_app(f"[{stage}] failed after {duration}s")
        logger.end_stage(stage, message=f"{stage} failed", metadata={"status": "error"})
        raise
    duration = round(time.perf_counter() - started, 3)
    _print_app(f"[{stage}] end after {duration}s")
    logger.end_stage(stage, message=f"{stage} completed", metadata={"duration_seconds": duration})
    return result


@st.cache_resource(show_spinner=False)
def _get_cached_knowledge_aggregator(include_hsdb: bool):
    registry = KnowledgeProviderRegistry()
    registry.register(Chem21Provider())
    registry.register(GskProvider())
    if include_hsdb:
        registry.register(HsdbProvider())
    return KnowledgeAggregator(registry)


def _render_processing_log(placeholder, events: list[PipelineEvent]) -> None:
    lines = []
    for event in events[-80:]:
        if event.event_type == "start":
            icon = "⏳"
        elif event.event_type == "end":
            icon = "✓"
        elif event.event_type == "progress":
            icon = "…"
        else:
            icon = "•"

        duration = (
            f" ({event.duration_seconds:.1f}s)"
            if event.duration_seconds is not None
            else ""
        )
        lines.append(f"{icon} {event.message}{duration}")

    placeholder.markdown("\n".join(lines) if lines else "No processing events yet.")


def _render_timing_summary(placeholder, timing_summary: dict[str, float]) -> None:
    if not timing_summary:
        return

    label_map = {
        "PDF extraction": "PDF extraction",
        "TXT extraction": "Text extraction",
        "Transformer NER": "NER",
        "PubChem validation": "PubChem",
        "RDKit descriptor generation": "RDKit",
        "Deduplication": "Deduplication",
        "Human verification preparation": "Human verification preparation",
        "Total": "Total",
    }
    rows = [
        {"Stage": label_map.get(stage, stage), "Duration (s)": duration}
        for stage, duration in timing_summary.items()
    ]
    with placeholder.container():
        st.subheader("Timing Summary")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def _render_chemical_intelligence(validated_chemicals, intelligence_rows) -> None:
    st.subheader("Chemical Intelligence")
    st.caption(
        "Provider facts are displayed with provenance. This section does not make "
        "recommendations, rankings, or regulatory classifications."
    )

    if not intelligence_rows:
        st.info("No CHEM21 or approved GSK facts were found for the validated chemicals.")
        return

    st.dataframe(
        pd.DataFrame(_chemical_intelligence_summary_rows(validated_chemicals, intelligence_rows)),
        hide_index=True,
        width="stretch",
    )

    for chemical in validated_chemicals:
        key = _chemical_identity_key(chemical.to_dict())
        rows = [
            row
            for row in intelligence_rows
            if _chemical_identity_key(row.get("chemical", {})) == key
        ]
        if not rows:
            continue

        display_name = chemical.canonical_name or chemical.original_surface_form
        cas = _first_fact_value(rows, "cas")
        chem21_rows = _source_rows(rows, "chem21")
        gsk_rows = _source_rows(rows, "gsk")

        with st.expander(f"{display_name} | CID: {chemical.pubchem_cid or 'unresolved'}"):
            st.markdown(
                "\n".join(
                    [
                        f"**Canonical name:** {chemical.canonical_name or 'Not resolved'}",
                        f"**Original mention:** {chemical.original_surface_form}",
                        f"**CAS:** {cas or 'Not available from providers'}",
                        f"**PubChem CID:** {chemical.pubchem_cid or 'Unresolved'}",
                    ]
                )
            )

            st.markdown("**CHEM21 Source Facts**")
            if chem21_rows:
                _render_fact_cards(chem21_rows, _CHEM21_DISPLAY_ATTRIBUTES)
                with st.expander("CHEM21 provenance", expanded=False):
                    st.dataframe(
                        pd.DataFrame(_provenance_rows(chem21_rows)),
                        hide_index=True,
                        width="stretch",
                    )
            else:
                st.write("No CHEM21 facts found.")

            st.markdown("**GSK Source Facts**")
            if gsk_rows:
                _render_fact_cards(gsk_rows, _GSK_DISPLAY_ATTRIBUTES)
                with st.expander("GSK provenance", expanded=False):
                    st.dataframe(
                        pd.DataFrame(_provenance_rows(gsk_rows)),
                        hide_index=True,
                        width="stretch",
                    )
            else:
                st.write("No approved GSK facts found.")

    with st.expander("Raw chemical_intelligence.json", expanded=False):
        st.json({"chemical_intelligence.json": intelligence_rows})


@st.cache_data(show_spinner=False, ttl=60)
def _lightweight_ai_system_checks() -> dict:
    rag_index_path = Path(RAG_INDEX_DIR) / "rag_evidence_index.pkl"
    source_docs_path = Path(RAG_SOURCE_DOCS_DIR)
    ollama_status = get_cached_ollama_status()
    return {
        "rag_index_exists": rag_index_path.exists(),
        "rag_index_path": str(rag_index_path),
        "rag_source_docs_exist": source_docs_path.exists()
        and any(source_docs_path.glob("*")),
        "rag_source_docs_path": str(source_docs_path),
        "ollama_status": ollama_status.get("status"),
        "ollama_error": ollama_status.get("error", ""),
        "ollama_model": OLLAMA_MODEL,
    }


def _render_ai_system_panel() -> None:
    with st.sidebar:
        st.subheader("AI System")
        checks = _lightweight_ai_system_checks()
        if checks["rag_index_exists"]:
            st.success("RAG Knowledge Base Ready")
        else:
            st.warning("RAG index missing. Run `venv310\\Scripts\\python.exe scripts\\build_rag_index.py`.")

        if checks["rag_source_docs_exist"]:
            st.success("Source documents available")
        else:
            st.warning("No RAG source documents found in `data/rag/source_docs/`.")

        if checks["ollama_status"] == "available":
            st.success("Ollama Connected")
        else:
            st.warning("Ollama not connected")
            if checks["ollama_error"]:
                st.caption(checks["ollama_error"])

        st.info("Models will load automatically when needed.")

        if (
            LLM_EXPLANATION_WARMUP_ON_START
            and not st.session_state.get("llm_explanation_warmup_complete")
        ):
            _startup_llm_explanation_warmup_internal()
            st.session_state["llm_explanation_warmup_complete"] = True

        warmup_mode = WARMUP_MODE if WARMUP_MODE in {"lazy", "startup"} else "lazy"
        if warmup_mode == "startup" and not st.session_state.get("startup_warmup_complete"):
            _startup_warmup_internal()
            st.session_state["startup_warmup_complete"] = True


def _startup_llm_explanation_warmup_internal() -> None:
    """Deployment-only explanation model warmup; intentionally not exposed as a control."""
    status = st.sidebar.empty()
    status.info("Preparing explanation model...")
    handle = warmup_cached_ollama_explanation_model(
        timeout_seconds=LLM_EXPLANATION_WARMUP_TIMEOUT_SECONDS
    )
    _record_resource_diagnostics(handle.diagnostics)
    if handle.diagnostics.get("status") == "warmed":
        status.success("Explanation model ready")
    else:
        status.warning("Explanation model warmup skipped or failed; fallback remains available.")
        if handle.diagnostics.get("error"):
            st.sidebar.caption(handle.diagnostics["error"])


def _startup_warmup_internal() -> None:
    """Developer/deployment warmup path; intentionally not exposed as UI config."""
    status = st.sidebar.empty()
    try:
        status.info("Preparing AI resources...")
        ner_handle = get_cached_ner_pipeline()
        _record_resource_diagnostics(ner_handle.diagnostics)
        rag_handle = get_cached_rag_store(
            use_dense=True,
            index_modified_timestamp=rag_index_modified_timestamp(),
        )
        _record_resource_diagnostics(rag_handle.diagnostics)
        status.success("AI resources ready.")
    except Exception as exc:
        status.error(f"Resource initialization failed: {exc}")


def _render_retrieved_evidence(evidence_rows: list[dict]) -> None:
    st.subheader("Retrieved Evidence")
    st.caption(
        "Local RAG retrieves supporting source-document snippets for facts that already exist. "
        "It does not create facts, score chemicals, use LLMs, or make recommendations."
    )

    if not evidence_rows:
        st.info(
            "No local evidence snippets were retrieved. Add source PDFs or text files under "
            "`data/rag/source_docs/` and rerun validation to build the local evidence index."
        )
        return

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "chemical": row.get("chemical_name"),
                    "source_type": row.get("source_type"),
                    "fact_attribute": row.get("fact_attribute"),
                    "topic_group": row.get("topic_group"),
                    "document": row.get("document_name"),
                    "page": row.get("page_number"),
                    "retrieval_method": row.get("retrieval_method"),
                    "similarity": row.get("similarity_score"),
                    "relevance": row.get("relevance_score"),
                    "contains_chemical": row.get("contains_chemical_name"),
                    "contains_cas": row.get("contains_cas"),
                    "contains_topic": row.get("contains_topic_keywords"),
                    "generic_penalty": row.get("generic_methodology_penalty"),
                    "snippet": row.get("evidence_snippet"),
                }
                for row in evidence_rows
            ]
        ),
        hide_index=True,
        width="stretch",
    )

    for row in evidence_rows:
        title = (
            f"{row.get('chemical_name')} | {row.get('source_type')} | "
            f"{row.get('fact_attribute')} | {row.get('document_name')}"
        )
        with st.expander(title):
            st.write(row.get("evidence_snippet"))
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "contains_chemical_name": row.get("contains_chemical_name"),
                            "contains_cas": row.get("contains_cas"),
                            "contains_topic_keywords": row.get("contains_topic_keywords"),
                            "generic_methodology_penalty": row.get("generic_methodology_penalty"),
                            "relevance_score": row.get("relevance_score"),
                        }
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
            st.caption(row.get("provenance"))


_CHEM21_DISPLAY_ATTRIBUTES = {
    "category",
    "safety_score",
    "health_score",
    "environment_score",
}
_GSK_DISPLAY_ATTRIBUTES = {
    "waste_score",
    "environmental_impact_score",
    "health_score",
    "flammability_explosion_score",
    "reactivity_stability_score",
    "life_cycle_score",
    "legislation_flag",
    "ehs_red_flag",
}


_ATTRIBUTE_LABELS = {
    "category": "Category",
    "safety_score": "Safety score",
    "health_score": "Health score",
    "environment_score": "Environment score",
    "waste_score": "Waste score",
    "environmental_impact_score": "Environmental impact",
    "flammability_explosion_score": "Flammability / explosion",
    "reactivity_stability_score": "Reactivity / stability",
    "life_cycle_score": "Life cycle",
    "legislation_flag": "Legislation flag",
    "ehs_red_flag": "EHS red flag",
}


def _chemical_intelligence_summary_rows(validated_chemicals, intelligence_rows) -> list[dict]:
    rows = []
    for chemical in validated_chemicals:
        key = _chemical_identity_key(chemical.to_dict())
        chemical_rows = [
            row
            for row in intelligence_rows
            if _chemical_identity_key(row.get("chemical", {})) == key
        ]
        if not chemical_rows:
            continue
        rows.append(
            {
                "chemical": chemical.canonical_name or chemical.original_surface_form,
                "cid": chemical.pubchem_cid,
                "cas": _first_fact_value(chemical_rows, "cas"),
                "chem21_facts": len(_source_rows(chemical_rows, "chem21")),
                "gsk_facts": len(_source_rows(chemical_rows, "gsk")),
                "sources": ", ".join(sorted({row.get("source", "") for row in chemical_rows})),
            }
        )
    return rows


def _chemical_identity_key(identity: dict) -> tuple:
    return (
        identity.get("pubchem_cid"),
        identity.get("canonical_name") or identity.get("original_surface_form"),
    )


def _first_fact_value(rows: list[dict], attribute: str):
    for row in rows:
        if row.get("attribute") == attribute and row.get("value"):
            return row["value"]
    return None


def _source_rows(rows: list[dict], source: str) -> list[dict]:
    return [row for row in rows if row.get("source") == source]


def _render_fact_cards(rows: list[dict], allowed_attributes: set[str]) -> None:
    display_rows = _display_rows(rows, allowed_attributes)
    if not display_rows:
        st.write("No displayable facts found for this provider.")
        return

    columns = st.columns(min(3, len(display_rows)))
    for index, row in enumerate(display_rows):
        with columns[index % len(columns)]:
            st.metric(
                label=_ATTRIBUTE_LABELS.get(row["attribute"], row["attribute"]),
                value=row["value"],
            )
            if row.get("evidence"):
                st.caption(row["evidence"])


def _display_rows(rows: list[dict], allowed_attributes: set[str]) -> list[dict]:
    return [
        {
            "attribute": row.get("attribute"),
            "value": row.get("value"),
            "confidence": row.get("confidence"),
            "evidence": row.get("evidence"),
            "provenance": row.get("provenance"),
        }
        for row in rows
        if row.get("attribute") in allowed_attributes
    ]


def _provenance_rows(rows: list[dict]) -> list[dict]:
    return [
        {
            "attribute": _ATTRIBUTE_LABELS.get(row.get("attribute"), row.get("attribute")),
            "value": row.get("value"),
            "confidence": row.get("confidence"),
            "provenance": row.get("provenance"),
        }
        for row in rows
    ]


if __name__ == "__main__":
    main()