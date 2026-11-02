# RAG Quality Report

## Scope

This report diagnoses the current Chemical Intelligence and RAG evidence quality based on the local project state.

Inputs inspected:

- `data/knowledge/gsk_extraction_review.csv`
- `knowledge/providers/gsk_provider.py`
- `services/rag_evidence.py`
- `data/rag/source_docs/`
- Local RAG smoke query: `acetone propanone 67-64-1 180 gsk health_score health`

No LLM, recommendation logic, or PubChem behavior was added or changed.

## GSK Facts Diagnosis

Finding: `gsk_facts = 0` is caused by incomplete GSK data ingestion, not by a disabled provider or matching failure.

Evidence:

- The GSK provider is registered in the Streamlit knowledge aggregator path.
- `GskProvider` loads only rows where `review_status = approved`.
- `data/knowledge/gsk_extraction_review.csv` currently exists but contains `0` rows.
- Because there are no approved rows, the provider index is empty.
- With an empty provider index, matching cannot return GSK facts for any chemical.

Conclusion:

- Provider status: enabled.
- Ingestion status: incomplete.
- Matching status: not meaningfully testable until approved GSK rows exist.

Required next data step:

- Populate `data/knowledge/gsk_extraction_review.csv` with manually reviewed/approved extraction rows, or ingest an official GSK supplementary spreadsheet if available.
- The current project correctly refuses to fabricate values from image-embedded PDF tables.

## CHEM21 Coverage

CHEM21 coverage remains sparse because the local CHEM21 workbook currently contains the alternative/new solvent table, not the full classical solvent guide. This is why common classical solvents may remain absent unless another official CHEM21 source file is added.

## RAG Source Coverage

Current source docs:

- `data/rag/source_docs/SSG-2011-published-paper.pdf`

The current RAG index has limited source diversity. It mainly contains the GSK paper text, which is methodology-heavy and includes figure captions. It does not contain a reviewed machine-readable GSK score table.

Implication:

- RAG can retrieve passages explaining how GSK scores are constructed.
- RAG often cannot retrieve chemical-specific GSK score evidence because those facts are not present as reviewed text/table rows in the indexed source documents.

## RAG Query Improvements Implemented

Previous query construction was too narrow for evidence retrieval.

The query now includes:

- chemical canonical name
- original surface form
- aliases
- PubChem synonyms when available in `chemical_intelligence.json`
- CAS values
- PubChem CID / CID
- provider source
- fact attribute
- topic group
- fact value

This improves exact matching for:

- CAS numbers
- abbreviations
- alternate solvent names
- provider-specific facts
- HSDB topic groups

## Evidence Relevance Diagnostics Implemented

Each retrieved evidence row now includes:

- `contains_chemical_name`
- `contains_cas`
- `contains_topic_keywords`
- `generic_methodology_penalty`
- `relevance_score`

These diagnostics make it visible when a snippet is chemically specific versus merely methodology-adjacent.

## Generic Methodology Penalty

Generic methodology chunks are now penalized when they do not contain the target chemical name or CAS.

Penalty triggers include methodology-oriented phrases such as:

- `methodology`
- `guide are calculated`
- `calculated without`
- `were gaps in the data`
- `life cycle scores`
- `high-level guide`
- `solvent selection guide`

The penalty is not applied when the chunk contains the target chemical name or CAS, because a methodology passage can still be useful if it is chemical-specific.

## Smoke Query Observation

Command:

```powershell
venv310\Scripts\python.exe scripts\query_rag.py "acetone propanone 67-64-1 180 gsk health_score health" --top-k 3
```

Observed behavior:

- BM25 candidates: `3`
- dense candidates: `3`
- merged candidates: `6`
- reranker: disabled
- top result included page 8 of the GSK paper and mentioned acetone recovery/life cycle score.
- other top results were broader GSK health-score methodology passages.

Interpretation:

- Exact chemical terms can retrieve chemical-specific passages when they exist in the paper text.
- Without reviewed GSK score rows or richer source documents, many returned snippets will still be methodology-level evidence.

## Remaining Limitations

- GSK provider cannot return facts until approved GSK extraction rows exist.
- RAG cannot create missing GSK facts and should not be used as a fact generator.
- The GSK PDF contains image/table information that is not reliably extracted as structured text.
- Current RAG source coverage is one GSK paper; it does not yet include CHEM21 methodology PDFs or regulatory source PDFs.
- Dense retrieval can still surface semantically related but non-chemical-specific methodology passages; diagnostics now expose that risk.

## Recommended Next Steps

1. Add reviewed GSK rows or an official GSK supplementary spreadsheet.
2. Add CHEM21 methodology/source documents to `data/rag/source_docs/`.
3. Rebuild the RAG index after adding source documents:

```powershell
venv310\Scripts\python.exe scripts\build_rag_index.py
```

4. Prefer snippets with `contains_chemical_name = true` or `contains_cas = true` for chemical-specific evidence.
5. Treat snippets with high `generic_methodology_penalty` as background methodology evidence, not chemical-specific support.
