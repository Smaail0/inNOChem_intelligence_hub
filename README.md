# InNOChem Intelligence Hub

Working vertical slice for document-based chemistry entity extraction.

## What It Does

The current Streamlit app lets a user upload a PDF or TXT file, extracts structured text, runs transformer-based chemistry NER, locally deduplicates and ranks candidates, lets a human reviewer select candidates, validates only selected entities with PubChem, computes RDKit descriptors, enriches chemicals with provider facts, estimates knowledge coverage/readiness, and displays grounded explanations with provenance. A React + FastAPI migration scaffold now lives beside the Streamlit app so the project can move toward a clearer production UI without losing existing functionality.

Gemma/Ollama cleanup is optional and disabled by default. The LLM is not the scientific decision-maker and must never receive the full source document in the extraction pipeline.

## Run The App

1. Optional: configure Gemini 2.5 Flash for managed AI explanations:

```powershell
$env:GEMINI_API_KEY="your-google-ai-studio-api-key"
$env:GEMINI_MODEL="gemini-2.5-flash"
```

Or create a local `.env` file from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Then paste your key into `.env`:

```text
GEMINI_API_KEY=your-google-ai-studio-api-key
GEMINI_MODEL=gemini-2.5-flash
```

Restart Streamlit after changing `.env` or PowerShell environment variables.

Gemini explanation mode receives only a compact packet of already validated facts, coverage, confidence, and selected evidence snippets. It does not receive full PDFs or raw document text.

2. Start Ollama only if you plan to enable optional local LLM cleanup or local LLM explanation polish:

```powershell
ollama serve
```

3. Confirm a local Ollama model is available if optional local LLM polish is enabled. The app defaults to:

```powershell
ollama list
```

If your local model uses a different tag, set `OLLAMA_MODEL` before launching Streamlit:

```powershell
$env:OLLAMA_MODEL="llama3.2:1b"
```

4. Install dependencies if needed:

```powershell
venv310\Scripts\python.exe -m pip install -r requirements.txt
```

5. Launch the existing Streamlit app from the project root:

```powershell
venv310\Scripts\python.exe -m streamlit run app/main.py
```

6. Launch the new FastAPI backend from the project root:

```powershell
venv310\Scripts\python.exe -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

7. Launch the new React frontend in a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

The React app expects the backend at `http://127.0.0.1:8000`. You can override this with `frontend/.env` using `VITE_API_BASE_URL`.

Current React/FastAPI migration scope:

- backend status and lightweight resource checks
- PDF/TXT upload through React
- structured text extraction through FastAPI
- document statistics and text preview display
- chemistry NER request from React through FastAPI
- locally deduplicated candidate review rows displayed in React

The full chemistry workflow remains available in Streamlit while editable human review, PubChem/RDKit validation, knowledge enrichment, RAG evidence, explanation, and screening are migrated step by step.

## AI System Behavior

The app keeps startup fast by performing only lightweight checks when the page opens:

- RAG index exists
- source documents are available
- Ollama is reachable when optional LLM functionality is configured
- required local paths are present

Heavy local objects are loaded automatically only when a workflow needs them, then cached through `services/resource_manager.py`:

- transformer NER pipeline
- BGE embedding model
- RAG evidence index
- optional reranker
- optional Ollama health/model status

Users do not need to warm up models manually. The sidebar shows a read-only `AI System` status such as:

- `RAG Knowledge Base Ready`
- `Ollama Connected`
- `Models will load automatically when needed`

When a heavy resource is first needed, the app displays contextual messages such as `Loading chemical recognition model...`, `Loading local RAG index without BGE...`, or `Loading local RAG reranker...`.

`WARMUP_MODE="lazy"` and `WARMUP_MODE="startup"` remain internal deployment settings for developers. They are intentionally not exposed in the user interface.

## Verify The Python 3.10 Environment

Use `venv310` for all project checks:

```powershell
venv310\Scripts\python.exe -m compileall app backend core services knowledge scripts tests
venv310\Scripts\python.exe -m unittest discover -s tests
```

## Current Pipeline

1. Upload a `.pdf` or `.txt` document.
2. Extract structured text locally, including page numbers and sentence contexts.
3. Run transformer-based chemistry NER with `teman67/matscibert-chem-ner`.
4. Locally normalize, deduplicate, rank, and filter raw NER mentions before any PubChem network calls.
5. Review local candidates in the Human Verification table.
6. Select candidates for PubChem validation and manually add missed chemicals.
7. Click `Validate selected candidates with PubChem`.
8. Query PubChem only for reviewer-selected candidates.
9. Validate SMILES and compute descriptors with RDKit.
10. Attach provider-backed Chemical Intelligence facts with provenance.
11. Measure Knowledge Coverage and Recommendation Confidence/readiness.
12. Retrieve optional RAG evidence snippets for existing facts.
13. Generate optional controlled AI explanations from validated data only.
14. Run deterministic preliminary risk and sustainability screening.
15. Display detected entities, validation-ready candidates, audit trail, validated chemicals, Chemical Intelligence, coverage, confidence, retrieved evidence, explanations, and preliminary screening.

Target architecture:

```text
PDF/TXT document
-> structured text extraction
-> transformer chemical NER
-> human verification
-> entity normalization
-> PubChem validation
-> RDKit descriptors
-> knowledge provider enrichment
-> knowledge coverage / confidence readiness
-> evidence-backed explanation
-> deterministic descriptor screening
-> downstream scoring/recommendation
```

The current implementation covers structured text extraction, transformer chemistry NER, local candidate deduplication/ranking/filtering, human verification before PubChem, normalization, PubChem validation for selected candidates only, RDKit descriptors, knowledge provider enrichment, knowledge coverage/readiness, controlled explanations, and preliminary deterministic screening. Recommendation is intentionally not implemented yet.

RDKit is pinned to run with `numpy<2` in `requirements.txt`, and project commands should use the Python 3.10 environment at `venv310`.

## Chemistry NER Backends

The NER layer is pluggable:

- `TransformerChemicalNERBackend`: primary backend using Hugging Face Transformers with `teman67/matscibert-chem-ner`.
- `ChemDataExtractorBackend`: optional fallback when ChemDataExtractor is installed and compatible.
- `LexiconChemistryBackend`: deterministic non-regex fallback when transformer/ChemDataExtractor extraction is unavailable.
- `CasNumberFallbackBackend`: regex fallback only for CAS Registry Numbers.

The transformer backend uses `AutoTokenizer`, `AutoModelForTokenClassification`, and the Transformers `token-classification` pipeline. Extracted document text is grouped by page, then split into tokenizer-aware chunks with overlap so the pipeline stays under transformer token limits without creating thousands of sentence-sized calls. Full documents are not sent to any LLM.

Set the model and chunk size through:

```powershell
$env:CHEM_NER_MODEL_NAME="teman67/matscibert-chem-ner"
$env:CHEM_NER_MAX_TOKENS_PER_CHUNK="384"
$env:CHEM_NER_STRIDE_TOKENS="64"
$env:CHEM_NER_MIN_CHARS_PER_CHUNK="100"
$env:CHEM_NER_BATCH_SIZE="8"
$env:CHEM_NER_SKIP_REFERENCES="true"
$env:CHEM_NER_RESULT_CACHE_DIR="data\cache\ner_cache"
$env:CHEM_NER_CACHE_DIR="models\huggingface"
```

Transformer inference is batched through the local Hugging Face pipeline with `CHEM_NER_BATCH_SIZE`. The app does not use Python threads for transformer inference.

By default, the NER stage skips content after a `References` or `Bibliography` heading to reduce noisy citation extraction. The Streamlit UI exposes this as `Skip references section during NER`.

NER results are cached at `data/cache/ner_cache/`. The cache key includes the document text hash, model name, max tokens per chunk, stride, skip-references flag, and debug chunk limit. On a cache hit, transformer inference is skipped and the app logs `[NER] Loaded entities from cache`.

The transformer backend uses local inference only. It first checks the local Hugging Face cache, then downloads model files once with standard Hugging Face model loading if they are missing. The cache script uses `snapshot_download`, then verifies `AutoTokenizer`, `AutoModelForTokenClassification`, and `pipeline("token-classification")` from local cache. After the model is cached, extraction can run offline. The app does not use the Hugging Face Inference API and does not send document text to Hugging Face or any external LLM service.

To pre-cache and verify the model from the project root:

```powershell
venv310\Scripts\python.exe scripts\cache_chem_ner_model.py
```

Set `CHEM_NER_DOWNLOAD_IF_MISSING=false` to require an already-cached model and prevent download attempts.

HunFlair2 is planned as a future advanced biomedical/chemical NER backend. The current primary backend is local MatSciBERT Chem NER because it supports standard Transformers loading and provides safetensors for more reliable local caching on Windows/Python 3.10.

ChemDataExtractor is not installed by default in this environment. If you install a compatible version later, the service can use it as a fallback without changing the Streamlit UI.

## PubChem Lookup Performance

Raw NER mentions are normalized, deduplicated, ranked, and filtered locally before PubChem is contacted. Local normalization lowercases candidates, strips boundary punctuation, normalizes whitespace, and converts common Greek symbols where possible while preserving original aliases and surface forms for human review.

Before PubChem name lookup, the app can resolve aliases through a local PubChem synonym SQLite database:

- `data/knowledge/pubchem_synonyms.sqlite`
- `services/alias_resolver.py`

This maps aliases such as `dimethyl ketone` or `2-propanone` to PubChem CID before property lookup. If the synonym database is unavailable, validation safely falls back to PubChem name lookup.

PubChem lookup uses:

- One network request per selected unique normalized candidate.
- In-memory cache during the current run.
- Persistent cache at `data/cache/pubchem_cache.json` for resolved and unresolved candidates.
- ThreadPoolExecutor parallelism only for selected network cache misses.
- `PUBCHEM_MAX_WORKERS=6`, capped at 10 to stay rate-limit friendly.
- `PUBCHEM_REQUEST_TIMEOUT_SECONDS=10` and `PUBCHEM_MAX_RETRIES=2` with exponential backoff.
- `PUBCHEM_VALIDATION_TIMEOUT_SECONDS=180` as a global safe timeout for a selected validation batch.
- Emergency stop if more than 200 network lookups are needed unless explicitly overridden in the Streamlit UI.

The Human Verification table receives locally deduplicated candidates with occurrence count, aliases, pages found, representative contexts, extraction methods, average confidence, and candidate priority score. PubChem is network-bound, so large candidate sets should be narrowed before validation.

PubChem progress logs include cache hits, selected network lookups, and completed lookup count. Cached candidates are not parallelized or sent to PubChem again.

To smoke test PubChem progress outside Streamlit:

```powershell
venv310\Scripts\python.exe scripts\validate_pubchem_smoke.py
```

Entity output includes:

```json
[
  {
    "surface_form": "benzene",
    "entity_type": "CHEMICAL_NAME",
    "sentence_context": "Benzene and ethanol were used as solvents.",
    "start_char": 0,
    "end_char": 7,
    "page_number": 1,
    "extraction_method": "transformer_chemical_ner",
    "confidence": 0.97
  }
]
```

## Human Verification

After transformer NER extraction and local candidate review preparation, the reviewer sees `Human Verification — Candidate Chemicals`.

The reviewer can:

- Focus the review table with `Show top N candidates`, defaulting to the top 100 priority-ranked candidates while keeping all candidates available through filters/search.
- Filter by minimum confidence, minimum occurrence count, low-information terms, search text, extraction method, and page range.
- Review normalized name, aliases/surface forms, occurrence count, pages found, representative contexts, extraction methods, average confidence, candidate priority score, and priority reason.
- Review `candidate_type`, `pubchem_validation_recommended`, and the PubChem recommendation reason.
- Uncheck `include_for_pubchem_validation` to remove false positives.
- Add missed chemicals with a comma-separated field, such as `benzene, ethanol, dimethylformamide`.
- Review the audit trail containing extracted entities, excluded entities, manually added entities, and timestamp.

Manually added entities use `extraction_method = "human_added"`, `confidence = 1.0`, and `sentence_context = "Manually added by human reviewer"`.

Only the top 100 locally ranked candidates are preselected for PubChem by default. The reviewer can select more candidates manually, but PubChem/RDKit validation does not run until `Validate selected candidates with PubChem` is clicked.

Candidate priority is deterministic and explainable. It combines occurrence count, average confidence, title/abstract/section-heading/introduction boosts, and unique page count. Each row includes a reason such as `Preselected because found in heading and appears 18 times`.

Candidate type is also deterministic. Specific compounds and abbreviations are recommended for PubChem by default. Generic classes, elements, reaction fragments, and generic terms stay visible but are unchecked by default with reasons such as `Not recommended for PubChem: generic chemical class` or `Not recommended for PubChem: element`.

## Preliminary Risk & Sustainability Screening

After PubChem/RDKit validation, the app computes transparent rule-based proxy scores. No LLM is used for scoring.

The current descriptor rules include:

- `LogP > 3`: higher bioaccumulation concern.
- `1 <= LogP <= 3`: moderate bioaccumulation concern.
- `LogP < 1`: low bioaccumulation concern by this proxy.
- `molecular_weight > 500`: high size/complexity concern.
- `TPSA < 40`: lower polarity / possible membrane permeability concern.
- Missing SMILES or unresolved PubChem entity: `cannot_score`.

Every chemical score includes the metric, input value, rule applied, score contribution, explanation, and limitation. The app displays this detail in an expandable explanation per chemical.

Disclaimer: Preliminary descriptor-based screening only. This is not a final toxicity, safety, or regulatory classification.

## Knowledge Providers

The future Green Chemistry and Regulatory layer uses a pluggable provider architecture under `knowledge/providers/`.

Providers implement `get_facts(chemical_identity)` and return standard facts with:

- `source`
- `attribute`
- `value`
- `confidence`
- `evidence`
- `provenance`

Provider metadata is configured in `config/provider_registry.yaml`. The registry defines provider labels, aliases, and decision categories:

- `identity`
- `safety`
- `sustainability`
- `regulatory`
- `evidence`

Knowledge Coverage, Recommendation Confidence, and AI Explanation read provider categories from this registry instead of hardcoding provider aliases in Python. Adding a future provider should require a YAML registry update plus the provider implementation, not edits to the coverage/confidence/explanation logic.

The provider registry and aggregator can combine facts from local files, APIs, databases, or future graph stores. Providers expose source facts only; they do not make final solvent decisions or recommendations yet.

HSDB is included as a Core Provider architecture layer and can be enabled with:

```powershell
$env:HSDB_ENABLED="true"
```

The HSDB catalog at `data/knowledge/hsdb_topic_catalog.json` is used only as a topic allowlist and priority map. It is not treated as compound-level evidence. Compound facts are retrieved by PubChem CID through the HSDB provider and normalized into `KnowledgeFact` records with source, attribute, value, confidence, evidence, and provenance.

Current HSDB topic groups include:

- Safety Intelligence: GHS Classification, NFPA Hazard Classification, Evidence for Carcinogenicity, Human Toxicity Values, IDLH, TLV, and NIOSH Recommendations.
- Regulatory Intelligence: OSHA Standards, TSCA Requirements, CERCLA Reportable Quantities, RCRA Requirements, Shipment Methods and Regulations, and DOT Emergency Guidelines.
- Environmental Intelligence: Ecotoxicity Values, Environmental Fate, Environmental Biodegradation, Environmental Bioconcentration, and Soil Adsorption/Mobility.

Raw guide sources live under:

- `data/raw/chem21/c5gc01008j1.xlsx`
- `data/raw/gsk/SSG-2011-published-paper.pdf`

Normalized knowledge datasets live under:

- `data/knowledge/chem21_solvents.csv`
- `data/knowledge/gsk_extraction_review.csv`

Run ingestion from the project root with `venv310`:

```powershell
venv310\Scripts\python.exe scripts\ingest_chem21.py
venv310\Scripts\python.exe scripts\ingest_gsk.py
```

To inspect the CHEM21 workbook structure and verify whether specific solvent names or CAS numbers exist in the raw source:

```powershell
venv310\Scripts\python.exe scripts\inspect_chem21_workbook.py
venv310\Scripts\python.exe scripts\debug_provider_matching.py
```

`scripts/ingest_chem21.py` reads the official CHEM21 workbook and writes normalized rows with source, version, reference, solvent identity, CAS, family, boiling/flash points, hazard-code fields, safety/health/environment scores, rankings, notes, and row-level provenance.

The current local CHEM21 workbook `data/raw/chem21/c5gc01008j1.xlsx` contains the alternative/new solvent table, not the full classical solvent guide. Inspection currently finds no workbook rows or XML text for THF/tetrahydrofuran `109-99-9`, DMF `68-12-2`, NMP `872-50-4`, or 1,4-dioxane `123-91-1`. The importer will ingest those rows automatically if a fuller CHEM21 workbook or supplementary sheet is added, but it does not fabricate missing source data.

`scripts/ingest_gsk.py` currently writes a review scaffold only. The available GSK PDF tables are image-embedded, so automatic table extraction is not considered reliable. The script fails clearly with: `Use the official GSK supplementary spreadsheet if available.` Reviewed GSK facts are loaded only when `review_status = approved`.

Provider matching supports normalized name, aliases, CAS numbers, and PubChem canonical names. The scoring layer can consume provider facts as provenance-backed evidence, but final recommendation logic is intentionally not implemented yet.

After PubChem/RDKit validation, the reviewer can run knowledge enrichment for each validated chemical. The `Chemical Intelligence` UI section shows identity, CHEM21 facts, approved GSK facts, evidence, and provenance. A unified `chemical_intelligence.json` payload is also displayed with:

- `chemical`
- `source`
- `attribute`
- `value`
- `confidence`
- `evidence`
- `provenance`

This is factual enrichment only. It does not use LLMs, RAG, recommendations, or final rankings.

## RAG Evidence Layer

The RAG Evidence Layer retrieves supporting passages for facts that already exist in Chemical Intelligence. It does not create facts, score chemicals, use LLMs, or make recommendations.

Source documents live under:

- `data/rag/source_docs/`

The current source folder includes the GSK SSG paper, CHEM21 paper `d4gc01962h.pdf`, and CHEM21 workbook `c5gc01008j1.xlsx`. Add future CHEM21 methodology papers, regulatory PDFs, or other source documents to the same folder.

The local index is stored under:

- `data/rag/index/`

The long-run retrieval architecture is hybrid:

- BM25 keyword retrieval for exact CAS numbers, abbreviations, hazard codes, and solvent names.
- Dense local embeddings with `BAAI/bge-base-en-v1.5` through `sentence-transformers`.
- Optional local reranking with `BAAI/bge-reranker-base`.
- Merge and deduplicate BM25 and dense results before returning snippets.

In the Streamlit UI, evidence retrieval defaults to a fast interactive mode:

- BM25-only retrieval by default, so the BGE embedding model is not loaded unless requested.
- Optional checkbox for dense local BGE retrieval when semantic matching is needed.
- Optional checkbox for local reranking, which is the slowest path.
- A bounded `Max fact queries for this evidence run` control so large Chemical Intelligence payloads do not feel stuck.
- Per-query progress messages in the Processing Log.

Configure with:

```powershell
$env:RAG_ENABLED="true"
$env:RAG_USE_DENSE="true"
$env:RAG_UI_USE_DENSE_DEFAULT="false"
$env:RAG_EMBEDDING_MODEL="BAAI/bge-base-en-v1.5"
$env:RAG_RERANKER_MODEL="BAAI/bge-reranker-base"
$env:RAG_USE_RERANKER="false"
$env:RAG_TOP_K="3"
$env:RAG_MAX_EVIDENCE_QUERIES="25"
```

The index preserves chunk provenance:

- document name
- page number
- spreadsheet sheet index for workbook sources
- section heading when detectable
- chunk text
- character offsets
- source type: methodology, regulatory, literature, or guide

Queries are built from the validated chemical name, provider source, fact attribute, topic group, and fact value. The Streamlit UI displays retrieved snippets in `Retrieved Evidence`.

Build or query the index directly with:

```powershell
venv310\Scripts\python.exe scripts\build_rag_index.py
venv310\Scripts\python.exe scripts\query_rag.py "acetone GSK health score"
```

## Coverage, Confidence, And Explanations

After knowledge enrichment, the app computes a non-recommendation readiness layer:

- `services/knowledge_coverage.py` measures provider completeness for identity, safety, sustainability, regulatory, and evidence coverage.
- `services/recommendation_confidence.py` estimates whether a future recommendation would be trustworthy enough to consider.
- `services/llm_explanation.py` can generate a controlled explanation from already validated facts, coverage, confidence, and retrieved evidence.

Important constraints:

- Recommendation confidence is not a recommendation engine.
- The explanation layer must not invent facts, scores, regulatory status, alternatives, or recommendations.
- Gemini 2.5 Flash is available as a managed polishing backend when `GEMINI_API_KEY` is configured.
- Local Ollama remains available as an optional local explanation backend.
- If Gemini/Ollama is unavailable or returns invalid JSON, the app uses a deterministic fallback explanation.
- Coverage/confidence are recomputed after RAG evidence retrieval so `rag_evidence` reflects the latest retrieved snippets.

The Streamlit button `Generate AI Explanation` appears after validation/coverage data is available. It is user-triggered and does not run automatically after PubChem validation. By default, `Fast deterministic + async Gemini polish` shows a deterministic explanation immediately, then runs Gemini in the background to polish the same validated evidence packet. The reviewer can keep working and use `Check Gemini result` to refresh the polished output when it is ready. Optional batch mode is capped at 10 chemicals and uses deterministic explanations for the async Gemini mode so the app never launches many Gemini calls by default. Generated explanations are cached by PubChem CID or chemical name and reused unless the reviewer chooses to regenerate them.

## Limitations

- Unresolved entities are kept in the validation table with an explicit status.
- PubChem/RDKit validation only receives human-selected and manually added entities.
- Human verification happens on locally deduplicated candidates, not repeated raw mentions.
- NER confidence values are extraction confidence only, not scientific validity.
- No RAG, recommendations, final safety assessment, or regulatory classification is implemented yet.
- Preliminary screening is descriptor-proxy based and should not be treated as a final decision.
- Optional LLM cleanup is disabled by default and receives only extracted candidate entities, never full document text.




