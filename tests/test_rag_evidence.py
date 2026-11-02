import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np

import services.rag_evidence as rag_evidence
from services.rag_evidence import (
    EvidenceChunk,
    RagEvidenceIndex,
    RetrievalResult,
    build_evidence_query,
    chunk_source_documents,
    evidence_relevance_diagnostics,
    merge_retrieval_results,
    retrieve_evidence_for_intelligence,
    retrieve_evidence_for_intelligence_with_store,
)


class FakeEmbeddingBackend:
    model_name = "fake-local-embedding"

    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.casefold()
            if any(term in lowered for term in ("flammable", "combustion", "ignition", "fire")):
                vectors.append([1.0, 0.0, 0.0])
            elif any(term in lowered for term in ("aquatic", "ecotoxicity", "fish")):
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.array(vectors)


class RagEvidenceTests(unittest.TestCase):
    def test_chunks_preserve_page_provenance_and_offsets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source_docs"
            source_dir.mkdir()
            (source_dir / "chem21_methodology.txt").write_text(
                "CHEM21 METHODOLOGY. Acetone is discussed as a solvent.",
                encoding="utf-8",
            )

            chunks = chunk_source_documents(source_dir)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].source_name, "chem21_methodology.txt")
        self.assertIsNone(chunks[0].page_number)
        self.assertEqual(chunks[0].start_char, 0)
        self.assertGreater(chunks[0].end_char, chunks[0].start_char)
        self.assertEqual(chunks[0].source_type, "methodology")

    def test_bm25_finds_exact_cas_and_chemical_names(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source_docs"
            index_dir = Path(temp_dir) / "index"
            source_dir.mkdir()
            (source_dir / "regulatory.txt").write_text(
                "CAS 67-64-1 identifies acetone in regulatory exposure documentation.\n"
                "This unrelated paragraph discusses water.",
                encoding="utf-8",
            )

            store = RagEvidenceIndex(source_dir, index_dir, use_dense=False).load_or_build()
            results = store.query("67-64-1 acetone", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].retrieval_method, "bm25")
        self.assertIn("67-64-1", results[0].chunk.chunk_text)

    def test_spreadsheet_source_docs_are_chunked_for_rag(self):
        pd = __import__("pandas")
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source_docs"
            source_dir.mkdir()
            workbook_path = source_dir / "chem21_source.xlsx"
            pd.DataFrame(
                [
                    {
                        "solvent_name": "acetone",
                        "cas": "67-64-1",
                        "final_ranking": "recommended",
                    }
                ]
            ).to_excel(workbook_path, index=False)

            chunks = chunk_source_documents(source_dir)

        self.assertTrue(chunks)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertIn("Spreadsheet sheet:", chunks[0].chunk_text)
        self.assertIn("67-64-1", chunks[0].chunk_text)

    def test_dense_retrieval_returns_semantic_matches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source_docs"
            index_dir = Path(temp_dir) / "index"
            source_dir.mkdir()
            (source_dir / "guide.txt").write_text(
                "This solvent has a low flash point and creates ignition risk.\n"
                "The second note is about fish and aquatic toxicity.",
                encoding="utf-8",
            )

            store = RagEvidenceIndex(
                source_dir,
                index_dir,
                embedding_backend=FakeEmbeddingBackend(),
            ).load_or_build()
            results = store.query("flammable combustion hazard", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertIn(results[0].retrieval_method, {"dense", "hybrid"})
        self.assertIn("ignition", results[0].chunk.chunk_text)

    def test_merged_results_deduplicate(self):
        chunk = self._chunk("same", "Acetone evidence.")
        bm25 = [RetrievalResult(chunk=chunk, score=1.2, retrieval_method="bm25", bm25_score=1.2)]
        dense = [RetrievalResult(chunk=chunk, score=0.8, retrieval_method="dense", dense_score=0.8)]

        merged = merge_retrieval_results(bm25, dense)

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].retrieval_method, "hybrid")
        self.assertAlmostEqual(merged[0].score, 2.0)

    def test_returned_evidence_contains_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source_docs"
            index_dir = Path(temp_dir) / "index"
            source_dir.mkdir()
            (source_dir / "gsk_ssg.txt").write_text(
                "The GSK solvent selection guide includes health score evidence for acetone.",
                encoding="utf-8",
            )

            evidence = retrieve_evidence_for_intelligence(
                [self._intelligence_row()],
                source_docs_dir=source_dir,
                index_dir=index_dir,
                top_k=1,
                embedding_backend=FakeEmbeddingBackend(),
            )

        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].chemical_name, "acetone")
        self.assertIn("gsk_ssg.txt", evidence[0].provenance)
        self.assertIn("chars=", evidence[0].provenance)
        self.assertIn("source_type=guide", evidence[0].provenance)

    def test_interactive_retrieval_can_be_bounded_with_progress(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source_docs"
            index_dir = Path(temp_dir) / "index"
            source_dir.mkdir()
            (source_dir / "chem21_guide.txt").write_text(
                "Acetone has CHEM21 health score evidence. "
                "Ethanol has CHEM21 safety score evidence.",
                encoding="utf-8",
            )
            store = RagEvidenceIndex(source_dir, index_dir, use_dense=False).load_or_build()
            progress_events = []

            evidence = retrieve_evidence_for_intelligence_with_store(
                [
                    self._intelligence_row(),
                    {
                        **self._intelligence_row(),
                        "attribute": "safety_score",
                        "value": "2",
                    },
                ],
                store,
                top_k=1,
                max_queries=1,
                progress_callback=progress_events.append,
            )

        self.assertLessEqual(len(evidence), 1)
        self.assertEqual(progress_events[0]["total"], 1)
        self.assertEqual(progress_events[-1]["stage"], "complete")

    def test_duplicate_evidence_queries_are_deduplicated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source_dir = Path(temp_dir) / "source_docs"
            index_dir = Path(temp_dir) / "index"
            source_dir.mkdir()
            (source_dir / "gsk_ssg.txt").write_text(
                "The GSK solvent guide includes health score evidence for acetone.",
                encoding="utf-8",
            )
            store = RagEvidenceIndex(source_dir, index_dir, use_dense=False).load_or_build()
            progress_events = []

            retrieve_evidence_for_intelligence_with_store(
                [self._intelligence_row(), self._intelligence_row()],
                store,
                top_k=1,
                progress_callback=progress_events.append,
            )

        self.assertEqual(progress_events[0]["total"], 1)

    def test_query_uses_chemical_source_attribute_and_topic_group(self):
        query = build_evidence_query(
            {
                "chemical": {
                    "canonical_name": "acetone",
                    "original_surface_form": "Acetone",
                    "aliases": ["propanone"],
                    "cas": ["67-64-1"],
                    "pubchem_cid": 180,
                    "cid": 180,
                },
                "source": "hsdb",
                "attribute": "safety_intelligence.ghs_classification",
                "value": "Flammable liquid category 2",
            }
        )

        self.assertIn("acetone", query)
        self.assertIn("hsdb", query)
        self.assertIn("propanone", query)
        self.assertIn("67-64-1", query)
        self.assertIn("180", query)
        self.assertIn("safety_intelligence.ghs_classification", query)
        self.assertIn("safety_intelligence", query)
        self.assertIn("Flammable liquid category 2", query)

    def test_relevance_diagnostics_detect_chemical_cas_and_topic(self):
        row = {
            "chemical": {
                "canonical_name": "acetone",
                "original_surface_form": "Acetone",
                "aliases": ["propanone"],
                "cas": ["67-64-1"],
            },
            "attribute": "safety_intelligence.ghs_classification",
        }
        diagnostics = evidence_relevance_diagnostics(
            row,
            self._chunk("diag", "Acetone CAS 67-64-1 has GHS hazard classification evidence."),
        )

        self.assertTrue(diagnostics["contains_chemical_name"])
        self.assertTrue(diagnostics["contains_cas"])
        self.assertTrue(diagnostics["contains_topic_keywords"])
        self.assertEqual(diagnostics["generic_methodology_penalty"], 0.0)

    def test_generic_methodology_chunk_is_penalized_without_chemical_match(self):
        row = {
            "chemical": {"canonical_name": "acetone", "cas": ["67-64-1"]},
            "attribute": "health_score",
        }
        diagnostics = evidence_relevance_diagnostics(
            row,
            self._chunk(
                "generic",
                "The solvent selection guide methodology explains how life cycle scores "
                "were calculated without solvent recovery.",
            ),
        )

        self.assertFalse(diagnostics["contains_chemical_name"])
        self.assertFalse(diagnostics["contains_cas"])
        self.assertGreater(diagnostics["generic_methodology_penalty"], 0.0)

    def test_empty_source_docs_returns_no_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            evidence = retrieve_evidence_for_intelligence(
                [self._intelligence_row()],
                source_docs_dir=Path(temp_dir) / "source_docs",
                index_dir=Path(temp_dir) / "index",
                top_k=1,
                embedding_backend=FakeEmbeddingBackend(),
            )

        self.assertEqual(evidence, [])

    def test_no_llm_is_called(self):
        source = inspect.getsource(rag_evidence).casefold()

        self.assertNotIn("ollama", source)
        self.assertNotIn("gemma", source)
        self.assertNotIn("openai", source)

    def _chunk(self, chunk_id, text):
        return EvidenceChunk(
            document_id="doc",
            source_name="source.txt",
            document_path="source.txt",
            page_number=None,
            section_heading="",
            chunk_text=text,
            start_char=0,
            end_char=len(text),
            source_type="guide",
            chunk_id=chunk_id,
        )

    def _intelligence_row(self):
        return {
            "chemical": {
                "canonical_name": "acetone",
                "original_surface_form": "Acetone",
                "pubchem_cid": 180,
            },
            "source": "gsk",
            "attribute": "health_score",
            "value": "3",
            "confidence": 1.0,
            "evidence": "Provider fact evidence.",
            "provenance": "fixture",
        }


if __name__ == "__main__":
    unittest.main()
