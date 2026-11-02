import unittest
from unittest.mock import patch
from unittest.mock import Mock

from services.llm_explanation import (
    build_explanation_input,
    build_fallback_explanation,
    compress_explanation_input,
    generate_chemical_explanation,
    generate_gemini_chemical_explanation,
    _parse_json_response,
)
from services.validator import ValidatedChemical


class LlmExplanationTests(unittest.TestCase):
    def test_fallback_explanation_uses_only_provided_data(self):
        explanation_input = build_explanation_input(
            validated_chemical=_validated_chemical(),
            intelligence_records=[
                {
                    "chemical": {"pubchem_cid": 180, "canonical_name": "propan-2-one"},
                    "source": "hsdb",
                    "attribute": "safety_intelligence.ghs_classification",
                    "value": "Flammable liquid category 2",
                    "evidence": "HSDB fixture evidence.",
                    "provenance": "fixture",
                },
                {
                    "chemical": {"pubchem_cid": 180, "canonical_name": "propan-2-one"},
                    "source": "hsdb",
                    "attribute": "regulatory_intelligence.osha_standards",
                    "value": "OSHA fixture standard",
                    "evidence": "OSHA fixture evidence.",
                    "provenance": "fixture",
                },
            ],
            coverage_row={
                "pubchem_cid": 180,
                "coverage_percent": 55,
                "missing_groups": ["sustainability"],
            },
            confidence_row={
                "pubchem_cid": 180,
                "confidence_label": "medium",
                "recommendation_status": "limited_support",
            },
            evidence_records=[
                {
                    "chemical_name": "propan-2-one",
                    "evidence_snippet": "Acetone evidence snippet.",
                    "document_name": "fixture.pdf",
                    "page_number": 1,
                }
            ],
        )

        explanation = build_fallback_explanation(explanation_input)

        self.assertIn("PubChem CID 180", explanation.summary)
        self.assertTrue(any("GHS" in risk or "ghs" in risk.lower() for risk in explanation.key_risks))
        self.assertTrue(any("OSHA" in note or "osha" in note.lower() for note in explanation.regulatory_notes))
        self.assertIn("sustainability", explanation.missing_information)

    def test_generate_explanation_falls_back_when_ollama_fails(self):
        explanation_input = build_explanation_input(
            validated_chemical=_validated_chemical(),
            coverage_row={"pubchem_cid": 180, "coverage_percent": 20},
        )

        with patch("services.llm_explanation.requests.post", side_effect=RuntimeError("offline")):
            explanation = generate_chemical_explanation(explanation_input, timeout_seconds=0.01)

        self.assertEqual(explanation.generated_by, "deterministic_fallback")
        self.assertTrue(explanation.warnings)
        self.assertLessEqual(explanation.timeout_seconds, 30.0)
        self.assertTrue(explanation.fallback_used)

    def test_generate_explanation_caps_timeout_at_thirty_seconds(self):
        explanation_input = build_explanation_input(
            validated_chemical=_validated_chemical(),
            coverage_row={"pubchem_cid": 180, "coverage_percent": 20},
        )

        with patch("services.llm_explanation.requests.post", side_effect=RuntimeError("offline")) as post:
            generate_chemical_explanation(explanation_input, timeout_seconds=120.0)

        self.assertEqual(post.call_args.kwargs["timeout"], 30.0)

    def test_prompt_compression_limits_repeated_rows_and_long_values(self):
        rows = [
            {
                "chemical": {"pubchem_cid": 180, "canonical_name": "propan-2-one"},
                "source": "hsdb",
                "attribute": "safety_intelligence.ghs_classification",
                "value": "Flammable " + ("very long value " * 40),
            },
            {
                "chemical": {"pubchem_cid": 180, "canonical_name": "propan-2-one"},
                "source": "hsdb",
                "attribute": "safety_intelligence.ghs_classification",
                "value": "Duplicate row should be removed",
            },
        ]
        explanation_input = build_explanation_input(
            validated_chemical=_validated_chemical(),
            intelligence_records=rows,
            evidence_records=[
                {
                    "chemical_name": "propan-2-one",
                    "evidence_snippet": "Evidence " + ("long " * 100),
                }
                for _ in range(5)
            ],
        )

        compressed = compress_explanation_input(explanation_input)

        self.assertEqual(len(compressed.intelligence_rows), 1)
        self.assertEqual(len(compressed.evidence_rows), 3)
        self.assertLessEqual(len(compressed.intelligence_rows[0]["value"]), 300)

    def test_gemini_explanation_uses_api_key_header_and_json_schema(self):
        explanation_input = build_explanation_input(
            validated_chemical=_validated_chemical(),
            coverage_row={"pubchem_cid": 180, "coverage_percent": 55},
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    '{"summary":"Gemini summary","key_risks":[],"sustainability_notes":[],'
                                    '"regulatory_notes":[],"evidence_used":[],"missing_information":[],'
                                    '"confidence_statement":"medium"}'
                                )
                            }
                        ]
                    }
                }
            ]
        }

        with patch("services.llm_explanation.requests.post", return_value=response) as post:
            explanation = generate_gemini_chemical_explanation(
                explanation_input,
                api_key="test-key",
                model="gemini-2.5-flash",
                timeout_seconds=30,
            )

        self.assertEqual(explanation.generated_by, "gemini:gemini-2.5-flash")
        self.assertFalse(explanation.fallback_used)
        self.assertEqual(post.call_args.kwargs["headers"]["x-goog-api-key"], "test-key")
        self.assertEqual(
            post.call_args.kwargs["json"]["generationConfig"]["responseMimeType"],
            "application/json",
        )

    def test_gemini_explanation_falls_back_without_api_key(self):
        explanation_input = build_explanation_input(
            validated_chemical=_validated_chemical(),
            coverage_row={"pubchem_cid": 180, "coverage_percent": 55},
        )

        with patch("services.llm_explanation.requests.post") as post:
            explanation = generate_gemini_chemical_explanation(
                explanation_input,
                api_key="",
                timeout_seconds=30,
            )

        post.assert_not_called()
        self.assertEqual(explanation.generated_by, "deterministic_fallback")
        self.assertTrue(explanation.fallback_used)
        self.assertTrue(any("GEMINI_API_KEY" in warning for warning in explanation.warnings))

    def test_parse_json_response_repairs_truncated_string(self):
        parsed = _parse_json_response('{"summary": "Partial Gemini output')

        self.assertEqual(parsed["summary"], "Partial Gemini output")


def _validated_chemical():
    return ValidatedChemical(
        original_surface_form="acetone",
        pubchem_cid=180,
        canonical_name="propan-2-one",
        molecular_formula="C3H6O",
        canonical_smiles="CC(=O)C",
        validation_status="validated_pubchem_rdkit",
        molecular_weight=58.08,
        logp=0.2,
        tpsa=17.1,
        h_bond_donors=0,
        h_bond_acceptors=1,
    )


if __name__ == "__main__":
    unittest.main()
