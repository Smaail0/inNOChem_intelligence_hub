import csv
from pathlib import Path
import tempfile
import unittest

from knowledge.providers.chem21_provider import Chem21Provider
from knowledge.providers.gsk_provider import GskProvider
from scripts.ingest_chem21 import ingest_chem21


class KnowledgeIngestionAndProviderTests(unittest.TestCase):
    def test_chem21_ingestion_creates_non_empty_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            rows = ingest_chem21(output_path=output_path)

            self.assertTrue(output_path.exists())
            self.assertGreater(len(rows), 0)
            with output_path.open("r", encoding="utf-8", newline="") as handle:
                persisted_rows = list(csv.DictReader(handle))
            self.assertGreater(len(persisted_rows), 0)

    def test_dmf_alias_resolves_to_problematic_when_source_row_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "chem21_solvents.csv"
            self._write_chem21_dataset(
                dataset_path,
                [
                    {
                        "solvent_name": "N,N-Dimethylformamide",
                        "normalized_name": "n,n-dimethylformamide",
                        "aliases": "DMF|dimethylformamide",
                        "cas": "68-12-2",
                        "ranking_by_default": "Problematic",
                        "final_ranking": "Problematic",
                        "provenance": "fixture::chem21 row 1",
                    }
                ],
            )

            facts = Chem21Provider(dataset_path).get_facts("DMF")

            ranking_values = {
                fact.value
                for fact in facts
                if fact.attribute in {"ranking_by_default", "final_ranking"}
            }
            self.assertIn("Problematic", ranking_values)

    def test_acetone_resolves_to_source_ranking_when_present(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            ingest_chem21(output_path=output_path)

            facts = Chem21Provider(output_path).get_facts(
                {"pubchem_canonical_name": "Acetone", "cas": "67-64-1"}
            )

            self.assertTrue(facts)
            ranking_values = {
                fact.value
                for fact in facts
                if fact.attribute in {"ranking_by_default", "final_ranking"}
            }
            self.assertEqual(ranking_values, {"Problematic"})

    def test_ingestion_expands_aliases_for_present_source_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            ingest_chem21(output_path=output_path)

            methf_facts = Chem21Provider(output_path).get_facts("2-MeTHF")
            sulfolane_facts = Chem21Provider(output_path).get_facts("sulfolane")

            self.assertTrue(methf_facts)
            self.assertTrue(sulfolane_facts)

    def test_ingestion_excludes_placeholder_solvent_name_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            rows = ingest_chem21(output_path=output_path)

            self.assertFalse(
                any(row["solvent_name"].casefold() == "solvent name" for row in rows)
            )

    def test_ingestion_preserves_sheet_and_table_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            rows = ingest_chem21(output_path=output_path)

            acetone = next(row for row in rows if row["cas"] == "67-64-1")

            self.assertIn("Results", acetone["provenance"])
            self.assertIn("solvent data sheet", acetone["provenance"])
            self.assertIn("source_table=", acetone["provenance"])

    def test_available_workbook_contains_known_source_cas_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            rows = ingest_chem21(output_path=output_path)

            cas_values = {row["cas"].strip() for row in rows}

            self.assertIn("67-64-1", cas_values)
            self.assertIn("64-17-5", cas_values)
            self.assertIn("96-47-9", cas_values)

    def test_available_workbook_does_not_fabricate_absent_classical_solvents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            rows = ingest_chem21(output_path=output_path)

            cas_values = {row["cas"].strip() for row in rows}

            self.assertNotIn("109-99-9", cas_values)
            self.assertNotIn("68-12-2", cas_values)
            self.assertNotIn("872-50-4", cas_values)
            self.assertNotIn("123-91-1", cas_values)

    def test_unknown_compound_returns_no_facts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            ingest_chem21(output_path=output_path)

            self.assertEqual(Chem21Provider(output_path).get_facts("not-a-real-solvent"), [])

    def test_provider_facts_include_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "chem21_solvents.csv"
            ingest_chem21(output_path=output_path)

            facts = Chem21Provider(output_path).get_facts("ethanol")

            self.assertTrue(facts)
            self.assertTrue(all(fact.provenance for fact in facts))

    def test_gsk_provider_loads_only_approved_review_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = Path(temp_dir) / "gsk_extraction_review.csv"
            self._write_gsk_review_dataset(dataset_path)

            provider = GskProvider(dataset_path)

            self.assertEqual(provider.get_facts("Pending Solvent"), [])
            approved_facts = provider.get_facts("Approved Solvent")
            self.assertTrue(approved_facts)
            self.assertTrue(all("review_status=approved" in fact.provenance for fact in approved_facts))

    def _write_chem21_dataset(self, dataset_path, rows):
        fieldnames = [
            "source",
            "source_version",
            "source_reference",
            "solvent_name",
            "normalized_name",
            "aliases",
            "cas",
            "family",
            "boiling_point_c",
            "flash_point_c",
            "worst_h3xx",
            "h4xx",
            "safety_score",
            "health_score",
            "environment_score",
            "ranking_by_default",
            "final_ranking",
            "notes",
            "provenance",
        ]
        with dataset_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                complete_row = {field: "" for field in fieldnames}
                complete_row.update(
                    {
                        "source": "CHEM21",
                        "source_version": "fixture",
                        "source_reference": "fixture source",
                    }
                )
                complete_row.update(row)
                writer.writerow(complete_row)

    def _write_gsk_review_dataset(self, dataset_path):
        fieldnames = [
            "solvent_name",
            "cas",
            "melting_point_c",
            "boiling_point_c",
            "waste_score",
            "environmental_impact_score",
            "health_score",
            "flammability_explosion_score",
            "reactivity_stability_score",
            "life_cycle_score",
            "legislation_flag",
            "ehs_red_flag",
            "extraction_confidence",
            "source_page",
            "source_figure",
            "review_status",
        ]
        rows = [
            {
                "solvent_name": "Pending Solvent",
                "health_score": "8",
                "extraction_confidence": "0.2",
                "source_page": "4",
                "source_figure": "1",
                "review_status": "pending",
            },
            {
                "solvent_name": "Approved Solvent",
                "health_score": "3",
                "waste_score": "2",
                "extraction_confidence": "1.0",
                "source_page": "5",
                "source_figure": "2",
                "review_status": "approved",
            },
        ]
        with dataset_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                complete_row = {field: "" for field in fieldnames}
                complete_row.update(row)
                writer.writerow(complete_row)


if __name__ == "__main__":
    unittest.main()
