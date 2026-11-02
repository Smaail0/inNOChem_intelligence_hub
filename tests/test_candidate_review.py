import unittest

from services.candidate_review import (
    apply_candidate_review_selection,
    build_candidate_review_records,
    candidate_records_to_rows,
)
from services.chemistry_ner import ChemicalEntity


class CountingPubChemClient:
    def __init__(self):
        self.lookup_count = 0

    def lookup_by_name(self, name):
        self.lookup_count += 1
        return None


class CandidateReviewTests(unittest.TestCase):
    def test_pubchem_is_not_called_before_human_selection(self):
        client = CountingPubChemClient()

        records = build_candidate_review_records([_entity("benzene"), _entity("benzene")])

        self.assertEqual(len(records), 1)
        self.assertEqual(client.lookup_count, 0)

    def test_top_100_are_preselected_by_default(self):
        entities = [_entity(f"candidate-{index}") for index in range(105)]

        records = build_candidate_review_records(entities)

        self.assertEqual(len(records), 105)
        self.assertEqual(sum(record.include_for_pubchem_validation for record in records), 100)

    def test_user_can_include_and_exclude_candidates(self):
        records = build_candidate_review_records([_entity("benzene"), _entity("ethanol")])
        rows = candidate_records_to_rows(records)
        rows[0]["include_for_pubchem_validation"] = False
        rows[1]["include_for_pubchem_validation"] = True

        selected = apply_candidate_review_selection(records, rows)

        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].surface_form, records[1].representative_entity.surface_form)

    def test_only_selected_candidates_are_returned_for_pubchem(self):
        records = build_candidate_review_records(
            [_entity("benzene"), _entity("ethanol"), _entity("acetone")]
        )
        rows = candidate_records_to_rows(records)
        for row in rows:
            row["include_for_pubchem_validation"] = row["normalized_name"] == "ethanol"

        selected = apply_candidate_review_selection(records, rows)

        self.assertEqual([entity.surface_form for entity in selected], ["ethanol"])

    def test_priority_score_ranks_heading_and_occurrence_higher(self):
        records = build_candidate_review_records(
            [
                _entity("benzene", context="Solvent screening heading: benzene was central."),
                _entity("benzene", context="benzene appeared again."),
                _entity("benzene", context="benzene appeared again.", page_number=2),
                _entity("ethanol", confidence=0.95, context="ethanol appeared once."),
            ]
        )

        self.assertEqual(records[0].normalized_name, "benzene")
        self.assertGreater(records[0].candidate_priority_score, records[1].candidate_priority_score)

    def test_priority_reason_explains_preselection(self):
        records = build_candidate_review_records(
            [
                _entity("benzene", context="Abstract: benzene was evaluated."),
                _entity("benzene", context="Introduction: benzene was selected.", page_number=2),
            ]
        )

        reason = records[0].priority_reason

        self.assertIn("Preselected because", reason)
        self.assertIn("found in abstract", reason)
        self.assertIn("appears 2 times", reason)
        self.assertIn("across 2 pages", reason)

    def test_pubchem_recommendation_defaults_by_candidate_type(self):
        expected = {
            "DMF": ("abbreviation", True),
            "NMP": ("abbreviation", True),
            "Cyrene": ("specific_compound", True),
            "1,4-dioxane": ("specific_compound", True),
            "hydrogen": ("element", False),
            "ether": ("chemical_class", False),
            "amide": ("chemical_class", False),
            "c-h": ("reaction_fragment", False),
            "aryl halides": ("chemical_class", False),
        }

        records = {
            record.aliases[0]: record
            for record in build_candidate_review_records(
                [_entity(name) for name in expected],
                preselect_limit=len(expected),
            )
        }

        for name, (candidate_type, recommended) in expected.items():
            with self.subTest(name=name):
                self.assertEqual(records[name].candidate_type, candidate_type)
                self.assertEqual(records[name].pubchem_validation_recommended, recommended)
                self.assertEqual(records[name].include_for_pubchem_validation, recommended)

    def test_non_recommended_candidates_remain_visible_but_unchecked(self):
        records = build_candidate_review_records(
            [_entity("hydrogen"), _entity("DMF")],
            preselect_limit=2,
        )

        by_alias = {record.aliases[0]: record for record in records}

        self.assertIn("hydrogen", by_alias)
        self.assertFalse(by_alias["hydrogen"].include_for_pubchem_validation)
        self.assertIn("Not recommended for PubChem: element", by_alias["hydrogen"].pubchem_recommendation_reason)
        self.assertTrue(by_alias["DMF"].include_for_pubchem_validation)
        self.assertEqual(by_alias["DMF"].pubchem_recommendation_reason, "Recommended: abbreviation")


def _entity(surface_form, confidence=0.9, page_number=1, context=None):
    return ChemicalEntity(
        surface_form=surface_form,
        entity_type="CHEMICAL_NAME",
        sentence_context=context or f"{surface_form} was found in the introduction.",
        start_char=0,
        end_char=len(surface_form),
        page_number=page_number,
        extraction_method="transformer_chemical_ner",
        confidence=confidence,
    )


if __name__ == "__main__":
    unittest.main()
