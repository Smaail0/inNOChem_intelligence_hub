from datetime import datetime, timezone
import unittest

from services.chemistry_ner import ChemicalEntity
from services.human_verification import apply_human_verification, prepare_verification_rows
from services.validator import PubChemCompoundRecord, validate_entities_with_pubchem_rdkit


class FakePubChemClient:
    records = {
        "benzene": PubChemCompoundRecord(
            cid=241,
            canonical_name="benzene",
            molecular_formula="C6H6",
            canonical_smiles="C1=CC=CC=C1",
        ),
        "ethanol": PubChemCompoundRecord(
            cid=702,
            canonical_name="ethanol",
            molecular_formula="C2H6O",
            canonical_smiles="CCO",
        ),
        "acetone": PubChemCompoundRecord(
            cid=180,
            canonical_name="propan-2-one",
            molecular_formula="C3H6O",
            canonical_smiles="CC(=O)C",
        ),
    }

    def lookup_by_name(self, name):
        return self.records.get(name)


class HumanVerificationTests(unittest.TestCase):
    def test_excluding_extracted_chemical_removes_it_from_validation_set(self):
        entities = [_entity("benzene"), _entity("ethanol")]
        rows = prepare_verification_rows(entities)
        rows[1]["include_for_validation"] = False

        result = apply_human_verification(
            extracted_entities=entities,
            verification_rows=rows,
            timestamp=datetime(2026, 5, 30, tzinfo=timezone.utc),
        )

        self.assertEqual([entity.surface_form for entity in result.included_entities], ["benzene"])
        self.assertEqual([entity.surface_form for entity in result.excluded_entities], ["ethanol"])
        self.assertEqual(result.audit_trail["timestamp"], "2026-05-30T00:00:00+00:00")
        self.assertEqual(len(result.audit_trail["extracted_entities"]), 2)
        self.assertEqual(len(result.audit_trail["excluded_entities"]), 1)

    def test_adding_missed_chemical_creates_human_added_entity(self):
        result = apply_human_verification(
            extracted_entities=[_entity("benzene")],
            verification_rows=prepare_verification_rows([_entity("benzene")]),
            manual_input="acetone, ethanol",
        )

        manual_entities = result.manually_added_entities

        self.assertEqual([entity.surface_form for entity in manual_entities], ["acetone", "ethanol"])
        self.assertTrue(all(entity.extraction_method == "human_added" for entity in manual_entities))
        self.assertTrue(all(entity.confidence == 1.0 for entity in manual_entities))
        self.assertTrue(
            all(
                entity.sentence_context == "Manually added by human reviewer"
                for entity in manual_entities
            )
        )

    def test_validation_uses_only_selected_and_manually_added_entities(self):
        entities = [_entity("benzene"), _entity("ethanol")]
        rows = prepare_verification_rows(entities)
        rows[1]["include_for_validation"] = False
        verification_result = apply_human_verification(
            extracted_entities=entities,
            verification_rows=rows,
            manual_input="acetone",
        )

        validated = validate_entities_with_pubchem_rdkit(
            verification_result.included_entities,
            FakePubChemClient(),
        )
        validated_names = [chemical.original_surface_form for chemical in validated]

        self.assertEqual(validated_names, ["benzene", "acetone"])
        self.assertNotIn("ethanol", validated_names)
        self.assertTrue(
            all(chemical.validation_status == "validated_pubchem_rdkit" for chemical in validated)
        )

    def test_verification_rows_show_deduplicated_candidate_metadata(self):
        entity = ChemicalEntity(
            surface_form="benzene",
            entity_type="CHEMICAL_NAME",
            sentence_context="benzene was extracted.",
            start_char=0,
            end_char=7,
            page_number=1,
            extraction_method="test",
            confidence=0.8,
            occurrence_count=3,
            aliases=("benzene", "phenyl hydride"),
            pages_found=(1, 4),
            representative_contexts=("benzene was extracted.", "phenyl hydride was extracted."),
        )

        rows = prepare_verification_rows([entity])

        self.assertEqual(rows[0]["occurrence_count"], 3)
        self.assertEqual(rows[0]["aliases"], "benzene, phenyl hydride")
        self.assertEqual(rows[0]["pages_found"], "1, 4")
        self.assertIn("phenyl hydride was extracted.", rows[0]["representative_contexts"])


def _entity(surface_form):
    return ChemicalEntity(
        surface_form=surface_form,
        entity_type="CHEMICAL_NAME",
        sentence_context=f"{surface_form} was extracted.",
        start_char=0,
        end_char=len(surface_form),
        page_number=1,
        extraction_method="test",
        confidence=0.8,
    )


if __name__ == "__main__":
    unittest.main()
