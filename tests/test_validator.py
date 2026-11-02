import unittest

import services.validator as validator
from services.chemistry_ner import ChemicalEntity
from services.normalizer import normalize_entities
from services.validator import (
    PubChemCompoundRecord,
    _capped_pubchem_workers,
    deduplicate_entities_by_pubchem_cid,
    validate_candidates_with_pubchem_rdkit,
    validate_entities_with_pubchem_rdkit,
)


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
        "N,N-dimethylformamide": PubChemCompoundRecord(
            cid=6228,
            canonical_name="N,N-dimethylformamide",
            molecular_formula="C3H7NO",
            canonical_smiles="CN(C)C=O",
        ),
        "phenyl hydride": PubChemCompoundRecord(
            cid=241,
            canonical_name="benzene",
            molecular_formula="C6H6",
            canonical_smiles="C1=CC=CC=C1",
        ),
    }

    def lookup_by_name(self, name):
        return self.records.get(name)


class CountingPubChemClient(FakePubChemClient):
    def __init__(self):
        self.lookup_names = []

    def lookup_by_name(self, name):
        self.lookup_names.append(name)
        return super().lookup_by_name(name)


class FailingPubChemClient:
    def lookup_by_name(self, name):
        raise AssertionError(f"Network lookup should not be called for cached candidate: {name}")


class PubChemRdkitValidationTests(unittest.TestCase):
    def setUp(self):
        validator._PUBCHEM_CACHE = {}

    def test_validates_common_chemicals_and_computes_descriptors(self):
        entities = [
            _entity("benzene"),
            _entity("ethanol"),
            _entity("acetone"),
            _entity("DMF", entity_type="ABBREVIATION"),
        ]
        validated = validate_entities_with_pubchem_rdkit(entities, FakePubChemClient())

        self.assertEqual(len(validated), 4)
        self.assertTrue(
            all(chemical.validation_status == "validated_pubchem_rdkit" for chemical in validated)
        )
        self.assertEqual(validated[0].pubchem_cid, 241)
        self.assertEqual(validated[1].pubchem_cid, 702)
        self.assertEqual(validated[2].pubchem_cid, 180)
        self.assertEqual(validated[3].pubchem_cid, 6228)
        self.assertTrue(all(chemical.canonical_smiles for chemical in validated))
        self.assertTrue(all(chemical.molecular_weight is not None for chemical in validated))
        self.assertTrue(all(chemical.logp is not None for chemical in validated))
        self.assertTrue(all(chemical.tpsa is not None for chemical in validated))
        self.assertTrue(all(chemical.h_bond_donors is not None for chemical in validated))
        self.assertTrue(all(chemical.h_bond_acceptors is not None for chemical in validated))

    def test_marks_nonsense_input_as_unresolved(self):
        candidates = normalize_entities([_entity("notachemicalxyz")])

        validated = validate_candidates_with_pubchem_rdkit(candidates, FakePubChemClient())

        self.assertEqual(len(validated), 1)
        self.assertEqual(validated[0].original_surface_form, "notachemicalxyz")
        self.assertEqual(validated[0].validation_status, "unresolved_pubchem")
        self.assertIsNone(validated[0].pubchem_cid)
        self.assertIsNone(validated[0].canonical_smiles)

    def test_deduplicates_entities_by_pubchem_cid_when_available(self):
        entities = [_entity("benzene"), _entity("phenyl hydride"), _entity("notachemicalxyz")]

        deduplicated = deduplicate_entities_by_pubchem_cid(entities, FakePubChemClient())

        self.assertEqual([entity.surface_form for entity in deduplicated], ["benzene", "notachemicalxyz"])

    def test_calls_pubchem_once_per_unique_local_candidate(self):
        entities = [
            _entity("benzene"),
            _entity("Benzene."),
            _entity(" benzene "),
            _entity("ethanol"),
        ]
        client = CountingPubChemClient()

        validated = validate_entities_with_pubchem_rdkit(entities, client)

        self.assertEqual(client.lookup_names.count("benzene"), 1)
        self.assertEqual(client.lookup_names.count("ethanol"), 1)
        self.assertEqual(len(client.lookup_names), 2)
        self.assertEqual([chemical.pubchem_cid for chemical in validated], [241, 241, 241, 702])

    def test_deduplicated_entities_include_human_review_metadata(self):
        entities = [
            _entity("benzene"),
            _entity("Benzene.", page_number=2),
            _entity("phenyl hydride", page_number=3),
        ]

        deduplicated = deduplicate_entities_by_pubchem_cid(entities, FakePubChemClient())

        self.assertEqual(len(deduplicated), 1)
        self.assertEqual(deduplicated[0].occurrence_count, 3)
        self.assertEqual(deduplicated[0].aliases, ("benzene", "Benzene.", "phenyl hydride"))
        self.assertEqual(deduplicated[0].pages_found, (1, 2, 3))

    def test_pubchem_cache_is_used_before_client_lookup(self):
        validator._PUBCHEM_CACHE = {
            "benzene": {
                "lookup_name": "benzene",
                "status": "resolved",
                "record": {
                    "cid": 241,
                    "canonical_name": "benzene",
                    "molecular_formula": "C6H6",
                    "canonical_smiles": "C1=CC=CC=C1",
                },
            }
        }
        client = CountingPubChemClient()

        validated = validate_entities_with_pubchem_rdkit([_entity("benzene")], client)

        self.assertEqual(client.lookup_names, [])
        self.assertEqual(validated[0].pubchem_cid, 241)

    def test_all_cached_candidates_complete_without_network_or_second_click(self):
        validator._PUBCHEM_CACHE = {
            "benzene": {
                "lookup_name": "benzene",
                "status": "resolved",
                "record": {
                    "cid": 241,
                    "canonical_name": "benzene",
                    "molecular_formula": "C6H6",
                    "canonical_smiles": "C1=CC=CC=C1",
                },
            },
            "ethanol": {
                "lookup_name": "ethanol",
                "status": "resolved",
                "record": {
                    "cid": 702,
                    "canonical_name": "ethanol",
                    "molecular_formula": "C2H6O",
                    "canonical_smiles": "CCO",
                },
            },
        }
        progress_events = []

        validated = validate_entities_with_pubchem_rdkit(
            [_entity("benzene"), _entity("ethanol")],
            FailingPubChemClient(),
            progress_callback=progress_events.append,
        )

        self.assertEqual([chemical.pubchem_cid for chemical in validated], [241, 702])
        self.assertEqual(progress_events[0]["stage"], "preflight")
        self.assertEqual(progress_events[0]["completed"], 2)
        self.assertEqual(progress_events[0]["total"], 2)
        self.assertEqual(progress_events[0]["cache_hits"], 2)
        self.assertEqual(progress_events[0]["network_lookups"], 0)
        self.assertEqual(progress_events[-1]["stage"], "cache_complete")
        self.assertEqual(progress_events[-1]["completed"], 2)
        self.assertEqual(progress_events[-1]["total"], 2)

    def test_parallel_pubchem_returns_same_result_structure_as_sequential(self):
        entities = [_entity("ethanol"), _entity("benzene"), _entity("acetone")]

        validated = validate_entities_with_pubchem_rdkit(entities, FakePubChemClient())

        self.assertEqual([chemical.original_surface_form for chemical in validated], ["ethanol", "benzene", "acetone"])
        self.assertEqual([chemical.pubchem_cid for chemical in validated], [702, 241, 180])
        self.assertTrue(all(chemical.canonical_smiles for chemical in validated))

    def test_output_order_is_stable(self):
        entities = [_entity("acetone"), _entity("benzene"), _entity("ethanol")]

        first = validate_entities_with_pubchem_rdkit(entities, FakePubChemClient())
        second = validate_entities_with_pubchem_rdkit(entities, FakePubChemClient())

        self.assertEqual(
            [chemical.original_surface_form for chemical in first],
            [chemical.original_surface_form for chemical in second],
        )

    def test_worker_count_is_capped(self):
        self.assertEqual(_capped_pubchem_workers(99), 10)
        self.assertEqual(_capped_pubchem_workers(0), 1)


def _entity(surface_form, entity_type="CHEMICAL_NAME", page_number=1):
    return ChemicalEntity(
        surface_form=surface_form,
        entity_type=entity_type,
        sentence_context=f"{surface_form} was mentioned.",
        start_char=0,
        end_char=len(surface_form),
        page_number=page_number,
        extraction_method="test",
        confidence=1.0,
    )


if __name__ == "__main__":
    unittest.main()
