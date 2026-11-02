import unittest

from knowledge.providers.base import KnowledgeFact
from services.chemical_intelligence import (
    chemical_intelligence_to_json,
    collect_chemical_intelligence,
)
from services.validator import ValidatedChemical


class FakeAggregator:
    seen_identities = []

    def get_facts(self, chemical_identity):
        self.seen_identities.append(chemical_identity)
        if chemical_identity["original_surface_form"] == "unknown":
            return []
        return [
            KnowledgeFact(
                source="chem21",
                attribute="safety_score",
                value="4",
                confidence=1.0,
                evidence="Fixture CHEM21 evidence.",
                provenance="fixture::chem21 row 1",
            ),
            KnowledgeFact(
                source="gsk",
                attribute="health_score",
                value="3",
                confidence=1.0,
                evidence="Fixture GSK evidence.",
                provenance="fixture::gsk row approved",
            ),
        ]


class FakeSynonymResolver:
    def get_synonyms(self, chemical):
        return ["67-64-1", "acetone", "dimethyl ketone"]


class ChemicalIntelligenceTests(unittest.TestCase):
    def test_collects_provider_facts_for_validated_chemical(self):
        aggregator = FakeAggregator()
        records = collect_chemical_intelligence(
            [self._validated_chemical("Acetone", 180, "propan-2-one")],
            aggregator=aggregator,
            synonym_resolver=FakeSynonymResolver(),
        )

        self.assertEqual(len(records), 2)
        self.assertEqual({record.source for record in records}, {"chem21", "gsk"})
        self.assertTrue(all(record.provenance for record in records))
        self.assertIn("67-64-1", aggregator.seen_identities[-1]["cas"])
        self.assertIn("dimethyl ketone", aggregator.seen_identities[-1]["aliases"])

    def test_unified_json_contains_required_fields(self):
        records = collect_chemical_intelligence(
            [self._validated_chemical("Acetone", 180, "propan-2-one")],
            aggregator=FakeAggregator(),
            synonym_resolver=FakeSynonymResolver(),
        )

        payload = chemical_intelligence_to_json(records)

        self.assertEqual(
            set(payload[0]),
            {
                "chemical",
                "source",
                "attribute",
                "value",
                "confidence",
                "evidence",
                "provenance",
            },
        )
        self.assertEqual(payload[0]["chemical"]["pubchem_cid"], 180)

    def test_unknown_compound_without_provider_facts_returns_empty_payload(self):
        records = collect_chemical_intelligence(
            [self._validated_chemical("unknown", None, None)],
            aggregator=FakeAggregator(),
            synonym_resolver=FakeSynonymResolver(),
        )

        self.assertEqual(records, [])

    def _validated_chemical(self, surface_form, cid, canonical_name):
        return ValidatedChemical(
            original_surface_form=surface_form,
            pubchem_cid=cid,
            canonical_name=canonical_name,
            molecular_formula="C3H6O" if cid else None,
            canonical_smiles="CC(=O)C" if cid else None,
            validation_status="validated_pubchem_rdkit" if cid else "unresolved_pubchem",
            molecular_weight=58.08 if cid else None,
            logp=0.2 if cid else None,
            tpsa=17.1 if cid else None,
            h_bond_donors=0 if cid else None,
            h_bond_acceptors=1 if cid else None,
        )


if __name__ == "__main__":
    unittest.main()
