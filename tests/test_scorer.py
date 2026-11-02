import unittest

from services.scorer import score_validated_chemicals
from services.validator import ValidatedChemical


class DescriptorScoringTests(unittest.TestCase):
    def test_scores_benzene(self):
        score = score_validated_chemicals(
            [_validated("benzene", 241, "C6H6", "C1=CC=CC=C1", 78.114, 1.6866, 0.0)]
        )[0]

        self.assertEqual(score.score_status, "scored")
        self.assertEqual(score.bioaccumulation_proxy_score, 1)
        self.assertEqual(score.size_complexity_proxy_score, 0)
        self.assertEqual(score.polarity_proxy_score, 1)
        self.assertEqual(score.preliminary_descriptor_risk_score, 2)
        self.assertEqual(score.risk_level, "moderate")
        self.assertEqual(len(score.rule_details), 3)

    def test_scores_ethanol(self):
        score = score_validated_chemicals(
            [_validated("ethanol", 702, "C2H6O", "CCO", 46.069, -0.0014, 20.23)]
        )[0]

        self.assertEqual(score.score_status, "scored")
        self.assertEqual(score.preliminary_descriptor_risk_score, 1)
        self.assertEqual(score.risk_level, "low")

    def test_scores_acetone(self):
        score = score_validated_chemicals(
            [_validated("acetone", 180, "C3H6O", "CC(=O)C", 58.08, 0.5953, 17.07)]
        )[0]

        self.assertEqual(score.score_status, "scored")
        self.assertEqual(score.preliminary_descriptor_risk_score, 1)
        self.assertEqual(score.risk_level, "low")

    def test_scores_dmf(self):
        score = score_validated_chemicals(
            [
                _validated(
                    "N,N-dimethylformamide",
                    6228,
                    "C3H7NO",
                    "CN(C)C=O",
                    73.095,
                    -0.829,
                    20.31,
                )
            ]
        )[0]

        self.assertEqual(score.score_status, "scored")
        self.assertEqual(score.preliminary_descriptor_risk_score, 1)
        self.assertEqual(score.risk_level, "low")

    def test_unresolved_entity_cannot_score(self):
        score = score_validated_chemicals(
            [
                ValidatedChemical(
                    original_surface_form="notachemicalxyz",
                    pubchem_cid=None,
                    canonical_name=None,
                    molecular_formula=None,
                    canonical_smiles=None,
                    validation_status="unresolved_pubchem",
                    molecular_weight=None,
                    logp=None,
                    tpsa=None,
                    h_bond_donors=None,
                    h_bond_acceptors=None,
                )
            ]
        )[0]

        self.assertEqual(score.score_status, "cannot_score")
        self.assertIsNone(score.preliminary_descriptor_risk_score)
        self.assertEqual(score.risk_level, "not_available")
        self.assertEqual(len(score.rule_details), 1)


def _validated(name, cid, formula, smiles, molecular_weight, logp, tpsa):
    return ValidatedChemical(
        original_surface_form=name,
        pubchem_cid=cid,
        canonical_name=name,
        molecular_formula=formula,
        canonical_smiles=smiles,
        validation_status="validated_pubchem_rdkit",
        molecular_weight=molecular_weight,
        logp=logp,
        tpsa=tpsa,
        h_bond_donors=0,
        h_bond_acceptors=0,
    )


if __name__ == "__main__":
    unittest.main()
