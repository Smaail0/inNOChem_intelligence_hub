import json
import tempfile
import unittest
from pathlib import Path

from knowledge.providers.hsdb_provider import (
    HsdbProvider,
    HsdbTopicCatalog,
    extract_hsdb_annotations_from_pug_view,
)


class FakeHsdbClient:
    def __init__(self):
        self.calls = []

    def get_annotations(self, cid, topics):
        self.calls.append((cid, topics))
        return [
            {
                "topic": "GHS Classification",
                "value": "Flammable liquids, category 2",
                "evidence": "Fixture PubChem PUG-View HSDB section.",
                "provenance": "fixture::pubchem_pug_view",
            }
        ]


class HsdbProviderTests(unittest.TestCase):
    def test_topic_allowlist_loads_from_catalog(self):
        catalog = HsdbTopicCatalog("data/knowledge/hsdb_topic_catalog.json")

        self.assertTrue(catalog.is_allowed("GHS Classification"))
        self.assertTrue(catalog.is_allowed("IDLH"))
        self.assertTrue(catalog.is_allowed("TSCA Requirements"))
        self.assertFalse(catalog.is_allowed("Unsupported Topic"))

    def test_provider_returns_normalized_knowledge_facts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = self._write_catalog(temp_dir)
            provider = HsdbProvider(catalog_path, annotation_client=FakeHsdbClient())

            facts = provider.get_facts(
                {
                    "pubchem_cid": 180,
                    "cas": ["67-64-1"],
                    "canonical_name": "acetone",
                }
            )

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].source, "hsdb")
        self.assertEqual(facts[0].attribute, "safety_intelligence.ghs_classification")
        self.assertEqual(facts[0].value, "Flammable liquids, category 2")
        self.assertEqual(facts[0].confidence, 1.0)

    def test_provenance_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = self._write_catalog(temp_dir)
            provider = HsdbProvider(catalog_path, annotation_client=FakeHsdbClient())

            fact = provider.get_facts({"cid": 180})[0]

        self.assertIn("fixture::pubchem_pug_view", fact.provenance)
        self.assertIn("cid=180", fact.provenance)
        self.assertIn("topic=GHS Classification", fact.provenance)
        self.assertIn("catalog_count=680", fact.provenance)

    def test_unsupported_topic_is_rejected_safely(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            catalog_path = self._write_catalog(temp_dir)
            client = FakeHsdbClient()
            provider = HsdbProvider(catalog_path, annotation_client=client)

            facts = provider.get_facts_for_topics({"cid": 180}, ["Unsupported Topic"])

        self.assertEqual(facts, [])
        self.assertEqual(client.calls, [])

    def test_pug_view_extraction_filters_requested_topics(self):
        payload = {
            "Record": {
                "Section": [
                    {
                        "TOCHeading": "Safety and Hazards",
                        "Section": [
                            {
                                "TOCHeading": "GHS Classification",
                                "Information": [
                                    {
                                        "Value": {
                                            "StringWithMarkup": [
                                                {"String": "Flammable liquid category 2"}
                                            ]
                                        }
                                    }
                                ],
                            },
                            {
                                "TOCHeading": "Unsupported Topic",
                                "Information": [{"Value": {"StringValue": "ignore me"}}],
                            },
                        ],
                    }
                ]
            }
        }

        annotations = extract_hsdb_annotations_from_pug_view(payload, ["GHS Classification"])

        self.assertEqual(len(annotations), 1)
        self.assertEqual(annotations[0]["topic"], "GHS Classification")
        self.assertEqual(annotations[0]["value"], "Flammable liquid category 2")

    def _write_catalog(self, directory):
        path = Path(directory) / "hsdb_topic_catalog.json"
        payload = {
            "InformationList": {
                "Annotation": [
                    {
                        "Heading": "GHS Classification",
                        "Description": "Fixture GHS topic.",
                        "Count": 680,
                    },
                    {
                        "Heading": "OSHA Standards",
                        "Description": "Fixture OSHA topic.",
                        "Count": 851,
                    },
                ]
            }
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path


if __name__ == "__main__":
    unittest.main()
