import unittest

from knowledge.aggregator import (
    KnowledgeAggregator,
    KnowledgeProviderRegistry,
    default_provider_registry,
)
from knowledge.provider_base import KnowledgeFact, chemical_name


class FakeProvider:
    name = "fake_file_provider"

    def get_facts(self, chemical):
        return [
            KnowledgeFact(
                source=self.name,
                attribute="local_file_attribute",
                value=chemical_name(chemical),
                confidence=0.8,
                evidence="Loaded from a fake local file provider.",
                provenance="tests::FakeProvider",
            )
        ]


class KnowledgeProviderTests(unittest.TestCase):
    def test_fact_schema_serializes_standard_fields(self):
        fact = KnowledgeFact(
            source="test",
            attribute="attribute",
            value="value",
            confidence=0.7,
            evidence="evidence text",
        )

        self.assertEqual(
            fact.to_dict(),
            {
                "source": "test",
                "attribute": "attribute",
                "value": "value",
                "confidence": 0.7,
                "evidence": "evidence text",
                "provenance": "",
            },
        )

    def test_default_registry_contains_expected_data_providers(self):
        registry = default_provider_registry()
        provider_names = {provider.name for provider in registry.active_providers()}

        self.assertEqual(provider_names, {"chem21", "gsk", "hsdb"})

    def test_aggregator_collects_facts_from_all_active_providers(self):
        registry = KnowledgeProviderRegistry()
        registry.register(FakeProvider())
        aggregator = KnowledgeAggregator(registry)

        facts = aggregator.collect_facts({"canonical_name": "benzene"})

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].source, "fake_file_provider")
        self.assertEqual(facts[0].value, "benzene")

    def test_aggregator_get_facts_alias_collects_facts(self):
        registry = KnowledgeProviderRegistry()
        registry.register(FakeProvider())
        aggregator = KnowledgeAggregator(registry)

        facts = aggregator.get_facts({"canonical_name": "benzene"})

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].source, "fake_file_provider")

    def test_registry_can_disable_provider_by_not_registering_it(self):
        registry = KnowledgeProviderRegistry()
        registry.register(FakeProvider(), active=False)
        aggregator = KnowledgeAggregator(registry)

        self.assertEqual(aggregator.collect_facts("ethanol"), [])

    def test_registry_can_unregister_provider(self):
        registry = KnowledgeProviderRegistry()
        registry.register(FakeProvider())
        registry.unregister("fake_file_provider")

        self.assertEqual(registry.active_providers(), [])

    def test_default_providers_do_not_return_solvent_decisions(self):
        facts = KnowledgeAggregator(default_provider_registry()).collect_facts("DMF")
        attributes = {fact.attribute for fact in facts}

        self.assertNotIn("solvent_recommendation", attributes)
        self.assertNotIn("solvent_decision", attributes)
        self.assertTrue(all(hasattr(fact, "evidence") for fact in facts))
        self.assertTrue(all(hasattr(fact, "provenance") for fact in facts))


if __name__ == "__main__":
    unittest.main()
