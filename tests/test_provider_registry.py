from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.provider_registry import (
    ProviderRegistry,
    get_provider_category,
    get_provider_registry,
    get_providers_by_category,
    normalize_provider_name,
)


class ProviderRegistryTests(unittest.TestCase):
    def test_alias_normalization_uses_yaml_registry(self) -> None:
        self.assertEqual(normalize_provider_name("pubchem_pug_view_hsdb"), "hsdb")
        self.assertEqual(normalize_provider_name("chem_21"), "chem21")
        self.assertEqual(normalize_provider_name("us_dot"), "dot")
        self.assertEqual(normalize_provider_name("RAG"), "rag_evidence")

    def test_category_lookup(self) -> None:
        self.assertEqual(get_provider_category("chem21"), "sustainability")
        self.assertEqual(get_provider_category("gsk_solvent_sustainability_guide"), "sustainability")
        self.assertEqual(get_provider_category("rag"), "evidence")
        self.assertEqual(get_provider_category("pubchem"), "identity")

        safety_names = {provider.name for provider in get_providers_by_category("safety")}
        self.assertIn("hsdb", safety_names)

    def test_unknown_provider_handling_is_safe(self) -> None:
        self.assertEqual(normalize_provider_name("future_provider"), "future_provider")
        self.assertIsNone(get_provider_category("future_provider"))
        self.assertIsNone(get_provider_registry().get_provider("future_provider"))

    def test_duplicate_alias_detection_keeps_first_provider(self) -> None:
        registry = ProviderRegistry.from_yaml(
            self._write_registry(
                """
providers:
  alpha:
    label: Alpha
    category: safety
    aliases:
      - shared_alias
  beta:
    label: Beta
    category: regulatory
    aliases:
      - shared_alias
"""
            )
        )

        self.assertEqual(registry.normalize_provider_name("shared_alias"), "alpha")
        self.assertTrue(
            any("Duplicate provider alias 'shared_alias'" in warning for warning in registry.warnings)
        )

    def test_malformed_provider_definitions_warn_and_continue(self) -> None:
        registry = ProviderRegistry.from_yaml(
            self._write_registry(
                """
providers:
  valid:
    label: Valid
    category: evidence
    aliases:
      - valid_alias
  missing_category:
    label: Missing Category
    aliases:
      - no_category
  bad_aliases:
    label: Bad Aliases
    category: safety
    aliases: not-a-list
"""
            )
        )

        self.assertEqual(registry.normalize_provider_name("valid_alias"), "valid")
        self.assertIsNone(registry.get_provider("missing_category"))
        self.assertIsNotNone(registry.get_provider("bad_aliases"))
        self.assertTrue(any("missing category" in warning for warning in registry.warnings))
        self.assertTrue(any("aliases must be a list" in warning for warning in registry.warnings))

    def _write_registry(self, content: str) -> str:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "provider_registry.yaml"
        path.write_text(content.strip(), encoding="utf-8")
        return str(path)


if __name__ == "__main__":
    unittest.main()
