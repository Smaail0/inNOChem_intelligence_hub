from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from services.resource_manager import warmup_cached_ollama_explanation_model


class ResourceManagerTests(unittest.TestCase):
    def tearDown(self) -> None:
        warmup_cached_ollama_explanation_model.clear()

    def test_ollama_explanation_warmup_success(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None

        with patch("services.resource_manager.requests.post", return_value=response) as post:
            handle = warmup_cached_ollama_explanation_model(
                base_url="http://localhost:11434",
                model="llama3.2:1b",
                timeout_seconds=10,
            )

        self.assertEqual(handle.diagnostics["status"], "warmed")
        self.assertEqual(post.call_args.kwargs["timeout"], 10.0)
        self.assertEqual(post.call_args.kwargs["json"]["options"]["num_predict"], 12)

    def test_ollama_explanation_warmup_failure_is_nonfatal(self) -> None:
        with patch("services.resource_manager.requests.post", side_effect=RuntimeError("offline")):
            handle = warmup_cached_ollama_explanation_model(
                base_url="http://localhost:11434",
                model="llama3.2:1b",
                timeout_seconds=10,
            )

        self.assertEqual(handle.diagnostics["status"], "warmup_failed")
        self.assertIn("offline", handle.diagnostics["error"])

    def test_ollama_explanation_warmup_timeout_is_capped(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None

        with patch("services.resource_manager.requests.post", return_value=response) as post:
            warmup_cached_ollama_explanation_model(
                base_url="http://localhost:11434",
                model="llama3.2:1b",
                timeout_seconds=60,
            )

        self.assertEqual(post.call_args.kwargs["timeout"], 10.0)


if __name__ == "__main__":
    unittest.main()
