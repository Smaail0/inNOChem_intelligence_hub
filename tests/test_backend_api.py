from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from backend.app.main import create_app


class BackendApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(create_app())

    def test_health_endpoint(self):
        response = self.client.get("/api/system/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")

    def test_status_endpoint_has_configured_models(self):
        response = self.client.get("/api/system/status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("configured_models", payload)
        self.assertIn("chemical_ner", payload["configured_models"])
        self.assertIn("resources", payload)

    def test_document_extract_txt_endpoint(self):
        response = self.client.post(
            "/api/documents/extract",
            files={"file": ("sample.txt", b"Benzene and ethanol were used.", "text/plain")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["file_type"], "txt")
        self.assertEqual(payload["statistics"]["pages"], 1)
        self.assertIn("Benzene", payload["text_preview"])

    def test_ner_extract_endpoint_returns_candidate_review_rows(self):
        response = self.client.post(
            "/api/ner/extract",
            json={
                "filename": "sample.txt",
                "file_type": "txt",
                "use_transformer": False,
                "pages": [
                    {
                        "page_number": 1,
                        "text": "Benzene and ethanol were used as solvents. THF was added later.",
                        "start_char": 0,
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["backend"], "chemistry_lexicon")
        self.assertGreaterEqual(payload["raw_entity_count"], 3)
        names = {row["normalized_name"] for row in payload["candidate_review_rows"]}
        self.assertIn("benzene", names)
        self.assertIn("ethanol", names)


if __name__ == "__main__":
    unittest.main()
