import unittest

from services.chemistry_ner import LexiconChemistryBackend, extract_chemical_entities
from services.document_extractor import ExtractedDocument, ExtractedTextPage, TextSegment
from services.normalizer import normalize_entities


class ChemistryNerServiceTests(unittest.TestCase):
    def test_extracts_chemical_entities_from_short_paragraph(self):
        text = "Benzene and ethanol were used as solvents. THF was added later."
        document = ExtractedDocument(
            filename="sample.txt",
            file_type="txt",
            text=text,
            pages=[ExtractedTextPage(page_number=1, text=text, start_char=0, end_char=len(text))],
            sentences=[
                TextSegment(
                    text="Benzene and ethanol were used as solvents.",
                    page_number=1,
                    start_char=0,
                    end_char=43,
                ),
                TextSegment(
                    text="THF was added later.",
                    page_number=1,
                    start_char=44,
                    end_char=len(text),
                ),
            ],
        )

        entities = extract_chemical_entities(document, preferred_backend=LexiconChemistryBackend())
        surface_forms = {entity.surface_form.lower() for entity in entities}

        self.assertIn("benzene", surface_forms)
        self.assertIn("ethanol", surface_forms)
        self.assertIn("thf", surface_forms)
        self.assertTrue(all(entity.sentence_context for entity in entities))
        self.assertTrue(all(entity.page_number == 1 for entity in entities))
        self.assertTrue(all(entity.extraction_method for entity in entities))

    def test_normalizer_marks_entities_as_pending_pubchem_validation(self):
        text = "Benzene and ethanol were used as solvents."
        document = ExtractedDocument(
            filename="sample.txt",
            file_type="txt",
            text=text,
            pages=[ExtractedTextPage(page_number=1, text=text, start_char=0, end_char=len(text))],
            sentences=[
                TextSegment(
                    text=text,
                    page_number=1,
                    start_char=0,
                    end_char=len(text),
                )
            ],
        )

        normalized = normalize_entities(
            extract_chemical_entities(document, preferred_backend=LexiconChemistryBackend())
        )

        self.assertTrue(normalized)
        self.assertTrue(
            all(candidate.validation_status == "pending_pubchem_validation" for candidate in normalized)
        )


if __name__ == "__main__":
    unittest.main()
