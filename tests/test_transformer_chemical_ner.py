import unittest

from services.chemistry_ner import TransformerChemicalNERBackend
from services.document_extractor import ExtractedDocument, ExtractedTextPage, TextSegment


class FakeTokenClassificationPipeline:
    terms = [
        "DMF",
        "NMP",
        "1,4-dioxane",
        "γ-valerolactone",
        "2-MeTHF",
        "CPME",
        "MTBE",
        "Cyrene",
        "sulfolane",
    ]

    def __call__(self, text):
        outputs = []
        for term in self.terms:
            start = text.index(term)
            outputs.append(
                {
                    "entity_group": "CHEMICAL",
                    "score": 0.97,
                    "word": term,
                    "start": start,
                    "end": start + len(term),
                }
            )
        return outputs


class SplitTokenPipeline:
    def __call__(self, text):
        return [
            {
                "entity_group": "CHEM",
                "score": 0.99,
                "word": "cy",
                "start": text.index("Cyrene"),
                "end": text.index("Cyrene") + 2,
            },
            {
                "entity_group": "CHEM",
                "score": 0.98,
                "word": "##rene",
                "start": text.index("Cyrene") + 2,
                "end": text.index("Cyrene") + len("Cyrene"),
            },
            {
                "entity_group": "CHEM",
                "score": 0.92,
                "word": ".",
                "start": len(text) - 1,
                "end": len(text),
            },
        ]


class FakeOffsetTokenizer:
    model_max_length = 512

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        cursor = 0
        input_ids = []
        offsets = []
        for index, token in enumerate(text.split()):
            start = text.index(token, cursor)
            end = start + len(token)
            input_ids.append(index)
            offsets.append((start, end))
            cursor = end

        result = {"input_ids": input_ids}
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class BatchTokenClassificationPipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, texts, batch_size=None):
        self.calls.append((list(texts), batch_size))
        outputs = []
        for text in texts:
            surface = text.split()[0]
            outputs.append(
                [
                    {
                        "entity_group": "CHEMICAL",
                        "score": 0.9,
                        "word": surface,
                        "start": text.index(surface),
                        "end": text.index(surface) + len(surface),
                    }
                ]
            )
        return outputs


class TransformerChemicalNerBackendTests(unittest.TestCase):
    def test_reconstructs_chemical_entities_from_transformer_outputs(self):
        paragraph = (
            "Solvent screening included DMF, NMP, 1,4-dioxane, γ-valerolactone, "
            "2-MeTHF, CPME, MTBE, Cyrene, and sulfolane."
        )
        document = ExtractedDocument(
            filename="screening.txt",
            file_type="txt",
            text=paragraph,
            pages=[ExtractedTextPage(page_number=1, text=paragraph, start_char=0, end_char=len(paragraph))],
            sentences=[
                TextSegment(
                    text=paragraph,
                    page_number=1,
                    start_char=0,
                    end_char=len(paragraph),
                )
            ],
        )
        backend = TransformerChemicalNERBackend(
            max_chars_per_chunk=500,
            ner_pipeline=FakeTokenClassificationPipeline(),
            enable_result_cache=False,
        )

        entities = backend.extract(document)
        surface_forms = [entity.surface_form for entity in entities]

        self.assertEqual(
            surface_forms,
            [
                "DMF",
                "NMP",
                "1,4-dioxane",
                "γ-valerolactone",
                "2-MeTHF",
                "CPME",
                "MTBE",
                "Cyrene",
                "sulfolane",
            ],
        )
        self.assertTrue(all(entity.extraction_method == "transformer_chemical_ner" for entity in entities))
        self.assertTrue(all(entity.entity_type == "CHEMICAL" for entity in entities))
        self.assertTrue(all(entity.page_number == 1 for entity in entities))
        self.assertTrue(all(entity.confidence == 0.97 for entity in entities))

    def test_merges_adjacent_subword_outputs_and_filters_punctuation(self):
        paragraph = "Cyrene was evaluated."
        document = ExtractedDocument(
            filename="screening.txt",
            file_type="txt",
            text=paragraph,
            pages=[ExtractedTextPage(page_number=1, text=paragraph, start_char=0, end_char=len(paragraph))],
            sentences=[
                TextSegment(
                    text=paragraph,
                    page_number=1,
                    start_char=0,
                    end_char=len(paragraph),
                )
            ],
        )
        backend = TransformerChemicalNERBackend(
            max_chars_per_chunk=500,
            ner_pipeline=SplitTokenPipeline(),
            enable_result_cache=False,
        )

        entities = backend.extract(document)

        self.assertEqual([entity.surface_form for entity in entities], ["Cyrene"])
        self.assertEqual(entities[0].confidence, 0.985)

    def test_uses_batch_pipeline_for_token_aware_chunks(self):
        paragraph = "benzene ethanol acetone methanol"
        pipeline = BatchTokenClassificationPipeline()
        document = ExtractedDocument(
            filename="screening.txt",
            file_type="txt",
            text=paragraph,
            pages=[ExtractedTextPage(page_number=1, text=paragraph, start_char=0, end_char=len(paragraph))],
            sentences=[TextSegment(text=paragraph, page_number=1, start_char=0, end_char=len(paragraph))],
        )
        backend = TransformerChemicalNERBackend(
            max_tokens_per_chunk=2,
            stride_tokens=0,
            min_chars_per_chunk=1,
            batch_size=8,
            ner_pipeline=pipeline,
            tokenizer=FakeOffsetTokenizer(),
            enable_result_cache=False,
        )

        entities = backend.extract(document)

        self.assertEqual(len(pipeline.calls), 1)
        self.assertEqual(pipeline.calls[0][1], 8)
        self.assertEqual(len(pipeline.calls[0][0]), 2)
        self.assertEqual([entity.surface_form for entity in entities], ["benzene", "acetone"])


if __name__ == "__main__":
    unittest.main()
