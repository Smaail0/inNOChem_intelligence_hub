import tempfile
import unittest

from services.chemistry_ner import (
    TransformerChemicalNERBackend,
    _build_ner_cache_key,
)
from services.document_extractor import ExtractedDocument, ExtractedTextPage, TextSegment


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


class KnownTermsBatchPipeline:
    def __init__(self, terms):
        self.terms = terms
        self.calls = 0

    def __call__(self, texts, batch_size=None):
        self.calls += 1
        if isinstance(texts, str):
            return self._outputs_for_text(texts)
        return [self._outputs_for_text(text) for text in texts]

    def _outputs_for_text(self, text):
        outputs = []
        for term in self.terms:
            if term in text:
                start = text.index(term)
                outputs.append(
                    {
                        "entity_group": "CHEMICAL",
                        "score": 0.91,
                        "word": term,
                        "start": start,
                        "end": start + len(term),
                    }
                )
        return outputs


class FailingPipeline:
    def __call__(self, texts, batch_size=None):
        raise AssertionError("Pipeline should not be called on NER cache hit.")


class NerCacheAndFilteringTests(unittest.TestCase):
    def test_batch_inference_preserves_all_entities(self):
        text = "benzene ethanol acetone methanol"
        pipeline = KnownTermsBatchPipeline(["benzene", "ethanol", "acetone", "methanol"])
        backend = TransformerChemicalNERBackend(
            max_tokens_per_chunk=1,
            stride_tokens=0,
            min_chars_per_chunk=1,
            batch_size=2,
            ner_pipeline=pipeline,
            tokenizer=FakeOffsetTokenizer(),
            enable_result_cache=False,
        )

        entities = backend.extract(_document(text), skip_references=False)

        self.assertEqual([entity.surface_form for entity in entities], ["benzene", "ethanol", "acetone", "methanol"])
        self.assertEqual([entity.chunk_id for entity in entities], [1, 2, 3, 4])
        self.assertTrue(all(entity.page_number == 1 for entity in entities))

    def test_references_section_is_skipped_when_enabled(self):
        text = "benzene was evaluated.\nReferences\nethanol was cited."
        pipeline = KnownTermsBatchPipeline(["benzene", "ethanol"])
        backend = TransformerChemicalNERBackend(
            ner_pipeline=pipeline,
            tokenizer=FakeOffsetTokenizer(),
            enable_result_cache=False,
        )

        entities = backend.extract(_document(text), skip_references=True)

        self.assertEqual([entity.surface_form for entity in entities], ["benzene"])

    def test_cache_key_changes_when_model_or_chunk_config_changes(self):
        document = _document("benzene was evaluated.")

        base = _build_ner_cache_key(document, "model-a", 384, 64, True, None)
        changed_model = _build_ner_cache_key(document, "model-b", 384, 64, True, None)
        changed_chunk = _build_ner_cache_key(document, "model-a", 256, 64, True, None)

        self.assertNotEqual(base, changed_model)
        self.assertNotEqual(base, changed_chunk)

    def test_cache_hit_avoids_model_call(self):
        document = _document("benzene was evaluated.")
        with tempfile.TemporaryDirectory() as cache_dir:
            first_pipeline = KnownTermsBatchPipeline(["benzene"])
            first_backend = TransformerChemicalNERBackend(
                ner_pipeline=first_pipeline,
                tokenizer=FakeOffsetTokenizer(),
                result_cache_dir=cache_dir,
            )
            first_entities = first_backend.extract(document, skip_references=False)

            second_backend = TransformerChemicalNERBackend(
                ner_pipeline=FailingPipeline(),
                tokenizer=FakeOffsetTokenizer(),
                result_cache_dir=cache_dir,
            )
            cached_entities = second_backend.extract(document, skip_references=False)

        self.assertEqual(first_pipeline.calls, 1)
        self.assertEqual([entity.surface_form for entity in cached_entities], [entity.surface_form for entity in first_entities])

    def test_batch_output_matches_single_chunk_output_structure(self):
        text = "benzene and ethanol were evaluated."
        pipeline = KnownTermsBatchPipeline(["benzene", "ethanol"])
        backend = TransformerChemicalNERBackend(
            max_tokens_per_chunk=128,
            stride_tokens=0,
            min_chars_per_chunk=1,
            batch_size=8,
            ner_pipeline=pipeline,
            tokenizer=FakeOffsetTokenizer(),
            enable_result_cache=False,
        )

        entities = backend.extract(_document(text), skip_references=False)

        self.assertEqual(len(entities), 2)
        self.assertEqual(entities[0].surface_form, "benzene")
        self.assertEqual(entities[0].start_char, 0)
        self.assertEqual(entities[0].sentence_context, text)
        self.assertEqual(entities[0].chunk_id, 1)


def _document(text):
    return ExtractedDocument(
        filename="test.txt",
        file_type="txt",
        text=text,
        pages=[ExtractedTextPage(page_number=1, text=text, start_char=0, end_char=len(text))],
        sentences=[TextSegment(text=text, page_number=1, start_char=0, end_char=len(text))],
    )


if __name__ == "__main__":
    unittest.main()
