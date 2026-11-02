import re
import unittest

from services.chemistry_ner import _iter_token_aware_page_chunks
from services.document_extractor import ExtractedTextPage


class FakeOffsetTokenizer:
    model_max_length = 512

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        matches = list(re.finditer(r"\S+", text))
        result = {"input_ids": list(range(len(matches)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(match.start(), match.end()) for match in matches]
        return result


class TokenAwareChunkingTests(unittest.TestCase):
    def test_long_page_text_creates_reasonable_number_of_chunks(self):
        page_text = " ".join(f"compound{i}" for i in range(1200))
        chunks = _iter_token_aware_page_chunks(
            pages=[ExtractedTextPage(page_number=1, text=page_text, start_char=0, end_char=len(page_text))],
            tokenizer=FakeOffsetTokenizer(),
            max_tokens_per_chunk=384,
            stride_tokens=64,
            min_chars_per_chunk=100,
        )

        self.assertLessEqual(len(chunks), 5)
        self.assertGreater(len(chunks), 1)

    def test_short_sentences_are_batched_together(self):
        page_text = "DMF was used. NMP was evaluated. THF was compared."
        chunks = _iter_token_aware_page_chunks(
            pages=[ExtractedTextPage(page_number=1, text=page_text, start_char=0, end_char=len(page_text))],
            tokenizer=FakeOffsetTokenizer(),
            max_tokens_per_chunk=384,
            stride_tokens=64,
            min_chars_per_chunk=100,
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, page_text)

    def test_chunk_token_count_does_not_exceed_max(self):
        page_text = " ".join(f"token{i}" for i in range(900))
        chunks = _iter_token_aware_page_chunks(
            pages=[ExtractedTextPage(page_number=1, text=page_text, start_char=0, end_char=len(page_text))],
            tokenizer=FakeOffsetTokenizer(),
            max_tokens_per_chunk=384,
            stride_tokens=64,
            min_chars_per_chunk=100,
        )

        self.assertTrue(all(chunk.token_count <= 384 for chunk in chunks))

    def test_page_numbers_are_preserved(self):
        first_page = " ".join(f"first{i}" for i in range(50))
        second_page = " ".join(f"second{i}" for i in range(50))
        chunks = _iter_token_aware_page_chunks(
            pages=[
                ExtractedTextPage(page_number=3, text=first_page, start_char=100, end_char=100 + len(first_page)),
                ExtractedTextPage(page_number=4, text=second_page, start_char=1000, end_char=1000 + len(second_page)),
            ],
            tokenizer=FakeOffsetTokenizer(),
            max_tokens_per_chunk=384,
            stride_tokens=64,
            min_chars_per_chunk=100,
        )

        self.assertEqual([chunk.page_number for chunk in chunks], [3, 4])
        self.assertEqual(chunks[0].start_char, 100)
        self.assertEqual(chunks[1].start_char, 1000)


if __name__ == "__main__":
    unittest.main()
