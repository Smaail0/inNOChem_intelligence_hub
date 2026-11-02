from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.ner import (
    CandidateReviewRowResponse,
    ChemicalEntityResponse,
    NerExtractionRequest,
    NerExtractionResponse,
)
from services.candidate_review import (
    build_candidate_review_records,
    candidate_records_to_rows,
)
from services.chemistry_ner import (
    LexiconChemistryBackend,
    TransformerChemicalNERBackend,
    extract_chemical_entities,
)
from services.document_extractor import ExtractedDocument, ExtractedTextPage, TextSegment
from services.resource_manager import get_cached_ner_pipeline


router = APIRouter(prefix="/api/ner", tags=["ner"])


@router.post("/extract", response_model=NerExtractionResponse)
def extract_ner_candidates(request: NerExtractionRequest) -> NerExtractionResponse:
    if not request.pages:
        raise HTTPException(status_code=400, detail="At least one page is required for NER.")

    document = _document_from_request(request)
    backend_name = "transformer_chemical_ner" if request.use_transformer else "chemistry_lexicon"

    try:
        if request.use_transformer:
            handle = get_cached_ner_pipeline()
            resource = handle.resource
            backend = TransformerChemicalNERBackend(
                ner_pipeline=resource["pipeline"],
                tokenizer=resource["tokenizer"],
            )
        else:
            backend = LexiconChemistryBackend()

        entities = extract_chemical_entities(
            document,
            preferred_backend=backend,
            limit_chunks_for_debugging=request.limit_chunks_for_debugging,
            skip_references=request.skip_references,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Chemical NER failed: {type(exc).__name__}: {exc}",
        ) from exc

    candidate_records = build_candidate_review_records(entities)
    candidate_rows = candidate_records_to_rows(candidate_records)
    return NerExtractionResponse(
        raw_entity_count=len(entities),
        candidate_count=len(candidate_records),
        entities=[ChemicalEntityResponse(**entity.to_dict()) for entity in entities],
        candidate_review_rows=[
            CandidateReviewRowResponse.model_validate(row)
            for row in candidate_rows
        ],
        backend=backend_name,
    )


def _document_from_request(request: NerExtractionRequest) -> ExtractedDocument:
    pages: list[ExtractedTextPage] = []
    document_parts: list[str] = []
    cursor = 0

    for page_input in request.pages:
        page_text = page_input.text.strip()
        if not page_text:
            continue
        if document_parts:
            document_parts.append("\n\n")
            cursor += 2
        start_char = cursor
        document_parts.append(page_text)
        cursor += len(page_text)
        pages.append(
            ExtractedTextPage(
                page_number=page_input.page_number,
                text=page_text,
                start_char=start_char,
                end_char=cursor,
            )
        )

    text = "".join(document_parts)
    return ExtractedDocument(
        filename=request.filename,
        file_type=request.file_type,
        text=text,
        pages=pages,
        sentences=_split_sentences(pages),
    )


def _split_sentences(pages: list[ExtractedTextPage]) -> list[TextSegment]:
    sentences: list[TextSegment] = []
    terminators = ".!?"

    for page in pages:
        sentence_start = 0
        index = 0
        while index < len(page.text):
            is_terminal = page.text[index] in terminators
            is_last_char = index == len(page.text) - 1
            if is_terminal or is_last_char:
                end = index + 1
                sentence = page.text[sentence_start:end].strip()
                if sentence:
                    leading_space = len(page.text[sentence_start:end]) - len(
                        page.text[sentence_start:end].lstrip()
                    )
                    start_char = page.start_char + sentence_start + leading_space
                    sentences.append(
                        TextSegment(
                            text=sentence,
                            page_number=page.page_number,
                            start_char=start_char,
                            end_char=start_char + len(sentence),
                        )
                    )
                sentence_start = end
            index += 1

    return sentences


