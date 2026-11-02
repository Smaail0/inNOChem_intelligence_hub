from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.app.schemas.documents import (
    DocumentExtractionResponse,
    ExtractedPageResponse,
    TextStatisticsResponse,
)
from services.document_extractor import (
    DocumentExtractionError,
    extract_text_from_upload,
    get_text_statistics,
)


router = APIRouter(prefix="/api/documents", tags=["documents"])


@dataclass(frozen=True)
class _UploadedBytes:
    name: str
    content: bytes

    def getvalue(self) -> bytes:
        return self.content


@router.post("/extract", response_model=DocumentExtractionResponse)
async def extract_document(file: UploadFile = File(...)) -> DocumentExtractionResponse:
    content = await file.read()
    upload = _UploadedBytes(name=file.filename or "uploaded_document", content=content)
    try:
        document = extract_text_from_upload(upload)
    except DocumentExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    stats = get_text_statistics(document)
    return DocumentExtractionResponse(
        filename=document.filename,
        file_type=document.file_type,
        text_preview=document.text[:3000],
        statistics=TextStatisticsResponse(
            characters=stats.characters,
            words=stats.words,
            pages=stats.pages,
            sentences=stats.sentences,
        ),
        pages=[
            ExtractedPageResponse(
                page_number=page.page_number,
                text=page.text,
                start_char=page.start_char,
                end_char=page.end_char,
            )
            for page in document.pages
        ],
    )

