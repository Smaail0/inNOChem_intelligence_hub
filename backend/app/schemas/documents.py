from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractedPageResponse(BaseModel):
    page_number: int
    text: str
    start_char: int
    end_char: int


class TextStatisticsResponse(BaseModel):
    characters: int
    words: int
    pages: int
    sentences: int


class DocumentExtractionResponse(BaseModel):
    filename: str
    file_type: str
    text_preview: str
    statistics: TextStatisticsResponse
    pages: list[ExtractedPageResponse] = Field(default_factory=list)

