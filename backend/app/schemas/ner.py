from __future__ import annotations

from pydantic import BaseModel, Field


class NerPageInput(BaseModel):
    page_number: int
    text: str
    start_char: int = 0
    end_char: int | None = None


class NerExtractionRequest(BaseModel):
    filename: str = "uploaded_document"
    file_type: str = "txt"
    pages: list[NerPageInput]
    use_transformer: bool = True
    skip_references: bool = True
    limit_chunks_for_debugging: bool = False


class ChemicalEntityResponse(BaseModel):
    surface_form: str
    entity_type: str
    sentence_context: str
    start_char: int | None = None
    end_char: int | None = None
    page_number: int | None = None
    extraction_method: str
    chunk_id: int | None = None
    confidence: float | None = None


class CandidateReviewRowResponse(BaseModel):
    include_for_pubchem_validation: bool
    normalized_name: str
    aliases_surface_forms: str = Field(alias="aliases/surface_forms")
    occurrence_count: int
    pages_found: str
    representative_contexts: str
    extraction_methods: str
    average_confidence: float
    candidate_priority_score: float
    priority_reason: str
    candidate_type: str
    pubchem_validation_recommended: bool
    pubchem_recommendation_reason: str

    model_config = {"populate_by_name": True}


class NerExtractionResponse(BaseModel):
    raw_entity_count: int
    candidate_count: int
    entities: list[ChemicalEntityResponse] = Field(default_factory=list)
    candidate_review_rows: list[CandidateReviewRowResponse] = Field(default_factory=list)
    backend: str

