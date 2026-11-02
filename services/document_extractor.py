"""Document text extraction service for uploaded PDF and TXT files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz

from services.pipeline_logger import PipelineLogger


class DocumentExtractionError(RuntimeError):
    """Raised when an uploaded document cannot be converted to text."""


@dataclass(frozen=True)
class ExtractedTextPage:
    page_number: int
    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class TextSegment:
    text: str
    page_number: int
    start_char: int
    end_char: int


@dataclass(frozen=True)
class ExtractedDocument:
    filename: str
    file_type: str
    text: str
    pages: list[ExtractedTextPage]
    sentences: list[TextSegment]


@dataclass(frozen=True)
class TextStatistics:
    characters: int
    words: int
    pages: int
    sentences: int


def extract_text_from_upload(uploaded_file, logger: PipelineLogger | None = None) -> ExtractedDocument:
    """Extract text from a Streamlit UploadedFile containing a PDF or TXT."""
    filename = uploaded_file.name or "uploaded_document"
    extension = Path(filename).suffix.lower()
    content = uploaded_file.getvalue()

    if not content:
        raise DocumentExtractionError("The uploaded file is empty.")

    stage = "PDF extraction" if extension == ".pdf" else "TXT extraction"
    if logger:
        logger.start_stage(stage, metadata={"filename": filename, "file_size_bytes": len(content)})

    try:
        if extension == ".pdf":
            page_texts = _extract_pdf_pages(content)
            file_type = "pdf"
        elif extension == ".txt":
            page_texts = [_extract_txt_text(content)]
            file_type = "txt"
        else:
            raise DocumentExtractionError("Only PDF and TXT files are supported.")

        if logger:
            logger.message(
                stage,
                f"{len(page_texts)} pages extracted",
                metadata={"page_count": len(page_texts)},
            )

        cleaned_text, pages = _build_document_text(page_texts)
        if not cleaned_text:
            raise DocumentExtractionError("No readable text was found in the uploaded file.")

        document = ExtractedDocument(
            filename=filename,
            file_type=file_type,
            text=cleaned_text,
            pages=pages,
            sentences=_split_sentences(pages),
        )
    except Exception:
        if logger:
            logger.end_stage(stage, metadata={"status": "error"})
        raise

    if logger:
        logger.end_stage(
            stage,
            metadata={
                "status": "success",
                "page_count": len(document.pages),
                "sentence_count": len(document.sentences),
                "character_count": len(document.text),
            },
        )

    return document


def get_text_statistics(document: ExtractedDocument) -> TextStatistics:
    """Return lightweight text stats for UI and downstream pipeline checks."""
    return TextStatistics(
        characters=len(document.text),
        words=len(document.text.split()),
        pages=len(document.pages),
        sentences=len(document.sentences),
    )


def _extract_pdf_pages(content: bytes) -> list[str]:
    try:
        with fitz.open(stream=content, filetype="pdf") as document:
            return [page.get_text("text") for page in document]
    except Exception as exc:
        raise DocumentExtractionError("Could not read the PDF file.") from exc


def _extract_txt_text(content: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DocumentExtractionError("Could not decode the TXT file.")


def _normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _build_document_text(page_texts: list[str]) -> tuple[str, list[ExtractedTextPage]]:
    document_parts = []
    pages = []
    cursor = 0

    for page_number, raw_text in enumerate(page_texts, start=1):
        page_text = _normalize_text(raw_text)
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
                page_number=page_number,
                text=page_text,
                start_char=start_char,
                end_char=cursor,
            )
        )

    return "".join(document_parts), pages


def _split_sentences(pages: list[ExtractedTextPage]) -> list[TextSegment]:
    sentences = []
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
