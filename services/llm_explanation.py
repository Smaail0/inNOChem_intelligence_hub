from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

import requests

from core.config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_EXPLANATION_TIMEOUT_SECONDS,
    GEMINI_MODEL,
    OLLAMA_MODEL,
)
from services.provider_registry import get_provider_category, normalize_provider_name


DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_INTERACTIVE_TIMEOUT_SECONDS = 30.0
MAX_PROMPT_INTELLIGENCE_ROWS = 8
MAX_PROMPT_EVIDENCE_ROWS = 3
MAX_PROMPT_FIELD_CHARS = 300


@dataclass(frozen=True)
class ChemicalExplanationInput:
    validated_chemical: dict[str, Any]
    intelligence_rows: list[dict[str, Any]] = field(default_factory=list)
    coverage_row: dict[str, Any] | None = None
    confidence_row: dict[str, Any] | None = None
    evidence_rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChemicalExplanation:
    chemical: str
    pubchem_cid: int | None
    summary: str
    key_risks: list[str]
    sustainability_notes: list[str]
    regulatory_notes: list[str]
    evidence_used: list[str]
    missing_information: list[str]
    confidence_statement: str
    generated_by: str = "llm"
    warnings: list[str] = field(default_factory=list)
    mode_used: str = "local_llm_polished_explanation"
    timeout_seconds: float | None = None
    intelligence_rows_sent: int = 0
    evidence_rows_sent: int = 0
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_explanation_input(
    validated_chemical: Any,
    intelligence_records: Iterable[Any] | None = None,
    coverage_row: Any | None = None,
    confidence_row: Any | None = None,
    evidence_records: Iterable[Any] | None = None,
) -> ChemicalExplanationInput:
    chemical_dict = _to_dict(validated_chemical)

    cid = chemical_dict.get("pubchem_cid") or chemical_dict.get("cid")
    canonical = chemical_dict.get("canonical_name")
    original = chemical_dict.get("original_surface_form") or chemical_dict.get("chemical")

    intelligence_rows = [
        _to_dict(row)
        for row in (intelligence_records or [])
        if _record_matches_chemical(row, cid=cid, canonical_name=canonical, original_name=original)
    ]

    evidence_rows = [
        _to_dict(row)
        for row in (evidence_records or [])
        if _record_matches_chemical(row, cid=cid, canonical_name=canonical, original_name=original)
    ]

    return ChemicalExplanationInput(
        validated_chemical=chemical_dict,
        intelligence_rows=intelligence_rows,
        coverage_row=_to_dict(coverage_row) if coverage_row is not None else None,
        confidence_row=_to_dict(confidence_row) if confidence_row is not None else None,
        evidence_rows=evidence_rows,
    )


def generate_chemical_explanation(
    explanation_input: ChemicalExplanationInput,
    model: str = OLLAMA_MODEL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    timeout_seconds: float = MAX_INTERACTIVE_TIMEOUT_SECONDS,
) -> ChemicalExplanation:
    """
    Generate a controlled, evidence-grounded explanation with a local Ollama model.

    The LLM is not allowed to create new facts, scores, or alternatives.
    If the LLM fails or returns invalid JSON, a deterministic fallback is returned.
    """
    timeout_seconds = min(float(timeout_seconds), MAX_INTERACTIVE_TIMEOUT_SECONDS)
    prompt_input = compress_explanation_input(explanation_input)
    prompt = _build_prompt(prompt_input)

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = str(payload.get("response", "")).strip()

        parsed = _parse_json_response(raw_text)
        explanation = _explanation_from_model_output(
            parsed,
            explanation_input=prompt_input,
            generated_by=f"ollama:{model}",
        )

        return ChemicalExplanation(
            **{
                **explanation.to_dict(),
                "mode_used": "local_llm_polished_explanation",
                "timeout_seconds": timeout_seconds,
                "intelligence_rows_sent": len(prompt_input.intelligence_rows),
                "evidence_rows_sent": len(prompt_input.evidence_rows),
                "fallback_used": False,
            }
        )

    except Exception as exc:
        fallback = build_fallback_explanation(prompt_input)
        return ChemicalExplanation(
            **{
                **fallback.to_dict(),
                "generated_by": "deterministic_fallback",
                "mode_used": "local_llm_polished_explanation",
                "timeout_seconds": timeout_seconds,
                "intelligence_rows_sent": len(prompt_input.intelligence_rows),
                "evidence_rows_sent": len(prompt_input.evidence_rows),
                "fallback_used": True,
                "warnings": [
                    *fallback.warnings,
                    f"LLM explanation failed or returned invalid output: {type(exc).__name__}: {exc}",
                ],
            }
        )


def generate_gemini_chemical_explanation(
    explanation_input: ChemicalExplanationInput,
    api_key: str = GEMINI_API_KEY,
    model: str = GEMINI_MODEL,
    base_url: str = GEMINI_BASE_URL,
    timeout_seconds: float = GEMINI_EXPLANATION_TIMEOUT_SECONDS,
) -> ChemicalExplanation:
    """
    Generate a managed Gemini explanation from a compact validated evidence packet.

    The request must not include full source documents or unvalidated raw extraction text.
    If Gemini is unavailable, the deterministic explanation is returned immediately.
    """
    timeout_seconds = min(float(timeout_seconds), MAX_INTERACTIVE_TIMEOUT_SECONDS)
    prompt_input = compress_explanation_input(explanation_input)

    if not api_key:
        fallback = build_fallback_explanation(prompt_input)
        return ChemicalExplanation(
            **{
                **fallback.to_dict(),
                "generated_by": "deterministic_fallback",
                "mode_used": "gemini_2_5_flash_polished_explanation",
                "timeout_seconds": timeout_seconds,
                "intelligence_rows_sent": len(prompt_input.intelligence_rows),
                "evidence_rows_sent": len(prompt_input.evidence_rows),
                "fallback_used": True,
                "warnings": [
                    *fallback.warnings,
                    "Gemini explanation skipped because GEMINI_API_KEY is not configured.",
                ],
            }
        )

    endpoint = f"{base_url.rstrip('/')}/models/{model}:generateContent"
    prompt = _build_prompt(prompt_input)

    try:
        response = requests.post(
            endpoint,
            headers={"x-goog-api-key": api_key},
            json={
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topP": 0.9,
                    "maxOutputTokens": 1200,
                    "responseMimeType": "application/json",
                },
            },
            timeout=timeout_seconds,
        )
        if not response.ok:
            raise RuntimeError(_http_error_detail(response, "Gemini request failed"))
        raw_text = _extract_gemini_text(response.json())
        parsed = _parse_json_response(raw_text)
        explanation = _explanation_from_model_output(
            parsed,
            explanation_input=prompt_input,
            generated_by=f"gemini:{model}",
        )
        return ChemicalExplanation(
            **{
                **explanation.to_dict(),
                "mode_used": "gemini_2_5_flash_polished_explanation",
                "timeout_seconds": timeout_seconds,
                "intelligence_rows_sent": len(prompt_input.intelligence_rows),
                "evidence_rows_sent": len(prompt_input.evidence_rows),
                "fallback_used": False,
            }
        )
    except Exception as exc:
        fallback = build_fallback_explanation(prompt_input)
        return ChemicalExplanation(
            **{
                **fallback.to_dict(),
                "generated_by": "deterministic_fallback",
                "mode_used": "gemini_2_5_flash_polished_explanation",
                "timeout_seconds": timeout_seconds,
                "intelligence_rows_sent": len(prompt_input.intelligence_rows),
                "evidence_rows_sent": len(prompt_input.evidence_rows),
                "fallback_used": True,
                "warnings": [
                    *fallback.warnings,
                    f"Gemini explanation failed or returned invalid output: {type(exc).__name__}: {exc}",
                ],
            }
        )


def compress_explanation_input(
    explanation_input: ChemicalExplanationInput,
    max_intelligence_rows: int = MAX_PROMPT_INTELLIGENCE_ROWS,
    max_evidence_rows: int = MAX_PROMPT_EVIDENCE_ROWS,
    max_field_chars: int = MAX_PROMPT_FIELD_CHARS,
) -> ChemicalExplanationInput:
    intelligence_rows: list[dict[str, Any]] = []
    seen_attributes: set[tuple[str, str]] = set()

    for row in explanation_input.intelligence_rows:
        compact = _truncate_mapping(row, max_field_chars)
        source = normalize_provider_name(str(compact.get("source") or ""))
        attribute = str(compact.get("attribute") or "").lower()
        marker = (source, attribute)
        if marker in seen_attributes:
            continue
        seen_attributes.add(marker)
        intelligence_rows.append(compact)
        if len(intelligence_rows) >= max_intelligence_rows:
            break

    evidence_rows = [
        _truncate_mapping(row, max_field_chars)
        for row in explanation_input.evidence_rows[:max_evidence_rows]
    ]

    return ChemicalExplanationInput(
        validated_chemical=_truncate_mapping(explanation_input.validated_chemical, max_field_chars),
        intelligence_rows=intelligence_rows,
        coverage_row=_truncate_mapping(explanation_input.coverage_row or {}, max_field_chars)
        if explanation_input.coverage_row is not None
        else None,
        confidence_row=_truncate_mapping(explanation_input.confidence_row or {}, max_field_chars)
        if explanation_input.confidence_row is not None
        else None,
        evidence_rows=evidence_rows,
    )


def build_fallback_explanation(
    explanation_input: ChemicalExplanationInput,
) -> ChemicalExplanation:
    explanation_input = compress_explanation_input(explanation_input)
    chemical = explanation_input.validated_chemical
    coverage = explanation_input.coverage_row or {}
    confidence = explanation_input.confidence_row or {}

    name = (
        chemical.get("canonical_name")
        or chemical.get("original_surface_form")
        or chemical.get("chemical")
        or "Unknown chemical"
    )
    cid = _safe_int(chemical.get("pubchem_cid") or chemical.get("cid"))

    validation_status = chemical.get("validation_status") or "unknown"
    coverage_percent = coverage.get("coverage_percent")
    confidence_label = confidence.get("confidence_label") or coverage.get("confidence") or "unknown"
    recommendation_status = (
        confidence.get("recommendation_status")
        or coverage.get("recommendation_readiness")
        or "unknown"
    )

    provider_sources = sorted(
        {
            str(row.get("source"))
            for row in explanation_input.intelligence_rows
            if row.get("source")
        }
    )

    summary = (
        f"{name} has validation status '{validation_status}'"
        + (f" and PubChem CID {cid}" if cid is not None else "")
        + "."
    )

    if coverage_percent is not None:
        summary += f" Current knowledge coverage is {coverage_percent}%."

    key_risks = _safety_notes(
        name,
        explanation_input.intelligence_rows,
        limit=5,
    )

    sustainability_notes = _sustainability_notes(
        explanation_input.intelligence_rows,
        limit=5,
    )

    regulatory_notes = _regulatory_notes(
        explanation_input.intelligence_rows,
        limit=5,
    )

    evidence_used = _evidence_summaries(explanation_input.evidence_rows, limit=5)

    missing_information = []
    missing_information.extend(_as_str_list(coverage.get("missing_groups")))
    missing_information.extend(_as_str_list(confidence.get("missing_for_strong_recommendation")))
    missing_information = _unique(missing_information)

    if not missing_information:
        missing_information = ["No missing information was explicitly reported by the coverage layer."]

    confidence_statement = (
        f"Recommendation confidence is '{confidence_label}' with status "
        f"'{recommendation_status}'. This reflects available provider coverage, not a final "
        "human-approved recommendation."
    )

    warnings = []
    if not provider_sources:
        warnings.append("No provider facts were available for the fallback explanation.")
    if not evidence_used:
        warnings.append("No retrieved RAG evidence was available for the fallback explanation.")

    return ChemicalExplanation(
        chemical=str(name),
        pubchem_cid=cid,
        summary=summary,
        key_risks=key_risks or ["No safety risk facts were available in the provided input."],
        sustainability_notes=sustainability_notes
        or ["No sustainability facts were available in the provided input."],
        regulatory_notes=regulatory_notes
        or ["No regulatory facts were available in the provided input."],
        evidence_used=evidence_used
        or ["No retrieved evidence snippets were provided."],
        missing_information=missing_information,
        confidence_statement=confidence_statement,
        generated_by="deterministic_fallback",
        warnings=warnings,
        mode_used="fast_deterministic_explanation",
        timeout_seconds=None,
        intelligence_rows_sent=len(explanation_input.intelligence_rows),
        evidence_rows_sent=len(explanation_input.evidence_rows),
        fallback_used=True,
    )


def _build_prompt(explanation_input: ChemicalExplanationInput) -> str:
    data = json.dumps(explanation_input.to_dict(), ensure_ascii=False, indent=2)

    return f"""
You are InNOChem's controlled chemistry explanation assistant.

Your role is to explain validated chemical intelligence in clear language for chemists.

Rules:
1. Use only the provided input data.
2. Do not invent chemical facts.
3. Do not invent scores.
4. Do not invent regulatory status.
5. Do not invent safer alternatives.
6. Do not claim something is safe or unsafe unless supported by the provided data.
7. If information is missing, explicitly say it is missing.
8. If evidence is weak or unavailable, say so.
9. Keep the chemist in control of the final decision.
10. Return valid JSON only. Do not wrap it in Markdown.

Required JSON schema:
{{
  "summary": "short paragraph",
  "key_risks": ["risk or limitation based only on provided data"],
  "sustainability_notes": ["note based only on CHEM21/GSK/provider data"],
  "regulatory_notes": ["note based only on regulatory provider data"],
  "evidence_used": ["short reference to evidence snippets actually provided"],
  "missing_information": ["important missing data"],
  "confidence_statement": "short statement using the provided confidence/coverage data"
}}

Input data:
{data}
""".strip()


def _parse_json_response(raw_text: str) -> dict[str, Any]:
    if not raw_text:
        raise ValueError("empty LLM response")

    raw_text = _strip_markdown_json_fence(raw_text)

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            repaired = _repair_partial_json_object(raw_text)
            parsed = json.loads(repaired)
        else:
            candidate = raw_text[start : end + 1]
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                repaired = _repair_partial_json_object(candidate)
                parsed = json.loads(repaired)

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON is not an object")

    return parsed


def _strip_markdown_json_fence(raw_text: str) -> str:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()
    return text


def _repair_partial_json_object(raw_text: str) -> str:
    """
    Best-effort salvage for model JSON that was truncated inside a string or array.

    This does not invent facts; it only closes JSON syntax so parsed fields already
    emitted by the model can be used. Missing fields still fall back deterministically.
    """
    text = raw_text.strip()
    start = text.find("{")
    if start > 0:
        text = text[start:]
    if not text.startswith("{"):
        raise json.JSONDecodeError("No JSON object found", raw_text, 0)

    in_string = False
    escaped = False
    stack: list[str] = []
    result: list[str] = []

    for char in text:
        result.append(char)
        if escaped:
            escaped = False
            continue
        if char == "\\" and in_string:
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in ("}", "]") and stack and stack[-1] == char:
            stack.pop()

    if in_string:
        result.append('"')

    repaired = "".join(result).rstrip()
    repaired = re.sub(r",\s*$", "", repaired)

    while stack:
        closer = stack.pop()
        if repaired.endswith(","):
            repaired = repaired[:-1].rstrip()
        repaired += closer

    return repaired


def _extract_gemini_text(payload: Mapping[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates:
        raise ValueError("Gemini response has no candidates")

    content = candidates[0].get("content") or {}
    parts = content.get("parts") or []
    text_parts = [str(part.get("text", "")) for part in parts if part.get("text")]
    raw_text = "\n".join(text_parts).strip()
    if not raw_text:
        raise ValueError("Gemini response candidate has no text")
    return raw_text


def _http_error_detail(response: requests.Response, prefix: str) -> str:
    body = ""
    try:
        body = response.text
    except Exception:
        body = ""
    body = _compact_text(body, 600)
    return f"{prefix}: HTTP {response.status_code} {response.reason}. {body}".strip()


def _explanation_from_model_output(
    parsed: Mapping[str, Any],
    explanation_input: ChemicalExplanationInput,
    generated_by: str,
) -> ChemicalExplanation:
    fallback = build_fallback_explanation(explanation_input)

    summary = _safe_str(parsed.get("summary")) or fallback.summary
    key_risks = _safe_list(parsed.get("key_risks")) or fallback.key_risks
    sustainability_notes = (
        _safe_list(parsed.get("sustainability_notes"))
        or fallback.sustainability_notes
    )
    regulatory_notes = _safe_list(parsed.get("regulatory_notes")) or fallback.regulatory_notes
    evidence_used = _safe_list(parsed.get("evidence_used")) or fallback.evidence_used
    missing_information = (
        _safe_list(parsed.get("missing_information"))
        or fallback.missing_information
    )
    confidence_statement = (
        _safe_str(parsed.get("confidence_statement"))
        or fallback.confidence_statement
    )

    warnings = _validate_model_output_against_input(
        parsed=parsed,
        explanation_input=explanation_input,
    )

    return ChemicalExplanation(
        chemical=fallback.chemical,
        pubchem_cid=fallback.pubchem_cid,
        summary=summary,
        key_risks=key_risks,
        sustainability_notes=sustainability_notes,
        regulatory_notes=regulatory_notes,
        evidence_used=evidence_used,
        missing_information=missing_information,
        confidence_statement=confidence_statement,
        generated_by=generated_by,
        warnings=warnings,
    )


def _validate_model_output_against_input(
    parsed: Mapping[str, Any],
    explanation_input: ChemicalExplanationInput,
) -> list[str]:
    warnings: list[str] = []

    output_text = json.dumps(parsed, ensure_ascii=False).lower()
    input_text = json.dumps(explanation_input.to_dict(), ensure_ascii=False).lower()

    risky_phrases = [
        "safe to use",
        "is safe",
        "no risk",
        "no hazards",
        "regulatory approved",
        "recommended alternative",
        "best alternative",
    ]

    for phrase in risky_phrases:
        if phrase in output_text and phrase not in input_text:
            warnings.append(
                f"Model output contains potentially unsupported phrase: '{phrase}'."
            )

    return warnings


def _fact_summaries(
    rows: Iterable[Mapping[str, Any]],
    preferred_categories: set[str],
    limit: int,
    included_attribute_prefixes: set[str] | None = None,
    excluded_attribute_prefixes: set[str] | None = None,
) -> list[str]:
    summaries: list[str] = []

    for row in rows:
        source = normalize_provider_name(str(row.get("source") or ""))
        source_category = get_provider_category(source)
        if source_category not in preferred_categories:
            continue

        attribute = row.get("attribute")
        attribute_text = str(attribute or "").lower()
        if included_attribute_prefixes and not any(
            attribute_text.startswith(prefix) for prefix in included_attribute_prefixes
        ):
            continue
        if excluded_attribute_prefixes and any(
            attribute_text.startswith(prefix) for prefix in excluded_attribute_prefixes
        ):
            continue
        value = row.get("value")
        evidence = row.get("evidence")

        if attribute is None and value is None:
            continue

        text = f"{source.upper()}: {attribute} = {value}"
        if evidence:
            text += f" ({evidence})"

        summaries.append(text)

        if len(summaries) >= limit:
            break

    return summaries


def _safety_notes(chemical_name: str, rows: Iterable[Mapping[str, Any]], limit: int) -> list[str]:
    notes: list[str] = []
    ghs_descriptors: list[str] = []

    for row in rows:
        source = normalize_provider_name(str(row.get("source") or ""))
        if get_provider_category(source) != "safety":
            continue

        attribute = str(row.get("attribute") or "").lower()
        if attribute.startswith("regulatory_intelligence."):
            continue

        value = _compact_text(row.get("value"), MAX_PROMPT_FIELD_CHARS)
        evidence = _compact_text(row.get("evidence"), 160)
        if not value:
            continue

        if "ghs" in attribute or "classification" in attribute:
            ghs_descriptors.extend(_hazard_keywords(value))
            continue

        label = _readable_attribute_label(attribute)
        note = f"{source.upper()} reports {label}: {value}."
        if evidence:
            note += f" Evidence: {evidence}."
        notes.append(note)
        if len(notes) >= limit:
            break

    if ghs_descriptors:
        descriptors = ", ".join(_unique(ghs_descriptors)[:5])
        notes.insert(
            0,
            (
                f"{chemical_name} has reported GHS hazard information including {descriptors}. "
                "Full classifications should be reviewed in the original provider data before lab use."
            ),
        )

    return notes[:limit]


def _sustainability_notes(rows: Iterable[Mapping[str, Any]], limit: int) -> list[str]:
    notes: list[str] = []

    for row in rows:
        source = normalize_provider_name(str(row.get("source") or ""))
        if get_provider_category(source) != "sustainability":
            continue

        attribute = _readable_attribute_label(str(row.get("attribute") or ""))
        value = _compact_text(row.get("value"), MAX_PROMPT_FIELD_CHARS)
        evidence = _compact_text(row.get("evidence"), 160)
        if not value:
            continue

        note = f"{source.upper()} reports {attribute}: {value}."
        if evidence:
            note += f" Evidence: {evidence}."
        notes.append(note)
        if len(notes) >= limit:
            break

    return notes


def _regulatory_notes(rows: Iterable[Mapping[str, Any]], limit: int) -> list[str]:
    notes: list[str] = []

    for row in rows:
        source = normalize_provider_name(str(row.get("source") or ""))
        source_category = get_provider_category(source)
        attribute = str(row.get("attribute") or "").lower()
        if source_category != "regulatory" and not attribute.startswith("regulatory_intelligence."):
            continue

        value = _compact_text(row.get("value"), MAX_PROMPT_FIELD_CHARS)
        evidence = _compact_text(row.get("evidence"), 160)
        if not value:
            continue

        label = _readable_attribute_label(attribute)
        note = f"{source.upper()} reports {label}: {value}."
        if evidence:
            note += f" Evidence: {evidence}."
        notes.append(note)
        if len(notes) >= limit:
            break

    return notes


def _evidence_summaries(rows: Iterable[Mapping[str, Any]], limit: int) -> list[str]:
    summaries: list[str] = []

    for row in rows:
        snippet = (
            row.get("evidence_snippet")
            or row.get("snippet")
            or row.get("text")
            or row.get("evidence")
        )
        document = row.get("document_name") or row.get("document") or row.get("source_doc")
        page = row.get("page_number") or row.get("page")

        if not snippet:
            continue

        prefix = ""
        if document:
            prefix += str(document)
        if page:
            prefix += f", page {page}"
        if prefix:
            prefix += ": "

        summaries.append(prefix + _compact_text(snippet, 300))

        if len(summaries) >= limit:
            break

    return summaries


def _truncate_mapping(value: Mapping[str, Any], max_chars: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in dict(value).items():
        if isinstance(item, str):
            result[key] = _compact_text(item, max_chars)
        elif isinstance(item, Mapping):
            result[key] = _truncate_mapping(item, max_chars)
        elif isinstance(item, list):
            result[key] = [
                _compact_text(entry, max_chars) if isinstance(entry, str) else entry
                for entry in item[:20]
            ]
        else:
            result[key] = item
    return result


def _compact_text(value: Any, max_chars: int) -> str:
    if value is None:
        return ""
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _readable_attribute_label(attribute: str) -> str:
    text = attribute.split(".")[-1]
    text = text.replace("_", " ").replace("-", " ").strip()
    return text or "provider fact"


def _hazard_keywords(value: str) -> list[str]:
    text = value.lower()
    keywords = [
        ("flammability", ("flammable", "flam.", "fire")),
        ("irritation-related warnings", ("irritat", "eye damage", "skin")),
        ("toxicity-related warnings", ("toxic", "harmful", "fatal")),
        ("carcinogenicity-related warnings", ("carcinogen", "cancer")),
        ("reproductive toxicity warnings", ("reproductive", "fertility", "unborn")),
        ("environmental hazard warnings", ("aquatic", "environment")),
        ("aspiration hazard warnings", ("aspiration",)),
        ("corrosion warnings", ("corros",)),
    ]
    matches = [label for label, needles in keywords if any(needle in text for needle in needles)]
    return matches or ["provider-reported hazard classifications"]


def _record_matches_chemical(
    record: Any,
    *,
    cid: Any,
    canonical_name: Any,
    original_name: Any,
) -> bool:
    record_dict = _to_dict(record)
    nested_chemical = record_dict.get("chemical")

    candidate_cids = [
        record_dict.get("pubchem_cid"),
        record_dict.get("cid"),
    ]

    if isinstance(nested_chemical, Mapping):
        candidate_cids.extend(
            [
                nested_chemical.get("pubchem_cid"),
                nested_chemical.get("cid"),
            ]
        )

    cid_str = str(cid) if cid is not None else None
    if cid_str:
        for candidate_cid in candidate_cids:
            if candidate_cid is not None and str(candidate_cid) == cid_str:
                return True

    target_names = {
        _normalize_name(canonical_name),
        _normalize_name(original_name),
    }
    target_names.discard("")

    candidate_names = [
        record_dict.get("chemical_name"),
        record_dict.get("chemical"),
        record_dict.get("canonical_name"),
        record_dict.get("original_surface_form"),
        record_dict.get("name"),
    ]

    if isinstance(nested_chemical, Mapping):
        candidate_names.extend(
            [
                nested_chemical.get("canonical_name"),
                nested_chemical.get("original_surface_form"),
                nested_chemical.get("name"),
            ]
        )

    for candidate_name in candidate_names:
        if _normalize_name(candidate_name) in target_names:
            return True

    return False


def _to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, Mapping):
        return dict(value)

    if hasattr(value, "to_dict"):
        return value.to_dict()

    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)

    return {}


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default

    if isinstance(obj, Mapping):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _safe_list(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]

    if isinstance(value, str) and value.strip():
        return [value.strip()]

    return []


def _as_str_list(value: Any) -> list[str]:
    return _safe_list(value)


def _normalize_name(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _unique(values: Iterable[str]) -> list[str]:
    seen = set()
    result: list[str] = []

    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result
