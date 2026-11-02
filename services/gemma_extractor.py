"""Optional local Ollama/Gemma cleanup service.

Gemma is not the scientific decision-maker. It may only clean up already
extracted candidate entities and must never receive full document text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests

from core.config import MAX_EXTRACTION_CHARS, OLLAMA_BASE_URL, OLLAMA_MODEL


class GemmaExtractionError(RuntimeError):
    """Raised when local Gemma extraction fails or returns unusable JSON."""


@dataclass(frozen=True)
class GemmaExtractionConfig:
    model: str = OLLAMA_MODEL
    base_url: str = OLLAMA_BASE_URL
    max_chars: int = MAX_EXTRACTION_CHARS


def extract_candidate_chemicals(
    text: str,
    config: GemmaExtractionConfig | None = None,
) -> dict[str, Any]:
    """Extract candidate chemical mentions with local Gemma via Ollama."""
    if not text.strip():
        raise GemmaExtractionError("No extracted text was provided to Gemma.")

    config = config or GemmaExtractionConfig()
    check_ollama_health(config)
    prompt = _build_prompt(text[: config.max_chars])
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(
            f"{config.base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise GemmaExtractionError(
            "Ollama request timed out while Gemma was extracting candidate chemicals. "
            "Try a shorter document or increase the request timeout later."
        ) from exc
    except requests.RequestException as exc:
        raise GemmaExtractionError(_format_ollama_request_error(exc, config.model)) from exc

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise GemmaExtractionError("Ollama returned a non-JSON API response.") from exc

    raw_response = response_payload.get("response", "")
    parsed = _parse_json_response(raw_response)
    return _normalize_extraction(parsed)


def cleanup_candidate_entities(
    entities: list[dict[str, Any]],
    config: GemmaExtractionConfig | None = None,
) -> dict[str, Any]:
    """Optionally clean up extracted entities without sending full documents."""
    if not entities:
        return {
            "candidate_chemicals": [],
            "process_summary": "No extracted entities were provided for optional LLM cleanup.",
            "limitations": ["Candidates are not yet validated by PubChem or RDKit."],
        }

    config = config or GemmaExtractionConfig()
    check_ollama_health(config)
    safe_entities = [
        {
            "surface_form": entity.get("surface_form", ""),
            "entity_type": entity.get("entity_type", ""),
            "sentence_context": entity.get("sentence_context", ""),
        }
        for entity in entities
    ]
    prompt = _build_cleanup_prompt(safe_entities)
    payload = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
        },
    }

    try:
        response = requests.post(
            f"{config.base_url.rstrip('/')}/api/generate",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
    except requests.Timeout as exc:
        raise GemmaExtractionError("Ollama request timed out during optional entity cleanup.") from exc
    except requests.RequestException as exc:
        raise GemmaExtractionError(_format_ollama_request_error(exc, config.model)) from exc

    try:
        response_payload = response.json()
    except ValueError as exc:
        raise GemmaExtractionError("Ollama returned a non-JSON API response.") from exc

    parsed = _parse_json_response(response_payload.get("response", ""))
    return _normalize_extraction(parsed)


def check_ollama_health(config: GemmaExtractionConfig | None = None) -> None:
    """Verify Ollama is reachable and the configured model is installed."""
    config = config or GemmaExtractionConfig()

    try:
        response = requests.get(f"{config.base_url.rstrip('/')}/api/tags", timeout=10)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise GemmaExtractionError(
            f"Ollama server is not reachable at {config.base_url}. Start Ollama with `ollama serve`."
        ) from exc
    except requests.Timeout as exc:
        raise GemmaExtractionError(
            f"Ollama health check timed out at {config.base_url}. Confirm the local server is responsive."
        ) from exc
    except requests.RequestException as exc:
        raise GemmaExtractionError(_format_ollama_request_error(exc, config.model)) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise GemmaExtractionError("Ollama health check returned a non-JSON response.") from exc

    installed_models = {
        model.get("name", "")
        for model in payload.get("models", [])
        if isinstance(model, dict)
    }
    if config.model not in installed_models:
        installed = ", ".join(sorted(installed_models)) or "none"
        raise GemmaExtractionError(
            f"Ollama model `{config.model}` was not found. Installed models: {installed}. "
            f"Run `ollama pull {config.model}` or set OLLAMA_MODEL to an installed Gemma model."
        )


def _build_prompt(text: str) -> str:
    return f"""
You are a language extraction component for InNOChem Intelligence Hub.
Your job is only to identify candidate chemical names mentioned in text.
Do not validate, rank, recommend, infer safety, or make scientific decisions.
PubChem and RDKit validation will happen later.

Return only valid JSON using exactly this schema:
{{
  "candidate_chemicals": [
    {{
      "name": "benzene",
      "context": "used as solvent",
      "confidence": 0.85
    }}
  ],
  "process_summary": "Short summary of what was extracted.",
  "limitations": [
    "Candidates are not yet validated by PubChem or RDKit."
  ]
}}

Rules:
- Include only names that appear to be chemicals, compounds, solvents, reagents, catalysts, polymers, or materials.
- Keep context short and grounded in the supplied text.
- Confidence is a language-extraction confidence from 0 to 1, not scientific validity.
- If no candidates are found, return an empty candidate_chemicals list.

Text:
\"\"\"{text}\"\"\"
""".strip()


def _build_cleanup_prompt(entities: list[dict[str, Any]]) -> str:
    return f"""
You are an optional cleanup component for InNOChem Intelligence Hub.
You are not validating chemicals and you are not making scientific decisions.
PubChem and RDKit validation will happen later.

You will receive only already-extracted candidate entities, not the full source document.
Return only valid JSON using this schema:
{{
  "candidate_chemicals": [
    {{
      "name": "benzene",
      "context": "used as solvent",
      "confidence": 0.85
    }}
  ],
  "process_summary": "Short summary of cleanup performed.",
  "limitations": [
    "Candidates are not yet validated by PubChem or RDKit."
  ]
}}

Candidate entities:
{json.dumps(entities, ensure_ascii=False)}
""".strip()


def _parse_json_response(raw_response: str) -> dict[str, Any]:
    if not raw_response.strip():
        raise GemmaExtractionError("Gemma returned an empty response instead of the expected JSON.")

    try:
        return json.loads(raw_response)
    except json.JSONDecodeError:
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise GemmaExtractionError(
                "Gemma did not return valid JSON. The response did not contain a JSON object."
            )

        try:
            return json.loads(raw_response[start : end + 1])
        except json.JSONDecodeError as exc:
            raise GemmaExtractionError(
                "Gemma returned malformed JSON. Try rerunning extraction or reducing the document size."
            ) from exc


def _normalize_extraction(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = payload.get("candidate_chemicals", [])
    if not isinstance(candidates, list):
        candidates = []

    normalized_candidates = []
    for item in candidates:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            continue

        normalized_candidates.append(
            {
                "name": name,
                "context": str(item.get("context", "")).strip(),
                "confidence": _coerce_confidence(item.get("confidence", 0)),
            }
        )

    limitations = payload.get("limitations")
    if not isinstance(limitations, list):
        limitations = []

    required_limitation = "Candidates are not yet validated by PubChem or RDKit."
    if required_limitation not in limitations:
        limitations.append(required_limitation)

    return {
        "candidate_chemicals": normalized_candidates,
        "process_summary": str(payload.get("process_summary", "")).strip()
        or "Candidate chemical names were extracted from the uploaded document text.",
        "limitations": [str(item) for item in limitations if str(item).strip()],
    }


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _format_ollama_request_error(exc: requests.RequestException, model: str) -> str:
    if exc.response is not None:
        detail = exc.response.text[:300].strip()
        if exc.response.status_code == 404:
            return (
                f"Ollama model `{model}` was not found by the generation API. "
                f"Run `ollama pull {model}` or set OLLAMA_MODEL to an installed Gemma model."
            )
        if detail:
            return f"Ollama request failed with HTTP {exc.response.status_code}. Ollama said: {detail}"
        return f"Ollama request failed with HTTP {exc.response.status_code}."

    return (
        "Could not complete the Ollama request. Confirm Ollama is running "
        f"and the Gemma model `{model}` is available."
    )
