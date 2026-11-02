import type {
  DocumentExtractionResponse,
  NerExtractionResponse,
  SystemStatus
} from "../types";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";

export async function getSystemStatus(): Promise<SystemStatus> {
  const response = await fetch(`${API_BASE_URL}/api/system/status`);
  if (!response.ok) {
    throw new Error(`API status request failed: ${response.status}`);
  }
  return response.json() as Promise<SystemStatus>;
}

export async function extractDocument(file: File): Promise<DocumentExtractionResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/documents/extract`, {
    method: "POST",
    body: formData
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Document extraction failed: ${response.status}`);
  }
  return response.json() as Promise<DocumentExtractionResponse>;
}

export async function extractNerCandidates(
  extraction: DocumentExtractionResponse,
  options: { useTransformer: boolean; skipReferences: boolean; limitChunksForDebugging: boolean }
): Promise<NerExtractionResponse> {
  const response = await fetch(`${API_BASE_URL}/api/ner/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      filename: extraction.filename,
      file_type: extraction.file_type,
      pages: extraction.pages,
      use_transformer: options.useTransformer,
      skip_references: options.skipReferences,
      limit_chunks_for_debugging: options.limitChunksForDebugging
    })
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `NER extraction failed: ${response.status}`);
  }
  return response.json() as Promise<NerExtractionResponse>;
}
