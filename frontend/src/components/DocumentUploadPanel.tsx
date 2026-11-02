import type { ChangeEvent, FormEvent } from "react";
import type { DocumentExtractionResponse } from "../types";

type DocumentUploadPanelProps = {
  extraction: DocumentExtractionResponse | null;
  error: string | null;
  loading: boolean;
  selectedFileName: string | null;
  onFileChange: (file: File | null) => void;
  onSubmit: () => void;
};

export function DocumentUploadPanel({
  extraction,
  error,
  loading,
  selectedFileName,
  onFileChange,
  onSubmit
}: DocumentUploadPanelProps) {
  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    onFileChange(event.target.files?.[0] ?? null);
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit();
  }

  return (
    <section className="upload-panel">
      <div className="section-heading">
        <span>Step 1</span>
        <h2>Upload and extract document text</h2>
      </div>
      <p className="muted">
        This first React/FastAPI slice sends only the selected PDF or TXT file to the
        local backend, extracts structured text, and returns page-level statistics.
      </p>

      <form className="upload-form" onSubmit={handleSubmit}>
        <label className="file-drop">
          <input accept=".pdf,.txt" type="file" onChange={handleFileChange} />
          <strong>{selectedFileName ?? "Choose a PDF or TXT file"}</strong>
          <span>Local extraction through FastAPI. No NER model is loaded yet.</span>
        </label>
        <button disabled={loading || !selectedFileName} type="submit">
          {loading ? "Extracting..." : "Extract Text"}
        </button>
      </form>

      {error && <p className="error-text">{error}</p>}

      {extraction && (
        <div className="extraction-result">
          <div className="metric-strip">
            <Metric label="Characters" value={extraction.statistics.characters.toLocaleString()} />
            <Metric label="Words" value={extraction.statistics.words.toLocaleString()} />
            <Metric label="Pages" value={String(extraction.statistics.pages)} />
            <Metric label="Sentences" value={String(extraction.statistics.sentences)} />
          </div>
          <div className="preview-card">
            <div>
              <span className="status-label">Text preview</span>
              <h3>{extraction.filename}</h3>
            </div>
            <pre>{extraction.text_preview || "No preview text returned."}</pre>
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

