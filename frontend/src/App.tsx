import { useEffect, useState } from "react";
import { extractDocument, extractNerCandidates, getSystemStatus } from "./api/client";
import { CandidateReviewPanel } from "./components/CandidateReviewPanel";
import { DocumentUploadPanel } from "./components/DocumentUploadPanel";
import type { DocumentExtractionResponse, NerExtractionResponse, SystemStatus } from "./types";

const workflowStages = [
  "Upload document",
  "Extract text",
  "Recognize chemicals",
  "Human review",
  "Validate with PubChem",
  "Enrich intelligence",
  "Retrieve evidence",
  "Explain and screen"
];

export default function App() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [extraction, setExtraction] = useState<DocumentExtractionResponse | null>(null);
  const [extracting, setExtracting] = useState(false);
  const [extractionError, setExtractionError] = useState<string | null>(null);
  const [nerResult, setNerResult] = useState<NerExtractionResponse | null>(null);
  const [nerLoading, setNerLoading] = useState(false);
  const [nerError, setNerError] = useState<string | null>(null);
  const [useTransformer, setUseTransformer] = useState(true);
  const [skipReferences, setSkipReferences] = useState(true);
  const [limitChunksForDebugging, setLimitChunksForDebugging] = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function loadStatus() {
      try {
        const nextStatus = await getSystemStatus();
        if (!cancelled) {
          setStatus(nextStatus);
          setError(null);
        }
      } catch (exc) {
        if (!cancelled) {
          setError(exc instanceof Error ? exc.message : "Could not reach the API.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    loadStatus();
    return () => {
      cancelled = true;
    };
  }, []);

  async function handleExtractDocument() {
    if (!selectedFile) {
      return;
    }
    setExtracting(true);
    setExtractionError(null);
    setExtraction(null);
    setNerResult(null);
    setNerError(null);
    try {
      const result = await extractDocument(selectedFile);
      setExtraction(result);
    } catch (exc) {
      setExtractionError(exc instanceof Error ? exc.message : "Document extraction failed.");
    } finally {
      setExtracting(false);
    }
  }

  async function handleRunNer() {
    if (!extraction) {
      return;
    }
    setNerLoading(true);
    setNerError(null);
    setNerResult(null);
    try {
      const result = await extractNerCandidates(extraction, {
        useTransformer,
        skipReferences,
        limitChunksForDebugging
      });
      setNerResult(result);
    } catch (exc) {
      setNerError(exc instanceof Error ? exc.message : "Chemical recognition failed.");
    } finally {
      setNerLoading(false);
    }
  }

  const readyCount = status?.resources.filter((resource) => resource.ready).length ?? 0;
  const resourceCount = status?.resources.length ?? 0;

  return (
    <main className="app-shell">
      <section className="hero-panel">
        <div className="eyebrow">React + FastAPI migration</div>
        <div className="hero-grid">
          <div>
            <h1>InNOChem Intelligence Hub</h1>
            <p>
              A guided chemistry workflow for extraction, validation, provider-backed
              intelligence, evidence retrieval, controlled explanation, and future solvent
              alternative recommendation.
            </p>
          </div>
          <div className="status-card">
            <span className="status-label">Backend status</span>
            <strong>{loading ? "Checking..." : error ? "Needs attention" : "Connected"}</strong>
            <p>{error ?? `${readyCount}/${resourceCount} lightweight resources ready`}</p>
          </div>
        </div>
      </section>

      <DocumentUploadPanel
        error={extractionError}
        extraction={extraction}
        loading={extracting}
        onFileChange={setSelectedFile}
        onSubmit={handleExtractDocument}
        selectedFileName={selectedFile?.name ?? null}
      />

      <CandidateReviewPanel
        disabled={!extraction}
        error={nerError}
        limitChunksForDebugging={limitChunksForDebugging}
        loading={nerLoading}
        onLimitChunksForDebuggingChange={setLimitChunksForDebugging}
        onRunNer={handleRunNer}
        onSkipReferencesChange={setSkipReferences}
        onUseTransformerChange={setUseTransformer}
        result={nerResult}
        skipReferences={skipReferences}
        useTransformer={useTransformer}
      />

      <section className="workflow-card">
        <div className="section-heading">
          <span>Workflow</span>
          <h2>Clear steps, no hidden model work</h2>
        </div>
        <div className="stage-grid">
          {workflowStages.map((stage, index) => (
            <article className="stage-tile" key={stage}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <h3>{stage}</h3>
            </article>
          ))}
        </div>
      </section>

      <section className="two-column">
        <article className="panel">
          <div className="section-heading">
            <span>AI System</span>
            <h2>Lightweight readiness checks</h2>
          </div>
          {loading && <p className="muted">Checking API resources...</p>}
          {error && <p className="error-text">{error}</p>}
          {status?.resources.map((resource) => (
            <div className="resource-row" key={resource.name}>
              <div>
                <strong>{resource.name}</strong>
                <p>{resource.detail}</p>
              </div>
              <span className={resource.ready ? "pill ready" : "pill warning"}>
                {resource.ready ? "Ready" : "Check"}
              </span>
            </div>
          ))}
        </article>

        <article className="panel">
          <div className="section-heading">
            <span>Configured models</span>
            <h2>Loaded only when needed</h2>
          </div>
          {status ? (
            Object.entries(status.configured_models).map(([name, model]) => (
              <div className="model-row" key={name}>
                <span>{name.replaceAll("_", " ")}</span>
                <strong>{model || "not configured"}</strong>
              </div>
            ))
          ) : (
            <p className="muted">Model configuration appears after the API responds.</p>
          )}
        </article>
      </section>
    </main>
  );
}
