import type { NerExtractionResponse } from "../types";

type CandidateReviewPanelProps = {
  disabled: boolean;
  error: string | null;
  loading: boolean;
  result: NerExtractionResponse | null;
  useTransformer: boolean;
  skipReferences: boolean;
  limitChunksForDebugging: boolean;
  onUseTransformerChange: (value: boolean) => void;
  onSkipReferencesChange: (value: boolean) => void;
  onLimitChunksForDebuggingChange: (value: boolean) => void;
  onRunNer: () => void;
};

export function CandidateReviewPanel({
  disabled,
  error,
  loading,
  result,
  useTransformer,
  skipReferences,
  limitChunksForDebugging,
  onUseTransformerChange,
  onSkipReferencesChange,
  onLimitChunksForDebuggingChange,
  onRunNer
}: CandidateReviewPanelProps) {
  const topRows = result?.candidate_review_rows.slice(0, 20) ?? [];

  return (
    <section className="panel candidate-panel">
      <div className="section-heading">
        <span>Step 2</span>
        <h2>Chemical recognition and local candidate review</h2>
      </div>
      <p className="muted">
        Run chemistry NER over the extracted pages, then review locally deduplicated
        candidates before any PubChem validation.
      </p>

      <div className="option-grid">
        <label>
          <input
            checked={useTransformer}
            onChange={(event) => onUseTransformerChange(event.target.checked)}
            type="checkbox"
          />
          Use transformer chemical NER
        </label>
        <label>
          <input
            checked={skipReferences}
            onChange={(event) => onSkipReferencesChange(event.target.checked)}
            type="checkbox"
          />
          Skip references
        </label>
        <label>
          <input
            checked={limitChunksForDebugging}
            onChange={(event) => onLimitChunksForDebuggingChange(event.target.checked)}
            type="checkbox"
          />
          Debug limit chunks
        </label>
      </div>

      <button className="secondary-action" disabled={disabled || loading} onClick={onRunNer}>
        {loading ? "Running chemical recognition..." : "Run Chemical Recognition"}
      </button>

      {error && <p className="error-text">{error}</p>}

      {result && (
        <div className="candidate-result">
          <div className="metric-strip light">
            <Metric label="Raw mentions" value={result.raw_entity_count.toLocaleString()} />
            <Metric label="Candidates" value={result.candidate_count.toLocaleString()} />
            <Metric label="Backend" value={result.backend.replaceAll("_", " ")} />
          </div>
          <div className="candidate-table-wrap">
            <table className="candidate-table">
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Type</th>
                  <th>Occurrences</th>
                  <th>Confidence</th>
                  <th>Recommended</th>
                </tr>
              </thead>
              <tbody>
                {topRows.map((row) => (
                  <tr key={row.normalized_name}>
                    <td>
                      <strong>{row.normalized_name}</strong>
                      <span>{row["aliases/surface_forms"]}</span>
                    </td>
                    <td>{row.candidate_type}</td>
                    <td>{row.occurrence_count}</td>
                    <td>{row.average_confidence.toFixed(2)}</td>
                    <td>{row.include_for_pubchem_validation ? "Yes" : "No"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-card light">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

