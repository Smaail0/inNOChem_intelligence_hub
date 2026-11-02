export type ResourceStatus = {
  name: string;
  ready: boolean;
  detail: string;
};

export type SystemStatus = {
  app_name: string;
  api_version: string;
  status: string;
  resources: ResourceStatus[];
  configured_models: Record<string, string>;
};

export type ExtractedPage = {
  page_number: number;
  text: string;
  start_char: number;
  end_char: number;
};

export type TextStatistics = {
  characters: number;
  words: number;
  pages: number;
  sentences: number;
};

export type DocumentExtractionResponse = {
  filename: string;
  file_type: string;
  text_preview: string;
  statistics: TextStatistics;
  pages: ExtractedPage[];
};

export type CandidateReviewRow = {
  include_for_pubchem_validation: boolean;
  normalized_name: string;
  "aliases/surface_forms": string;
  occurrence_count: number;
  pages_found: string;
  representative_contexts: string;
  extraction_methods: string;
  average_confidence: number;
  candidate_priority_score: number;
  priority_reason: string;
  candidate_type: string;
  pubchem_validation_recommended: boolean;
  pubchem_recommendation_reason: string;
};

export type NerExtractionResponse = {
  raw_entity_count: number;
  candidate_count: number;
  backend: string;
  candidate_review_rows: CandidateReviewRow[];
};
