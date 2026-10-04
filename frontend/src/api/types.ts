/**
 * TypeScript types mirroring the FastAPI schemas in `app/schemas/`.
 *
 * These are hand-written rather than generated. For a project this size that is
 * fine and easier to read; if the API grows, generate them with
 * `npx openapi-typescript http://localhost:8000/openapi.json -o src/api/schema.ts`.
 */

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------
export interface LLMStatus {
  configured: boolean;
  provider: string;
  model: string;
  /** Plain-language explanation when `configured` is false. */
  reason: string | null;
  /** Which env var is missing, when that is the cause. */
  env_var?: string | null;
}

export interface HealthResponse {
  status: string;
  environment: string;
  database: string;
  embedding_provider: string;
  embedding_dim: number;
  llm_enabled: boolean;
  llm_status?: LLMStatus;
  candidate_count: number;
  chunk_count: number;
  details: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------
export interface IngestSummary {
  candidate_id: string;
  created: boolean;
  chunk_count: number;
  skill_count: number;
  text_length: number;
  full_name: string | null;
  extraction_method: string | null;
  warnings: string[];
}

export interface UploadResponse {
  batch_size: number;
  succeeded: number;
  failed: number;
  results: IngestSummary[];
  errors: { filename: string; error: string }[];
}

// ---------------------------------------------------------------------------
// Candidates
// ---------------------------------------------------------------------------
export interface SkillOut {
  canonical: string;
  display: string;
  occurrences: number;
  is_core: boolean;
}

export interface CandidateSummary {
  id: string;
  full_name: string | null;
  current_title: string | null;
  current_company: string | null;
  location: string | null;
  total_years_experience: number | null;
  seniority: string | null;
  email: string | null;
  phone: string | null;
  summary: string | null;
  status: string;
  created_at: string | null;
}

export interface ChunkOut {
  id: string;
  chunk_index: number;
  section: string;
  heading: string | null;
  content: string;
  token_count: number;
}

export interface CandidateDetail extends CandidateSummary {
  links: Record<string, string>;
  education: { level?: string; field?: string; institution?: string; gpa?: number };
  languages: string[];
  skills: SkillOut[];
  resume_filename: string | null;
  extraction_method: string | null;
  text_length: number;
  chunk_count: number;
  chunks: ChunkOut[];
  updated_at: string | null;
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------
export interface EvidenceOut {
  chunk_id: string;
  candidate_id: string;
  chunk_index: number;
  section: string;
  heading: string | null;
  content: string;
  score: number;
  vector_score: number | null;
  keyword_score: number | null;
  rerank_score: number | null;
}

export interface CandidateResult {
  id: string;
  full_name: string | null;
  current_title: string | null;
  current_company: string | null;
  location: string | null;
  total_years_experience: number | null;
  seniority: string | null;
  summary: string | null;
  email: string | null;
  phone: string | null;
  score: number;
  skills: SkillOut[];
  evidence: EvidenceOut[];
}

export interface CitationOut {
  index: number;
  candidate_id: string;
  candidate_name: string | null;
  section: string | null;
  heading: string | null;
  excerpt: string;
}

export interface AnswerOut {
  text: string;
  citations: CitationOut[];
  used_llm: boolean;
  model: string | null;
  latency_ms: number | null;
  insufficient_evidence: boolean;
  had_invalid_citations: boolean;
}

/** Mirrors `app/schemas/search.py::SearchRequest`. */
export interface SearchRequest {
  query?: string;
  skills?: string[];
  skills_mode?: "all" | "any";
  location?: string | null;
  min_years_experience?: number | null;
  max_years_experience?: number | null;
  current_title?: string | null;
  company?: string | null;
  seniority?: string[];
  languages?: string[];
  education_level?: string[];
  has_email?: boolean | null;
  has_phone?: boolean | null;
  top_k?: number;
  use_rerank?: boolean | null;
  generate_answer?: boolean;
}

export interface SearchResponse {
  query: string;
  total: number;
  results: CandidateResult[];
  answer: AnswerOut | null;
  filters_applied: Record<string, unknown>;
  retrieval_ms: number;
  total_ms: number;
  diagnostics: Record<string, number | string | null>;
}

// ---------------------------------------------------------------------------
// JD matching (Phase 4)
// ---------------------------------------------------------------------------
export interface JDMatchResult {
  candidate_id: string;
  full_name: string | null;
  current_title: string | null;
  score: number;
  skills_match: number | null;
  experience_match: number | null;
  seniority_match: number | null;
  missing_critical_skills: string[];
  strengths: Record<string, unknown>[];
  summary: string | null;
  evidence: EvidenceOut[];
}

/** Mirrors `app/schemas/search.py::JDMatchRequest`. */
export interface JDMatchRequest {
  job_description: string;
  top_k?: number;
  min_score?: number;
  generate_analysis?: boolean;
}

export interface JDMatchResponse {
  job_description_excerpt: string;
  total: number;
  results: JDMatchResult[];
  used_llm: boolean;
  retrieval_ms: number;
  total_ms: number;
}

// ---------------------------------------------------------------------------
// Facets
// ---------------------------------------------------------------------------
export interface SkillFacet {
  canonical: string;
  display: string;
  candidate_count: number;
  is_core: boolean;
}

export interface FilterOptions {
  seniority: { value: string; count: number }[];
  locations: { value: string; count: number }[];
  education_levels: string[];
  skills_mode_options: string[];
}