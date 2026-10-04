/**
 * Thin API client.
 *
 * Everything the UI knows about HTTP lives here. Components call typed methods
 * and get either data or an `ApiError` - they never see fetch(), Response, or a
 * status code. That keeps error handling in one place instead of 20 `try/catch`
 * blocks.
 */

import type {
  CandidateDetail,
  CandidateSummary,
  FilterOptions,
  HealthResponse,
  JDMatchRequest,
  JDMatchResponse,
  SearchRequest,
  SearchResponse,
  SkillFacet,
  UploadResponse,
} from "./types";

/**
 * Base URL. Empty string means "same origin", which is what you want because
 * `vite.config.ts` proxies /api to the backend in dev and nginx does it in prod.
 * Set VITE_API_URL to point at a different host.
 */
const BASE_URL = import.meta.env.VITE_API_URL ?? "";

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly detail?: unknown,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

/**
 * Extract a human-readable message from a FastAPI error body.
 * FastAPI returns `{detail: "..."}` for HTTPException and
 * `{detail: [{loc, msg, type}, ...]}` for validation errors.
 */
function extractMessage(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;

    if (typeof detail === "string") return detail;

    if (Array.isArray(detail)) {
      // Validation errors: surface the first readable message.
      const first = detail[0] as { msg?: string; loc?: string[] } | undefined;
      if (first?.msg) {
        const field = first.loc?.filter((p) => p !== "body").join(".");
        return field ? `${field}: ${first.msg}` : first.msg;
      }
    }
  }
  return `Request failed with status ${status}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE_URL}${path}`, {
    ...init,
    headers: {
      // Do NOT set Content-Type for FormData: the browser must add the
      // multipart boundary itself, and a manual header breaks the upload.
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = null;
    }
    throw new ApiError(extractMessage(body, response.status), response.status, body);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------
export const api = {
  health: () => request<HealthResponse>("/health"),

  // -- ingestion --------------------------------------------------------
  /**
   * Upload one or many resumes.
   * FormData (not JSON) because the backend expects multipart/form-data and
   * appending every file under the SAME field name "files" is how FastAPI's
   * `files: list[UploadFile]` receives a batch.
   */
  uploadResumes: (files: File[]) => {
    const form = new FormData();
    files.forEach((file) => form.append("files", file));
    return request<UploadResponse>("/api/candidates/upload", {
      method: "POST",
      body: form,
    });
  },

  ingestText: (text: string) =>
    request<{ candidate_id: string }>("/api/candidates/text", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),

  // -- candidates -------------------------------------------------------
  listCandidates: (params: Record<string, string | number> = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).map(([k, v]) => [k, String(v)]),
    ).toString();
    return request<CandidateSummary[]>(`/api/candidates?${query}`);
  },

  getCandidate: (id: string) => request<CandidateDetail>(`/api/candidates/${id}`),

  deleteCandidate: (id: string) =>
    request<void>(`/api/candidates/${id}`, { method: "DELETE" }),

  // -- search -----------------------------------------------------------
  search: (payload: SearchRequest) =>
    request<SearchResponse>("/api/search", {
      method: "POST",
      body: JSON.stringify(payload),
    }),

  skillFacets: (limit = 60) =>
    request<{ skills: SkillFacet[] }>(`/api/search/skills?limit=${limit}`),

  filterOptions: () => request<FilterOptions>("/api/search/filters"),

  /** Phase 4: rank the pool against a pasted job description. */
  matchJobDescription: (payload: JDMatchRequest) =>
    request<JDMatchResponse>("/api/match", {
      method: "POST",
      body: JSON.stringify(payload),
    }),
};