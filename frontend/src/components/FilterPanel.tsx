/**
 * The structured half of the search UI.
 *
 * Every control here maps 1:1 to a column or table the backend filters on.
 * This is the visible counterpart of the "don't use pure vector search" idea:
 * exact constraints are exact, and the UI makes that explicit.
 */

import { useEffect, useMemo, useState } from "react";

import { api } from "@/api/client";
import type { SearchRequest } from "@/api/types";
import { useAsync } from "@/hooks/useAsync";

export interface FilterState {
  skills: string[];
  skillsMode: "all" | "any";
  location: string;
  minYears: string;
  maxYears: string;
  seniority: string[];
  title: string;
}

export const EMPTY_FILTERS: FilterState = {
  skills: [],
  skillsMode: "all",
  location: "",
  minYears: "",
  maxYears: "",
  seniority: [],
  title: "",
};

/**
 * Convert UI state into the API payload.
 *
 * Note the `null` vs `undefined` choice: the backend distinguishes "filter not
 * provided" (null/undefined) from "filter set to empty", and empty strings sent
 * as values would produce LIKE '%%' which matches everything and is misleading.
 */
export function filtersToRequest(filters: FilterState, query: string, topK: number): SearchRequest {
  const toNumber = (value: string) => (value.trim() === "" ? null : Number(value));

  return {
    query: query.trim(),
    skills: filters.skills,
    skills_mode: filters.skillsMode,
    location: filters.location.trim() || null,
    min_years_experience: toNumber(filters.minYears),
    max_years_experience: toNumber(filters.maxYears),
    current_title: filters.title.trim() || null,
    seniority: filters.seniority,
    top_k: topK,
  };
}

export function isFilterActive(filters: FilterState): boolean {
  return (
    filters.skills.length > 0 ||
    filters.location.trim() !== "" ||
    filters.minYears.trim() !== "" ||
    filters.maxYears.trim() !== "" ||
    filters.seniority.length > 0 ||
    filters.title.trim() !== ""
  );
}

/**
 * Can this payload actually be searched?
 *
 * `POST /api/search` rejects a body with neither a question nor a filter with
 * 422 - which is the right behaviour for a search endpoint. The UI therefore
 * needs this check so it never fires a request it knows will fail.
 *
 * This mirrors `SearchRequest.has_text_or_filters` in
 * `backend/app/schemas/search.py`. If you add a filter field there, add it
 * here too, otherwise the UI will happily 422 on a request it thinks is valid.
 */
export function isSearchable(payload: SearchRequest): boolean {
  const present = (value: number | null | undefined): boolean =>
    value !== null && value !== undefined;

  return Boolean(
    payload.query?.trim() ||
      payload.skills?.length ||
      payload.location ||
      payload.current_title ||
      payload.company ||
      payload.seniority?.length ||
      payload.languages?.length ||
      payload.education_level?.length ||
      present(payload.min_years_experience) ||
      present(payload.max_years_experience) ||
      payload.has_email !== null && payload.has_email !== undefined ||
      payload.has_phone !== null && payload.has_phone !== undefined,
  );
}

export default function FilterPanel({
  filters,
  onChange,
}: {
  filters: FilterState;
  onChange: (next: FilterState) => void;
}) {
  // Facets come from the database, so the dropdowns always reflect the pool you
  // actually have instead of a hardcoded list.
  const [facetNonce, setFacetNonce] = useState(0);
  const {
    data: skillData,
    error: skillError,
    loading: skillsLoading,
  } = useAsync(() => api.skillFacets(80), [facetNonce]);
  const { data: options } = useAsync(() => api.filterOptions(), [facetNonce]);

  // Uploading resumes changes which skills and locations exist. UploadPanel
  // announces that with a window event so the facet lists refresh without the
  // parent having to thread a prop down here.
  useEffect(() => {
    const handler = () => setFacetNonce((n) => n + 1);
    window.addEventListener("candidates-changed", handler);
    return () => window.removeEventListener("candidates-changed", handler);
  }, []);

  const skills = useMemo(() => skillData?.skills ?? [], [skillData]);

  useEffect(() => {
    // Drop selected skills that no longer exist in the corpus.
    if (skills.length === 0 || filters.skills.length === 0) return;
    const valid = new Set(skills.map((s) => s.canonical));
    const next = filters.skills.filter((s) => valid.has(s));
    if (next.length !== filters.skills.length) onChange({ ...filters, skills: next });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [skills]);

  function toggleSkill(canonical: string) {
    const next = filters.skills.includes(canonical)
      ? filters.skills.filter((s) => s !== canonical)
      : [...filters.skills, canonical];
    onChange({ ...filters, skills: next });
  }

  function toggleSeniority(value: string) {
    const next = filters.seniority.includes(value)
      ? filters.seniority.filter((s) => s !== value)
      : [...filters.seniority, value];
    onChange({ ...filters, seniority: next });
  }

  return (
    <aside className="card p-5">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="text-sm font-semibold text-slate-900">Structured filters</h2>
        {isFilterActive(filters) && (
          <button
            type="button"
            onClick={() => onChange(EMPTY_FILTERS)}
            className="text-xs font-medium text-brand-600 hover:underline"
          >
            Clear all
          </button>
        )}
      </div>

      {/* -- skills ---------------------------------------------------- */}
      <div className="mb-5">
        <span className="label">Skills</span>
        <div className="mb-2 flex gap-2 text-xs">
          {(["all", "any"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => onChange({ ...filters, skillsMode: mode })}
              className={`rounded px-2 py-1 font-medium ${
                filters.skillsMode === mode
                  ? "bg-brand-600 text-white"
                  : "bg-slate-100 text-slate-600 hover:bg-slate-200"
              }`}
            >
              {mode === "all" ? "match all" : "match any"}
            </button>
          ))}
        </div>
        <div className="flex max-h-40 flex-wrap gap-1.5 overflow-y-auto">
          {skills.map((skill) => (
            <button
              key={skill.canonical}
              type="button"
              onClick={() => toggleSkill(skill.canonical)}
              title={`${skill.candidate_count} candidate(s)`}
              className={`chip ${
                filters.skills.includes(skill.canonical) ? "chip-active" : "hover:bg-slate-100"
              }`}
            >
              {skill.display}
              <span className="text-[10px] text-slate-400">{skill.candidate_count}</span>
            </button>
          ))}

          {/* Distinguish "no skills in the corpus" from "could not ask". Showing
              "No skills indexed yet" when the API is down reads like an empty
              database and sends people hunting for the wrong bug. */}
          {skills.length === 0 && skillError && (
            <p className="text-xs text-red-600">
              Could not load skills — the API is unreachable.
            </p>
          )}
          {skills.length === 0 && !skillError && skillsLoading && (
            <p className="text-xs text-slate-400">Loading skills…</p>
          )}
          {skills.length === 0 && !skillError && !skillsLoading && (
            <p className="text-xs text-slate-400">
              No skills indexed yet. Upload a resume to populate this list.
            </p>
          )}
        </div>
      </div>

      {/* -- location -------------------------------------------------- */}
      <div className="mb-4">
        <label className="label" htmlFor="location">
          Location
        </label>
        <input
          id="location"
          className="input"
          placeholder="Berlin"
          value={filters.location}
          onChange={(e) => onChange({ ...filters, location: e.target.value })}
          list="location-options"
        />
        <datalist id="location-options">
          {(options?.locations ?? []).map((l) => (
            <option key={l.value} value={l.value} />
          ))}
        </datalist>
      </div>

      {/* -- experience ------------------------------------------------ */}
      <div className="mb-4">
        <span className="label">Experience (years)</span>
        <div className="flex items-center gap-2">
          <input
            className="input"
            type="number"
            min={0}
            max={60}
            placeholder="min"
            aria-label="Minimum years of experience"
            value={filters.minYears}
            onChange={(e) => onChange({ ...filters, minYears: e.target.value })}
          />
          <span className="text-slate-400">–</span>
          <input
            className="input"
            type="number"
            min={0}
            max={60}
            placeholder="max"
            aria-label="Maximum years of experience"
            value={filters.maxYears}
            onChange={(e) => onChange({ ...filters, maxYears: e.target.value })}
          />
        </div>
      </div>

      {/* -- title ----------------------------------------------------- */}
      <div className="mb-4">
        <label className="label" htmlFor="title">
          Job title
        </label>
        <input
          id="title"
          className="input"
          placeholder="Backend Engineer"
          value={filters.title}
          onChange={(e) => onChange({ ...filters, title: e.target.value })}
        />
      </div>

      {/* -- seniority -------------------------------------------------- */}
      <div>
        <span className="label">Seniority</span>
        <div className="flex flex-wrap gap-1.5">
          {(options?.seniority ?? []).map((s) => (
            <button
              key={s.value}
              type="button"
              onClick={() => toggleSeniority(s.value)}
              className={`chip ${
                filters.seniority.includes(s.value) ? "chip-active" : "hover:bg-slate-100"
              }`}
            >
              {s.value}
              <span className="text-[10px] text-slate-400">{s.count}</span>
            </button>
          ))}
        </div>
      </div>
    </aside>
  );
}