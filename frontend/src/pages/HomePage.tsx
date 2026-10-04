/**
 * Search page: natural-language query + structured filters + grounded answer.
 *
 * The layout mirrors the backend pipeline, left to right:
 *   filters (structured)  ->  query (semantic + keyword)  ->  answer  ->  results
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";

import AnswerPanel from "@/components/AnswerPanel";
import CandidateCard from "@/components/CandidateCard";
import FilterPanel, {
  EMPTY_FILTERS,
  filtersToRequest,
  isFilterActive,
  isSearchable,
  type FilterState,
} from "@/components/FilterPanel";
import UploadPanel from "@/components/UploadPanel";
import { useHealth } from "@/hooks/useHealth";
import { useSearch } from "@/hooks/useSearch";
import type { SearchRequest } from "@/api/types";

const EXAMPLES = [
  "Senior Python engineer who has worked with Kubernetes",
  "Anyone with retrieval augmented generation or vector search experience?",
  "React developer in Berlin",
  "People who have run Kafka pipelines in production",
];

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [filters, setFilters] = useState<FilterState>(EMPTY_FILTERS);
  const [topK, setTopK] = useState(10);
  // Tracks whether the user has actually searched yet. The page deliberately
  // starts idle: `POST /api/search` requires a question or a filter and answers
  // 422 otherwise, so firing a request on mount would only produce a red error.
  const [hasSearched, setHasSearched] = useState(false);

  const { response, loading, error, submit, schedule } = useSearch();
  const { offline } = useHealth();

  // The single payload every interaction builds from. Memoised so the
  // `useEffect` below has a stable dependency.
  const payload = useMemo(
    () => filtersToRequest(filters, query, topK),
    [filters, query, topK],
  );

  /** Single place that actually talks to the search endpoint. */
  function runSearch(next: SearchRequest) {
    // Guard: never send a request the API is guaranteed to reject.
    if (!isSearchable(next)) {
      setHasSearched(false);
      return;
    }
    setHasSearched(true);
    submit(next);
  }

  // Re-run the search whenever the filters change (debounced inside the hook).
  // The text query is excluded on purpose: it fires on Enter / button only, so
  // we are not firing a request per keystroke.
  useEffect(() => {
    if (isFilterActive(filters) && isSearchable(payload)) schedule(payload);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters]);

  // Re-run the last real search when the API comes back up, so the page
  // recovers on its own instead of waiting for the user to search again.
  const wasOffline = useRef(offline);
  useEffect(() => {
    if (wasOffline.current && !offline && hasSearched && isSearchable(payload)) {
      submit(payload);
    }
    wasOffline.current = offline;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [offline]);

  const hasQuery = query.trim().length > 0;
  const showEmptyState = !hasSearched && !loading && response === null;

  return (
    <div className="space-y-6">
      {/* -- Query bar -------------------------------------------------- */}
      <section className="card p-5">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            runSearch(payload);
          }}
        >
          <label className="label" htmlFor="query">
            Ask about the candidate pool
          </label>
          <div className="flex flex-col gap-3 sm:flex-row">
            <input
              id="query"
              className="input flex-1"
              placeholder="e.g. senior Python engineer with Kubernetes and PostgreSQL"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500" htmlFor="topk">
                Top
              </label>
              <select
                id="topk"
                className="input w-20"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
              >
                {[5, 10, 20, 50].map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? "Searching…" : "Search"}
              </button>
            </div>
          </div>
        </form>

        <div className="mt-3 flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-slate-400">Try:</span>
          {EXAMPLES.map((example) => (
            <button
              key={example}
              type="button"
              onClick={() => {
                setQuery(example);
                runSearch(filtersToRequest(filters, example, topK));
              }}
              className="chip hover:bg-slate-100"
            >
              {example}
            </button>
          ))}
        </div>
      </section>

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <div className="space-y-6">
          <FilterPanel filters={filters} onChange={setFilters} />
          <UploadPanel
            onUploaded={() => {
              // Facets and the health pill refresh themselves via the
              // `candidates-changed` event. Re-run the search only when there
              // is something to search for - an empty payload is a 422.
              if (isSearchable(payload)) runSearch(payload);
            }}
          />
        </div>

        <div className="space-y-4">
          {error && (
            <p className="rounded-lg bg-red-50 px-4 py-3 text-sm text-red-700">
              {offline
                ? "Search is unavailable while the API is unreachable — see the banner at the top of the page."
                : error}
            </p>
          )}

          {response && (
            <div className="flex flex-wrap items-center gap-3 text-xs text-slate-500">
              <span>
                {response.total} candidate{response.total === 1 ? "" : "s"}
              </span>
              <span>· retrieval {Math.round(response.retrieval_ms)} ms</span>
              <span>· total {Math.round(response.total_ms)} ms</span>
              {/* Exposing the retrieval diagnostics makes the architecture visible. */}
              {typeof response.diagnostics.vector_hits === "number" &&
                typeof response.diagnostics.keyword_hits === "number" && (
                  <span className="text-slate-400">
                    · vector {response.diagnostics.vector_hits} · bm25{" "}
                    {response.diagnostics.keyword_hits}
                  </span>
                )}
            </div>
          )}

          {hasQuery && <AnswerPanel answer={response?.answer ?? null} />}

          {response && response.results.length === 0 && (
            <div className="card p-8 text-center text-sm text-slate-500">
              No candidates matched. Try removing a filter, or upload more resumes.
            </div>
          )}

          {loading && response === null && (
            <div className="card p-8 text-center text-sm text-slate-400">Searching…</div>
          )}

          {/* Idle state. Deliberately not a request: an empty payload is a 422. */}
          {showEmptyState && (
            <div className="card p-10 text-center">
              <h2 className="text-sm font-semibold text-slate-900">
                Ask a question or pick a filter
              </h2>
              <p className="mx-auto mt-2 max-w-md text-sm text-slate-500">
                Search combines structured filters with BM25 and vector similarity,
                then writes a cited answer. Try one of the examples above, or
                browse the full pool on the{" "}
                <Link to="/candidates" className="font-medium text-brand-600 hover:underline">
                  Candidates
                </Link>{" "}
                tab.
              </p>
            </div>
          )}

          <div className="space-y-3">
            {response?.results.map((candidate, index) => (
              <CandidateCard key={candidate.id} candidate={candidate} rank={index + 1} />
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}