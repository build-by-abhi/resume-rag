/**
 * Candidate directory: the plain "table view" an HR user expects.
 *
 * This page hits `GET /api/candidates`, which is purely structured SQL - no
 * embeddings involved. Having both a browse view and a semantic search view is
 * the practical payoff of keeping structured extraction in the design.
 */

import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

import { api } from "@/api/client";
import { useAsync } from "@/hooks/useAsync";
import { initialsOf, short, yearsLabel } from "@/lib/format";

const SORTS = [
  { value: "recent", label: "Newest first" },
  { value: "experience", label: "Most experienced" },
  { value: "name", label: "Name (A–Z)" },
] as const;

export default function CandidatesPage() {
  const [search, setSearch] = useState("");
  const [location, setLocation] = useState("");
  const [sort, setSort] = useState<(typeof SORTS)[number]["value"]>("recent");
  const [nonce, setNonce] = useState(0);

  const { data: candidates, loading, error } = useAsync(
    () =>
      api.listCandidates({
        search: search.trim(),
        location: location.trim(),
        sort,
        limit: 100,
      }),
    // `nonce` is in the dependency list so `setNonce` can force a reload.
    [search, location, sort, nonce],
  );

  // Fetch the list again after an upload completes.
  useEffect(() => {
    const handler = () => setNonce((n) => n + 1);
    window.addEventListener("candidates-changed", handler);
    return () => window.removeEventListener("candidates-changed", handler);
  }, []);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-slate-900">Candidates</h1>
          <p className="text-sm text-slate-500">
            {candidates?.length ?? 0} record{candidates?.length === 1 ? "" : "s"} · structured
            filters only (no embeddings)
          </p>
        </div>
        <div className="flex flex-wrap items-end gap-3">
          <div>
            <label className="label" htmlFor="cand-search">
              Search
            </label>
            <input
              id="cand-search"
              className="input w-56"
              placeholder="name, title, company"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <div>
            <label className="label" htmlFor="cand-location">
              Location
            </label>
            <input
              id="cand-location"
              className="input w-40"
              placeholder="Berlin"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </div>
          <div>
            <label className="label" htmlFor="cand-sort">
              Sort
            </label>
            <select
              id="cand-sort"
              className="input w-40"
              value={sort}
              onChange={(e) => setSort(e.target.value as typeof sort)}
            >
              {SORTS.map((s) => (
                <option key={s.value} value={s.value}>
                  {s.label}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {error && <p className="text-sm text-red-600">{error}</p>}
      {loading && <p className="text-sm text-slate-400">Loading…</p>}

      <div className="card divide-y divide-slate-100">
        {(candidates ?? []).map((candidate) => {
          const years = yearsLabel(candidate.total_years_experience);
          return (
            <Link
              key={candidate.id}
              to={`/candidates/${candidate.id}`}
              className="flex items-start gap-4 p-4 transition hover:bg-slate-50"
            >
              <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-600">
                {initialsOf(candidate.full_name)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-slate-900">
                    {candidate.full_name ?? "Unknown"}
                  </span>
                  {candidate.seniority && (
                    <span className="chip border-indigo-200 bg-indigo-50 text-indigo-700">
                      {candidate.seniority}
                    </span>
                  )}
                </div>
                <p className="text-sm text-slate-600">
                  {[candidate.current_title, candidate.current_company].filter(Boolean).join(" · ")}
                  {candidate.location && ` — ${candidate.location}`}
                  {years && ` — ${years}`}
                </p>
                {candidate.summary && (
                  <p className="mt-1 text-xs text-slate-400">{short(candidate.summary, 140)}</p>
                )}
              </div>
            </Link>
          );
        })}

        {candidates?.length === 0 && (
          <p className="p-10 text-center text-sm text-slate-400">
            No candidates yet. Upload a resume from the Search or Upload page.
          </p>
        )}
      </div>
    </div>
  );
}