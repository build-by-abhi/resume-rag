/**
 * Phase 4: job-description → candidate match.
 *
 * Retrieval is the same hybrid search as the HR query (a JD is just another
 * query). On top of it, the LLM scores each retrieved candidate against the JD
 * and returns a structured scorecard with citations.
 */

import { useState } from "react";
import { Link } from "react-router-dom";

import { api } from "@/api/client";
import type { JDMatchResponse } from "@/api/types";
import { initialsOf } from "@/lib/format";

const SAMPLE_JD = `We are hiring a Senior Backend Engineer to work on our real-time
data platform.

Requirements:
- 5+ years of backend engineering experience
- Strong Python (FastAPI or Django)
- Production experience with PostgreSQL and event streaming (Kafka)
- Kubernetes and Docker for deployment
- Exposure to vector search or retrieval systems is a plus`;

function scoreTone(score: number): string {
  if (score >= 75) return "bg-emerald-500";
  if (score >= 50) return "bg-amber-500";
  return "bg-slate-400";
}

export default function MatchPage() {
  const [jobDescription, setJobDescription] = useState(SAMPLE_JD);
  const [result, setResult] = useState<JDMatchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    if (jobDescription.trim().length < 40) {
      setError("Paste a job description of at least 40 characters.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      setResult(
        await api.matchJobDescription({
          job_description: jobDescription,
          top_k: 10,
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Matching failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-900">Job description match</h1>
        <p className="mt-1 text-sm text-slate-500">
          The JD becomes a query. Hybrid retrieval shortlists candidates, then the
          LLM scores each one against the requirements.
        </p>
      </div>

      <section className="card p-5">
        <label className="label" htmlFor="jd">
          Job description
        </label>
        <textarea
          id="jd"
          rows={9}
          className="input font-mono text-xs"
          value={jobDescription}
          onChange={(e) => setJobDescription(e.target.value)}
        />
        <div className="mt-3 flex items-center gap-3">
          <button type="button" className="btn-primary" onClick={run} disabled={loading}>
            {loading ? "Matching…" : "Match candidates"}
          </button>
          {result && (
            <span className="text-xs text-slate-500">
              {result.total} scored · retrieval {Math.round(result.retrieval_ms)} ms · LLM{" "}
              {result.used_llm ? "on" : "off (fallback score)"}
            </span>
          )}
        </div>
        {error && <p className="mt-3 text-sm text-red-600">{error}</p>}
      </section>

      <div className="space-y-3">
        {result?.results.map((candidate, index) => (
          <article key={candidate.candidate_id} className="card p-5">
            <div className="flex items-start gap-4">
              <div className="flex h-10 w-10 items-center justify-center rounded-full bg-brand-100 text-sm font-bold text-brand-700">
                {initialsOf(candidate.full_name)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-3">
                  <Link
                    to={`/candidates/${candidate.candidate_id}`}
                    className="font-semibold text-slate-900 hover:text-brand-700"
                  >
                    {candidate.full_name ?? "Unknown"}
                  </Link>
                  <span className="text-sm text-slate-500">{candidate.current_title}</span>
                  <span className="ml-auto text-2xl font-bold text-slate-900">
                    {Math.round(candidate.score)}
                  </span>
                  <span className="text-xs text-slate-400">/ 100</span>
                </div>

                <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-100">
                  <div
                    className={`h-full rounded-full ${scoreTone(candidate.score)}`}
                    style={{ width: `${Math.min(100, candidate.score)}%` }}
                  />
                </div>

                {/* Sub-scores are the actionable part: they tell the recruiter
                    which dimension is weak. */}
                <div className="mt-3 grid gap-2 text-xs sm:grid-cols-3">
                  {(
                    [
                      ["Skills", candidate.skills_match],
                      ["Experience", candidate.experience_match],
                      ["Seniority", candidate.seniority_match],
                    ] as const
                  ).map(([label, value]) =>
                    value === null ? null : (
                      <div key={label}>
                        <div className="flex justify-between text-slate-500">
                          <span>{label}</span>
                          <span>{Math.round(value)}</span>
                        </div>
                        <div className="mt-1 h-1 overflow-hidden rounded-full bg-slate-100">
                          <div
                            className="h-full rounded-full bg-brand-400"
                            style={{ width: `${Math.min(100, value)}%` }}
                          />
                        </div>
                      </div>
                    ),
                  )}
                </div>

                {candidate.summary && (
                  <p className="mt-3 text-sm text-slate-600">{candidate.summary}</p>
                )}

                {candidate.missing_critical_skills.length > 0 && (
                  <p className="mt-2 text-xs text-amber-700">
                    Missing: {candidate.missing_critical_skills.join(", ")}
                  </p>
                )}

                {candidate.strengths.length > 0 && (
                  <ul className="mt-2 space-y-1 text-xs text-slate-600">
                    {candidate.strengths.map((strength, i) => (
                      <li key={i}>
                        {typeof strength.point === "string" ? strength.point : JSON.stringify(strength)}
                        {typeof strength.citation === "number" && (
                          <span className="ml-1 font-semibold text-brand-600">
                            [{strength.citation}]
                          </span>
                        )}
                      </li>
                    ))}
                  </ul>
                )}

                {candidate.evidence.length > 0 && (
                  <details className="mt-3 text-xs">
                    <summary className="cursor-pointer font-semibold text-brand-600">
                      Evidence ({candidate.evidence.length})
                    </summary>
                    <div className="mt-2 space-y-2">
                      {candidate.evidence.map((chunk) => (
                        <div key={chunk.chunk_id} className="rounded-lg bg-slate-50 p-2.5">
                          <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-400">
                            {chunk.section}
                            {chunk.heading && ` · ${chunk.heading}`}
                          </p>
                          <p className="whitespace-pre-wrap text-xs text-slate-600">
                            {chunk.content}
                          </p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            </div>
            {index === 0 && result.total > 1 && (
              <p className="mt-3 border-t border-slate-100 pt-2 text-xs text-emerald-700">
                Best match in this pool.
              </p>
            )}
          </article>
        ))}
      </div>
    </div>
  );
}