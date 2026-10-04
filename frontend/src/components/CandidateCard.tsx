/**
 * One candidate in the results list, with the evidence that matched.
 *
 * The score bars are the point of the card: they show which leg of the hybrid
 * search contributed (BM25 / vector / rerank), so a surprising ranking can be
 * diagnosed instead of just accepted.
 */

import { useState } from "react";
import { Link } from "react-router-dom";

import type { CandidateResult } from "@/api/types";
import { initialsOf, short, yearsLabel } from "@/lib/format";

function ScoreBar({ value, className }: { value: number | null; className: string }) {
  if (value === null) return null;
  // Normalise to 0..1 for the bar width. BM25 ranks are unbounded, so use a
  // log scale - otherwise one 12.0 rank makes every other bar invisible.
  const width = value > 0 ? Math.min(100, Math.log10(1 + value) * 40) : 0;
  return (
    <div className="flex items-center gap-1.5" title={`${value.toFixed(3)}`}>
      <div className="h-1.5 w-14 overflow-hidden rounded-full bg-slate-200">
        <div className={`h-full rounded-full ${className}`} style={{ width: `${width}%` }} />
      </div>
    </div>
  );
}

export default function CandidateCard({
  candidate,
  rank,
}: {
  candidate: CandidateResult;
  rank: number;
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const years = yearsLabel(candidate.total_years_experience);

  return (
    <article className="card p-5 transition hover:border-brand-300">
      <div className="flex items-start gap-4">
        {/* Rank + avatar */}
        <div className="flex flex-col items-center gap-1">
          <span className="text-xs font-bold text-slate-400">#{rank}</span>
          <div className="flex h-11 w-11 items-center justify-center rounded-full bg-brand-100 text-sm font-bold text-brand-700">
            {initialsOf(candidate.full_name)}
          </div>
        </div>

        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
            <Link
              to={`/candidates/${candidate.id}`}
              className="text-base font-semibold text-slate-900 hover:text-brand-700"
            >
              {candidate.full_name ?? "Unknown candidate"}
            </Link>
            {candidate.seniority && (
              <span className="chip border-indigo-200 bg-indigo-50 text-indigo-700">
                {candidate.seniority}
              </span>
            )}
            <span className="text-xs text-slate-400">score {candidate.score.toFixed(4)}</span>
          </div>

          <p className="mt-0.5 text-sm text-slate-600">
            {[candidate.current_title, candidate.current_company].filter(Boolean).join(" · ")}
            {candidate.location && (
              <span className="text-slate-400"> — {candidate.location}</span>
            )}
            {years && <span className="text-slate-400"> — {years}</span>}
          </p>

          {candidate.summary && (
            <p className="mt-2 text-sm leading-relaxed text-slate-600">
              {short(candidate.summary, 220)}
            </p>
          )}

          {candidate.skills.length > 0 && (
            <div className="mt-3 flex flex-wrap gap-1.5">
              {candidate.skills.slice(0, 10).map((skill) => (
                <span
                  key={skill.canonical}
                  className={skill.is_core ? "chip chip-active" : "chip"}
                  title={`${skill.occurrences} mention(s)`}
                >
                  {skill.display}
                </span>
              ))}
              {candidate.skills.length > 10 && (
                <span className="chip">+{candidate.skills.length - 10} more</span>
              )}
            </div>
          )}

          {/* Contact details are only rendered because the recruiter asked -
              this is personal data, so it is never shown by default elsewhere. */}
          {(candidate.email || candidate.phone) && (
            <p className="mt-3 text-xs text-slate-500">
              {candidate.email}
              {candidate.email && candidate.phone && " · "}
              {candidate.phone}
            </p>
          )}

          {/* Why did this candidate match? */}
          {candidate.evidence.length > 0 && (
            <div className="mt-4 border-t border-slate-100 pt-3">
              <button
                type="button"
                onClick={() => setShowEvidence((v) => !v)}
                className="text-xs font-semibold text-brand-600 hover:underline"
              >
                {showEvidence ? "Hide" : "Show"} why this matched ({candidate.evidence.length})
              </button>

              {showEvidence && (
                <div className="mt-2 space-y-2">
                  {candidate.evidence.map((chunk) => (
                    <div key={chunk.chunk_id} className="rounded-lg bg-slate-50 p-3">
                      <div className="mb-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                        <span className="font-semibold uppercase tracking-wide">
                          {chunk.section}
                        </span>
                        {chunk.heading && <span>· {chunk.heading}</span>}
                        <span className="ml-auto flex items-center gap-2">
                          {chunk.vector_score !== null && (
                            <>
                              <span className="text-brand-600">vector</span>
                              <ScoreBar value={chunk.vector_score} className="bg-brand-500" />
                            </>
                          )}
                          {chunk.keyword_score !== null && (
                            <>
                              <span className="text-amber-600">bm25</span>
                              <ScoreBar value={chunk.keyword_score} className="bg-amber-500" />
                            </>
                          )}
                          {chunk.rerank_score !== null && (
                            <>
                              <span className="text-emerald-600">rerank</span>
                              <ScoreBar value={chunk.rerank_score} className="bg-emerald-500" />
                            </>
                          )}
                        </span>
                      </div>
                      <p className="whitespace-pre-wrap text-xs leading-relaxed text-slate-600">
                        {chunk.content}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}