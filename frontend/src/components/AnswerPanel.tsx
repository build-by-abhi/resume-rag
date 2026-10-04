/**
 * The grounded answer, with clickable citations.
 *
 * Two things make this trustworthy rather than decorative:
 *   1. `[n]` markers in the answer become links to the exact excerpt
 *   2. when no LLM is configured the panel says so, instead of pretending the
 *      extractive digest was generated
 */

import { useState } from "react";

import type { AnswerOut } from "@/api/types";
import { useHealth } from "@/hooks/useHealth";

export default function AnswerPanel({ answer }: { answer: AnswerOut | null }) {
  const [openCitation, setOpenCitation] = useState<number | null>(null);
  const { health } = useHealth();

  if (!answer) return null;

  const citationByIndex = new Map(answer.citations.map((c) => [c.index, c]));
  // Prefer the backend's explanation over a generic label: "LLM_PROVIDER is set
  // but HUGGINGFACE_API_KEY is empty" tells you what to do, whereas
  // "no LLM key configured" just tells you it is broken.
  const fallbackReason = health?.llm_status?.reason ?? "no LLM configured";

  /**
   * Split the answer on `[1]` / `[2,3]` markers so they can be rendered as
   * buttons. A plain `.split(/(\[[\d,\s]+\])/)` keeps the delimiters in the
   * output array, alternating text / marker / text.
   */
  const parts = answer.text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);

  return (
    <section className="card border-brand-200 bg-brand-50/40 p-6">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-slate-900">Answer</h2>
        {answer.used_llm ? (
          <span className="chip border-violet-300 bg-violet-50 text-violet-700">
            generated · {answer.model}
          </span>
        ) : (
          <span
            className="chip border-amber-300 bg-amber-50 text-amber-700"
            title={fallbackReason}
          >
            extractive — LLM off
          </span>
        )}
        {typeof answer.latency_ms === "number" && (
          <span className="text-xs text-slate-400">{Math.round(answer.latency_ms)} ms</span>
        )}
      </div>

      {answer.insufficient_evidence && (
        <p className="mb-3 rounded-lg bg-amber-100 px-3 py-2 text-sm text-amber-800">
          The retrieved context did not contain enough evidence to answer this.
        </p>
      )}

      <div className="whitespace-pre-wrap text-sm leading-relaxed text-slate-800">
        {parts.map((part, index) => {
          const match = /^\[(\d+(?:\s*,\s*\d+)*)\]$/.exec(part);
          if (!match) return <span key={index}>{part}</span>;

          const numbers = match[1].split(",").map((n) => Number(n.trim()));
          const active = openCitation !== null && numbers.includes(openCitation);

          return (
            <button
              key={index}
              type="button"
              onClick={() => setOpenCitation(active ? null : numbers[0])}
              className={`mx-0.5 rounded px-1.5 py-0.5 align-middle text-xs font-semibold transition ${
                active
                  ? "bg-brand-600 text-white"
                  : "bg-white text-brand-700 ring-1 ring-brand-200 hover:bg-brand-100"
              }`}
            >
              {numbers.join(",")}
            </button>
          );
        })}
      </div>

      {answer.had_invalid_citations && (
        <p className="mt-3 text-xs text-amber-600">
          The model cited references that did not exist; they were removed.
        </p>
      )}

      {/* The whole answer is factual (it quotes retrieved text), it just was not
          written by a model. Say exactly why, so the reader can fix it. */}
      {!answer.used_llm && (
        <p className="mt-3 rounded-lg bg-amber-50 px-3 py-2 text-xs text-amber-800">
          Showing an extractive summary built from the retrieved text — no model
          wrote this sentence. {fallbackReason}
          {health?.llm_status?.env_var && (
            <>
              {" "}
              Add your token in{" "}
              <code className="rounded bg-amber-100 px-1">
                backend/.env
              </code>{" "}
              as{" "}
              <code className="rounded bg-amber-100 px-1">
                {health.llm_status.env_var}=hf_...
              </code>
              , then restart the API and verify with{" "}
              <code className="rounded bg-amber-100 px-1">
                python scripts/check_llm.py
              </code>
              .
            </>
          )}
        </p>
      )}

      {openCitation !== null && citationByIndex.has(openCitation) && (
        <blockquote className="mt-4 rounded-lg border-l-4 border-brand-400 bg-white p-3 text-sm text-slate-700">
          <p className="text-xs font-semibold text-slate-500">
            [{openCitation}]{" "}
            {citationByIndex.get(openCitation)?.candidate_name} ·{" "}
            {citationByIndex.get(openCitation)?.section}
            {citationByIndex.get(openCitation)?.heading
              ? ` · ${citationByIndex.get(openCitation)?.heading}`
              : ""}
          </p>
          <p className="mt-1 leading-relaxed">
            {citationByIndex.get(openCitation)?.excerpt}
          </p>
        </blockquote>
      )}

      {answer.citations.length > 0 && (
        <div className="mt-5 border-t border-slate-200 pt-4">
          <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
            Sources ({answer.citations.length})
          </h3>
          <ol className="space-y-1.5">
            {answer.citations.map((c) => (
              <li key={c.index} className="flex gap-2 text-xs text-slate-600">
                <span className="shrink-0 rounded bg-slate-200 px-1.5 font-semibold text-slate-700">
                  {c.index}
                </span>
                <span>
                  <span className="font-medium text-slate-800">
                    {c.candidate_name ?? "Unknown"}
                  </span>
                  {c.section && <span className="text-slate-400"> · {c.section}</span>}
                </span>
              </li>
            ))}
          </ol>
        </div>
      )}
    </section>
  );
}