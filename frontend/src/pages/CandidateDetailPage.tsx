/**
 * Full candidate profile, including the chunks that were actually embedded.
 *
 * Showing the chunks matters: it makes the RAG pipeline inspectable. If a resume
 * parsed badly, you can see it here instead of wondering why search missed it.
 */

import { Link, useNavigate, useParams } from "react-router-dom";

import { api } from "@/api/client";
import { useAsync } from "@/hooks/useAsync";
import { initialsOf, yearsLabel } from "@/lib/format";

const SECTION_STYLES: Record<string, string> = {
  summary: "border-brand-200 bg-brand-50/40",
  experience: "border-emerald-200 bg-emerald-50/40",
  education: "border-amber-200 bg-amber-50/40",
  skills: "border-violet-200 bg-violet-50/40",
  projects: "border-sky-200 bg-sky-50/40",
  certifications: "border-slate-200 bg-slate-50",
  languages: "border-slate-200 bg-slate-50",
};

export default function CandidateDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { data, error, loading, reload } = useAsync(
    () => (id ? api.getCandidate(id) : Promise.reject(new Error("missing id"))),
    [id],
  );

  if (loading) return <p className="text-sm text-slate-400">Loading profile…</p>;
  if (error) return <p className="text-sm text-red-600">{error}</p>;
  if (!data) return null;

  const years = yearsLabel(data.total_years_experience);
  const chunksBySection = data.chunks.reduce<Record<string, typeof data.chunks>>((acc, chunk) => {
    (acc[chunk.section] ??= []).push(chunk);
    return acc;
  }, {});

  async function remove() {
    if (!id) return;
    if (!window.confirm("Delete this candidate and all of their chunks?")) return;
    await api.deleteCandidate(id);
    navigate("/candidates");
  }

  return (
    <div className="space-y-6">
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          <div className="flex h-14 w-14 items-center justify-center rounded-full bg-brand-100 text-lg font-bold text-brand-700">
            {initialsOf(data.full_name)}
          </div>
          <div>
            <h1 className="text-xl font-bold text-slate-900">
              {data.full_name ?? "Unknown candidate"}
            </h1>
            <p className="text-sm text-slate-600">
              {[data.current_title, data.current_company].filter(Boolean).join(" · ")}
            </p>
            <p className="text-xs text-slate-400">
              {[data.location, years, data.seniority].filter(Boolean).join(" · ")}
            </p>
          </div>
        </div>
        <button type="button" onClick={remove} className="btn-secondary text-red-600">
          Delete
        </button>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_320px]">
        <div className="space-y-6">
          {/* -- contact: personal data, only shown on the detail page --- */}
          <section className="card p-5">
            <h2 className="mb-3 text-sm font-semibold text-slate-900">Contact</h2>
            <dl className="grid gap-2 text-sm sm:grid-cols-2">
              {data.email && (
                <div>
                  <dt className="text-xs text-slate-400">Email</dt>
                  <dd className="text-slate-800">{data.email}</dd>
                </div>
              )}
              {data.phone && (
                <div>
                  <dt className="text-xs text-slate-400">Phone</dt>
                  <dd className="text-slate-800">{data.phone}</dd>
                </div>
              )}
              {data.location && (
                <div>
                  <dt className="text-xs text-slate-400">Location</dt>
                  <dd className="text-slate-800">{data.location}</dd>
                </div>
              )}
              {data.languages.length > 0 && (
                <div>
                  <dt className="text-xs text-slate-400">Languages</dt>
                  <dd className="text-slate-800">{data.languages.join(", ")}</dd>
                </div>
              )}
              {Object.entries(data.links).map(([key, url]) => (
                <div key={key}>
                  <dt className="text-xs capitalize text-slate-400">{key}</dt>
                  <dd>
                    <a
                      href={url}
                      target="_blank"
                      rel="noreferrer noopener"
                      className="text-brand-600 hover:underline"
                    >
                      {url}
                    </a>
                  </dd>
                </div>
              ))}
            </dl>
          </section>

          {data.summary && (
            <section className="card p-5">
              <h2 className="mb-2 text-sm font-semibold text-slate-900">Summary</h2>
              <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700">
                {data.summary}
              </p>
            </section>
          )}

          {/* -- the embedded chunks, grouped by detected section -------- */}
          <section>
            <h2 className="mb-3 text-sm font-semibold text-slate-900">
              Embedded chunks ({data.chunk_count})
            </h2>
            {Object.entries(chunksBySection).map(([section, chunks]) => (
              <div key={section} className="mb-4">
                <h3 className="mb-2 text-xs font-semibold uppercase tracking-wide text-slate-500">
                  {section} ({chunks.length})
                </h3>
                <div className="space-y-2">
                  {chunks.map((chunk) => (
                    <div
                      key={chunk.id}
                      className={`rounded-lg border p-3 ${
                        SECTION_STYLES[chunk.section] ?? "border-slate-200 bg-white"
                      }`}
                    >
                      <p className="mb-1 text-[11px] text-slate-400">
                        #{chunk.chunk_index} · {chunk.token_count} tokens
                        {chunk.heading && ` · ${chunk.heading}`}
                      </p>
                      <p className="whitespace-pre-wrap text-xs leading-relaxed text-slate-700">
                        {chunk.content}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </section>
        </div>

        {/* -- sidebar ------------------------------------------------- */}
        <aside className="space-y-4">
          <section className="card p-5">
            <h2 className="mb-3 text-sm font-semibold text-slate-900">
              Skills ({data.skills.length})
            </h2>
            <div className="flex flex-wrap gap-1.5">
              {data.skills.map((skill) => (
                <span
                  key={skill.canonical}
                  className={skill.is_core ? "chip chip-active" : "chip"}
                  title={`${skill.occurrences} mention(s)`}
                >
                  {skill.display}
                </span>
              ))}
            </div>
          </section>

          {Object.keys(data.education).length > 0 && (
            <section className="card p-5">
              <h2 className="mb-2 text-sm font-semibold text-slate-900">Education</h2>
              <dl className="space-y-1 text-sm">
                {Object.entries(data.education).map(([key, value]) => (
                  <div key={key} className="flex justify-between gap-3">
                    <dt className="capitalize text-slate-400">{key}</dt>
                    <dd className="text-right text-slate-700">{String(value)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          )}

          <section className="card p-5 text-xs text-slate-500">
            <h2 className="mb-2 text-sm font-semibold text-slate-900">Provenance</h2>
            <dl className="space-y-1">
              <div className="flex justify-between gap-3">
                <dt>File</dt>
                <dd className="truncate text-right">{data.resume_filename ?? "–"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt>Extraction</dt>
                <dd>{data.extraction_method ?? "–"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt>Text length</dt>
                <dd>{data.text_length.toLocaleString()} chars</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt>Ingested</dt>
                <dd>{data.created_at ? new Date(data.created_at).toLocaleDateString() : "–"}</dd>
              </div>
            </dl>
            <button type="button" onClick={reload} className="btn-secondary mt-3 w-full">
              Refresh
            </button>
            <Link to="/candidates" className="btn-secondary mt-2 w-full">
              Back to list
            </Link>
          </section>
        </aside>
      </div>
    </div>
  );
}