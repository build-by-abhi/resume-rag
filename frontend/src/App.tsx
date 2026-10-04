/**
 * App shell: header with live backend status + routes.
 *
 * The header polls /health so you can see at a glance whether the backend, the
 * database and the LLM are wired up. That status panel is also the fastest way
 * to debug a fresh clone.
 */

import { NavLink, Route, Routes } from "react-router-dom";

import BackendBanner from "@/components/BackendBanner";
import { useHealth } from "@/hooks/useHealth";
import { HealthProvider } from "@/hooks/useHealth";
import CandidatesPage from "@/pages/CandidatesPage";
import CandidateDetailPage from "@/pages/CandidateDetailPage";
import HomePage from "@/pages/HomePage";
import MatchPage from "@/pages/MatchPage";
import UploadPage from "@/pages/UploadPage";

function HealthPill() {
  const { health, error, loading } = useHealth();

  // Three distinct states, because they mean different things to a developer:
  //   error  -> the API is not running at all
  //   !db ok -> the API is up but Postgres is unreachable
  //   ok     -> everything is wired up
  if (error) {
    return (
      <span className="chip border-red-300 bg-red-50 text-red-700" title={error}>
        backend: offline
      </span>
    );
  }
  if (loading && !health) {
    return <span className="chip">backend: …</span>;
  }
  if (!health) {
    return <span className="chip border-amber-300 bg-amber-50 text-amber-700">
      backend: unknown
    </span>;
  }
  if (health.database !== "ok") {
    return (
      <span
        className="chip border-red-300 bg-red-50 text-red-700"
        title={String((health.details as { error?: string })?.error ?? "database down")}
      >
        db: down
      </span>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="chip border-emerald-300 bg-emerald-50 text-emerald-700">db ok</span>
      <span className="chip" title="Active embedding backend">
        emb: {health.embedding_provider} · {health.embedding_dim}d
      </span>
      <span
        className={
          health.llm_enabled
            ? "chip border-violet-300 bg-violet-50 text-violet-700"
            : "chip border-amber-300 bg-amber-50 text-amber-700"
        }
        // Show the precise cause on hover: "provider selected but key empty"
        // is a different fix from "no provider selected".
        title={
          health.llm_enabled
            ? `LLM: ${health.llm_status?.provider} / ${health.llm_status?.model}`
            : (health.llm_status?.reason ?? "no LLM configured")
        }
      >
        llm: {health.llm_enabled ? "on" : "off"}
      </span>
      <span className="chip" title="Candidates and searchable chunks in the database">
        {health.candidate_count} people · {health.chunk_count} chunks
      </span>
    </div>
  );
}

const NAV = [
  { to: "/", label: "Search", end: true },
  { to: "/candidates", label: "Candidates", end: false },
  { to: "/upload", label: "Upload", end: false },
  { to: "/match", label: "JD Match", end: false },
];

export default function App() {
  return (
    <HealthProvider>
      <div className="min-h-screen">
        <header className="border-b border-slate-200 bg-white">
          <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-4 md:flex-row md:items-center md:justify-between">
            <div>
              <h1 className="text-lg font-bold tracking-tight text-slate-900">
                Resume RAG
                <span className="ml-2 text-sm font-normal text-slate-500">
                  hybrid search over pgvector
                </span>
              </h1>
              <nav className="mt-2 flex gap-1">
                {NAV.map((item) => (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.end}
                    className={({ isActive }) =>
                      `rounded-lg px-3 py-1.5 text-sm font-medium transition ${
                        isActive
                          ? "bg-brand-50 text-brand-700"
                          : "text-slate-600 hover:bg-slate-100"
                      }`
                    }
                  >
                    {item.label}
                  </NavLink>
                ))}
              </nav>
            </div>
            <HealthPill />
          </div>
        </header>

        {/* Only rendered when the API is unreachable. */}
        <BackendBanner />

        <main className="mx-auto max-w-7xl px-6 py-8">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/candidates" element={<CandidatesPage />} />
            <Route path="/candidates/:id" element={<CandidateDetailPage />} />
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/match" element={<MatchPage />} />
          </Routes>
        </main>

        <footer className="border-t border-slate-200 py-6 text-center text-xs text-slate-400">
          Structured filters → BM25 + vector search → RRF fusion → optional rerank →
          grounded answer
        </footer>
      </div>
    </HealthProvider>
  );
}