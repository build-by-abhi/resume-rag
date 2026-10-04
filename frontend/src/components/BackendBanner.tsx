/**
 * Banner shown when the API cannot be reached.
 *
 * This exists because of a specific, easy-to-hit failure: `npm run dev` works,
 * `python -m app.run` was never started (or crashed), and Vite's proxy answers
 * every `/api` request with 500. Without this banner the UI just shows
 * "backend: ..." and "No skills indexed yet" - which looks like an empty
 * database, not a dead server.
 */

import { useHealth } from "@/hooks/useHealth";

export default function BackendBanner() {
  const { offline, error, retry, loading, health } = useHealth();

  if (!offline) return null;

  return (
    <div className="border-b border-red-200 bg-red-50">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-6 py-4 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="text-sm font-semibold text-red-800">
            Cannot reach the API
          </p>
          <p className="mt-0.5 text-xs text-red-700">
            {error ?? "The backend did not respond."} The Vite dev server proxies
            <code className="mx-1 rounded bg-red-100 px-1">/api</code>
            to <code className="mx-1 rounded bg-red-100 px-1">localhost:8000</code>,
            so the API has to be running too.
          </p>
          <p className="mt-1.5 text-xs text-red-800">
            <span className="font-semibold">Start it:</span>{" "}
            <code className="rounded bg-red-100 px-1.5 py-0.5">
              cd backend &amp;&amp; python -m app.run --reload
            </code>
          </p>
          {health === null && (
            <p className="mt-1 text-[11px] text-red-600">
              If it fails to start on Windows, use <code>python -m app.run</code>{" "}
              rather than <code>uvicorn app.main:app</code> - uvicorn picks an
              event loop psycopg cannot use. See the README.
            </p>
          )}
        </div>
        <button
          type="button"
          onClick={retry}
          disabled={loading}
          className="btn-secondary shrink-0 border-red-300 text-red-700 hover:bg-red-100"
        >
          {loading ? "Retrying…" : "Retry now"}
        </button>
      </div>
    </div>
  );
}