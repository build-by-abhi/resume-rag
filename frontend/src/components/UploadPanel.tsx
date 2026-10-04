/**
 * Resume upload panel.
 *
 * Shows exactly what the backend returned per file - chunk count, extracted
 * skills, and any warnings - because that transparency is what makes extraction
 * quality debuggable instead of mysterious.
 */

import { useRef, useState } from "react";

import { api, ApiError } from "@/api/client";
import type { UploadResponse } from "@/api/types";

const ACCEPTED = ".pdf,.docx,.txt";

export default function UploadPanel({ onUploaded }: { onUploaded?: () => void }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function upload(files: File[]) {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    setResult(null);

    try {
      const response = await api.uploadResumes(files);
      setResult(response);
      // Let any mounted list refresh itself without prop drilling.
      window.dispatchEvent(new Event("candidates-changed"));
      onUploaded?.();
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Upload failed. Is the backend running on port 8000?",
      );
    } finally {
      setUploading(false);
    }
  }

  return (
    <section className="card p-6">
      <h2 className="text-base font-semibold text-slate-900">Upload resumes</h2>
      <p className="mt-1 text-sm text-slate-500">
        PDF, DOCX or TXT. Each file is parsed, cleaned, split into section-aware
        chunks, structured into fields and embedded. Duplicates are detected by
        file hash.
      </p>

      {/* Drag-and-drop zone. `onDragOver` must preventDefault or the browser
          refuses the drop. */}
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          void upload(Array.from(e.dataTransfer.files));
        }}
        onClick={() => inputRef.current?.click()}
        className={`mt-4 cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition ${
          dragging
            ? "border-brand-400 bg-brand-50"
            : "border-slate-300 hover:border-brand-300 hover:bg-slate-50"
        }`}
      >
        <p className="text-sm font-medium text-slate-700">
          {uploading ? "Processing…" : "Drop resumes here or click to browse"}
        </p>
        <p className="mt-1 text-xs text-slate-400">Up to 20 files per batch</p>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={ACCEPTED}
          className="hidden"
          onChange={(e) => {
            void upload(Array.from(e.target.files ?? []));
            // Reset so re-selecting the same file still fires onChange.
            e.target.value = "";
          }}
        />
      </div>

      {error && (
        <p className="mt-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">
          {error}
        </p>
      )}

      {result && (
        <div className="mt-5 space-y-2">
          <p className="text-sm font-medium text-slate-700">
            {result.succeeded} ingested · {result.failed} failed
          </p>

          {result.results.map((r) => (
            <div key={r.candidate_id} className="rounded-lg border border-slate-200 p-3">
              <div className="flex items-center justify-between gap-3">
                <span className="text-sm font-medium text-slate-800">
                  {r.full_name ?? "Unnamed candidate"}
                </span>
                <span className="text-xs text-slate-500">
                  {r.chunk_count} chunks · {r.skill_count} skills ·{" "}
                  {r.text_length} chars
                </span>
              </div>
              {r.warnings.length > 0 && (
                <p className="mt-1 text-xs text-amber-600">{r.warnings.join(" · ")}</p>
              )}
            </div>
          ))}

          {result.errors.map((e) => (
            <div
              key={e.filename}
              className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700"
            >
              <span className="font-medium">{e.filename}</span>: {e.error}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}