import UploadPanel from "@/components/UploadPanel";

export default function UploadPage() {
  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="text-xl font-bold text-slate-900">Ingest resumes</h1>
        <p className="mt-1 text-sm text-slate-500">
          What happens to each file: parse → clean text → extract structured fields →
          chunk by section → embed chunks and the full profile → store in Postgres
          with pgvector. Uploading the same file twice is a no-op.
        </p>
      </div>
      <UploadPanel />
    </div>
  );
}