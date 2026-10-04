/** Formatting helpers shared by the components. */

/** Human label for a score in 0..1 (RRF or cosine), as a percentage. */
export function scorePercent(score: number | null | undefined): string {
  if (typeof score !== "number" || Number.isNaN(score)) return "–";
  // RRF scores are small (e.g. 0.032). Rescale with a log so the bar is
  // readable instead of always looking like zero.
  const scaled = score >= 1 ? score : Math.min(1, Math.log10(1 + score * 30) + 0.15);
  return `${Math.round(scaled * 100)}%`;
}

export function yearsLabel(years: number | null | undefined): string | null {
  if (typeof years !== "number") return null;
  return `${Number.isInteger(years) ? years : years.toFixed(1)} yrs`;
}

export function initialsOf(name: string | null | undefined, fallback = "?"): string {
  if (!name) return fallback;
  const parts = name.trim().split(/\s+/).slice(0, 2);
  return parts.map((p) => p[0]?.toUpperCase() ?? "").join("") || fallback;
}

/** Truncate for a single-line preview. */
export function short(text: string | null | undefined, limit = 120): string {
  if (!text) return "";
  return text.length <= limit ? text : `${text.slice(0, limit).trimEnd()}…`;
}