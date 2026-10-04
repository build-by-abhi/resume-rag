/**
 * Backend health as shared app state.
 *
 * WHY A CONTEXT INSTEAD OF A HOOK IN EACH COMPONENT
 * -------------------------------------------------
 * The header pill, the offline banner and the filter panel all need to know
 * whether the API is reachable. If each called `useAsync(api.health)` we would
 * poll `/health` three times on every render pass, and they could disagree.
 *
 * One provider = one poll, one truth.
 *
 * It also gives us a single place to detect the failure mode that actually
 * wastes beginners' time: the frontend runs, the backend does not, and every
 * request 500s. In that state the UI used to show "backend: ..." and
 * "No skills indexed yet", which reads like an empty database rather than a
 * server that is not running.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { api } from "@/api/client";
import type { HealthResponse } from "@/api/types";

/** Re-check periodically so the UI recovers once the backend is started. */
const POLL_INTERVAL_MS = 10_000;

export interface HealthState {
  health: HealthResponse | null;
  error: string | null;
  loading: boolean;
  /** True when the API could not be reached at all (network/proxy failure). */
  offline: boolean;
  retry: () => void;
}

const HealthContext = createContext<HealthState | null>(null);

export function HealthProvider({ children }: { children: ReactNode }) {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let cancelled = false;

    async function probe() {
      try {
        const result = await api.health();
        if (cancelled) return;
        setHealth(result);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        // Keep the last known health so the header does not flicker, but clear
        // it when we have never had a successful response.
        setHealth((prev) => prev);
        setError(err instanceof Error ? err.message : "Backend unreachable");
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void probe();
    const timer = setInterval(probe, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [nonce]);

  const retry = useCallback(() => {
    setLoading(true);
    setNonce((n) => n + 1);
  }, []);

  const value = useMemo<HealthState>(
    () => ({
      health,
      error,
      loading,
      // A 500 from the dev proxy means the API is not listening. A 200 with
      // database="down" is a different problem and gets its own message.
      offline: error !== null,
      retry,
    }),
    [health, error, loading, retry],
  );

  return <HealthContext.Provider value={value}>{children}</HealthContext.Provider>;
}

export function useHealth(): HealthState {
  const ctx = useContext(HealthContext);
  if (!ctx) throw new Error("useHealth must be used inside <HealthProvider>");
  return ctx;
}