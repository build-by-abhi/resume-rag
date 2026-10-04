/**
 * `useSearch` - owns the search form state and the request lifecycle.
 *
 * Why debounce here and not in the input component: form state and the network
 * call are different concerns. Keeping them apart means typing stays instant
 * while the (expensive) hybrid search fires once the user pauses.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "@/api/client";
import type { SearchRequest, SearchResponse } from "@/api/types";

export interface UseSearchResult {
  response: SearchResponse | null;
  loading: boolean;
  error: string | null;
  /** Fire immediately (Enter key, Search button). */
  submit: (payload: SearchRequest) => void;
  /** Fire after a short pause (filter dropdowns, checkboxes). */
  schedule: (payload: SearchRequest) => void;
  reset: () => void;
}

const DEBOUNCE_MS = 350;

export function useSearch(): UseSearchResult {
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Only the newest request may write to state.
  const requestId = useRef(0);
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const run = useCallback(async (payload: SearchRequest) => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);

    try {
      const result = await api.search(payload);
      if (id === requestId.current) {
        setResponse(result);
        setLoading(false);
      }
    } catch (err) {
      if (id === requestId.current) {
        setError(err instanceof Error ? err.message : "Search failed");
        setLoading(false);
      }
    }
  }, []);

  const submit = useCallback((payload: SearchRequest) => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    void run(payload);
  }, [run]);

  const schedule = useCallback(
    (payload: SearchRequest) => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
      debounceTimer.current = setTimeout(() => void run(payload), DEBOUNCE_MS);
    },
    [run],
  );

  const reset = useCallback(() => {
    requestId.current += 1;
    setResponse(null);
    setError(null);
    setLoading(false);
  }, []);

  // Cancel a pending debounce on unmount so no request fires after teardown.
  useEffect(
    () => () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    },
    [],
  );

  return { response, loading, error, submit, schedule, reset };
}