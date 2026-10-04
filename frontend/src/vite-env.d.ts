/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional absolute backend URL. Empty = same origin (proxied by Vite). */
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}