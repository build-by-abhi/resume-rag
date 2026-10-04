# Resume RAG + HR Search

An HR recruiting tool: upload resumes, ask questions in plain English, get back
ranked candidates with **evidence** and a **cited answer**.

The central design decision: **this is not pure vector search.** Resume search
needs structured extraction, metadata filtering and hybrid retrieval, because an
embedding cannot reliably answer "6+ years in Berlin" and a keyword index cannot
answer "someone who has shipped a vector search pipeline".

```
HR question
   │
   ├─ 1. STRUCTURED FILTERS   exact SQL on name / location / years / skills
   │                          (embeddings are lossy - exact beats approximate)
   │
   ├─ 2a. VECTOR SEARCH       pgvector cosine distance over resume chunks
   │ 2b. KEYWORD SEARCH       Postgres BM25 (ts_rank_cd) over the same chunks
   │
   ├─ 3. RRF FUSION           rank-based fusion of the two lists
   │
   ├─ 4. CROSS-ENCODER RERANK optional; reads (query, chunk) pairs together
   │
   └─ 5. GROUNDED ANSWER      LLM answer with [1][2] citations, or an
                              extractive digest when no key is configured
```

---

## Table of contents

1. [Quick start](#quick-start)
2. [What to try once it is running](#what-to-try-once-it-is-running)
3. [Why hybrid, not pure vector](#why-hybrid-not-pure-vector)
4. [Project structure](#project-structure)
5. [How the pieces work](#how-the-pieces-work)
6. [Configuration](#configuration)
7. [Testing](#testing)
8. [The roadmap this implements](#the-roadmap-this-implements)
9. [Known limitations](#known-limitations)

---

## Quick start

### Prerequisites

| Tool | Needed? | Notes |
|---|---|---|
| Python 3.11+ | yes | developed on 3.11–3.14 |
| Node 18+ | yes | only for the React UI |
| Docker | recommended | runs Postgres + pgvector |

### 1. Start the database

```bash
docker compose up -d db
```

This starts Postgres 16 with the `vector` and `pg_trgm` extensions on
`localhost:5432`.

### 2. Backend

```bash
cd backend

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
copy .env.example .env          # Windows
# cp .env.example .env          # macOS/Linux

# Create the sample talent pool (4 resumes)
python -m scripts.seed --reset

# Run the API  ->  http://127.0.0.1:8000/docs
python -m app.run --reload
```

> **Windows note.** Use `python -m app.run`, not `uvicorn app.main:app --reload`.
> uvicorn builds its own event loop and picks `ProactorEventLoop` on Windows,
> which psycopg's async driver cannot use. `app/run.py` passes an explicit
> `SelectorEventLoop`. (`uvicorn app.main:app --reload` happens to work too,
> because reload mode forces the subprocess path, but do not rely on it.)

### 3. Frontend

```bash
cd frontend
npm install
npm run dev          # -> http://localhost:5173
```

Vite proxies `/api` to `http://localhost:8000`, so there is no CORS setup in
development.

### Everything in Docker

```bash
copy backend\.env.example backend\.env     # set your keys first
docker compose up --build                  # web on :5173, api on :8000
```

### First run downloads two models

`sentence-transformers` fetches `all-MiniLM-L6-v2` (~90 MB) for embeddings and
`ms-marco-MiniLM-L-6-v2` (~90 MB) for reranking. Both are cached afterwards.
If you want a zero-download setup (useful in CI), set
`EMBEDDING_PROVIDER=hashing`, `EMBEDDING_DIM=1536`, `RERANK_ENABLED=false` and
re-seed with `python -m scripts.seed --drop`.

---

## What to try once it is running

| Try this | Shows off |
|---|---|
| `senior backend engineer who has worked with Kubernetes and PostgreSQL` | semantic + keyword + rerank working together |
| `someone who built a hybrid search system using BM25 and vector databases` | semantic leg alone (no shared keywords) |
| Skills `k8s` + `match any` | taxonomy aliasing — `k8s` is stored as `kubernetes` |
| `min years 5`, skills Python + PostgreSQL | exact structured filtering |
| Filters only, no question | "table view" mode, pure SQL |
| Click **Show why this matched** | the retrieved chunks and each leg's score |
| Click a `[n]` citation in the answer | the exact excerpt the claim came from |
| Upload the same PDF twice | sha256 dedupe — the second upload is a no-op |
| JD Match page | a job description scored against the pool |

---

## Why hybrid, not pure vector

Each retrieval strategy fails in a different, well-understood way.

**Pure vector search fails at:**

- exact tokens — `C++` vs `C`, or `Kubernetes CRD operator`
- hard numbers — a 6-year candidate can outrank a 7-year one
- negation and exclusion — "not a manager"
- anything needing `>=` or `=` rather than "looks similar"

**Pure keyword (BM25) fails at:**

- synonyms — "ML" and "machine learning" never meet
- seniority — happily returns a 2-year resume for "senior engineer"
- implicit requirements — "someone who can own a migration"

**So each layer does what it is actually good at:**

| Query | Layer that answers it |
|---|---|
| `location = 'Berlin'` | structured filter (indexed, exact) |
| `skills && ['python','kubernetes']` | `candidate_skills` + GIN index |
| `total_years_experience >= 5` | numeric comparison, indexed |
| "built a vector search pipeline" | vector similarity over chunks |
| "Terraform" | BM25 over the generated `tsvector` |

And they are combined with **Reciprocal Rank Fusion**, not raw scores:

```
RRF(d) = Σ  1 / (k + rank_i(d)),   k = 60
```

Ranks are used rather than scores because a cosine similarity of `0.71` and a
BM25 score of `3.2` are not on the same scale. Fusing raw scores requires
inventing a normalisation that is really just a guess. Fusion by rank is
scale-free and is the standard approach.

---

## Project structure

```
resume_rag/
├── docker-compose.yml            Postgres+pgvector, API, web
├── docker/postgres/init.sql      installs the vector / pg_trgm extensions
│
├── backend/
│   ├── app/
│   │   ├── main.py               FastAPI app factory, CORS, lifespan
│   │   ├── run.py                uvicorn entrypoint (Windows event loop)
│   │   ├── core/
│   │   │   ├── config.py         all settings, one cached instance
│   │   │   ├── logging.py
│   │   │   └── runtime.py        event-loop compatibility
│   │   ├── db/
│   │   │   ├── base.py           declarative base
│   │   │   ├── session.py        engine, session, schema creation
│   │   │   └── models.py         candidates, candidate_skills, resume_chunks
│   │   ├── schemas/              Pydantic request/response models
│   │   ├── api/routes/
│   │   │   ├── upload.py         POST /api/candidates/upload | /text
│   │   │   ├── candidates.py     browse, detail, facets, delete
│   │   │   ├── search.py         POST /api/search | /api/match, facets
│   │   │   └── health.py         /health, /stats, /health/ready
│   │   ├── services/
│   │   │   ├── ingestion/        pdf_loader, pipeline (the orchestrator)
│   │   │   ├── extraction/       rule_extractor, llm_extractor, skills, merge
│   │   │   ├── chunking/         section-aware, overlapping chunker
│   │   │   ├── embeddings/       provider interface + 4 implementations
│   │   │   ├── vectordb/         pgvector queries
│   │   │   ├── retrieval/        filters, hybrid (RRF), rerank, answer
│   │   │   └── llm/              provider-agnostic client (4 backends)
│   │   ├── prompts/              every prompt, in one reviewable file
│   │   └── utils/                text cleaning, upload safety
│   ├── scripts/
│   │   ├── seed.py               populate the demo database
│   │   ├── inspect_vectors.py    print stored vectors next to their text
│   │   ├── check_llm.py          verify the configured LLM provider
│   │   ├── make_sample_files.py  generate the PDF/DOCX fixtures
│   │   └── sample_resumes/       6 realistic sample resumes (txt, pdf, docx)
│   └── tests/                    226 tests (unit + integration)
│
└── frontend/                     React 18 + TypeScript + Vite + Tailwind
    └── src/
        ├── api/                  typed client, types mirroring the API
        ├── components/           UploadPanel, FilterPanel, AnswerPanel, ...
        ├── hooks/                useAsync, useSearch
        └── pages/                Search, Candidates, Detail, Upload, JD Match
```

---

## How the pieces work

### Ingestion (`services/ingestion/pipeline.py`)

The one place that knows the whole order of operations:

```
parse PDF/DOCX/TXT → clean text → sha256 dedupe → extract fields
→ section-aware chunking → embed chunks + profile → ONE transaction
```

All-or-nothing persistence is deliberate: a candidate with a profile but no
chunks is invisible to search and breaks every result-count expectation.

### Structured extraction (`services/extraction/`)

Rule-based extraction **always** runs; the LLM extractor runs after and only
fills gaps. The merge policy is explicit in `normalizer.py`:

| Field | Winner | Why |
|---|---|---|
| email, phone, links | rules always | transcription is not a language task |
| name, title, company | rules, else LLM | |
| years of experience | rules, else LLM | rules merge date ranges properly |
| summary | the longer one | |
| skills | union, counts summed | |

Each field is validated before it can reach the database — LLM output is
untrusted input.

### Skill taxonomy (`services/extraction/skills.py`)

Resumes write `React.js`, `ReactJS`, `React JS` and `React`. Without collapsing
them, one person becomes four rows and four filters. The taxonomy maps every
surface form to one canonical key, and it is reused at query time for
**alias expansion**: searching `postgres` also matches `PostgreSQL`, because
BM25 only sees literal tokens and `pg_trgm` cannot bridge a 3-letter gap.

### Chunking (`services/chunking/chunker.py`)

Not "split every 1000 characters":

1. section headers are detected and never merged across
2. a breadcrumb (`[EXPERIENCE | Acme Corp]`) is prepended so an isolated chunk
   still carries its context into the embedding
3. chunks break on paragraph → sentence → word boundaries, with token overlap
4. a stable `heading` lets results be cited as "Experience | Zalando | 2021"

### The database (`db/models.py`)

```sql
candidates        structured columns + profile_embedding vector(384)
                  + GIN expression index over the profile fields
candidate_skills  one row per (candidate, canonical skill)
resume_chunks     content + embedding vector(384)
                  + search_tsv tsvector GENERATED ALWAYS AS ...
                  + HNSW index (m=16, ef_construction=64) for cosine search
search_queries    audit log: query, filters, result count, latency
```

`search_tsv` is a **generated column**, so Postgres keeps it in sync with
`content` and Python never writes to it.

### Answering (`services/retrieval/answer.py`)

Two paths behind one interface:

- **LLM** — numbered context, instructions to answer only from context and cite
- **extractive** — a factual digest of the top hits when no key is configured

The extractive path is not a placeholder; it is the guaranteed-correct floor.
A recruiter must never get a blank box because a key expired. Citations the
model invents (`[9]` when only 4 excerpts exist) are detected and stripped.

---

## Choosing an LLM (including HuggingFace)

The LLM is used for two things: writing the cited answer, and (optionally)
extracting structured fields at ingest time. **Retrieval never depends on it** —
with no key configured the app still searches, filters and ranks candidates, and
the answer panel serves an extractive digest instead.

| `LLM_PROVIDER` | Model example | Needs |
|---|---|---|
| `none` | – | nothing (default) |
| `huggingface` | `meta-llama/Llama-3.2-3B-Instruct` | `HUGGINGFACE_API_KEY` |
| `openai` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `anthropic` | `claude-3-5-haiku-latest` | `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-2.0-flash` | `GEMINI_API_KEY` |

### Using HuggingFace

```dotenv
# backend/.env
LLM_PROVIDER=huggingface
LLM_MODEL=Qwen/Qwen3-8B
HUGGINGFACE_API_KEY=hf_...
```

**Qwen3 is not gated** — a plain read token from
<https://huggingface.co/settings/tokens> is enough, no licence click. (If you
ever switch to `meta-llama/*`, you *do* need to accept the licence at
<https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct> **and** create the
token with *"Read access to contents of all public gated repos"*.)

Then verify before restarting anything:

```bash
cd backend
python scripts/check_llm.py
```

```
  LLM_PROVIDER = huggingface
  LLM_MODEL    = Qwen/Qwen3-8B
  HUGGINGFACE_API_KEY = hf_abc...wxyz (37 chars)

  client = huggingface  model = Qwen/Qwen3-8B

  [1/2] plain completion
  ok in 812ms -> 'Paris is the capital of France.'
  tokens: prompt=24 completion=8

  [2/2] JSON round trip (used by structured extraction)
  ok -> {'skills': ['python'], 'location': 'Berlin', ...}

  RESULT: provider healthy.
```

The script exists because provider problems are otherwise invisible: an expired
key silently downgrades you to the extractive fallback and search still "works".
`check_llm.py` also reports whether the model actually returns parseable JSON,
which is what structured extraction depends on.

### The free tier is small — check your budget

| Account | Monthly HF credits |
|---|---|
| **Free** | **$0.10** |
| PRO ($9/mo) | $2.00 |

That is enough for a portfolio demo (hundreds of calls on an 8B model), but it
runs out fast if you re-ingest repeatedly with LLM extraction enabled. Watch it
at <https://huggingface.co/settings/billing>. When credits run dry the app
degrades to the extractive answer instead of erroring — but the written answer
disappears, so keep an eye on it.

If the cap is the problem, running the same model locally has no limits at all:

```bash
pip install ollama && ollama pull qwen3:8b
```

### Qwen3 is a *reasoning* model — this is handled for you

`Qwen/Qwen3-*` emits a `<think>...</think>` block before answering unless the
chat template is told not to. Left on, that block would:

- roughly triple latency for a six-field extraction
- sit between the prompt and the JSON, making extraction fall back to rules
- be rendered verbatim in the answer panel

So the app sets it automatically:

```python
# app/core/config.py -> Settings.llm_extra_body_merged
if model.startswith("qwen/qwen3") and "enable_thinking" not in extra:
    extra["enable_thinking"] = False
```

which is sent as `extra_body={"enable_thinking": false}`. Override it if you
want reasoning for hard cases:

```dotenv
LLM_EXTRA_BODY={"enable_thinking": true}
```

There is also a second line of defence: `strip_reasoning()` removes `<think>`,
`<thinking>` and `<reasoning>` blocks (including an unterminated one, which is
what you get when `max_tokens` cuts the trace off mid-sentence). Verified against
a model that ignores the flag entirely — the answer and the JSON both still come
out clean.

### Model size guidance

| Size | Verdict |
|---|---|
| ≤ 3B (e.g. `Llama-3.2-3B`) | invents citation markers, flaky JSON — extraction silently degrades to rules |
| **8B (e.g. `Qwen3-8B`)** | **the sweet spot** — good JSON, decent grounding, fast on Groq |
| 14B+ | better grounding, costs ~2× the credits per call |

`/health` shows `llm: on/off`, and a candidate's `extraction_method` shows
`rules` vs `hybrid` — if you see `rules` everywhere, the JSON step is failing
and step 2 of `check_llm.py` is where to look.

### HuggingFace embeddings (optional)

The same key also works for embeddings, if you would rather not run a local
model:

```dotenv
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5   # 384-dim
EMBEDDING_DIM=384
python -m scripts.seed --drop            # re-create the vector columns
```

Keep `sentence-transformers` for bulk ingestion — every HF embedding call is a
network round trip, so a 400-chunk upload becomes 400 HTTP requests.

---

## Inspecting the vectors

A `vector` column is a 384-number list, which is unreadable in a data grid. Use
the script instead — it prints each vector *next to the text it came from*.

```bash
cd backend

python scripts/inspect_vectors.py                      # counts, dimensions, sample chunks
python scripts/inspect_vectors.py --candidate Priya    # one candidate in full
python scripts/inspect_vectors.py --nearest "kafka pipelines"   # live similarity search
python scripts/inspect_vectors.py --indexes            # indexes + extensions
python scripts/inspect_vectors.py --raw                 # one complete float list
```

| Flag | What it shows |
|---|---|
| *(none)* | candidate/chunk/skill counts, null-embedding check, declared dimensions vs `EMBEDDING_DIM`, and the first chunks with their vector heads |
| `--candidate NAME` | every stored field for one person, plus their skill rows and profile vector |
| `--nearest TEXT` | embeds TEXT with the **same provider used at ingest** and runs the raw pgvector query, then the BM25 query on the same text |
| `--indexes` | every index with its DDL, plus installed extensions |
| `--raw` | the full 384-float list, 6 per line, with the L2 norm |
| `--head N` | how many vector numbers to print (default 8) |
| `--text-width N` | snippet width (default 70) |
| `--limit N` | rows to show (default 10) |

`--nearest` is the most instructive one. It also highlights *why* this project
is hybrid — for a concept query the vector leg returns results while BM25
returns none:

```
  NEAREST CHUNKS TO: 'event streaming pipelines at scale'
  embedding model    : all-MiniLM-L6-v2

  cosine similarity (1 = identical, 0 = opposite)
  1. similarity 0.7445   (cosine distance 0.5110)
  2. similarity 0.7338   (cosine distance 0.5323)

  BM25 (keyword) RESULTS FOR THE SAME QUERY
  no keyword matches
  -> This is the case that justifies hybrid search: the semantic leg
     found relevant chunks while exact keyword matching found none,
     because the resumes never use these exact words.
```

### Using a GUI instead

pgAdmin (<http://localhost:8081/browser/>) or DBeaver work fine. Register a
server with:

| Field | Value |
|---|---|
| Host | `localhost` |
| Port | `5432` |
| Maintenance database | `postgres` |
| Username | `postgres` |
| Password | `postgres` |
| Your database | `resume_rag` |

Browse `resume_rag → Schemas → public → Tables → resume_chunks`. Useful Query
Tool snippets:

```sql
SELECT count(*) FROM resume_chunks;
SELECT vector_dims(embedding) FROM resume_chunks LIMIT 1;

-- text and vector together (note: slice in SQL needs an integer literal,
-- so `embedding[1:9]` works but `embedding[1:$1]` does not)
SELECT c.full_name, rc.section, rc.embedding[1:8] AS vector_head,
       left(rc.content, 60) AS text
FROM resume_chunks rc
JOIN candidates c ON c.id = rc.candidate_id
LIMIT 10;

-- the HNSW and GIN indexes behind the two retrieval legs
SELECT indexname, indexdef FROM pg_indexes WHERE tablename = 'resume_chunks';
```

> Opening `http://localhost:5432` in a browser will always fail with
> `ERR_EMPTY_RESPONSE`. Browsers speak HTTP; PostgreSQL speaks its own wire
> protocol. Port 5432 is not a web server.

---

## Configuration

Everything lives in `backend/.env`. The values you are most likely to change:

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_PROVIDER` | `sentence-transformers` | or `huggingface`, `openai`, `hashing` |
| `EMBEDDING_DIM` | `384` | must match the model; re-seed with `--drop` after changing |
| `LLM_PROVIDER` | `huggingface` | `openai` / `anthropic` / `gemini` / `none` |
| `LLM_MODEL` | `Qwen/Qwen3-8B` | see the reasoning-model note below |
| `HUGGINGFACE_API_KEY` | – | covers both the LLM and HF embeddings |
| `LLM_EXTRA_BODY` | auto | JSON forwarded to the chat endpoint |
| `RERANK_ENABLED` | `true` | ~300–500 ms per query on CPU |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `350` / `60` | approximate tokens |
| `TOP_K` | `10` | |
| `MAX_UPLOAD_MB` | `10` | enforced while streaming, not after |

> Changing `EMBEDDING_DIM` after ingesting fails with
> `expected 1536 dimensions, not 384`, because the vector width is baked into
> `vector(N)` at `CREATE TABLE`. The app detects this and tells you to run
> `python -m scripts.seed --drop`.

---

## Testing

```bash
cd backend

pytest                    # everything
pytest -m "not integration"   # fast unit tests only, no database needed
pytest -m integration         # full stack against Postgres + pgvector
ruff check .
```

- **103 unit tests** — no external services. Chunking, cleaning, skill
  taxonomy, rule extraction, the rules/LLM merge policy, RRF maths, filter SQL
  compilation (including SQL-injection payloads), embedding determinism,
  citation sanitising.
- **28 integration tests** — real Postgres, real SQL, real vectors. Ingestion,
  facets, hybrid search, alias expansion, filters, JD matching, cascade
  deletes. They run against a separate `resume_rag_test` database that is
  created automatically, so they never touch your dev data.

Integration tests skip automatically when no database is reachable, so
`pytest` stays green without Docker.

---

## The roadmap this implements

| Phase | Status |
|---|---|
| **1.** Upload PDF → extract → candidate + chunks → embeddings → pgvector | done |
| **2.** Natural-language HR search | done |
| **3.** Hybrid search (BM25 + vector + RRF + reranking) | done |
| **4.** JD → candidate matching with scorecards | done |
| **5.** Production security + evaluation | partly |

What Phase 5 would add:

- Alembic migrations (currently `create_all`, which is fine for a demo)
- API-key / OAuth auth — `api/deps.py` has the `require_api_key` hook ready
- object-level access control, so an HR user cannot search another tenant's pool
- encryption or tokenisation of PII columns; resume data is personal data
- an eval harness over the `search_queries` audit log (recall@k, no-answer rate,
  latency percentiles) plus a labelled golden set
- rate limiting, background job queue for bulk ingestion, object storage for
  resume files

---

## Known limitations

Honest list, because a portfolio project that hides these gets caught in the
interview:

1. **Rule extraction is regex + heuristics.** It handles conventional resume
   layouts well and two-column PDFs poorly. The LLM extractor covers the gap
   when a key is configured; without one, some fields are simply null. The
   upload panel surfaces these gaps as warnings instead of hiding them.
2. **Skills come from a curated taxonomy.** A technology not in
   `SKILL_TAXONOMY` is still stored (lowercased, from the skills section), but
   it will not match an alias.
3. **Scanned/image-only PDFs need OCR.** The parser detects the empty text layer
   and says so rather than ingesting an empty profile.
4. **The `hashing` embedding provider is not semantic.** It is a lexical
   hashing trick: deterministic, free, offline, and good enough to exercise the
   plumbing in CI. Use `sentence-transformers` for real semantic quality.
5. **Reranking costs latency on CPU** (~300–500 ms per query after the model is
   loaded). It is worth it for quality; turn it off for bulk browsing.
6. **Search is not permission-aware.** Every caller sees the whole pool.
7. **No OCR, no resume parsing of complex tables, no multi-language stemming**
   (`websearch_to_tsquery` uses the `english` config).

---

## API reference

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | DB status, embedding provider, LLM status, counts |
| `GET` | `/health/ready` | strict readiness probe |
| `GET` | `/stats` | corpus statistics |
| `POST` | `/api/candidates/upload` | multipart upload (PDF/DOCX/TXT) |
| `POST` | `/api/candidates/text` | ingest pasted resume text |
| `GET` | `/api/candidates` | structured browse, filter, sort, paginate |
| `GET` | `/api/candidates/facets` | distinct values for the filters |
| `GET` | `/api/candidates/{id}` | full profile + embedded chunks |
| `DELETE` | `/api/candidates/{id}` | delete (cascades to chunks and skills) |
| `POST` | `/api/search` | hybrid search + cited answer |
| `GET` | `/api/search/skills` | skill facets with candidate counts |
| `GET` | `/api/search/filters` | seniority / location / education options |
| `POST` | `/api/match` | JD → candidate scorecards |

Interactive docs at `http://127.0.0.1:8000/docs`.