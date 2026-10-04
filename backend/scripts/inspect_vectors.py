"""
Inspect the stored vectors from the terminal.

    python scripts/inspect_vectors.py             # overview + sample rows
    python scripts/inspect_vectors.py --candidate Priya
    python scripts/inspect_vectors.py --nearest "kafka pipelines"   # embed a query
    python scripts/inspect_vectors.py --indexes
    python scripts/inspect_vectors.py --raw       # full float list for one chunk

    # narrow or widen the output
    python scripts/inspect_vectors.py --head 16 --text-width 120
    python scripts/inspect_vectors.py --limit 50

WHY THIS EXISTS
---------------
A vector column is a 384-number list in Postgres, which is useless to read in a
data grid. A GUI like pgAdmin shows all of it and none of it at the same time.
This script prints the vector *next to the text it came from*, which is the only
view that makes the abstraction concrete.

`--nearest` is the interesting one: it embeds a question with the SAME provider
used at ingest and runs the raw pgvector query, so you can see the exact cosine
distance the app sees before BM25, RRF and reranking get involved.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Allow `python scripts/inspect_vectors.py` from the backend directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.logging import setup_logging  # noqa: E402
from app.core.runtime import configure_event_loop  # noqa: E402

configure_event_loop()

from app.db.session import SessionLocal, close_engine, embedding_dim_mismatch  # noqa: E402

logger = __name__  # module logger; setup_logging() configures handlers

WIDTH = 78


def rule(title: str = "") -> None:
    """Horizontal separator. `title` is centred when provided."""
    if not title:
        print("-" * WIDTH)
        return
    pad = max(0, (WIDTH - len(title) - 2) // 2)
    print(f"\n{'=' * WIDTH}")
    print(f"{' ' * pad}{title}")
    print("=" * WIDTH)


def parse_vector(raw: str | None, head: int) -> list[float]:
    """
    Read the first `head` numbers out of a `vector` rendered as text.

    WHY SLICE IN PYTHON RATHER THAN IN SQL
    -------------------------------------
    The obvious query is `embedding[1:9]`, but that needs the `::text` cast in a
    specific spot and the operator precedence fights you:

        embedding[1:9]::text   -> parsed as embedding[ (1:9)::text ] -> the error
                                    "cannot subscript type vector because it does
                                    not support subscripting"

    Subsetting the whole column and slicing in Python is version-independent
    (it works on every pgvector release), avoids the precedence trap entirely, and
    for ten rows costs nothing. The full column is only transferred when
    `--head` is set to a large value.
    """
    if not raw:
        return []
    values = [float(v) for v in raw.strip("[]").split(",") if v.strip()]
    return values[: max(1, head)]


def format_vector(values: list[float], head: int) -> str:
    """Human-readable slice of a vector, with the trailing count elided."""
    if not values:
        return "<null>"
    shown = ", ".join(f"{v:+.3f}" for v in values[:head])
    return shown if len(values) <= head else f"{shown}, ..."


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------
async def show_overview(session, args) -> None:
    """Counts, dimensions and a sample of chunk rows."""
    from sqlalchemy import text as sql_text

    rule("OVERVIEW")
    # Three sequential awaits rather than a generator expression: `await` inside
    # a generator expression creates an async generator, which cannot be unpacked.
    async def _count(table: str) -> int:
        result = await session.execute(sql_text(f"SELECT count(*) FROM {table}"))
        return result.scalar_one()

    candidates = await _count("candidates")
    chunks = await _count("resume_chunks")
    skills = await _count("candidate_skills")
    print(f"  candidates      {candidates:>6}")
    print(f"  chunks          {chunks:>6}")
    print(f"  skill rows      {skills:>6}")

    # How many vectors are actually present vs NULL. A NULL embedding means the
    # chunk is invisible to vector search, which is a silent failure otherwise.
    null_chunks = (
        await session.execute(
            sql_text(
                "SELECT count(*) FROM resume_chunks WHERE embedding IS NULL"
            )
        )
    ).scalar_one()
    null_candidates = (
        await session.execute(
            sql_text(
                "SELECT count(*) FROM candidates WHERE profile_embedding IS NULL"
            )
        )
    ).scalar_one()
    print(f"  null embeddings {null_chunks + null_candidates:>6}   <- must be 0")
    if null_chunks or null_candidates:
        print("    WARNING: some rows have no vector and can never be vector-searched.")

    rule("VECTOR DIMENSIONS")
    # `format_type` returns e.g. 'vector(384)'. Comparing the declared width
    # against the live setting catches the EMBEDDING_DIM mismatch early.
    for table, column in (
        ("candidates", "profile_embedding"),
        ("resume_chunks", "embedding"),
    ):
        declared = (
            await session.execute(
                sql_text(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod)
                    FROM pg_attribute a
                    JOIN pg_class t ON t.oid = a.attrelid
                    WHERE t.relname = :table AND a.attname = :column
                    """,
                ),
                {"table": table, "column": column},
            )
        ).scalar_one_or_none()
        print(f"  {table}.{column:<22} {declared or '<column missing>'}")

    mismatch = await embedding_dim_mismatch()
    if mismatch:
        print(f"\n  CONFIGURATION ERROR\n  {mismatch}\n")
    else:
        print("\n  Dimensions match EMBEDDING_DIM in .env")

    rule("CHUNKS")
    if args.candidate:
        rows = (
            await session.execute(
                sql_text(
                    """
                    SELECT c.full_name, rc.chunk_index, rc.section, rc.heading,
                           rc.token_count,
                           rc.embedding::text AS vec,
                           vector_dims(rc.embedding) AS dim,
                           left(rc.content, :text) AS snippet
                    FROM resume_chunks rc
                    JOIN candidates c ON c.id = rc.candidate_id
                    WHERE c.full_name ILIKE :like
                    ORDER BY c.full_name, rc.chunk_index
                    LIMIT :limit
                    """
                ),
                {
                    "text": args.text_width,
                    "like": f"%{args.candidate}%",
                    "limit": args.limit,
                },
            )
        ).mappings().all()
    else:
        rows = (
            await session.execute(
                sql_text(
                    """
                    SELECT c.full_name, rc.chunk_index, rc.section, rc.heading,
                           rc.token_count,
                           rc.embedding::text AS vec,
                           vector_dims(rc.embedding) AS dim,
                           left(rc.content, :text) AS snippet
                    FROM resume_chunks rc
                    JOIN candidates c ON c.id = rc.candidate_id
                    ORDER BY c.full_name, rc.chunk_index
                    LIMIT :limit
                    """
                ),
                {"text": args.text_width, "limit": args.limit},
            )
        ).mappings().all()

    if not rows:
        print("  No chunks found. Run:  python -m scripts.seed --reset")
        return

    for row in rows:
        # ASCII separators only: the Windows console defaults to a codepage that
        # renders "·" as a replacement character.
        print(
            f"\n  {row['full_name'] or 'Unnamed'}  |  chunk {row['chunk_index']}"
            f"  |  {row['section']}  |  {row['token_count']} tokens  |  {row['dim']}d"
        )
        if row["heading"]:
            print(f"    heading: {row['heading']}")
        # The text and its vector together - the whole point of this script.
        print(f"    text  : {row['snippet']}...")
        print(f"    vector: [{format_vector(parse_vector(row['vec'], args.head), args.head)}]")


async def show_candidate(session, args) -> None:
    """Everything stored for one candidate, including profile-level fields."""
    from sqlalchemy import text as sql_text

    rule(f"CANDIDATE: {args.candidate}")

    row = (
        await session.execute(
            sql_text(
                """
                SELECT id, full_name, email, phone, location, current_title,
                       current_company, total_years_experience, seniority,
                       education, summary, resume_filename, extraction_method,
                       char_length(raw_text) AS text_len,
                       profile_embedding::text AS vec,
                       vector_dims(profile_embedding) AS dim
                FROM candidates
                WHERE full_name ILIKE :like
                LIMIT 1
                """
            ),
            {"like": f"%{args.candidate}%"},
        )
    ).mappings().first()

    if row is None:
        print("  Not found.")
        return

    print(f"  id                {row['id']}")
    for label, key in (
        ("title", "current_title"),
        ("company", "current_company"),
        ("location", "location"),
        ("experience", "total_years_experience"),
        ("seniority", "seniority"),
        ("email", "email"),
        ("phone", "phone"),
        ("education", "education"),
        ("file", "resume_filename"),
        ("extraction", "extraction_method"),
    ):
        value = row[key]
        if value not in (None, "", {}):
            print(f"  {label:<17} {value}")

    print(f"  {'text length':<17} {row['text_len']} chars")
    print(f"  {'profile vector':<17} {row['dim']}d  "
          f"[{format_vector(parse_vector(row['vec'], args.head), args.head)}]")

    chunk_count = (
        await session.execute(
            sql_text(
                "SELECT count(*) FROM resume_chunks WHERE candidate_id = :cid"
            ),
            {"cid": row["id"]},
        )
    ).scalar_one()
    print(f"  {'chunks':<17} {chunk_count}")

    skill_rows = (
        await session.execute(
            sql_text(
                """
                SELECT skill, display, occurrences, is_core
                FROM candidate_skills
                WHERE candidate_id = :cid
                ORDER BY is_core DESC, occurrences DESC, skill
                LIMIT 20
                """
            ),
            {"cid": row["id"]},
        )
    ).mappings().all()
    if skill_rows:
        print("\n  skills:")
        for skill in skill_rows:
            core = "*" if skill["is_core"] else " "
            print(f"   {core} {skill['display']:<22} x{skill['occurrences']}")


async def show_nearest(session, args) -> None:
    """
    Embed a query and run the raw pgvector similarity search.

    This shows the pure semantic leg, before BM25, RRF and reranking. It is the
    clearest way to see what the embedding model actually considers similar.
    """
    from app.services.embeddings import get_embedder
    from app.services.extraction.skills import expand_query_for_skills
    from app.services.vectordb.pgvector_store import PgVectorStore

    rule(f"NEAREST CHUNKS TO: {args.nearest!r}")

    embedder = get_embedder()
    print(f"  embedding provider : {settings_embedding_provider()}")
    # Providers name the attribute differently: OpenAI/HF use `model`,
    # sentence-transformers uses `model_name`.
    model_name = getattr(embedder, "model", None) or getattr(embedder, "model_name", "?")
    print(f"  embedding model    : {model_name}")
    print("  embedding query... ", end="", flush=True)
    vector = await embedder.embed_query(args.nearest)
    print(f" ok ({len(vector)}d)")

    store = PgVectorStore(session)
    rows = await store.semantic_chunk_search(vector, limit=args.limit)
    if not rows:
        print("\n  No chunks stored. Run:  python -m scripts.seed --reset")
        return

    # `vector_score` from the store is the 0..1 SIMILARITY
    # (1 - cosine_distance / 2). Both numbers are shown because they are the
    # ones that appear in different places: the UI prints the similarity, while
    # `embedding <=> query` in SQL returns the distance.
    print("\n  cosine similarity (1 = identical, 0 = opposite)")
    for rank, row in enumerate(rows, start=1):
        similarity = float(row["vector_score"])
        distance = 2.0 * (1.0 - similarity)
        print(f"\n  {rank}. similarity {similarity:.4f}   (cosine distance {distance:.4f})")
        print(f"     section: {row['section']}")
        print(f"     text   : {row['content'][: args.text_width]}...")

    # The BM25 leg on the same query, so the difference is visible side by side.
    expanded = expand_query_for_skills(args.nearest)
    keyword_rows = await store.keyword_chunk_search(expanded, limit=args.limit)
    rule("BM25 (keyword) RESULTS FOR THE SAME QUERY")
    if expanded != args.nearest:
        print(f"  skill-alias expansion: {expanded}")
    if not keyword_rows:
        print(
            "  no keyword matches\n"
            "  -> This is the case that justifies hybrid search: the semantic leg\n"
            "     found relevant chunks while exact keyword matching found none,\n"
            "     because the resumes never use these exact words."
        )
    else:
        for rank, row in enumerate(keyword_rows, start=1):
            print(f"\n  {rank}. rank {row['keyword_score']:.4f}  [{row['section']}]")
            print(f"     {row['content'][: args.text_width]}...")

    rule("NOTE")
    print("  The app does not use either list directly - it fuses them with RRF")
    print("  and reranks. Run a real query in the UI to see the final ranking.")


def settings_embedding_provider() -> str:
    from app.core.config import settings

    return f"{settings.embedding_provider} ({settings.embedding_dim}d)"


async def show_indexes(session, args) -> None:
    """The indexes that make hybrid search fast."""
    from sqlalchemy import text as sql_text

    rule("INDEXES")
    rows = (
        await session.execute(
            sql_text(
                """
                SELECT tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname
                """
            )
        )
    ).mappings().all()

    current = None
    for row in rows:
        if row["tablename"] != current:
            current = row["tablename"]
            print(f"\n  {current}")
        # Wrap at a space so words stay intact. Splitting mid-token ("USI / NG
        # btree") makes DDL hard to read.
        definition = row["indexdef"]
        indent = "    "
        while len(definition) > WIDTH - 6:
            cut = definition.rfind(" ", 0, WIDTH - 6)
            cut = cut if cut > 20 else WIDTH - 6
            print(f"{indent}{definition[:cut]}")
            definition = definition[cut:].lstrip()
        if definition:
            print(f"{indent}{definition}")

    rule("EXTENSIONS")
    exts = (
        await session.execute(
            sql_text("SELECT extname, extversion FROM pg_extension ORDER BY extname")
        )
    ).mappings().all()
    for ext in exts:
        print(f"  {ext['extname']:<12} {ext['extversion']}")

    print(
        "\n  HNSW  = approximate nearest-neighbour index for the vector leg\n"
        "  GIN   = inverted index for the BM25 tsvector leg\n"
        "  BTREE = exact equality/range filters (skills, location, years)"
    )


async def show_raw(session, args) -> None:
    """The complete float list for a single chunk."""
    from sqlalchemy import text as sql_text

    rule("RAW VECTOR")
    row = (
        await session.execute(
            sql_text(
                """
                SELECT c.full_name, rc.chunk_index, rc.embedding::text AS vec
                FROM resume_chunks rc
                JOIN candidates c ON c.id = rc.candidate_id
                ORDER BY rc.chunk_index
                LIMIT 1
                """
            )
        )
    ).mappings().first()

    if row is None or row["vec"] is None:
        print("  No embeddings stored.")
        return

    values = [float(v) for v in row["vec"].strip("[]").split(",")]
    print(f"  {row['full_name']} | chunk {row['chunk_index']}")
    print(f"  {len(values)} dimensions\n")

    # 6 per line keeps the columns narrow enough for a standard terminal.
    for start in range(0, len(values), 6):
        chunk = values[start : start + 6]
        label = f"  [{start:>3}]"
        print(f"{label} " + "  ".join(f"{v:+.5f}" for v in chunk))

    norm = sum(v * v for v in values) ** 0.5
    print(f"\n  L2 norm = {norm:.6f}  <- ~1.0 because vectors are normalised")
    print("  (that is what makes cosine distance a plain dot product)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the vectors stored in Postgres.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--candidate", help="show one candidate in full (matches full_name, fuzzy)"
    )
    parser.add_argument(
        "--nearest", metavar="TEXT", help="embed TEXT and run the raw pgvector search"
    )
    parser.add_argument("--indexes", action="store_true", help="list indexes + extensions")
    parser.add_argument("--raw", action="store_true", help="print one full float list")
    parser.add_argument(
        "--head", type=int, default=8, help="how many vector numbers to show (default 8)"
    )
    parser.add_argument(
        "--text-width", type=int, default=70, help="text snippet width (default 70)"
    )
    parser.add_argument("--limit", type=int, default=10, help="rows to show (default 10)")
    args = parser.parse_args()

    setup_logging()

    if not any([args.candidate, args.nearest, args.indexes, args.raw]):

        async def _overview() -> int:
            async with SessionLocal() as session:
                await show_overview(session, args)
            await close_engine()
            return 0

        return asyncio.run(_overview())

    async def _run() -> int:
        async with SessionLocal() as session:
            if args.indexes:
                await show_indexes(session, args)
            elif args.raw:
                await show_raw(session, args)
            elif args.candidate:
                await show_candidate(session, args)
            else:
                await show_nearest(session, args)
        await close_engine()
        return 0

    try:
        return asyncio.run(_run())
    except Exception as exc:
        print(f"\n  ERROR: {type(exc).__name__}: {exc}\n")
        print("  Is the database running?  docker compose up -d db")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
