"""
Shared pytest fixtures.

TEST SPLIT
----------
* Pure unit tests (chunking, extraction, skills, RRF, filters, prompts) run with
  no external services. They should always pass.
* Database/API tests need Postgres + pgvector. They are marked `integration` and
  SKIP automatically when no database is reachable, so `pytest` stays green on
  a laptop without Docker.

ISOLATION
---------
Integration tests run against a SEPARATE database (`<POSTGRES_DB>_test`), created
automatically on first run. This matters: the ingestion routes call
`session.commit()`, so you cannot simply wrap a test in a transaction and hope -
and running the tests against your dev database would mean either destroying your
seeded data or having the assertions depend on whether you seeded first.

Run everything with:            pytest
Run only the fast unit tests:   pytest -m "not integration"
Run only the full-stack tests:  pytest -m integration
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
from sqlalchemy.engine import make_url

# ---------------------------------------------------------------------------
# HERMETIC TESTS: never call a paid, rate-limited LLM from the test suite.
#
# This must run BEFORE any `app.*` import, because `app.core.config` reads the
# environment once and caches it. Without this, setting LLM_PROVIDER=huggingface
# in your .env makes every search test issue a real billed API call - which
# burns your $0.10 of monthly credits, makes the suite minutes slower, and turns
# a flaky third-party endpoint into a red build.
#
# Opt in to a live call with the `llm_live` marker (see `llm_live` fixture).
# ---------------------------------------------------------------------------
os.environ["LLM_PROVIDER"] = "none"

from app.core.runtime import configure_event_loop  # noqa: E402

configure_event_loop()


# ---------------------------------------------------------------------------
# Database availability
# ---------------------------------------------------------------------------
def _postgres_reachable() -> bool:
    """TCP probe so a missing database is a skip, not a wall of tracebacks."""
    from app.core.config import settings

    try:
        with socket.create_connection(
            (settings.postgres_host, settings.postgres_port), timeout=2
        ):
            return True
    except OSError:
        return False


POSTGRES_AVAILABLE = _postgres_reachable()


def _test_database_url() -> tuple[str, str]:
    """
    Return (admin_url, test_url).

    `admin_url` points at the default `postgres` database - you cannot
    `CREATE DATABASE` from inside the database you are creating. `test_url` is
    the same server with the database name swapped for `<name>_test`.
    """
    from urllib.parse import urlsplit, urlunsplit

    from app.core.config import settings

    parts = urlsplit(settings.sqlalchemy_url)
    test_name = f"{settings.postgres_db}_test"

    def rebuild(dbname: str) -> str:
        return urlunsplit(parts._replace(path=f"/{dbname}"))

    return rebuild("postgres"), rebuild(test_name)


def _ensure_test_database(admin_url: str, test_name: str) -> None:
    """Create the test database if it does not exist yet. Synchronous on purpose."""
    import psycopg

    # psycopg wants a DSN string, not a SQLAlchemy URL object.
    dsn = make_url(admin_url).set(drivername="postgresql").render_as_string(hide_password=False)
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (test_name,))
        if cur.fetchone() is None:
            # An identifier cannot be a bound parameter; test_name is derived
            # from our own settings, not user input.
            cur.execute(f'CREATE DATABASE "{test_name}"')


def _create_extensions(connection) -> None:
    """Synchronous on purpose: this is invoked through `run_sync`."""
    for ext in ("vector", "pg_trgm"):
        connection.exec_driver_sql(f"CREATE EXTENSION IF NOT EXISTS {ext}")


requires_db = pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason=(
        "Postgres with pgvector is not reachable. Start it with "
        "`docker compose up -d db` from the project root."
    ),
)


@pytest.fixture
async def session():
    """
    A database session against a dedicated test database.

    Every test starts from a clean slate and its changes are rolled back, so
    tests are independent of each other AND of your development data. Using a
    separate database (rather than a wrapped transaction) is required because the
    ingestion routes call `session.commit()`, which releases the savepoint.
    """
    from sqlalchemy import text as sql_text
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.db import models  # noqa: F401  (import registers the tables)
    from app.db.base import Base

    admin_url, test_url = _test_database_url()
    test_name = test_url.rstrip("/").rsplit("/", 1)[-1]
    _ensure_test_database(admin_url, test_name)

    engine = create_async_engine(test_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(_create_extensions)
        await conn.run_sync(Base.metadata.create_all)

    # TRUNCATE ... CASCADE also clears chunks and skills, so a previous test's
    # vectors can never leak into this one's assertions.
    async with engine.begin() as conn:
        await conn.execute(
            sql_text(
                "TRUNCATE candidates, resume_chunks, candidate_skills, "
                "search_queries RESTART IDENTITY CASCADE"
            )
        )

    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.rollback()

    await engine.dispose()


@pytest.fixture
async def client(session):
    """
    ASGI client bound to the test database.

    `ASGITransport` (rather than `TestClient`) keeps everything on a single event
    loop, which the async session requires.
    """
    from httpx import ASGITransport, AsyncClient

    from app.db.session import get_session
    from app.main import create_app

    app = create_app()

    async def _override():
        yield session

    app.dependency_overrides[get_session] = _override

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac
    app.dependency_overrides.clear()


@pytest.fixture
def llm_client():
    """
    A live LLM client for tests explicitly marked `@pytest.mark.llm_live`.

    Skipped unless the provider and its key are both configured, so the default
    suite never touches the network for generation.
    """
    from app.services.llm.client import NullLLMClient, build_llm_client

    client = build_llm_client()
    if isinstance(client, NullLLMClient):
        pytest.skip("no LLM configured - see backend/.env")
    return client


# ---------------------------------------------------------------------------
# Sample resumes
# ---------------------------------------------------------------------------
SAMPLE_SENIOR_PYTHON = """\
Priya Raghunathan
Berlin, Germany
priya.raghunathan@example.com | +49 30 5551234
linkedin.com/in/priya-raghunathan | github.com/praghu

SUMMARY
Senior backend engineer with 8 years building data-intensive services. Led the
migration of a monolithic billing system to an event-driven architecture.

EXPERIENCE
Staff Backend Engineer | Zalando | 2021 - Present
- Rebuilt the search pipeline on PostgreSQL and pgvector, cutting p95 query
  latency from 1200 ms to 180 ms.
- Designed a Kafka-based ingestion flow handling 40k events per second.
- Mentored 4 engineers and introduced code review standards.

Backend Engineer | Delivery Hero | 2017 - 2021
- Built REST APIs in Python and FastAPI serving 2M requests per day.
- Migrated the reporting stack from MySQL to PostgreSQL with zero downtime.
- Introduced Docker and Kubernetes for the deployment pipeline.

EDUCATION
M.Sc. Computer Science | Technische Universitat Berlin | 2017

SKILLS
Languages: Python, JavaScript, SQL, Go
Frameworks: FastAPI, Django, React
Data: PostgreSQL, Kafka, Airflow, Redis, pgvector
Cloud: AWS, Docker, Kubernetes, Terraform

CERTIFICATIONS
AWS Certified Solutions Architect
"""


SAMPLE_JUNIOR_REACT = """\
Marcus Webb
Austin, TX, USA
marcus.webb@example.com | (512) 555-9876
github.com/marcuswebb

PROFILE
Frontend developer focused on React and design systems. Enjoys turning complex
product requirements into accessible interfaces.

EXPERIENCE
Frontend Developer | Atlassian | 2022 - Present
- Rebuilt the component library in React and TypeScript, adopted by 9 teams.
- Cut bundle size by 38% using code splitting and lazy loading.
- Added end-to-end tests with Playwright.

Junior Web Developer | SiteGround | 2020 - 2022
- Built marketing sites with HTML, CSS and vanilla JavaScript.
- Introduced npm and ESLint to the team workflow.

EDUCATION
B.Sc. Computer Science | University of Texas at Austin | 2020

SKILLS
Languages: JavaScript, TypeScript, HTML, CSS
Frameworks: React, Next.js, Vue
Tools: Jest, Cypress, Figma, Git
"""


SAMPLE_ML_RESEARCHER = """\
Dr. Ananya Iyer
Bengaluru, India
ananya.iyer@example.com | +91 80 4000 1234
linkedin.com/in/ananya-iyer

PROFESSIONAL SUMMARY
Applied ML researcher specialising in retrieval-augmented generation and
evaluation of large language models for enterprise search.

EXPERIENCE
Principal ML Engineer | Sarvam AI | 2022 - Present
- Built a hybrid retrieval pipeline combining BM25 and pgvector with reciprocal
  rank fusion, improving recall@10 by 34%.
- Designed a reranking stage with a cross-encoder over MiniLM checkpoints.
- Published three papers on hallucination detection in grounded generation.

Machine Learning Engineer | Flipkart | 2018 - 2022
- Fine-tuned transformer models with PyTorch and Hugging Face.
- Deployed model serving on Kubernetes with Docker images.
- Built feature stores on PostgreSQL for 12 business domains.

EDUCATION
Ph.D. Machine Learning | Indian Institute of Science | 2018

SKILLS
Languages: Python, R, SQL
ML: PyTorch, TensorFlow, scikit-learn, Hugging Face, LangChain
Data: PostgreSQL, Pinecone, FAISS, Airflow, Spark
Cloud: GCP, Docker, Kubernetes, MLflow

LANGUAGES
English, Hindi, Kannada
"""


SAMPLE_DEVOPS = """\
Tomasz Nowak
Krakow, Poland
tomasz.nowak@example.com | +48 12 555 2233

SUMMARY
Infrastructure engineer focused on reliability, Kubernetes and CI/CD. Reduced
paging volume by building better observability.

EXPERIENCE
Senior DevOps Engineer | Allegro | 2020 - Present
- Operated 40 Kubernetes clusters across three cloud regions.
- Built a CI/CD pipeline with GitHub Actions and ArgoCD cutting deploy time
  from 45 to 8 minutes.
- Introduced Terraform modules that standardised 200+ cloud resources.

Systems Administrator | CD Projekt | 2016 - 2020
- Managed Linux servers, Apache and Nginx in production.
- Automated backups with Bash and Python scripts.

EDUCATION
B.Eng. Computer Engineering | AGH University of Science and Technology | 2016

SKILLS
Cloud: AWS, GCP, Terraform, Docker, Kubernetes
Languages: Bash, Python, Go
Tooling: GitHub Actions, Jenkins, Grafana, Prometheus, Ansible
"""


ALL_SAMPLES = {
    "priya_backend_python": SAMPLE_SENIOR_PYTHON,
    "marcus_frontend_react": SAMPLE_JUNIOR_REACT,
    "ananya_ml_research": SAMPLE_ML_RESEARCHER,
    "tomasz_devops": SAMPLE_DEVOPS,
}


@pytest.fixture
def sample_resumes() -> dict[str, str]:
    return dict(ALL_SAMPLES)


@pytest.fixture
def tmp_resume_files(tmp_path: Path) -> Path:
    """Writes each sample resume to a .txt file and returns the directory."""
    for name, text in ALL_SAMPLES.items():
        (tmp_path / f"{name}.txt").write_text(text, encoding="utf-8")
    return tmp_path