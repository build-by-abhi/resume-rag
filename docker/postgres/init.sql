-- Runs automatically the first time the Postgres container starts.
-- It installs the pgvector extension. Your app's SQLAlchemy models will
-- create the tables/columns; this file only needs the extension itself.

CREATE EXTENSION IF NOT EXISTS vector;

-- Optional: pg_trgm speeds up fuzzy skill/company matching.
CREATE EXTENSION IF NOT EXISTS pg_trgm;