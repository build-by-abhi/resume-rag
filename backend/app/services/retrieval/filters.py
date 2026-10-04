"""
Structured / metadata filters.

This module is the reason this project is not "pure vector search".

A recruiter asking "senior Python engineers in Berlin with 5+ years" wants:
  * `total_years_experience >= 5`   -> a numeric comparison (exact, indexed)
  * `location ILIKE '%berlin%'`     -> a string match (exact, indexed)
  * `skills && ['python']`         -> an array intersection (exact, GIN indexed)

Vector similarity cannot answer any of those reliably: embeddings are lossy, so
a 6-year candidate can still rank above a 7-year one, and "Berlin" as a concept
will match "Munich" if the embedding is fuzzy. So filters run FIRST and cut the
corpus down, and only then does similarity rank what is left.

THE ALIAS
---------
Generated SQL refers to the candidates table by an alias. Raw-SQL callers pass
the default alias `"c"` and write `FROM candidates c`. ORM callers
(`select(Candidate)`) pass `alias=None`, which emits bare column names so the
fragment composes with an existing ORM select instead of inventing a second FROM.

SECURITY NOTE
-------------
`build_filters` returns a SQL fragment plus a parameter dict. User values are
ONLY ever passed as bound parameters (`:_f_min_exp_1`), never interpolated into
SQL. That makes this class SQL-injection-proof by construction, which matters
the moment you add auth (Phase 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from app.services.extraction.skills import normalize_skill

#: Alias used when the caller supplies its own FROM clause.
CANDIDATE_ALIAS = "c"


@dataclass
class SearchFilters:
    """Everything the API layer accepts as structured filters."""

    skills: list[str] = field(default_factory=list)
    # Match ALL listed skills (AND) or ANY (OR). AND is the recruiter default:
    # "Python AND Kubernetes" should not return a Python-only resume.
    skills_mode: str = "all"  # all | any
    location: str | None = None
    min_years_experience: float | None = None
    max_years_experience: float | None = None
    current_title: str | None = None
    company: str | None = None
    seniority: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    education_level: list[str] = field(default_factory=list)
    candidate_ids: list[str] | None = None
    has_email: bool | None = None
    has_phone: bool | None = None

    def is_empty(self) -> bool:
        return not any(
            (
                self.skills,
                self.location,
                self.min_years_experience is not None,
                self.max_years_experience is not None,
                self.current_title,
                self.company,
                self.seniority,
                self.languages,
                self.education_level,
                self.candidate_ids is not None,
                self.has_email is not None,
                self.has_phone is not None,
            )
        )

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot for the query log / API echo."""
        return {
            "skills": self.skills,
            "skills_mode": self.skills_mode,
            "location": self.location,
            "min_years_experience": self.min_years_experience,
            "max_years_experience": self.max_years_experience,
            "current_title": self.current_title,
            "company": self.company,
            "seniority": self.seniority,
            "languages": self.languages,
            "education_level": self.education_level,
            "candidate_ids": self.candidate_ids,
        }


class FilterSqlBuilder:
    """
    Compiles `SearchFilters` into a SQL fragment + bound params.

    Design rule: one responsibility per filter, one bound parameter per value,
    and no string concatenation of user input into SQL.
    """

    def __init__(self, filters: SearchFilters, alias: str | None = CANDIDATE_ALIAS) -> None:
        self.filters = filters
        self.alias = alias
        self.clauses: list[str] = []
        self.params: dict[str, object] = {}
        self._counter = 0

    def col(self, column: str) -> str:
        """`c.location` when aliased, plain `location` when composing with ORM."""
        return f"{self.alias}.{column}" if self.alias else column

    def _register(self, prefix: str, value) -> str:
        """Bind a parameter and return its generated name."""
        self._counter += 1
        name = f"_f_{prefix}_{self._counter}"
        self.params[name] = value
        return name

    def _add(self, clause: str) -> None:
        self.clauses.append(clause)

    def build(self) -> tuple[str | None, dict[str, object]]:
        """Return (sql_fragment_or_None, params)."""
        self._add_skills()
        self._add_location()
        self._add_experience()
        self._add_text_match()
        self._add_seniority()
        self._add_languages()
        self._add_education()
        self._add_id_set()
        self._add_presence()

        if not self.clauses:
            return None, {}
        return " AND ".join(self.clauses), self.params

    # -- individual filters -------------------------------------------------

    def _add_skills(self) -> None:
        """
        AND-mode emits one EXISTS per skill; OR-mode emits a single EXISTS with
        `= ANY(array)`. Both hit the (candidate_id, skill) unique index.
        """
        f = self.filters
        canonical = [
            value
            for value in (normalize_skill(raw) or (raw or "").strip().lower() for raw in f.skills)
            if value
        ]
        if not canonical:
            return

        candidate_id = self.col("id")
        if f.skills_mode == "any":
            name = self._register("skill_any", canonical)
            self._add(
                "EXISTS (SELECT 1 FROM candidate_skills cs "
                f"WHERE cs.candidate_id = {candidate_id} AND cs.skill = ANY(:{name}))"
            )
            return

        for skill in canonical:
            name = self._register("skill_all", skill)
            self._add(
                "EXISTS (SELECT 1 FROM candidate_skills cs "
                f"WHERE cs.candidate_id = {candidate_id} AND cs.skill = :{name})"
            )

    def _add_location(self) -> None:
        if not self.filters.location:
            return
        location = self.col("location")
        name = self._register("loc", f"%{self.filters.location.strip().lower()}%")
        # Whole-field match OR city-only match, so "Berlin" hits "Berlin, Germany".
        self._add(
            f"(lower(coalesce({location},'')) LIKE :{name} "
            f"OR lower(split_part(coalesce({location},''), ',', 1)) LIKE :{name})"
        )

    def _add_experience(self) -> None:
        column = self.col("total_years_experience")
        f = self.filters
        if f.min_years_experience is not None:
            name = self._register("min_exp", float(f.min_years_experience))
            self._add(f"{column} IS NOT NULL AND {column} >= :{name}")
        if f.max_years_experience is not None:
            name = self._register("max_exp", float(f.max_years_experience))
            self._add(f"{column} IS NOT NULL AND {column} <= :{name}")

    def _add_text_match(self) -> None:
        f = self.filters
        if f.current_title:
            column = self.col("current_title")
            name = self._register("title", f"%{f.current_title.strip().lower()}%")
            self._add(f"lower(coalesce({column},'')) LIKE :{name}")
        if f.company:
            column = self.col("current_company")
            name = self._register("comp", f"%{f.company.strip().lower()}%")
            self._add(f"lower(coalesce({column},'')) LIKE :{name}")

    def _add_seniority(self) -> None:
        if self.filters.seniority:
            name = self._register("seniority", [s.title() for s in self.filters.seniority])
            self._add(f"{self.col('seniority')} = ANY(:{name})")

    def _add_languages(self) -> None:
        if not self.filters.languages:
            return
        column = self.col("languages")
        name = self._register(
            "lang", [lang.strip().lower() for lang in self.filters.languages]
        )
        # unnest() so we compare element by element instead of substring-matching
        # the array's text form (which produced false positives like "en" in "den").
        self._add(
            f"EXISTS (SELECT 1 FROM unnest(coalesce({column}, ARRAY[]::varchar[])) AS lang "
            f"WHERE lower(lang) = ANY(:{name}))"
        )

    def _add_education(self) -> None:
        if not self.filters.education_level:
            return
        column = self.col("education")
        name = self._register("edu", [e.upper() for e in self.filters.education_level])
        self._add(f"upper(coalesce({column}->>'level','')) = ANY(:{name})")

    def _add_id_set(self) -> None:
        if self.filters.candidate_ids is None:
            return
        name = self._register("ids", [str(i) for i in self.filters.candidate_ids])
        self._add(f"({self.col('id')}::text = ANY(:{name}))")

    def _add_presence(self) -> None:
        f = self.filters
        email = self.col("email")
        phone = self.col("phone")
        if f.has_email is True:
            self._add(f"{email} IS NOT NULL AND {email} <> ''")
        elif f.has_email is False:
            self._add(f"({email} IS NULL OR {email} = '')")
        if f.has_phone is True:
            self._add(f"{phone} IS NOT NULL AND {phone} <> ''")
        elif f.has_phone is False:
            self._add(f"({phone} IS NULL OR {phone} = '')")


def build_filters(
    filters: SearchFilters | None, *, alias: str | None = CANDIDATE_ALIAS
) -> tuple[str | None, dict[str, object]]:
    """
    Compile filters into a SQL fragment.

    Args:
        alias: `"c"` (default) for raw SQL that supplies `FROM candidates c`;
            pass `None` when composing with `select(Candidate)`.

    Usage with raw SQL:
        sql, params = build_filters(filters)
        SELECT c.id::text FROM candidates c WHERE {sql}
    """
    if filters is None or filters.is_empty():
        return None, {}
    return FilterSqlBuilder(filters, alias=alias).build()


async def resolve_filtered_candidate_ids(
    session,
    filters: SearchFilters,
    *,
    limit: int = 20_000,
) -> tuple[list[str], str | None, dict[str, object]]:
    """
    Run the structured filters first and return the matching candidate ids.

    Chunk-level searches receive this id list and add
    `chunk.candidate_id IN (...)`, which is how the structured layer constrains
    the semantic layer.

    Returns (ids, sql_fragment, params) so callers can reuse both.
    """
    from sqlalchemy import text as sql_text

    sql, params = build_filters(filters)
    if not sql:
        return [], None, {}

    stmt = sql_text(f"SELECT c.id::text FROM candidates c WHERE {sql} LIMIT :_f_limit")
    rows = (await session.execute(stmt, {"_f_limit": limit, **params})).scalars().all()
    return [str(r) for r in rows], sql, params


SKILL_FACET_SQL = """
    SELECT cs.skill,
           min(cs.display)                  AS display,
           count(DISTINCT cs.candidate_id)  AS candidate_count,
           bool_or(cs.is_core)              AS is_core
    FROM candidate_skills cs
    JOIN candidates c ON c.id = cs.candidate_id
    GROUP BY cs.skill
    ORDER BY candidate_count DESC, display ASC
    LIMIT :_facet_limit
"""