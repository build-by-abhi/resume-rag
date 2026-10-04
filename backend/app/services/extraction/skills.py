"""
Skill taxonomy + normalisation.

WHY NOT JUST SPLIT THE SKILLS SECTION ON COMMAS?
------------------------------------------------
Resumes are inconsistent: "React.js", "ReactJS", "react js", "React" are the
same skill, and a comma split produces 4 database rows and 4 different filters.
The `candidates_skills` table is only useful if the same skill collapses to one
canonical token.

So: a curated alias map turns every surface form into a canonical key
("react"), and everything downstream (exact filters, facets, "who knows
Kafka?") uses the canonical key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Skill:
    canonical: str  # storage key, e.g. "postgresql"
    display: str  # human label, e.g. "PostgreSQL"
    category: str  # language | framework | database | cloud | ml | tooling | ...
    aliases: tuple[str, ...] = ()


def _s(canonical: str, display: str, category: str, *aliases: str) -> Skill:
    return Skill(canonical, display, category, tuple(aliases))


# ---------------------------------------------------------------------------
# The taxonomy. Extend this freely -- it is data, not logic.
# Keep it domain-focused; a 2000-entry list mostly adds noise.
# ---------------------------------------------------------------------------
SKILL_TAXONOMY: tuple[Skill, ...] = (
    # Languages
    _s("python", "Python", "language", "python3", "py"),
    _s("java", "Java", "language"),
    _s("javascript", "JavaScript", "language", "js", "ecmascript"),
    _s("typescript", "TypeScript", "language", "ts"),
    _s("csharp", "C#", "language", "c#", "c sharp", "dotnet", ".net", "asp.net"),
    _s("cpp", "C++", "language", "c++"),
    _s("go", "Go", "language", "golang"),
    _s("rust", "Rust", "language"),
    _s("scala", "Scala", "language"),
    _s("kotlin", "Kotlin", "language"),
    _s("swift", "Swift", "language"),
    _s("r", "R", "language"),
    _s("php", "PHP", "language"),
    _s("sql", "SQL", "language"),

    # Frontend
    _s("react", "React", "framework", "react.js", "reactjs", "react js"),
    _s("nextjs", "Next.js", "framework", "next.js", "nextjs"),
    _s("vue", "Vue", "framework", "vue.js", "vuejs", "vue.js"),
    _s("angular", "Angular", "framework", "angularjs"),
    _s("svelte", "Svelte", "framework"),
    _s("html", "HTML", "language", "html5"),
    _s("css", "CSS", "language", "css3", "scss", "sass", "less"),
    _s("tailwind", "Tailwind CSS", "framework", "tailwindcss"),
    _s("redux", "Redux", "framework"),

    # Backend
    _s("fastapi", "FastAPI", "framework"),
    _s("django", "Django", "framework"),
    _s("flask", "Flask", "framework"),
    _s("spring", "Spring Boot", "framework", "spring boot", "springboot"),
    _s("nodejs", "Node.js", "runtime", "node", "node.js"),
    _s("express", "Express.js", "framework", "express.js", "expressjs"),
    _s("nestjs", "NestJS", "framework", "nest.js"),
    _s("graphql", "GraphQL", "framework"),

    # Data / ML
    _s("machine-learning", "Machine Learning", "ml", "ml", "machine learning"),
    _s("deep-learning", "Deep Learning", "ml", "deep learning"),
    _s("nlp", "NLP", "ml", "natural language processing", "nlp"),
    _s("pytorch", "PyTorch", "ml", "torch"),
    _s("tensorflow", "TensorFlow", "ml"),
    _s("scikit-learn", "scikit-learn", "ml", "sklearn", "scikit learn", "sci-kit learn"),
    _s("pandas", "pandas", "data"),
    _s("numpy", "NumPy", "data"),
    _s("spark", "Apache Spark", "data", "pyspark", "apache spark"),
    _s("airflow", "Airflow", "data", "apache airflow"),
    _s("dbt", "dbt", "data"),
    _s("llm", "LLMs", "ml", "llms", "large language model", "large language models"),
    _s("rag", "RAG", "ml", "retrieval augmented generation", "retrieval-augmented generation"),
    _s("mlops", "MLOps", "ml"),
    _s("langchain", "LangChain", "ml"),
    _s("huggingface", "Hugging Face", "ml", "hugging face"),

    # Databases
    _s("postgresql", "PostgreSQL", "database", "postgres", "psql", "postgresql"),
    _s("mysql", "MySQL", "database"),
    _s("sqlite", "SQLite", "database"),
    _s("mongodb", "MongoDB", "database", "mongo"),
    _s("redis", "Redis", "database"),
    _s("elasticsearch", "Elasticsearch", "database", "elastic search", "opensearch"),
    _s("cassandra", "Cassandra", "database"),
    _s("oracle", "Oracle DB", "database"),
    _s("sqlserver", "SQL Server", "database", "mssql", "sql server"),
    _s("snowflake", "Snowflake", "database"),

    # Vector search / infra
    _s("pgvector", "pgvector", "database"),
    _s("pinecone", "Pinecone", "vectordb"),
    _s("chroma", "ChromaDB", "vectordb", "chroma db", "chromadb"),
    _s("faiss", "FAISS", "vectordb"),
    _s("weaviate", "Weaviate", "vectordb"),
    _s("qdrant", "Qdrant", "vectordb"),
    _s("milvus", "Milvus", "vectordb"),
    _s("docker", "Docker", "devops"),
    _s("kubernetes", "Kubernetes", "devops", "k8s"),
    _s("terraform", "Terraform", "devops"),
    _s("aws", "AWS", "cloud", "amazon web services"),
    _s("gcp", "GCP", "cloud", "google cloud"),
    _s("azure", "Azure", "cloud", "microsoft azure"),
    _s("linux", "Linux", "devops"),
    _s("ci-cd", "CI/CD", "devops", "ci cd", "cicd", "github actions", "jenkins", "gitlab ci"),
    _s("git", "Git", "tooling"),
    _s("kafka", "Kafka", "messaging", "apache kafka"),
    _s("rabbitmq", "RabbitMQ", "messaging"),
    _s("celery", "Celery", "messaging"),

    # Practices / tooling
    _s("agile", "Agile", "practice", "agile methodology", "scrum", "kanban"),
    _s("tdd", "TDD", "practice", "test driven development", "test-driven development"),
    _s("pytest", "pytest", "testing"),
    _s("jest", "Jest", "testing"),
    _s("selenium", "Selenium", "testing"),
    _s("playwright", "Playwright", "testing"),
    _s("figma", "Figma", "tooling"),
    _s("jira", "Jira", "tooling"),
    _s("microservices", "Microservices", "practice"),
    _s("system-design", "System Design", "practice", "system design"),
    _s("data-modeling", "Data Modeling", "practice", "data modelling"),
)

# alias -> Skill, built once at import.
_ALIAS_INDEX: dict[str, Skill] = {}
for _skill in SKILL_TAXONOMY:
    for _alias in (_skill.canonical, _skill.display, *_skill.aliases):
        _ALIAS_INDEX[_alias.lower().strip()] = _skill

# Aliases must be matched on word boundaries, and "." / "#" / "+" inside them
# break plain \b. We escape and add explicit lookarounds.
def _alias_pattern(alias: str) -> re.Pattern[str]:
    escaped = re.escape(alias.lower().strip())
    # \b fails next to . # + / so use non-word lookarounds instead.
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])", re.IGNORECASE)


_ALIAS_PATTERNS: list[tuple[Skill, re.Pattern[str]]] = [
    # Longest alias first so "react native" wins over "react".
    # Deduplicate case-insensitively: without this, a skill whose display name
    # equals its canonical name ("python" / "Python") would compile TWO matching
    # patterns and every occurrence would be double counted.
    (skill, _alias_pattern(alias))
    for skill in SKILL_TAXONOMY
    for alias in sorted(
        {a.lower() for a in (skill.canonical, skill.display, *skill.aliases)},
        key=len,
        reverse=True,
    )
]

CATEGORY_LABELS = {
    "language": "Languages",
    "framework": "Frameworks",
    "database": "Databases",
    "vectordb": "Vector DBs",
    "ml": "AI / ML",
    "data": "Data",
    "cloud": "Cloud",
    "devops": "DevOps",
    "messaging": "Messaging",
    "tooling": "Tooling",
    "practice": "Practices",
    "testing": "Testing",
    "other": "Other",
}


def normalize_skill(raw: str) -> str | None:
    """Map any surface form to its canonical key, or None if unknown."""
    if not raw:
        return None
    key = raw.strip().lower()
    key = re.sub(r"\s+", " ", key).strip(" .-")
    skill = _ALIAS_INDEX.get(key)
    return skill.canonical if skill else None


def display_name(canonical: str) -> str:
    for skill in SKILL_TAXONOMY:
        if skill.canonical == canonical:
            return skill.display
    # Unknown skill: "ci-cd" -> "Ci Cd" would be wrong, so drop the hyphens first.
    return canonical.replace("-", " ").replace("_", " ").title()


def category_of(canonical: str) -> str:
    for skill in SKILL_TAXONOMY:
        if skill.canonical == canonical:
            return skill.category
    return "other"


def extract_skills(text: str) -> dict[str, int]:
    """
    Scan text for every known skill and count occurrences.

    Returns {canonical_skill: occurrences}. Counts matter: a skill listed in the
    skills section AND used in 3 bullets is far more relevant than one mention.
    """
    if not text:
        return {}

    counts: dict[str, int] = {}
    for skill, pattern in _ALIAS_PATTERNS:
        found = len(pattern.findall(text))
        if found:
            counts[skill.canonical] = counts.get(skill.canonical, 0) + found
    return counts


_CANONICAL_INDEX: dict[str, Skill] = {s.canonical: s for s in SKILL_TAXONOMY}


def expand_query_for_skills(query: str) -> str:
    """
    Rewrite a search query so the KEYWORD leg also knows the skill aliases.

    Why this matters
    ----------------
    An HR user types "postgres". Postgres' text search only sees the literal
    token `postgres`, but the resume says "PostgreSQL" - so BM25 returns
    nothing. `pg_trgm` fuzziness cannot bridge a 3-letter difference either.

    We use the taxonomy we already have at ingest time to expand each
    recognised skill token into an OR group of its aliases:

        "python postgres"  ->  'python' ("postgres" or "postgresql" or "psql")

    The output is valid `websearch_to_tsquery` syntax, which supports `or`, `and`
    and parentheses. Adjacent terms are ANDed, so this only ever *widens* each
    individual skill term - it never turns an AND into an OR.
    """
    if not query:
        return query

    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#.\-_/]*", query)
    if not tokens:
        return query

    parts: list[str] = []
    for token in tokens:
        canonical = normalize_skill(token)
        skill = _CANONICAL_INDEX.get(canonical) if canonical else None
        if skill is None:
            parts.append(token)
            continue

        variants = {
            token.lower(),
            skill.canonical.replace("-", " "),
            skill.display,
            *skill.aliases,
        }
        # A phrase must not contain quotes or the tsquery syntax breaks.
        cleaned = sorted(
            {v.strip().lower() for v in variants if v and v.strip() and '"' not in v}
        )
        # Cap the group so a skill with many aliases does not bloat the query.
        parts.append("(" + " or ".join(f'"{v}"' for v in cleaned[:6]) + ")")

    return " ".join(parts) or query


_CATEGORY_LABEL = re.compile(
    # Every label an HR resume is likely to put in front of a skill list.
    # Without this, "Data: PostgreSQL, Kafka" would store a skill literally
    # called "data: postgresql" and the taxonomy filter would miss it.
    #
    # The trailing `\b` is essential, not decorative. Without it "Datadog" was
    # split into the label "Data" plus the leftover "dog", inventing a skill
    # called "dog". It also protects "Cloudflare" (from "Cloud"),
    # "Toolset" (from "Tools") and every other word that merely *starts* with a
    # category name.
    r"^("
    r"languages?|programming\s+languages?|tools?|toolings?|tool\s*stack|"
    r"frameworks?|libraries?|databases?|db|data(?:\s+engineering|\s+platform)?|"
    r"cloud|cloud\s+platforms?|infra(?:structure)?|devops|dev\s*ops|"
    r"messaging|message\s+queue|testing|test\s+tools?|ai(?:\s*/\s*ml)?|"
    r"machine\s+learning|ml|analytics|methodolog(?:y|ies)|expertise|"
    r"proficienc(?:y|ies)|competencies|skills?|soft\s+skills?"
    r")\b\s*[:&\-]?\s*",
    re.IGNORECASE,
)

#: Fallback for category labels the list above does not name ("Tooling:",
#: "Environment:", "Nice to have:").
#:
#: The earlier version only knew a fixed vocabulary, so "Tooling: Prometheus,
#: Grafana" became the single bogus skill "ing: Prometheus, Grafana" - it
#: stripped the prefix "Tool" and kept "ing".
#:
#: The three constraints are what keep this from eating real skills:
#:   * letters, spaces, `&` and `-` only  -> "C++: templates" is left alone
#:   * the label is at least 3 characters   -> "a:b" is left alone
#:   * a space must follow the colon        -> a label is always followed by one
_GENERIC_LABEL = re.compile(r"^[A-Za-z][A-Za-z /&-]{2,24}:\s+")


def _strip_category_label(token: str) -> str:
    """Remove a leading category label such as ``Tooling:`` or ``Cloud -``."""
    original = token.strip(" \t-–—:.\n")
    stripped = _CATEGORY_LABEL.sub("", original)
    # If the vocabulary match left something starting with ':' or a stray
    # fragment ("ing: Prometheus"), fall back to the generic rule.
    if ":" in stripped:
        generic = _GENERIC_LABEL.match(stripped)
        if generic:
            stripped = stripped[generic.end() :].strip()
    stripped = stripped.strip()

    # Stripping must never consume the whole token. A standalone "Database" or
    # "Cloud" is a (harmless) word, not a label introducing anything, and
    # returning "" would silently drop it.
    return stripped or original


def parse_skill_list(raw: str) -> list[str]:
    """
    Extract skills from a free-form block such as
    "Languages: Python, JavaScript | Tools: Docker, k8s".

    Category labels ("Data:", "Cloud:", "Tooling:") are stripped so they never
    become skills. Newlines are handled here too, so callers may pass a whole
    skills block or a single line.
    """
    if not raw:
        return []

    # Split on commas/semicolons/pipes and on newlines, so a multi-line block
    # behaves the same as several single lines.
    parts = re.split(r"[,;|•·]|\t+|\n|\s{2,}", raw)
    out: list[str] = []
    for part in parts:
        token = _strip_category_label(part)
        if 1 < len(token) <= 40 and not re.match(r"^\d+$", token):
            out.append(token)
    return out