"""
MCP server tests.

Two groups:

* **Surface tests** (no database) - assert the server declares the primitives we
  intend, with descriptions and output schemas a model can actually use. These
  are the contracts; if a decorator changes, they fail loudly.
* **Behaviour tests** (database) - drive the real handlers through
  `call_tool` / `read_resource` against the dedicated test database.

The handlers are invoked in-process rather than over a transport, so the tests
exercise SQL and retrieval without a client, a socket, or a subprocess.
"""

from __future__ import annotations

import json

import pytest

from app.mcp.schemas import (
    IngestResult,
    MatchResult,
    SearchResult,
    SkillResult,
)

# ---------------------------------------------------------------------------
# Surface: no database required
# ---------------------------------------------------------------------------


@pytest.fixture
def server():
    from app.mcp.server import build_server

    return build_server()


def _by_name(items, name):
    """Find an item by `.name`. Plain sync - callers already awaited the list."""
    return next(i for i in items if i.name == name)


class TestDeclaredSurface:
    """The contract the model sees. Descriptive enough to act on."""

    async def test_exactly_the_intended_tools(self, server):
        names = {t.name for t in await server.list_tools()}
        assert names == {
            "search_candidates",
            "match_job_description",
            "list_available_skills",
            "ingest_resume",
        }

    async def test_every_tool_declares_an_output_schema(self, server):
        """
        A tool returning bare `dict` gets NO outputSchema, which means the
        client gets prose it must parse. The TypedDict return annotations are
        what generate these - if one is removed this test fails.
        """
        for tool in await server.list_tools():
            assert tool.output_schema, f"{tool.name} has no outputSchema"
            assert tool.output_schema.get("properties"), (
                f"{tool.name} has an empty outputSchema"
            )

    async def test_read_only_tools_are_annotated(self, server):
        tools = {t.name: t for t in await server.list_tools()}
        for name in ("search_candidates", "match_job_description", "list_available_skills"):
            ann = tools[name].annotations
            assert ann is not None, name
            assert ann.read_only_hint is True, name

    async def test_write_tool_is_not_marked_read_only(self, server):
        """
        The client uses this to decide whether to prompt for confirmation.
        `ingest_resume` creates rows, so it must not claim to be read-only.
        """
        ann = _by_name(await server.list_tools(), "ingest_resume").annotations
        assert ann.read_only_hint is not True

    async def test_tool_descriptions_tell_the_model_when_NOT_to_use_them(
        self, server
    ):
        """
        The description is the only thing the model reads before choosing. A tool
        that says what it is for but not what it is NOT for gets misused.
        """
        tools = {t.name: t for t in await server.list_tools()}
        assert "do not use" in tools["search_candidates"].description.lower()
        assert "do not use" in tools["match_job_description"].description.lower()
        # The alias behaviour is documented, because silently returning nothing
        # for 'k8s' is the most likely confusion.
        assert "alias" in tools["search_candidates"].description.lower()

    async def test_required_arguments_are_declared(self, server):
        tools = {t.name: t for t in await server.list_tools()}
        assert tools["match_job_description"].input_schema["required"] == [
            "job_description"
        ]
        assert tools["ingest_resume"].input_schema["required"] == ["resume_text"]

    async def test_search_arguments_are_all_optional(self, server):
        """
        A filter-only search is a legitimate call, so `query` must not be
        required. The handler rejects a wholly empty request instead.
        """
        schema = _by_name(await server.list_tools(), "search_candidates").input_schema
        assert not schema.get("required")

    async def test_resources_and_templates(self, server):
        uris = {str(r.uri) for r in await server.list_resources()}
        assert uris == {
            "resume://candidates",
            "talent://facets/skills",
            "talent://corpus/stats",
        }
        templates = {t.uri_template for t in await server.list_resource_templates()}
        assert "resume://candidates/{candidate_id}" in templates

    async def test_prompt_is_registered_with_arguments(self, server):
        prompt = _by_name(await server.list_prompts(), "screen_against_jd")
        args = {a.name for a in (prompt.arguments or [])}
        assert {"job_description", "max_candidates"} <= args

    async def test_server_instructions_state_the_rules(self, server):
        """
        Server-level instructions are injected into the model's context. The two
        that matter: only claim what evidence shows, and check corpus_stats when
        nothing matches.
        """
        instructions = server.instructions
        assert "evidence" in instructions.lower()
        assert "corpus_stats" in instructions
        assert "ingest_resume" in instructions


class TestRejectionPaths:
    """
    Handlers must return a usable error, not raise.

    A tool that throws surfaces as `UnexpectedToolError` and burns a turn. One
    that returns `{"error": ...}` lets the model branch and tell the user.
    """

    async def test_empty_search_is_refused_with_guidance(self, server):
        result = await server.call_tool("search_candidates", {})
        payload = json.loads(result.content[0].text)
        assert payload["total"] == 0
        assert "query" in payload["error"]
        assert "filter" in payload["error"]

    async def test_short_jd_is_refused(self, server):
        result = await server.call_tool("match_job_description", {"job_description": "hire"})
        payload = json.loads(result.content[0].text)
        assert "error" in payload
        assert "40 characters" in payload["error"]

    async def test_short_resume_is_refused(self, server):
        result = await server.call_tool("ingest_resume", {"resume_text": "too short"})
        payload = json.loads(result.content[0].text)
        assert "80 characters" in payload["error"]

    async def test_unknown_candidate_id_returns_a_hint(self, server):
        async for contents in _read(
            server, "resume://candidates/00000000-0000-0000-0000-000000000000"
        ):
            payload = json.loads(contents.content)
            assert "error" in payload
            # The hint tells the model how to recover, which is the whole point.
            assert "search_candidates" in payload["hint"]


class TestOutputSchemas:
    """The generated schemas must actually describe the payload."""

    @staticmethod
    def _resolve(schema: dict, node: dict) -> dict:
        """
        Follow a local `$ref` so nested shapes can be inspected.

        The SDK emits `{"$ref": "#/$defs/SkillRow"}` for a nested TypedDict
        rather than inlining it. That is legal JSON Schema, but it means any
        consumer has to resolve refs - which is exactly why the spec requires
        implementations to handle them.
        """
        ref = node.get("$ref")
        if not ref:
            return node
        if not ref.startswith("#/$defs/"):
            raise AssertionError(f"unexpected external $ref: {ref}")
        return schema.get("$defs", {}).get(ref.removeprefix("#/$defs/"), {})

    async def test_search_result_schema_shape(self, server):
        tool = _by_name(await server.list_tools(), "search_candidates")
        schema = tool.output_schema
        assert {"total", "results", "diagnostics"} <= set(schema["properties"])

        # Walk into results -> items -> (resolved) properties.
        items = schema["properties"]["results"]["items"]
        row_props = self._resolve(schema, items)["properties"]
        assert "evidence" in row_props, "evidence makes every claim checkable"

    async def test_error_key_is_declared_on_every_tool(self, server):
        """
        Every handler can return `error` instead of its normal payload. If the
        schema omits it, a client validating strictly will reject the failure -
        which is worse than not having validation.
        """
        for tool in await server.list_tools():
            assert "error" in tool.output_schema["properties"], tool.name

    async def test_output_schemas_use_local_refs_only(self, server):
        """
        A `$ref` pointing at a network URI must not appear: auto-dereferencing
        would turn schema validation into an SSRF vector.
        """
        for tool in await server.list_tools():
            blob = json.dumps(tool.output_schema)
            assert "#/$defs/" in blob or "properties" in blob
            assert "http://" not in blob.replace("http://json-schema.org", "")
            assert "https://" not in blob.replace("https://json-schema.org", "")

    async def test_typed_dicts_are_not_empty(self):
        """Guards the `-> dict[str, Any]` trap, which yields an empty schema."""
        for schema_type in (SearchResult, MatchResult, SkillResult, IngestResult):
            assert schema_type.__total__ is False, schema_type.__name__
            assert schema_type.__annotations__, schema_type.__name__


class TestPromptRendering:
    async def test_prompt_includes_the_job_description_and_arguments(self, server):
        result = await server.get_prompt(
            "screen_against_jd",
            {"job_description": "Senior Python engineer", "max_candidates": "3"},
        )
        text = result.messages[0].content.text
        assert "Senior Python engineer" in text
        # The argument is threaded into the workflow, not ignored.
        assert "3" in text
        # The gap section is the most useful part for a recruiter - do not drop it.
        assert "GAPS" in text.upper()

    async def test_prompt_tells_the_model_to_avoid_a_second_llm_call(self, server):
        result = await server.get_prompt("screen_against_jd", {"job_description": "x" * 50})
        text = result.messages[0].content.text
        assert "generate_answer=false" in text


# ---------------------------------------------------------------------------
# Behaviour: needs the database
# ---------------------------------------------------------------------------
pytestmark_db = pytest.mark.integration

RESUME_A = """\
Dana Whitfield
Cape Town, South Africa
dana.whitfield@example.com | +27 21 555 0199

SUMMARY
Site reliability engineer with 11 years automating infrastructure for
high-traffic fintech platforms across Africa and Europe.

EXPERIENCE
Staff SRE | Paystack | 2019 - Present
Designed a GitOps platform using ArgoCD and Flux across 12 Kubernetes clusters.
Introduced Terraform modules standardised across 200 cloud resources.
Cut mean time to recovery from 47 minutes to 6 minutes.

EDUCATION
B.Eng. Software Engineering | University of Cape Town | 2014

SKILLS
Cloud: AWS, GCP, Terraform, Kubernetes
Tooling: Prometheus, Grafana, ArgoCD, Ansible, Datadog
Languages: Go, Python, Bash
"""

RESUME_B = """\
Marcus Webb
Austin, TX, USA
marcus.webb@example.com | (512) 555-9876

PROFILE
Frontend developer focused on React and design systems.

EXPERIENCE
Frontend Developer | Atlassian | 2022 - Present
Rebuilt the component library in React and TypeScript, adopted by 9 teams.
Added end-to-end tests with Playwright.

EDUCATION
B.Sc. Computer Science | University of Texas at Austin | 2020

SKILLS
Languages: JavaScript, TypeScript, HTML, CSS
Frameworks: React, Next.js, Vue
"""

#: A third resume, used by the dedupe test so the first ingest is not a
#: duplicate of something the fixture already created.
RESUME_C = RESUME_A.replace("Dana Whitfield", "Jordan Blake")


@pytest.fixture
async def mcp_server(session, monkeypatch):
    """
    An MCPServer wired to the TEST database.

    `app.mcp.tools` holds a module-level `database` reference for the dev
    database. Rather than standing up a second engine (which fights the
    connection pool and produces MissingGreenlet on teardown), we pin the MCP
    layer to the very session the conftest `session` fixture already manages -
    same test database, same truncate-before/rollback-after isolation.
    """
    from app.mcp import context as mcp_context
    from app.mcp import tools as mcp_tools
    from app.mcp.server import build_server

    pinned = mcp_context.Database()
    pinned.force_session(session)
    monkeypatch.setattr(mcp_context, "database", pinned)
    monkeypatch.setattr(mcp_tools, "database", pinned)

    server = build_server()

    # Seed through the real tool, so the tests also cover ingest.
    await server.call_tool("ingest_resume", {"resume_text": RESUME_A})
    await server.call_tool("ingest_resume", {"resume_text": RESUME_B})

    return server


async def _read(server, uri):
    """
    Read a resource and yield its contents.

    `read_resource` is awaited (it is a coroutine) and may return either a
    sync iterable or an async generator depending on SDK version.
    """
    result = await server.read_resource(uri)
    if hasattr(result, "__aiter__"):
        async for item in result:
            yield item
    else:
        for item in result:
            yield item


def _payload(result) -> dict:
    return json.loads(result.content[0].text)


@pytestmark_db
class TestToolsAgainstTheDatabase:
    async def test_list_available_skills_returns_real_facets(self, mcp_server):
        payload = _payload(await mcp_server.call_tool("list_available_skills", {}))
        canonical = {s["canonical"] for s in payload["skills"]}
        assert {"kubernetes", "terraform", "python", "react"} <= canonical
        # Counts must reflect the two seeded resumes.
        kubernetes = next(
            s for s in payload["skills"] if s["canonical"] == "kubernetes"
        )
        assert kubernetes["candidate_count"] == 1

    async def test_semantic_search_returns_evidence(self, mcp_server):
        payload = _payload(
            await mcp_server.call_tool(
                "search_candidates",
                {"query": "reduced incident recovery time with observability", "top_k": 3},
            )
        )
        assert payload["total"] >= 1
        top = payload["results"][0]
        # The semantic leg must find Dana even though none of the query words
        # appear in the resume.
        assert top["full_name"] == "Dana Whitfield"
        # Evidence is what stops the model inventing a reason.
        assert top["evidence"], "every result must carry supporting chunks"
        assert top["evidence"][0]["content"]

    async def test_structured_filters_are_exact(self, mcp_server):
        payload = _payload(
            await mcp_server.call_tool(
                "search_candidates",
                {"skills": ["kubernetes", "terraform"], "min_years_experience": 5},
            )
        )
        # Dana has both skills and ~6 computed years; Marcus has neither.
        # A pure-vector search would happily return Marcus on "engineer", which
        # is exactly what this test exists to prevent.
        assert [r["full_name"] for r in payload["results"]] == ["Dana Whitfield"]

    async def test_skill_aliases_resolve(self, mcp_server):
        """'k8s' must behave identically to 'kubernetes'."""
        alias = _payload(
            await mcp_server.call_tool("search_candidates", {"skills": ["k8s"]})
        )
        canonical = _payload(
            await mcp_server.call_tool("search_candidates", {"skills": ["kubernetes"]})
        )
        assert [r["id"] for r in alias["results"]] == [
            r["id"] for r in canonical["results"]
        ]

    async def test_top_k_is_clamped(self, mcp_server):
        """Model-supplied bounds are not trusted."""
        payload = _payload(
            await mcp_server.call_tool(
                "search_candidates", {"query": "engineer", "top_k": 10_000}
            )
        )
        assert len(payload["results"]) <= 50

    async def test_years_are_clamped_to_a_sane_range(self, mcp_server):
        payload = _payload(
            await mcp_server.call_tool(
                "search_candidates",
                {"query": "engineer", "min_years_experience": 9_999},
            )
        )
        assert payload["total"] == 0, "clamped to 60, nobody has that"

    async def test_match_job_description_ranks_relevant_first(self, mcp_server):
        payload = _payload(
            await mcp_server.call_tool(
                "match_job_description",
                {
                    "job_description": (
                        "Site reliability engineer wanted. You will run "
                        "Kubernetes and Terraform in production and cut "
                        "incident recovery time."
                    )
                },
            )
        )
        assert payload["total"] >= 1
        assert payload["results"][0]["full_name"] == "Dana Whitfield"
        # Without an LLM we must not imply this is a fit judgement.
        assert payload["scoring"] == "retrieval_only"
        assert "shortlist" in payload["note"].lower()

    async def test_ingest_is_idempotent_by_content_hash(self, mcp_server):
        # A resume NOT already seeded by the fixture, so the first call is fresh.
        fresh = RESUME_C.replace("Priya", "Jordan")
        first = _payload(await mcp_server.call_tool("ingest_resume", {"resume_text": fresh}))
        second = _payload(await mcp_server.call_tool("ingest_resume", {"resume_text": fresh}))
        assert first["created"] is True
        assert second["created"] is False
        assert second["deduplicated"] is True
        assert first["candidate_id"] == second["candidate_id"]

    async def test_ingest_does_not_leave_a_file_on_disk(self, mcp_server):
        """Pasted text is written to disk only to reuse the pipeline; then removed."""
        from app.core.config import settings

        before = set(settings.upload_dir.glob("mcp_*.txt"))
        await mcp_server.call_tool("ingest_resume", {"resume_text": RESUME_A})
        assert set(settings.upload_dir.glob("mcp_*.txt")) == before

    async def test_ingest_reports_extraction_gaps(self, mcp_server):
        """A resume with no skills must say so rather than looking fine."""
        sparse = (
            "Someone Anonymous\n\nSUMMARY\n" + "Some words about work. " * 10
        )
        payload = _payload(await mcp_server.call_tool("ingest_resume", {"resume_text": sparse}))
        assert payload["warnings"], "expected warnings for a resume with no skills"


@pytestmark_db
class TestResourcesAgainstTheDatabase:
    async def test_corpus_stats(self, mcp_server):
        async for contents in _read(mcp_server, "talent://corpus/stats"):
            data = json.loads(contents.content)
            assert data["candidates"] == 2
            assert data["chunks"] > 0
            # The whole point of this resource: distinguish causes of emptiness.
            assert "chunks_without_vector" in data
            assert "llm_enabled" in data
            assert "embedding_provider" in data

    async def test_skill_facets(self, mcp_server):
        async for contents in _read(mcp_server, "talent://facets/skills"):
            data = json.loads(contents.content)
            canonical = {s["canonical"] for s in data["skills"]}
            assert "kubernetes" in canonical

    async def test_candidate_directory_is_ordered_by_experience(self, mcp_server):
        async for contents in _read(mcp_server, "resume://candidates"):
            data = json.loads(contents.content)
            years = [c["total_years_experience"] for c in data["candidates"]]
            assert years == sorted(years, reverse=True)

    async def test_candidate_profile_includes_chunks(self, mcp_server):
        async for contents in _read(mcp_server, "resume://candidates"):
            directory = json.loads(contents.content)

        target = next(
            c for c in directory["candidates"] if c["full_name"] == "Dana Whitfield"
        )
        async for contents in _read(
            mcp_server, f"resume://candidates/{target['id']}"
        ):
            profile = json.loads(contents.content)
            assert profile["full_name"] == "Dana Whitfield"
            assert profile["chunk_count"] > 0
            assert profile["skills"]
            assert "raw_text" not in profile, "do not dump the whole resume"