"""
MCP prompts: the user-controlled surface.

A prompt is a slash command. The USER decides when it runs - the model cannot
invoke one on its own, which is exactly what you want for a workflow that costs
money, takes minutes, or writes something.

Why put workflow logic here rather than in a system prompt the user types?

    versioned with the code      - a fix ships to every user, not to whoever
                                   remembered to update their prompt
    reviewable                   - it is a reviewable artefact in the repo
    parameterised                - arguments are declared and validated
    composable                   - the host can enumerate them

Each prompt returns fully-rendered messages, so the model receives a complete,
correctly ordered instruction set rather than a fragment.
"""

from __future__ import annotations

from typing import Annotated


def register_prompts(server) -> None:
    """Attach every user-invocable workflow."""

    # -----------------------------------------------------------------------
    @server.prompt(
        name="screen_against_jd",
        title="Screen candidates against a job description",
        description=(
            "Full screening workflow: extract the requirements, search the pool, "
            "score each candidate, and produce a shortlist with evidence and an "
            "explicit list of gaps. Use when a user supplies a role spec."
        ),
    )
    async def screen_against_jd(
        job_description: Annotated[str, "The job description to screen against."],
        max_candidates: Annotated[str, "How many candidates to shortlist."] = "5",
    ) -> str:
        return f"""You are screening candidates for a role. Work strictly from the
candidate pool; never invent experience, contact details or dates.

JOB DESCRIPTION
---
{job_description}
---

METHOD (follow in order)

1. REQUIREMENTS
   List the must-have requirements separately from the nice-to-haves. Keep each
   one to a single phrase.

2. SEARCH
   Call `list_available_skills` once to learn the valid skill names, then call
   `search_candidates` with those skills plus `min_years_experience` if the role
   states a seniority floor. Fetch at most {max_candidates} candidates.

3. EVALUATE
   For each candidate, read the `evidence` excerpts. A candidate matches only if
   the excerpts show it. Quote the specific excerpt for every claim you make.

4. SHORTLIST
   Return the top {max_candidates} as a table with: name, current role, years,
   location, why they match (with the quoted excerpt), and what is missing.

5. GAPS
   List requirements that NO candidate in the pool satisfies. This is the most
   useful section for the recruiter - do not skip it, and do not soften it.

RULES

- If the pool has nobody suitable, say so plainly rather than nominating the
  closest poor match.
- Quote resume text for every claim. Never paraphrase into a strength the
  excerpt does not support.
- Use `generate_answer=false` on search; you are writing the summary yourself
  from the evidence, which avoids paying for a second LLM call."""


__all__ = ["register_prompts"]