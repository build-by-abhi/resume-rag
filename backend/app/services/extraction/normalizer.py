"""
Merge rule-based and LLM extraction results into one `Extraction`.

MERGE POLICY (this is the important part)
-----------------------------------------
| Field                | Winner                                    |
|----------------------|-------------------------------------------|
| email, phone, links  | rules always (exact text, no hallucination)|
| full_name            | rules, else LLM                           |
| total_years          | whichever produced a value; LLM refines   |
| current_title/company| rules, else LLM                           |
| location             | rules, else LLM                           |
| summary              | LLM if longer/richer, else rules          |
| education            | merge field by field                      |
| skills               | UNION, summing occurrence counts          |

Why: the LLM is better at *interpretation* (summary, seniority, implied
skills) and worse at *transcription* (an email address, a phone number).
Trust each for what it is good at, and never let a hallucination overwrite a
value we actually read off the page.
"""

from __future__ import annotations

from app.core.logging import get_logger
from app.services.extraction.rule_extractor import Extraction
from app.services.extraction.skills import CATEGORY_LABELS, category_of, display_name

logger = get_logger(__name__)


def merge_extractions(
    rules: Extraction, llm: Extraction | None, *, llm_enabled: bool = False
) -> Extraction:
    """Combine two extractions using the policy documented above."""
    if llm is None or not llm_enabled:
        return rules

    merged = Extraction()

    # --- transcription-critical fields: rules win, unconditionally ---
    merged.full_name = rules.full_name or llm.full_name
    merged.email = rules.email or llm.email
    merged.phone = rules.phone or llm.phone
    merged.links = {**llm.links, **(rules.links or {})}  # rules override

    # --- interpreted fields: LLM fills gaps ---
    merged.location = rules.location or llm.location
    merged.current_title = rules.current_title or llm.current_title
    merged.current_company = rules.current_company or llm.current_company
    merged.languages = rules.languages or llm.languages

    # Years: trust the rule's date arithmetic; use the LLM's number only when
    # the rules found nothing (unusual layouts hide the date ranges).
    merged.total_years_experience = rules.total_years_experience or llm.total_years_experience

    # Seniority: recompute once both titles are known.
    merged.seniority = rules.seniority
    if not merged.seniority and llm.current_title:
        from app.services.extraction.rule_extractor import RuleExtractor

        merged.seniority = RuleExtractor()._extract_seniority(  # noqa: SLF001
            llm.current_title, ""
        )

    # Summary: prefer whichever is more substantial.
    rule_len = len(rules.summary or "")
    llm_len = len(llm.summary or "")
    merged.summary = rules.summary if rule_len >= llm_len else llm.summary

    # Education: per-field merge, rules preferred.
    merged.education = {**(llm.education or {}), **(rules.education or {})}

    # Skills: union with summed counts. `is_core` sticks once set.
    merged.skills = dict(rules.skills or {})
    for key, value in (llm.skills or {}).items():
        existing = merged.skills.get(key)
        if existing:
            existing["occurrences"] += value.get("occurrences", 1)
            existing["is_core"] = existing.get("is_core") or value.get("is_core", False)
        else:
            merged.skills[key] = dict(value)

    logger.debug(
        "merged extraction: %d rule skills + %d llm skills -> %d total",
        len(rules.skills or {}),
        len(llm.skills or {}),
        len(merged.skills),
    )
    return merged


def group_skills_by_category(
    skills: dict[str, dict],
) -> dict[str, list[dict]]:
    """
    Group skills for the UI: {"Languages": [{"name","canonical","core"}, ...]}.

    Ordering: core skills first, then by frequency, then alphabetically - so
    the strongest signal appears first.
    """
    grouped: dict[str, list[dict]] = {}
    for canonical, meta in (skills or {}).items():
        category = meta.get("category") or category_of(canonical)
        label = CATEGORY_LABELS.get(category, "Other")
        grouped.setdefault(label, []).append(
            {
                "canonical": canonical,
                "name": meta.get("display") or display_name(canonical),
                "occurrences": meta.get("occurrences", 1),
                "core": bool(meta.get("is_core")),
            }
        )

    for items in grouped.values():
        items.sort(key=lambda item: (not item["core"], -item["occurrences"], item["name"]))
    return grouped