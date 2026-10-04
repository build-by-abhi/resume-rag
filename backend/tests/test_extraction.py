"""Rule-based structured extraction and the rules/LLM merge policy."""

from __future__ import annotations

from app.services.extraction import Extraction, merge_extractions, rule_extractor
from app.services.extraction.normalizer import group_skills_by_category


class TestIdentity:
    def test_name_email_phone(self, sample_resumes):
        out = rule_extractor.extract(sample_resumes["priya_backend_python"])
        assert out.full_name == "Priya Raghunathan"
        assert out.email == "priya.raghunathan@example.com"
        assert "+49 30 5551234" in (out.phone or "")

    def test_links_are_classified(self, sample_resumes):
        out = rule_extractor.extract(sample_resumes["priya_backend_python"])
        assert "linkedin.com/in/priya-raghunathan" in out.links["linkedin"]
        assert "github.com/praghu" in out.links["github"]

    def test_location_including_city_country(self, sample_resumes):
        assert rule_extractor.extract(sample_resumes["priya_backend_python"]).location == "Berlin, Germany"
        assert rule_extractor.extract(sample_resumes["ananya_ml_research"]).location == "Bengaluru, India"

    def test_rejects_non_names(self):
        out = rule_extractor.extract(
            "SENIOR SOFTWARE ENGINEER\njane@example.com\nBerlin, Germany\n"
        )
        # The job title must not be mistaken for the person's name.
        assert out.full_name != "Senior Software Engineer"


class TestWork:
    def test_current_role_and_company(self, sample_resumes):
        out = rule_extractor.extract(sample_resumes["priya_backend_python"])
        assert out.current_title == "Staff Backend Engineer"
        assert out.current_company == "Zalando"

    def test_seniority(self, sample_resumes):
        assert rule_extractor.extract(sample_resumes["priya_backend_python"]).seniority == "Senior"
        assert rule_extractor.extract(sample_resumes["marcus_frontend_react"]).seniority == "Junior"
        assert rule_extractor.extract(sample_resumes["ananya_ml_research"]).seniority == "Principal"

    def test_years_from_explicit_claim(self):
        out = rule_extractor.extract("Jane Doe\nSummary\nSenior engineer with 9+ years of experience.")
        assert out.total_years_experience == 9.0

    def test_years_from_date_ranges(self, sample_resumes):
        out = rule_extractor.extract(sample_resumes["priya_backend_python"])
        # Zalando 2021->now plus Delivery Hero 2017->2021, merged without double counting.
        assert 7 <= (out.total_years_experience or 0) <= 11

    def test_overlapping_ranges_are_not_double_counted(self):
        # Same job listed twice must not inflate the total.
        text = (
            "Jane Doe\nEXPERIENCE\nEngineer | Acme | 2018 - 2022\n"
            "Engineer | Acme | 2019 - 2020\n"
        )
        out = rule_extractor.extract(text)
        assert out.total_years_experience == 3.0

    def test_rejects_implausible_digit_runs(self):
        out = rule_extractor.extract("Jane Doe\nContact 1111111111111 or +49 30 5551234")
        assert "1111111111111" not in (out.phone or "")


class TestEducationAndSkills:
    def test_degree_and_institution(self, sample_resumes):
        edu = rule_extractor.extract(sample_resumes["priya_backend_python"]).education
        assert edu.get("level") == "MS"
        assert "Berlin" in (edu.get("institution") or "") or "Technische" in (edu.get("institution") or "")

    def test_phd_detected(self, sample_resumes):
        assert rule_extractor.extract(sample_resumes["ananya_ml_research"]).education.get("level") == "PhD"

    def test_field_is_not_taken_from_inside_a_university_name(self):
        """
        Regression: the field pattern used a bare `(in|of)` and matched the "of"
        inside "University of Cape Town", so B.Eng. Software Engineering was
        recorded as the field "Cape Town".
        """
        out = rule_extractor.extract(
            "Dana Whitfield\n\nEDUCATION\n"
            "B.Eng. Software Engineering | University of Cape Town | 2014\n"
        )
        edu = out.education
        assert edu.get("level") == "BEng", edu
        assert "Cape Town" not in (edu.get("field") or ""), edu
        assert edu.get("field") == "Software Engineering", edu

    def test_field_still_found_with_a_preposition(self):
        out = rule_extractor.extract(
            "Jane Doe\n\nEDUCATION\nMSc in Computer Science | University of Berlin\n"
        )
        assert out.education.get("field") == "Computer Science"

    def test_institution_is_just_the_institution_segment(self):
        out = rule_extractor.extract(
            "Jane Doe\n\nEDUCATION\n"
            "M.Sc. Computer Science | Technische Universitat Berlin | 2017\n"
        )
        inst = out.education.get("institution") or ""
        # The whole pipe-delimited line used to be stored, degree included.
        assert "Universitat Berlin" in inst
        assert "M.Sc." not in inst
        assert "|" not in inst

    def test_skills_are_canonicalised(self, sample_resumes):
        skills = rule_extractor.extract(sample_resumes["priya_backend_python"]).skills
        assert "python" in skills
        assert "postgresql" in skills
        assert "kubernetes" in skills
        assert "k8s" not in skills, "alias must collapse to the canonical key"
        # Explicitly listed in the SKILLS section -> flagged core.
        assert skills["python"]["is_core"] is True

    def test_frequency_ranks_real_expertise(self, sample_resumes):
        skills = rule_extractor.extract(sample_resumes["ananya_ml_research"]).skills
        assert skills["pytorch"]["occurrences"] >= 1
        assert skills["python"]["occurrences"] >= 3

    def test_languages_section(self, sample_resumes):
        # "Languages: English, Hindi" must not leak in as a skill.
        skills = rule_extractor.extract(sample_resumes["ananya_ml_research"]).skills
        assert "english" not in skills

    def test_summary_extracted(self, sample_resumes):
        summary = rule_extractor.extract(sample_resumes["priya_backend_python"]).summary
        assert summary and "backend" in summary.lower()


class TestMergePolicy:
    def test_rules_win_for_contact_details(self):
        rules = Extraction(email="real@example.com", phone="+49 30 5551234")
        llm = Extraction(email="hallucinated@example.com", phone="+1 555 000 0000")
        merged = merge_extractions(rules, llm, llm_enabled=True)
        assert merged.email == "real@example.com"
        assert merged.phone == "+49 30 5551234"

    def test_llm_fills_gaps(self):
        rules = Extraction(email="real@example.com")
        llm = Extraction(full_name="Jane Doe", current_title="Data Engineer", location="Lisbon")
        merged = merge_extractions(rules, llm, llm_enabled=True)
        assert merged.email == "real@example.com"  # rule kept
        assert merged.full_name == "Jane Doe"        # LLM filled the gap

    def test_years_fall_back_to_llm_when_rules_found_nothing(self):
        merged = merge_extractions(
            Extraction(), Extraction(total_years_experience=12.0), llm_enabled=True
        )
        assert merged.total_years_experience == 12.0

    def test_skills_are_unioned_with_summed_counts(self):
        rules = Extraction(skills={"python": {"display": "Python", "occurrences": 3, "is_core": True}})
        llm = Extraction(skills={
            "python": {"display": "Python", "occurrences": 1, "is_core": False},
            "rust": {"display": "Rust", "occurrences": 1, "is_core": True},
        })
        merged = merge_extractions(rules, llm, llm_enabled=True)
        assert merged.skills["python"]["occurrences"] == 4
        assert merged.skills["python"]["is_core"] is True
        assert "rust" in merged.skills

    def test_llm_disabled_means_rules_only(self):
        rules = Extraction(full_name="Jane Doe")
        llm = Extraction(full_name="Wrong Name", current_title="Invented")
        merged = merge_extractions(rules, llm, llm_enabled=False)
        assert merged.full_name == "Jane Doe"
        assert merged.current_title is None

    def test_education_merges_field_by_field(self):
        rules = Extraction(education={"level": "BSc"})
        llm = Extraction(education={"level": "PhD", "field": "Machine Learning"})
        merged = merge_extractions(rules, llm, llm_enabled=True)
        assert merged.education["level"] == "BSc"
        assert merged.education["field"] == "Machine Learning"


class TestGrouping:
    def test_groups_by_category_with_core_first(self):
        skills = {
            "docker": {"display": "Docker", "occurrences": 2, "is_core": True, "category": "devops"},
            "kubernetes": {"display": "Kubernetes", "occurrences": 9, "is_core": False, "category": "devops"},
        }
        grouped = group_skills_by_category(skills)
        assert "DevOps" in grouped
        # Core skills sort first even with a lower occurrence count.
        assert grouped["DevOps"][0]["name"] == "Docker"