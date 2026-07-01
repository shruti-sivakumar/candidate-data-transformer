"""Tests for Module 3 normalizers + orchestrator.

Behavioral tests use inline minimal RawRecords for precise control; integration
tests run real fixtures through the source adapters then the normalizers, pinning
real-data properties (confidence math per method, conflict survival, location
parsing). Expected values for integration tests were confirmed against the
fixture inventory.
"""
from pathlib import Path

from src.transformer.ingest import read_file
from src.transformer.models import NormalizedProject, RawRecord
from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.github_source import GitHubSource
from src.transformer.normalize.orchestrator import normalize_record
from src.transformer.normalize.csv_normalizer import normalize_csv
from src.transformer.normalize.ats_normalizer import normalize_ats
from src.transformer.normalize.github_normalizer import normalize_github
from src.transformer.normalize.notes_normalizer import normalize_notes


SAMPLES = Path("samples")


def _load(adapter, rel_path: str) -> RawRecord:
    """Read a real fixture through its adapter, return the first RawRecord."""
    records = adapter.extract(read_file(SAMPLES / rel_path))
    assert records, f"adapter produced no records for {rel_path}"
    return records[0]


# ---------------------------------------------------------------------------
# Behavioral — CSV normalizer (inline minimal records)
# ---------------------------------------------------------------------------

class TestCSVNormalizerBehavioral:
    def _raw(self, **cols) -> RawRecord:
        return RawRecord(source="recruiter_csv", raw_fields=cols)

    def test_name_joined_and_direct_confidence(self):
        r = normalize_csv(self._raw(first_name="Kelsey", last_name="Hightower"))
        assert r.full_name.value == "Kelsey Hightower"
        assert r.full_name.method == "direct"
        assert r.full_name.confidence == 0.8  # 0.80 x 1.0 x 1.0

    def test_skills_split_and_direct(self):
        r = normalize_csv(self._raw(top_skills="Kubernetes, Go; Python"))
        vals = [s.value for s in r.skills]
        assert vals == ["Kubernetes", "Go", "Python"]
        assert all(s.method == "direct" and s.confidence == 0.8 for s in r.skills)

    def test_skill_aliases_are_canonicalized(self):
        r = normalize_csv(self._raw(top_skills="k8s; golang; torch"))
        assert [s.value for s in r.skills] == ["Kubernetes", "Go", "PyTorch"]

    def test_failed_email_omitted(self):
        # Unparseable email -> field absent (empty list), not a None-valued entry.
        r = normalize_csv(self._raw(email="not-an-email"))
        assert r.emails == []

    def test_missing_company_yields_no_experience(self):
        # Company is the experience anchor.
        r = normalize_csv(self._raw(current_title="Engineer"))  # no company
        assert r.experience == []

    def test_location_positional(self):
        r = normalize_csv(self._raw(location="Washington, DC, USA"))
        loc = r.location.value
        assert (loc.city, loc.region, loc.country) == ("Washington", "DC", "US")

    def test_empty_record_produces_empty_normalized(self):
        r = normalize_csv(self._raw())
        assert r.full_name is None and r.emails == [] and r.skills == []


# ---------------------------------------------------------------------------
# Behavioral — ATS normalizer
# ---------------------------------------------------------------------------

class TestATSNormalizerBehavioral:
    def _raw(self, **fields) -> RawRecord:
        return RawRecord(source="ats_json", raw_fields=fields)

    def test_email_array_all_entries_kept(self):
        r = normalize_ats(self._raw(email_addresses=[
            {"value": "WORK@x.com", "type": "work"},
            {"value": "Personal@x.com", "type": "personal"},
        ]))
        assert [e.value for e in r.emails] == ["work@x.com", "personal@x.com"]
        assert all(e.method == "direct" and e.confidence == 0.9 for e in r.emails)

    def test_employments_become_experience(self):
        r = normalize_ats(self._raw(employments=[
            {"company_name": "Stripe Inc.", "title": "Staff", "start_date": "2021-03", "end_date": None},
        ]))
        e = r.experience[0].value
        assert e.company == "Stripe Inc." and e.start == "2021-03" and e.end is None

    def test_educations_become_education_with_int_year(self):
        r = normalize_ats(self._raw(educations=[
            {"school_name": "GT", "degree": "B.S.", "discipline": "CS", "end_date": "2007"},
        ]))
        edu = r.education[0].value
        assert edu.institution == "GT" and edu.end_year == 2007

    def test_no_skills_field_yields_empty(self):
        r = normalize_ats(self._raw(first_name="X"))
        assert r.skills == []

    def test_explicit_skills_field_is_normalized(self):
        r = normalize_ats(self._raw(skills=["k8s", {"name": "golang"}, "torch"]))
        assert [skill.value for skill in r.skills] == ["Kubernetes", "Go", "PyTorch"]
        assert all(skill.method == "direct" and skill.confidence == 0.9 for skill in r.skills)

    def test_skill_like_custom_fields_are_normalized(self):
        r = normalize_ats(self._raw(custom_fields=[
            {"name": "Technical Skills", "value": "Python, k8s"},
            {"name": "Interview status", "value": "strong"},
        ]))
        assert [skill.value for skill in r.skills] == ["Python", "Kubernetes"]


# ---------------------------------------------------------------------------
# Behavioral — GitHub normalizer
# ---------------------------------------------------------------------------

class TestGitHubNormalizerBehavioral:
    def _raw(self, profile=None, repos=None) -> RawRecord:
        return RawRecord(source="github",
                         raw_fields={"profile": profile or {}, "repos": repos or []})

    def test_language_becomes_inferred_skill(self):
        r = normalize_github(self._raw(repos=[
            {"name": "p", "language": "Go", "fork": False, "html_url": "https://github.com/x/p"},
        ]))
        assert r.skills[0].value == "Go"
        assert r.skills[0].method == "inferred"
        assert round(r.skills[0].confidence, 3) == 0.35  # 0.70 x 0.5 x 1.0

    def test_fork_dropped_from_projects_and_skills(self):
        r = normalize_github(self._raw(repos=[
            {"name": "forked", "language": "Rust", "fork": True, "html_url": "https://github.com/x/forked"},
        ]))
        assert r.projects == [] and r.skills == []

    def test_null_language_contributes_no_skill_but_project_kept(self):
        r = normalize_github(self._raw(repos=[
            {"name": "proj", "language": None, "fork": False, "html_url": "https://github.com/x/proj"},
        ]))
        assert r.skills == []
        assert isinstance(r.projects[0].value, NormalizedProject)
        assert r.projects[0].value.primary_language is None

    def test_languages_deduped_across_repos(self):
        r = normalize_github(self._raw(repos=[
            {"name": "a", "language": "Go", "fork": False, "html_url": "https://github.com/x/a"},
            {"name": "b", "language": "Go", "fork": False, "html_url": "https://github.com/x/b"},
        ]))
        assert [s.value for s in r.skills] == ["Go"]

    def test_bio_becomes_headline(self):
        r = normalize_github(self._raw(profile={"name": "A", "bio": "I train nets."}))
        assert r.headline.value == "I train nets."


# ---------------------------------------------------------------------------
# Behavioral — Notes normalizer
# ---------------------------------------------------------------------------

class TestNotesNormalizerBehavioral:
    def _raw(self, **fields) -> RawRecord:
        return RawRecord(source="recruiter_notes", raw_fields=fields)

    def test_contacts_are_regex_method(self):
        r = normalize_notes(self._raw(emails=["a@b.com"], phones=["+12025550142"]))
        assert r.emails[0].method == "regex"
        assert round(r.emails[0].confidence, 3) == 0.385  # 0.55 x 0.7 x 1.0

    def test_urls_classified_into_links(self):
        r = normalize_notes(self._raw(urls=[
            "https://linkedin.com/in/x", "https://github.com/x", "https://blog.x",
        ]))
        links = r.links.value
        assert "linkedin.com" in links.linkedin
        assert "github.com" in links.github
        assert any("blog.x" in u for u in links.other)

    def test_no_name_extracted_from_notes(self):
        # Prose name extraction is descoped.
        r = normalize_notes(self._raw(emails=["a@b.com"]))
        assert r.full_name is None


# ---------------------------------------------------------------------------
# Behavioral — orchestrator dispatch
# ---------------------------------------------------------------------------

class TestOrchestrator:
    def test_dispatches_by_source(self):
        r = normalize_record(RawRecord(source="recruiter_csv",
                                       raw_fields={"first_name": "K", "last_name": "H"}))
        assert r is not None and r.source == "recruiter_csv"
        assert r.full_name.value == "K H"

    def test_unknown_source_returns_none(self):
        r = normalize_record(RawRecord(source="mystery", raw_fields={}))
        assert r is None


# ---------------------------------------------------------------------------
# Integration — real fixtures end-to-end (adapter -> normalizer)
# ---------------------------------------------------------------------------

class TestIntegrationRealFixtures:
    def test_csv_kelsey_full_mapping(self):
        norm = normalize_record(_load(CSVSource(), "recruiter_csv/kelsey_hightower.csv"))
        assert norm.full_name.value == "Kelsey Hightower"
        assert norm.emails[0].value == "kelsey.hightower@gmail.com"
        assert norm.phones[0].value == "+12025550142"
        loc = norm.location.value
        assert (loc.city, loc.region, loc.country) == ("Washington", "DC", "US")
        # CSV skills are direct/0.8
        assert all(s.method == "direct" and s.confidence == 0.8 for s in norm.skills)
        assert "Kubernetes" in [s.value for s in norm.skills]

    def test_csv_experience_conflict_values_survive(self):
        norm = normalize_record(_load(CSVSource(), "recruiter_csv/kelsey_hightower.csv"))
        companies = [e.value.company for e in norm.experience]
        # CSV side of the seeded conflicts.
        assert "Stripe" in companies and "Google" in companies

    def test_ats_kelsey_conflict_values_survive(self):
        norm = normalize_record(_load(ATSSource(), "ats_json/kelsey_hightower.json"))
        companies = [e.value.company for e in norm.experience]
        # ATS side of the seeded conflicts.
        assert "Stripe Inc." in companies and "Google LLC" in companies

    def test_ats_email_agreement_value_present(self):
        # The personal email that also appears in CSV — the merge agreement case.
        norm = normalize_record(_load(ATSSource(), "ats_json/kelsey_hightower.json"))
        assert "kelsey.hightower@gmail.com" in [e.value for e in norm.emails]

    def test_github_skill_is_inferred(self):
        norm = normalize_record(_load(GitHubSource(), "github/kelsey_hightower.json"))
        assert norm.skills, "expected at least one inferred skill from repo languages"
        assert all(s.method == "inferred" for s in norm.skills)
        assert "Go" in [s.value for s in norm.skills]

    def test_github_forks_excluded(self):
        # Kelsey's 'appdash' is a fork -> must not appear as a project.
        norm = normalize_record(_load(GitHubSource(), "github/kelsey_hightower.json"))
        names = [p.value.name for p in norm.projects]
        assert "appdash" not in names

    def test_github_location_is_bare_city(self):
        norm = normalize_record(_load(GitHubSource(), "github/kelsey_hightower.json"))
        loc = norm.location.value
        assert loc.city == "Washington" and loc.region is None and loc.country is None
