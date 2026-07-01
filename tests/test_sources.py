"""Full test suite for Module 2 source adapters.

Structure
---------
Each adapter has two sections:
  - Behavioral tests  — hand-crafted minimal payloads; precise contract control.
  - Integration tests — real samples/ fixtures; every expected value was confirmed
                        by reading the fixture before writing the assertion.

The 8 targeted regression tests (restkey + skill lookaround) live in the Notes
and CSV sections and are preserved here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.github_source import GitHubSource
from src.transformer.sources.notes_source import NotesSource

# ── fixture-loading helper ────────────────────────────────────────────────────

_SAMPLES = Path(__file__).parent.parent / "samples"


def sample_text(rel: str) -> str:
    """Read a samples/ fixture by relative path and return its raw UTF-8 text."""
    return (_SAMPLES / rel).read_text(encoding="utf-8")


# =============================================================================
# CSVSource
# =============================================================================

# ── behavioral ────────────────────────────────────────────────────────────────


def test_csv_empty_payload_returns_empty():
    assert CSVSource().extract("") == []


def test_csv_header_only_returns_empty():
    assert CSVSource().extract("name,email\n") == []


def test_csv_malformed_never_raises():
    """Whatever CSVSource does with broken input, it must never raise."""
    try:
        result = CSVSource().extract('"unclosed_quote\nname\nAlice')
        assert isinstance(result, list)
    except Exception as exc:
        pytest.fail(f"extract() raised {type(exc).__name__}: {exc}")


def test_csv_values_are_whitespace_stripped():
    payload = "name,role\n  Alice  ,  Engineer  \n"
    rf = CSVSource().extract(payload)[0].raw_fields
    assert rf["name"] == "Alice"
    assert rf["role"] == "Engineer"


# restkey regression — over-wide row must survive, not be silently dropped
def test_csv_over_wide_row_is_preserved_not_dropped():
    src = CSVSource()
    payload = "name,email\nKelsey,k@x.com,EXTRA_VALUE"
    records = src.extract(payload)
    assert len(records) == 1, "whole-row drop regression: expected 1 record, got 0"
    assert records[0].raw_fields["name"] == "Kelsey"
    assert records[0].raw_fields["email"] == "k@x.com"


def test_csv_over_wide_row_overflow_lands_in_extra():
    src = CSVSource()
    payload = "name,email\nKelsey,k@x.com,EXTRA_VALUE"
    rf = src.extract(payload)[0].raw_fields
    assert "_extra" in rf
    assert rf["_extra"] == ["EXTRA_VALUE"]


def test_csv_normal_row_has_no_extra_key():
    src = CSVSource()
    payload = "name,email\nKelsey,k@x.com"
    rf = src.extract(payload)[0].raw_fields
    assert "_extra" not in rf


# ── integration ───────────────────────────────────────────────────────────────


def test_csv_kelsey_yields_one_record():
    records = CSVSource().extract(sample_text("recruiter_csv/kelsey_hightower.csv"))
    assert len(records) == 1


def test_csv_kelsey_has_22_columns():
    # Confirmed by reading the file: 13 original + 9 education/experience columns.
    rf = CSVSource().extract(sample_text("recruiter_csv/kelsey_hightower.csv"))[0].raw_fields
    assert len(rf) == 22


def test_csv_kelsey_real_field_values():
    rf = CSVSource().extract(sample_text("recruiter_csv/kelsey_hightower.csv"))[0].raw_fields
    assert rf["first_name"] == "Kelsey"
    assert rf["last_name"] == "Hightower"
    # CSV says "Stripe" — the ATS says "Stripe Inc." — intentional conflict for merge layer
    assert rf["current_company"] == "Stripe"
    assert rf["email"] == "kelsey.hightower@gmail.com"
    assert rf["years_experience"] == "12"
    assert rf["prev_company"] == "Google"
    assert rf["education_degree"] == "B.S."


def test_csv_andrej_yields_one_record():
    records = CSVSource().extract(sample_text("recruiter_csv/andrej_karpathy.csv"))
    assert len(records) == 1


def test_csv_andrej_real_company_and_email():
    rf = CSVSource().extract(sample_text("recruiter_csv/andrej_karpathy.csv"))[0].raw_fields
    assert rf["current_company"] == "Eureka Labs"
    assert rf["email"] == "andrej.karpathy@gmail.com"
    assert rf["prev_company"] == "OpenAI"


# =============================================================================
# ATSSource
# =============================================================================

# ── behavioral ────────────────────────────────────────────────────────────────


def test_ats_single_object_yields_one_record():
    result = ATSSource().extract('{"id": 1, "name": "Test"}')
    assert len(result) == 1
    assert result[0].raw_fields["name"] == "Test"


def test_ats_json_array_of_two_yields_two_records():
    result = ATSSource().extract('[{"id": 1}, {"id": 2}]')
    assert len(result) == 2


def test_ats_empty_object_yields_one_record_with_empty_fields():
    result = ATSSource().extract("{}")
    assert len(result) == 1
    assert result[0].raw_fields == {}


def test_ats_malformed_json_returns_empty():
    assert ATSSource().extract("{not json}") == []


def test_ats_bare_string_top_level_returns_empty():
    assert ATSSource().extract('"just a string"') == []


def test_ats_bare_number_top_level_returns_empty():
    assert ATSSource().extract("42") == []


def test_ats_non_dict_entries_in_array_skipped_rest_survive():
    payload = '[{"id": 1}, "bad_entry", {"id": 2}]'
    result = ATSSource().extract(payload)
    assert len(result) == 2
    assert {r.raw_fields["id"] for r in result} == {1, 2}


# ── integration ───────────────────────────────────────────────────────────────


def test_ats_kelsey_email_addresses_is_list_of_two():
    rf = ATSSource().extract(sample_text("ats_json/kelsey_hightower.json"))[0].raw_fields
    assert isinstance(rf["email_addresses"], list)
    assert len(rf["email_addresses"]) == 2


def test_ats_kelsey_personal_email_type_tag_intact():
    """The merge layer reads type tags — the personal-typed entry must survive verbatim."""
    rf = ATSSource().extract(sample_text("ats_json/kelsey_hightower.json"))[0].raw_fields
    personal = next(e for e in rf["email_addresses"] if e["type"] == "personal")
    assert personal["value"] == "kelsey.hightower@gmail.com"


def test_ats_kelsey_work_email_type_tag_intact():
    rf = ATSSource().extract(sample_text("ats_json/kelsey_hightower.json"))[0].raw_fields
    work = next(e for e in rf["email_addresses"] if e["type"] == "work")
    assert work["value"] == "kelsey@stripe.com"


def test_ats_kelsey_employments_count_and_structure():
    rf = ATSSource().extract(sample_text("ats_json/kelsey_hightower.json"))[0].raw_fields
    emps = rf["employments"]
    assert len(emps) == 2
    # Current role: end_date is null (None after JSON parse)
    assert emps[0]["company_name"] == "Stripe Inc."
    assert emps[0]["end_date"] is None
    # Prior role
    assert emps[1]["company_name"] == "Google LLC"
    assert emps[1]["end_date"] == "2021-02"


def test_ats_kelsey_educations_nested_intact():
    rf = ATSSource().extract(sample_text("ats_json/kelsey_hightower.json"))[0].raw_fields
    edu = rf["educations"]
    assert len(edu) == 1
    assert edu[0]["school_name"] == "Georgia Institute of Technology"
    assert edu[0]["degree"] == "Bachelor of Science"


def test_ats_andrej_personal_email_correct():
    rf = ATSSource().extract(sample_text("ats_json/andrej_karpathy.json"))[0].raw_fields
    personal = next(e for e in rf["email_addresses"] if e["type"] == "personal")
    assert personal["value"] == "andrej.karpathy@gmail.com"


def test_ats_andrej_current_employment_has_null_end_date():
    rf = ATSSource().extract(sample_text("ats_json/andrej_karpathy.json"))[0].raw_fields
    current = rf["employments"][0]
    assert current["company_name"] == "Eureka Labs"
    assert current["end_date"] is None


def test_ats_andrej_education_degree_is_doctor_of_philosophy():
    rf = ATSSource().extract(sample_text("ats_json/andrej_karpathy.json"))[0].raw_fields
    assert rf["educations"][0]["degree"] == "Doctor of Philosophy"


# =============================================================================
# GitHubSource
# =============================================================================

# ── behavioral ────────────────────────────────────────────────────────────────


def test_github_malformed_json_returns_empty():
    assert GitHubSource().extract("{not json}") == []


def test_github_non_dict_top_level_returns_empty():
    assert GitHubSource().extract("[1, 2, 3]") == []


def test_github_neither_profile_nor_repos_returns_empty():
    assert GitHubSource().extract('{"other": "stuff"}') == []


def test_github_null_language_repo_is_carried():
    """language=None must be preserved in the slimmed repo, not dropped."""
    doc = json.dumps({
        "profile": {"login": "x"},
        "repos": [{"name": "r", "description": None, "html_url": "",
                   "language": None, "fork": False, "topics": []}],
    })
    rf = GitHubSource().extract(doc)[0].raw_fields
    assert rf["repos"][0]["language"] is None


def test_github_fork_repo_is_carried_with_fork_true():
    """fork=True must survive — the normalize layer decides what to do with forks."""
    doc = json.dumps({
        "profile": {"login": "x"},
        "repos": [{"name": "r", "description": None, "html_url": "",
                   "language": "Go", "fork": True, "topics": []}],
    })
    rf = GitHubSource().extract(doc)[0].raw_fields
    assert rf["repos"][0]["fork"] is True


# ── integration ───────────────────────────────────────────────────────────────

_EXPECTED_PROFILE_KEYS = {"login", "name", "location", "bio", "blog", "company", "email", "html_url"}
_EXPECTED_REPO_KEYS = {"name", "description", "html_url", "language", "fork", "topics"}


def test_github_kelsey_profile_has_exactly_the_selected_keys():
    rf = GitHubSource().extract(sample_text("github/kelsey_hightower.json"))[0].raw_fields
    assert set(rf["profile"].keys()) == _EXPECTED_PROFILE_KEYS


def test_github_kelsey_profile_login():
    rf = GitHubSource().extract(sample_text("github/kelsey_hightower.json"))[0].raw_fields
    assert rf["profile"]["login"] == "kelseyhightower"


def test_github_kelsey_profile_location():
    rf = GitHubSource().extract(sample_text("github/kelsey_hightower.json"))[0].raw_fields
    assert rf["profile"]["location"] == "Washington"


def test_github_kelsey_profile_bio_is_none():
    # Confirmed in the live API response: bio field is null for kelseyhightower.
    rf = GitHubSource().extract(sample_text("github/kelsey_hightower.json"))[0].raw_fields
    assert rf["profile"]["bio"] is None


def test_github_kelsey_repos_is_list_of_10():
    rf = GitHubSource().extract(sample_text("github/kelsey_hightower.json"))[0].raw_fields
    assert isinstance(rf["repos"], list)
    assert len(rf["repos"]) == 10


def test_github_kelsey_every_repo_has_exactly_the_selected_keys():
    rf = GitHubSource().extract(sample_text("github/kelsey_hightower.json"))[0].raw_fields
    for repo in rf["repos"]:
        assert set(repo.keys()) == _EXPECTED_REPO_KEYS


def test_github_kelsey_fork_repo_carried_with_flag():
    """'appdash' is a fork in the fixture — must survive with fork=True."""
    rf = GitHubSource().extract(sample_text("github/kelsey_hightower.json"))[0].raw_fields
    fork_repos = [r for r in rf["repos"] if r["fork"] is True]
    assert len(fork_repos) == 1
    assert fork_repos[0]["name"] == "appdash"


def test_github_andrej_profile_login():
    rf = GitHubSource().extract(sample_text("github/andrej_karpathy.json"))[0].raw_fields
    assert rf["profile"]["login"] == "karpathy"


def test_github_andrej_profile_bio():
    rf = GitHubSource().extract(sample_text("github/andrej_karpathy.json"))[0].raw_fields
    assert rf["profile"]["bio"] == "I like to train Deep Neural Nets on large datasets."


def test_github_andrej_null_language_fork_carried():
    """'cpython' has language=None and fork=True — both must be preserved as-is."""
    rf = GitHubSource().extract(sample_text("github/andrej_karpathy.json"))[0].raw_fields
    cpython = next(r for r in rf["repos"] if r["name"] == "cpython")
    assert cpython["language"] is None
    assert cpython["fork"] is True


def test_github_andrej_repo_topics_carried():
    rf = GitHubSource().extract(sample_text("github/andrej_karpathy.json"))[0].raw_fields
    arxiv = next(r for r in rf["repos"] if r["name"] == "arxiv-sanity-lite")
    assert "machine-learning" in arxiv["topics"]
    assert "deep-learning" in arxiv["topics"]


# =============================================================================
# NotesSource
# =============================================================================

# ── behavioral ────────────────────────────────────────────────────────────────


def test_notes_empty_string_returns_empty():
    assert NotesSource().extract("") == []


def test_notes_whitespace_only_returns_empty():
    assert NotesSource().extract("   \n  ") == []


def test_notes_valid_international_phone_kept():
    rf = NotesSource().extract("Call me at +1 650 555 0287 anytime.")[0].raw_fields
    assert "+1 650 555 0287" in rf["phones"]


def test_notes_invalid_phone_shape_dropped():
    # "+0 000 000 0000" matches the shape regex (has +, digits, right length) but
    # country code 0 is invalid — phonenumbers rejects it.
    rf = NotesSource().extract("Reach out at +0 000 000 0000 please.")[0].raw_fields
    assert rf["phones"] == []


def test_notes_valid_bare_indian_mobile_kept():
    rf = NotesSource().extract("Candidate said WhatsApp is best at 9876543210 after 6pm.")[0].raw_fields
    assert "9876543210" in rf["phones"]
    assert rf["phone_default_region"] == "IN"
    assert rf["phone_recognition_methods"]["9876543210"] == "region_bare"


def test_notes_random_bare_ten_digit_number_rejected():
    rf = NotesSource().extract("Comp note: internal ID 1111111111 was copied from the sheet.")[0].raw_fields
    assert rf["phones"] == []


def test_notes_url_trailing_period_stripped():
    # Regression: _URL_RE used to capture the trailing period as part of the URL.
    rf = NotesSource().extract("See https://kelsey.dev.")[0].raw_fields
    assert "https://kelsey.dev" in rf["urls"]
    assert "https://kelsey.dev." not in rf["urls"]


def test_notes_empty_vocabulary_yields_no_skills():
    rf = NotesSource(skill_vocabulary=[]).extract("Expert in Python and Kubernetes.")[0].raw_fields
    assert rf["skills"] == []


def test_notes_none_vocabulary_uses_default_taxonomy():
    rf = NotesSource(skill_vocabulary=None).extract("Expert in Python and Kubernetes.")[0].raw_fields
    assert "Python" in rf["skills"]
    assert "Kubernetes" in rf["skills"]


# skill lookaround regressions — punctuation-bearing skills must be recognized

_PUNCT_VOCAB = ["C++", "C#", ".NET", "Node.js", "Go", "Python"]
_PUNCT_TEXT = (
    "Strong in C++ and C#, did some .NET and Node.js. Python and Go too. Works at Google."
)


def test_skills_cxx_recognized():
    assert "C++" in NotesSource(skill_vocabulary=_PUNCT_VOCAB)._extract_skills(_PUNCT_TEXT)


def test_skills_csharp_recognized():
    assert "C#" in NotesSource(skill_vocabulary=_PUNCT_VOCAB)._extract_skills(_PUNCT_TEXT)


def test_skills_dotnet_recognized():
    assert ".NET" in NotesSource(skill_vocabulary=_PUNCT_VOCAB)._extract_skills(_PUNCT_TEXT)


def test_skills_go_does_not_match_inside_google():
    assert NotesSource(skill_vocabulary=["Go"])._extract_skills("Works at Google.") == []


def test_skills_go_does_not_match_inside_hyphenated_words():
    assert NotesSource(skill_vocabulary=["Go"])._extract_skills("Handled a go-live.") == []


def test_skills_unknown_skill_not_recognized():
    assert "Rust" not in NotesSource(skill_vocabulary=["Python"])._extract_skills(
        "Expert in Rust and Python."
    )


# ── integration ───────────────────────────────────────────────────────────────

# Vocabulary confirmed present verbatim in kelsey_hightower.txt:
#   "Kubernetes" — "early Kubernetes days"
#   "platform engineering" — "right abstractions for platform engineering"
#   "infrastructure" — "infrastructure and cloud advisory work"
_KELSEY_VOCAB = ["Kubernetes", "platform engineering", "infrastructure"]


def test_notes_kelsey_recognizes_personal_email():
    rf = NotesSource().extract(sample_text("recruiter_notes/kelsey_hightower.txt"))[0].raw_fields
    assert "kelsey.hightower@gmail.com" in rf["emails"]


def test_notes_kelsey_recognizes_work_email():
    rf = NotesSource().extract(sample_text("recruiter_notes/kelsey_hightower.txt"))[0].raw_fields
    assert "kelsey@stripe.com" in rf["emails"]


def test_notes_kelsey_recognizes_phone():
    # +1 202 555 0142 confirmed valid via phonenumbers.is_valid_number before writing.
    rf = NotesSource().extract(sample_text("recruiter_notes/kelsey_hightower.txt"))[0].raw_fields
    assert "+1 202 555 0142" in rf["phones"]


def test_notes_kelsey_recognizes_linkedin_url():
    rf = NotesSource().extract(sample_text("recruiter_notes/kelsey_hightower.txt"))[0].raw_fields
    assert "https://linkedin.com/in/kelsey-hightower" in rf["urls"]


def test_notes_kelsey_recognizes_expected_skills():
    src = NotesSource()
    rf = src.extract(sample_text("recruiter_notes/kelsey_hightower.txt"))[0].raw_fields
    assert "Kubernetes" in rf["skills"]
    assert "Platform Engineering" in rf["skills"]
    assert "Infrastructure" in rf["skills"]


def test_notes_kelsey_does_not_invent_sign_off_initial():
    # Regression (AUDIT A3): "— Riya R." must not be mined as skill "R".
    rf = NotesSource().extract(sample_text("recruiter_notes/kelsey_hightower.txt"))[0].raw_fields
    assert "R" not in rf["skills"]


def test_notes_kelsey_skill_extraction_is_deterministic():
    text = sample_text("recruiter_notes/kelsey_hightower.txt")
    src = NotesSource()
    assert src.extract(text)[0].raw_fields["skills"] == src.extract(text)[0].raw_fields["skills"]


def test_notes_skill_audit_records_match_method():
    _, audit = NotesSource().extract_with_audit("Strong in Kubernets and platform engineering.")
    skill_events = [event for event in audit if event.field == "skills" and event.kind == "value_recognized"]
    assert skill_events
    assert {event.details["match_method"] for event in skill_events}.issubset({"exact", "fuzzy"})


def test_notes_andrej_missing_file_graceful_degradation():
    # Andrej has no recruiter_notes/ file — by design, tests the missing-source path.
    # "Missing source" is an IngestError at the ingest layer (not the adapter's job).
    # At the adapter boundary the contract is: empty string payload → [].
    assert NotesSource().extract("") == []
