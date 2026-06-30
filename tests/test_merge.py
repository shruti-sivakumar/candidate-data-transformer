"""Tests for Module 4 merge behavior."""
from __future__ import annotations

from pathlib import Path

from src.transformer.ingest import read_file
from src.transformer.merge import merge_records
from src.transformer.normalize.orchestrator import normalize_record
from src.transformer.score import score_profile
from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.github_source import GitHubSource
from src.transformer.sources.notes_source import NotesSource

SAMPLES = Path("samples")


def _normalize(adapter, rel_path: str):
    records = adapter.extract(read_file(SAMPLES / rel_path))
    assert records, f"adapter produced no records for {rel_path}"
    norm = normalize_record(records[0])
    assert norm is not None
    return norm


def _kelsey_records():
    return [
        _normalize(CSVSource(), "recruiter_csv/kelsey_hightower.csv"),
        _normalize(ATSSource(), "ats_json/kelsey_hightower.json"),
        _normalize(GitHubSource(), "github/kelsey_hightower.json"),
        _normalize(NotesSource(skill_vocabulary=[]), "recruiter_notes/kelsey_hightower.txt"),
    ]


class TestMergeRealFixtures:
    def test_duplicate_email_is_collapsed_and_confidence_boosted(self):
        profile, _ = merge_records(_kelsey_records())

        matches = [email for email in profile.emails if email.value == "kelsey.hightower@gmail.com"]
        assert len(matches) == 1
        merged = matches[0]
        assert merged.sources == ["ats_json", "recruiter_csv", "recruiter_notes"]
        assert round(merged.confidence, 3) == 0.988

    def test_experience_entries_merge_on_company_anchor(self):
        profile, _ = merge_records(_kelsey_records())

        companies = [entry.company for entry in profile.experience]
        assert companies == ["Google LLC", "Stripe Inc."]
        stripe = next(entry for entry in profile.experience if "Stripe" in entry.company)
        assert stripe.sources == ["ats_json", "recruiter_csv"]
        assert stripe.title == "Staff Platform Engineer"

    def test_links_merge_per_subfield(self):
        profile, _ = merge_records(_kelsey_records())

        links = profile.links.value
        assert "linkedin.com" in (links.linkedin or "")
        assert "github.com" in (links.github or "")

    def test_candidate_id_is_surrogate_not_raw_email(self):
        profile, _ = merge_records(_kelsey_records())

        assert profile.candidate_id.startswith("cand_")
        assert "@" not in profile.candidate_id
        assert profile.candidate_id != "kelsey.hightower@gmail.com"

    def test_conflict_audit_event_is_emitted(self):
        _, audit_log = merge_records(_kelsey_records())

        events = [
            event for event in audit_log
            if event.field == "experience.title" and event.kind == "conflict_resolved"
        ]
        assert events
        assert events[0].details["winner_source"] == "ats_json"

    def test_merge_does_not_compute_overall_confidence(self):
        profile, _ = merge_records(_kelsey_records())

        assert profile.overall_confidence == 0.0

    def test_score_stage_computes_weighted_overall_confidence(self):
        merged, _ = merge_records(_kelsey_records())
        profile = score_profile(merged)

        assert 0.0 <= profile.overall_confidence <= 1.0
        assert profile.overall_confidence > 0.8
