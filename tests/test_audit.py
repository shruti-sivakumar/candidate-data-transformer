"""Tests for structured audit sidecars across extract and normalize."""
from __future__ import annotations

from src.transformer.models import RawRecord
from src.transformer.normalize.orchestrator import normalize_record_with_audit
from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource


def test_csv_extract_with_audit_reports_empty_payload():
    records, audit_log = CSVSource().extract_with_audit("")

    assert records == []
    assert any(
        event.stage == "extract"
        and event.kind == "source_empty"
        and event.reason == "empty_payload"
        for event in audit_log
    )


def test_normalize_with_audit_distinguishes_missing_vs_dropped_email():
    _, missing_log = normalize_record_with_audit(
        RawRecord(source="recruiter_csv", raw_fields={})
    )
    _, dropped_log = normalize_record_with_audit(
        RawRecord(source="recruiter_csv", raw_fields={"email": "not-an-email"})
    )

    assert any(
        event.field == "emails"
        and event.kind == "field_missing"
        for event in missing_log
    )
    assert any(
        event.field == "emails"
        and event.kind == "value_dropped"
        and event.reason == "failed_normalization"
        for event in dropped_log
    )


def test_github_fork_drop_emits_structured_audit_event():
    record = RawRecord(
        source="github",
        raw_fields={
            "profile": {},
            "repos": [
                {"name": "forked", "language": "Rust", "fork": True, "html_url": "https://github.com/x/forked"}
            ],
        },
    )

    _, audit_log = normalize_record_with_audit(record)

    assert any(
        event.field == "projects"
        and event.kind == "entry_dropped"
        and event.reason == "fork_repo_excluded"
        for event in audit_log
    )


def test_unknown_source_returns_audit_event():
    normalized, audit_log = normalize_record_with_audit(
        RawRecord(source="mystery", raw_fields={})
    )

    assert normalized is None
    assert any(
        event.kind == "source_failed"
        and event.reason == "unknown_source"
        for event in audit_log
    )


def test_ats_extract_with_audit_reports_non_object_entries():
    records, audit_log = ATSSource().extract_with_audit('[{"id": 1}, "oops"]')

    assert len(records) == 1
    assert any(
        event.stage == "extract"
        and event.kind == "entry_dropped"
        and event.reason == "non_object_candidate"
        for event in audit_log
    )
