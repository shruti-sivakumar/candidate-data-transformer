"""Tests for Module 5 scoring."""
from __future__ import annotations

from pathlib import Path

from src.transformer.merge import merge_records
from src.transformer.score import field_family_confidence, overall_confidence, score_profile
from src.transformer.normalize.orchestrator import normalize_record
from src.transformer.ingest import read_file
from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.github_source import GitHubSource

SAMPLES = Path("samples")


def _normalize(adapter, rel_path: str):
    records = adapter.extract(read_file(SAMPLES / rel_path))
    assert records
    normalized = normalize_record(records[0])
    assert normalized is not None
    return normalized


def _kelsey_core_records():
    return [
        _normalize(CSVSource(), "recruiter_csv/kelsey_hightower.csv"),
        _normalize(ATSSource(), "ats_json/kelsey_hightower.json"),
        _normalize(GitHubSource(), "github/kelsey_hightower.json"),
    ]


def test_field_family_confidence_only_contains_populated_families():
    merged, _ = merge_records(_kelsey_core_records())

    scores = field_family_confidence(merged)

    assert scores
    assert all(score > 0.0 for score in scores.values())
    assert "overall_confidence" not in scores


def test_score_profile_populates_overall_confidence():
    merged, _ = merge_records(_kelsey_core_records())

    scored = score_profile(merged)

    assert scored.overall_confidence == overall_confidence(merged)
    assert scored.overall_confidence > 0.0
