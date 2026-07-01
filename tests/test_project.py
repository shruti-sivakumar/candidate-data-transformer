"""Tests for Module 6 projection and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.transformer.ingest import read_file
from src.transformer.merge import merge_records
from src.transformer.normalize.orchestrator import normalize_record
from src.transformer.project import (
    ProjectionConfig,
    ProjectionError,
    build_json_schema,
    project_profile,
)
from src.transformer.score import score_profile
from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.github_source import GitHubSource
from src.transformer.sources.notes_source import NotesSource

SAMPLES = Path("samples")


def _normalize(adapter, rel_path: str):
    records = adapter.extract(read_file(SAMPLES / rel_path))
    assert records
    normalized = normalize_record(records[0])
    assert normalized is not None
    return normalized


def _kelsey_profile():
    merged, _ = merge_records(
        [
            _normalize(CSVSource(), "recruiter_csv/kelsey_hightower.csv"),
            _normalize(ATSSource(), "ats_json/kelsey_hightower.json"),
            _normalize(GitHubSource(), "github/kelsey_hightower.json"),
            _normalize(NotesSource(skill_vocabulary=[]), "recruiter_notes/kelsey_hightower.txt"),
        ]
    )
    return score_profile(merged)


def test_project_subset_and_path_remap():
    profile = _kelsey_profile()
    config = {
        "fields": [
            {"name": "name", "path": "full_name", "type": "string"},
            {"name": "primary_email", "path": "emails[0]", "type": "string"},
            {"name": "skill_names", "path": "skills[].name", "type": "array"},
        ]
    }

    projected = project_profile(profile, config)

    assert projected["name"] == "Kelsey Hightower"
    assert projected["primary_email"] == "kelsey.hightower@gmail.com"
    assert "Go" in projected["skill_names"]


def test_project_missing_null_and_omit_policies():
    profile = _kelsey_profile()
    config = {
        "fields": [
            {"name": "headline", "path": "headline", "type": "string", "on_missing": "null"},
            {"name": "missing_field", "path": "does.not.exist", "type": "string", "on_missing": "omit"},
        ]
    }

    projected = project_profile(profile, config)

    assert "headline" in projected
    assert "missing_field" not in projected


def test_project_missing_error_policy_raises():
    profile = _kelsey_profile()
    config = {
        "fields": [
            {"name": "missing_field", "path": "does.not.exist", "type": "string", "on_missing": "error"},
        ]
    }

    with pytest.raises(ProjectionError):
        project_profile(profile, config)


def test_project_validation_rejects_type_mismatch():
    profile = _kelsey_profile()
    config = {
        "fields": [
            {"name": "bad_name", "path": "full_name", "type": "number"},
        ]
    }

    with pytest.raises(ProjectionError):
        project_profile(profile, config)


def test_project_can_include_confidence_and_provenance():
    profile = _kelsey_profile()
    config = ProjectionConfig.model_validate(
        {
            "fields": [
                {"name": "name", "path": "full_name", "type": "string"},
                {"name": "skill_names", "path": "skills[].name", "type": "array"},
            ],
            "include_confidence": True,
            "include_provenance": True,
        }
    )

    projected = project_profile(profile, config)

    assert "confidence" in projected
    assert "provenance" in projected
    assert sorted(projected["confidence"]) == ["name", "skill_names"]
    assert projected["confidence"]["name"] == profile.full_name.confidence
    assert projected["confidence"]["skill_names"] == [skill.confidence for skill in profile.skills]
    assert isinstance(projected["provenance"], list)


def test_project_confidence_sidecar_uses_output_field_names_only():
    profile = _kelsey_profile()
    config = {
        "fields": [
            {"name": "primary_email", "path": "emails[0]", "type": "string"},
            {"name": "city", "path": "location.city", "type": "string", "nullable": True},
        ],
        "include_confidence": True,
    }

    projected = project_profile(profile, config)

    assert sorted(projected["confidence"]) == ["city", "primary_email"]
    assert projected["confidence"]["primary_email"] == profile.emails[0].confidence
    assert projected["confidence"]["city"] == profile.location.confidence


def test_project_accepts_ps_example_vocabulary_verbatim():
    # The PS's own example config, reproduced exactly: "path" as output name,
    # "from" as source path, "required": true, "type": "string[]", per-field
    # "normalize", and a top-level "on_missing" default.
    profile = _kelsey_profile()
    config = {
        "fields": [
            {"path": "full_name", "type": "string", "required": True},
            {"path": "primary_email", "from": "emails[0]", "type": "string", "required": True},
            {"path": "phone", "from": "phones[0]", "type": "string", "normalize": "E164"},
            {"path": "skills", "from": "skills[].name", "type": "string[]", "normalize": "canonical"},
        ],
        "on_missing": "null",
    }

    projected = project_profile(profile, config)

    assert projected["full_name"] == "Kelsey Hightower"
    assert projected["primary_email"] == "kelsey.hightower@gmail.com"
    assert projected["phone"] == "+12025550142"
    assert "Go" in projected["skills"]


def test_ps_vocabulary_normalizes_to_internal_name_and_path():
    config = ProjectionConfig.model_validate(
        {
            "fields": [
                {"path": "full_name", "type": "string", "required": True},
                {"path": "primary_email", "from": "emails[0]", "type": "string"},
                {"path": "skills", "from": "skills[].name", "type": "string[]"},
            ],
            "on_missing": "omit",
        }
    )

    by_name = {field.name: field for field in config.fields}

    # "path" without "from" is both output name and source path.
    assert by_name["full_name"].path == "full_name"
    # "path"/"from" split: output name from "path", source path from "from".
    assert by_name["primary_email"].path == "emails[0]"
    # "string[]" collapses to an array type.
    assert by_name["skills"].type == "array"
    # "required": true -> error; other fields inherit the top-level default.
    assert by_name["full_name"].on_missing == "error"
    assert by_name["primary_email"].on_missing == "omit"


def test_ps_top_level_on_missing_is_overridden_by_per_field():
    config = ProjectionConfig.model_validate(
        {
            "fields": [
                {"path": "headline", "type": "string", "on_missing": "null"},
                {"path": "note", "from": "does.not.exist", "type": "string"},
            ],
            "on_missing": "omit",
        }
    )

    by_name = {field.name: field for field in config.fields}
    assert by_name["headline"].on_missing == "null"
    assert by_name["note"].on_missing == "omit"


def test_build_json_schema_marks_error_fields_required():
    config = ProjectionConfig.model_validate(
        {
            "fields": [
                {"name": "name", "path": "full_name", "type": "string", "on_missing": "error"},
                {"name": "headline", "path": "headline", "type": "string", "on_missing": "null"},
            ]
        }
    )

    schema = build_json_schema(config)

    assert schema["required"] == ["name"]
