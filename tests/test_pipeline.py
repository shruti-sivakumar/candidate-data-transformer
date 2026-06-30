"""Tests for the shared end-to-end pipeline helper."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.transformer.pipeline import PipelineInputs, run_pipeline

SAMPLES = Path("samples").resolve()


def test_run_pipeline_returns_default_projection_payload():
    payload = run_pipeline(
        PipelineInputs(
            csv_payload=(SAMPLES / "recruiter_csv/kelsey_hightower.csv").read_text(encoding="utf-8"),
            ats_payload=(SAMPLES / "ats_json/kelsey_hightower.json").read_text(encoding="utf-8"),
            github_payload=(SAMPLES / "github/kelsey_hightower.json").read_text(encoding="utf-8"),
        )
    )

    assert payload["full_name"] == "Kelsey Hightower"
    assert payload["candidate_id"].startswith("cand_")
    assert "overall_confidence" in payload
    assert "provenance" in payload


def test_run_pipeline_requires_at_least_one_source():
    with pytest.raises(ValueError, match="At least one source"):
        run_pipeline(PipelineInputs())
