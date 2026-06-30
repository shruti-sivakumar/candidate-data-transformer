"""Tests for the end-to-end CLI."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from src.transformer.cli import app

runner = CliRunner()
SAMPLES = Path("samples").resolve()


def test_cli_transform_uses_default_projection():
    result = runner.invoke(
        app,
        [
            "--csv",
            str(SAMPLES / "recruiter_csv/kelsey_hightower.csv"),
            "--ats",
            str(SAMPLES / "ats_json/kelsey_hightower.json"),
            "--github",
            str(SAMPLES / "github/kelsey_hightower.json"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["full_name"] == "Kelsey Hightower"
    assert payload["candidate_id"].startswith("cand_")
    assert "overall_confidence" in payload
    assert "provenance" in payload


def test_cli_transform_can_include_audit_log():
    result = runner.invoke(
        app,
        [
            "--csv",
            str(SAMPLES / "recruiter_csv/kelsey_hightower.csv"),
            "--notes",
            str(SAMPLES / "recruiter_notes/kelsey_hightower.txt"),
            "--include-audit",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert "output" in payload
    assert "audit_log" in payload
    assert isinstance(payload["audit_log"], list)
