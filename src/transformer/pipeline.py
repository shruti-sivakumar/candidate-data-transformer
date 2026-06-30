"""Shared end-to-end pipeline runner for CLI and UI surfaces."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.transformer.merge import merge_records
from src.transformer.normalize.orchestrator import normalize_record_with_audit
from src.transformer.project import ProjectionConfig, default_projection_config, project_profile
from src.transformer.score import score_profile
from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.github_source import GitHubSource
from src.transformer.sources.notes_source import NotesSource


@dataclass(frozen=True)
class PipelineInputs:
    """Raw source payloads for one candidate bundle."""

    csv_payload: str | None = None
    ats_payload: str | None = None
    github_payload: str | None = None
    notes_payload: str | None = None
    config_payload: str | None = None


def _load_projection_config(config_payload: str | None) -> ProjectionConfig:
    """Load a projection config payload, or fall back to the default projection."""
    if config_payload is None:
        return default_projection_config()
    return ProjectionConfig.model_validate(json.loads(config_payload))


def _extract_and_normalize(
    adapter: object,
    payload: str,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Run extract + normalize for one source, returning records and audit rows."""
    extract_with_audit = getattr(adapter, "extract_with_audit")
    raw_records, audit_log = extract_with_audit(payload)
    normalized = []
    for record in raw_records:
        item, normalize_audit = normalize_record_with_audit(record)
        audit_log.extend(normalize_audit)
        if item is not None:
            normalized.append(item)
    return normalized, [event.model_dump(mode="python") for event in audit_log]


def run_pipeline(
    inputs: PipelineInputs,
    *,
    include_audit: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline and return the projected output payload."""
    normalized_records = []
    audit_log: list[dict[str, Any]] = []

    if inputs.csv_payload:
        records, audit = _extract_and_normalize(CSVSource(), inputs.csv_payload)
        normalized_records.extend(records)
        audit_log.extend(audit)
    if inputs.ats_payload:
        records, audit = _extract_and_normalize(ATSSource(), inputs.ats_payload)
        normalized_records.extend(records)
        audit_log.extend(audit)
    if inputs.github_payload:
        records, audit = _extract_and_normalize(GitHubSource(), inputs.github_payload)
        normalized_records.extend(records)
        audit_log.extend(audit)
    if inputs.notes_payload:
        records, audit = _extract_and_normalize(
            NotesSource(skill_vocabulary=[]),
            inputs.notes_payload,
        )
        normalized_records.extend(records)
        audit_log.extend(audit)

    if not normalized_records:
        raise ValueError("At least one source must produce a normalized record.")

    merged, merge_audit = merge_records(normalized_records)
    scored = score_profile(merged)
    config = _load_projection_config(inputs.config_payload)
    projected = project_profile(scored, config)

    if include_audit:
        return {
            "output": projected,
            "audit_log": audit_log + [event.model_dump(mode="python") for event in merge_audit],
        }
    return projected


def read_text(path: Path) -> str:
    """Read a UTF-8 text file from disk."""
    return path.read_text(encoding="utf-8")
