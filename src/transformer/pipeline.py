"""Shared end-to-end pipeline runner for CLI and UI surfaces."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.transformer.merge import group_records_by_candidate, merge_records
from src.transformer.models import NormalizedRecord
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
) -> tuple[list[tuple[NormalizedRecord, list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Run extract + normalize for one source.

    Returns ``(record_pairs, shared_audit)`` where each pair carries a normalized
    record and its own normalize-stage audit rows, and ``shared_audit`` holds the
    input-level events (extract parsing, plus the audit of records that dropped
    out during normalization) that belong to no single surviving candidate.
    """
    extract_with_audit = getattr(adapter, "extract_with_audit")
    raw_records, extract_audit = extract_with_audit(payload)
    shared_audit = [event.model_dump(mode="python") for event in extract_audit]
    record_pairs: list[tuple[NormalizedRecord, list[dict[str, Any]]]] = []
    for record in raw_records:
        item, normalize_audit = normalize_record_with_audit(record)
        audit_rows = [event.model_dump(mode="python") for event in normalize_audit]
        if item is not None:
            record_pairs.append((item, audit_rows))
        else:
            shared_audit.extend(audit_rows)
    return record_pairs, shared_audit


def run_pipeline(
    inputs: PipelineInputs,
    *,
    include_audit: bool = False,
) -> dict[str, Any] | list[dict[str, Any]]:
    """Run the full pipeline for one or more candidates.

    Records are grouped per candidate BEFORE merge, then the unchanged
    extract→normalize→merge→score→project engine runs once per group. The return
    is a single object when exactly one candidate is detected (unchanged shape),
    or a JSON array of such objects when a batch (e.g. a multi-row recruiter CSV)
    resolves to multiple distinct candidates.
    """
    record_pairs: list[tuple[NormalizedRecord, list[dict[str, Any]]]] = []
    shared_audit: list[dict[str, Any]] = []

    for adapter, payload in (
        (CSVSource(), inputs.csv_payload),
        (ATSSource(), inputs.ats_payload),
        (GitHubSource(), inputs.github_payload),
        (NotesSource(), inputs.notes_payload),
    ):
        if payload:
            pairs, audit = _extract_and_normalize(adapter, payload)
            record_pairs.extend(pairs)
            shared_audit.extend(audit)

    if not record_pairs:
        raise ValueError("At least one source must produce a normalized record.")

    records = [record for record, _ in record_pairs]
    config = _load_projection_config(inputs.config_payload)
    groups = group_records_by_candidate(records)

    results: list[dict[str, Any]] = []
    for group in groups:
        group_records = [records[i] for i in group]
        merged, merge_audit = merge_records(group_records)
        scored = score_profile(merged)
        projected = project_profile(scored, config)
        if include_audit:
            group_audit = [row for i in group for row in record_pairs[i][1]]
            merge_rows = [event.model_dump(mode="python") for event in merge_audit]
            results.append(
                {
                    "output": projected,
                    "audit_log": shared_audit + group_audit + merge_rows,
                }
            )
        else:
            results.append(projected)

    # Exactly one candidate keeps the historical single-object shape; only a
    # genuine multi-candidate batch widens to a list.
    if len(results) == 1:
        return results[0]
    return results


def read_text(path: Path) -> str:
    """Read a UTF-8 text file from disk."""
    return path.read_text(encoding="utf-8")
