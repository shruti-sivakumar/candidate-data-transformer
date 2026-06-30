"""CLI entry point for the candidate data transformer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from src.transformer.ingest import read_file
from src.transformer.merge import merge_records
from src.transformer.normalize.orchestrator import normalize_record_with_audit
from src.transformer.project import ProjectionConfig, default_projection_config, project_profile
from src.transformer.score import score_profile
from src.transformer.sources.ats_source import ATSSource
from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.github_source import GitHubSource
from src.transformer.sources.notes_source import NotesSource

app = typer.Typer(add_completion=False, no_args_is_help=True)


def _load_projection_config(config_path: Path | None) -> ProjectionConfig:
    """Load a projection config file, or fall back to the default projection."""
    if config_path is None:
        return default_projection_config()
    payload = json.loads(read_file(config_path))
    return ProjectionConfig.model_validate(payload)


def _extract_and_normalize(
    adapter: object,
    path: Path,
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Run extract + normalize for one source, returning records and audit rows."""
    payload = read_file(path)
    extract_with_audit = getattr(adapter, "extract_with_audit")
    raw_records, audit_log = extract_with_audit(payload)
    normalized = []
    for record in raw_records:
        item, normalize_audit = normalize_record_with_audit(record)
        audit_log.extend(normalize_audit)
        if item is not None:
            normalized.append(item)
    return normalized, [event.model_dump(mode="python") for event in audit_log]


@app.command()
def transform(
    csv_path: Path | None = typer.Option(None, "--csv", exists=True, dir_okay=False),
    ats_path: Path | None = typer.Option(None, "--ats", exists=True, dir_okay=False),
    github_path: Path | None = typer.Option(None, "--github", exists=True, dir_okay=False),
    notes_path: Path | None = typer.Option(None, "--notes", exists=True, dir_okay=False),
    config_path: Path | None = typer.Option(None, "--config", exists=True, dir_okay=False),
    output_path: Path | None = typer.Option(None, "--output", dir_okay=False),
    include_audit: bool = typer.Option(False, "--include-audit"),
) -> None:
    """Run the full pipeline for one candidate bundle and emit projected JSON."""
    normalized_records = []
    audit_log: list[dict[str, Any]] = []

    if csv_path:
        records, audit = _extract_and_normalize(CSVSource(), csv_path)
        normalized_records.extend(records)
        audit_log.extend(audit)
    if ats_path:
        records, audit = _extract_and_normalize(ATSSource(), ats_path)
        normalized_records.extend(records)
        audit_log.extend(audit)
    if github_path:
        records, audit = _extract_and_normalize(GitHubSource(), github_path)
        normalized_records.extend(records)
        audit_log.extend(audit)
    if notes_path:
        records, audit = _extract_and_normalize(NotesSource(skill_vocabulary=[]), notes_path)
        normalized_records.extend(records)
        audit_log.extend(audit)

    if not normalized_records:
        raise typer.BadParameter("At least one source must produce a normalized record.")

    merged, merge_audit = merge_records(normalized_records)
    scored = score_profile(merged)
    config = _load_projection_config(config_path)
    projected = project_profile(scored, config)

    result: dict[str, Any]
    if include_audit:
        result = {
            "output": projected,
            "audit_log": audit_log + [event.model_dump(mode="python") for event in merge_audit],
        }
    else:
        result = projected

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        typer.echo(rendered)


if __name__ == "__main__":
    app()
