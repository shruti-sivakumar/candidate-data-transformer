"""CLI entry point for the candidate data transformer."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from src.transformer.pipeline import PipelineInputs, read_text, run_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)


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
    try:
        result = run_pipeline(
            PipelineInputs(
                csv_payload=read_text(csv_path) if csv_path else None,
                ats_payload=read_text(ats_path) if ats_path else None,
                github_payload=read_text(github_path) if github_path else None,
                notes_payload=read_text(notes_path) if notes_path else None,
                config_payload=read_text(config_path) if config_path else None,
            ),
            include_audit=include_audit,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    rendered = json.dumps(result, indent=2, sort_keys=True)
    if output_path is not None:
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        typer.echo(rendered)


if __name__ == "__main__":
    app()
