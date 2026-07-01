# Multi-Source Candidate Data Transformer

This project turns candidate data from multiple sources into a canonical,
schema-valid JSON profile. It handles conflicting values, normalizes common
formats, deduplicates across sources, and records confidence and provenance so a
downstream consumer can see both the chosen value and why it was chosen.

Implemented source types:

- Structured: recruiter CSV, ATS JSON
- Unstructured: GitHub profile/repositories JSON, recruiter notes text

The core pipeline is:

```text
extract -> normalize -> merge -> score -> project -> validate
```

`project` is driven by a runtime JSON config, so callers can request a different
output shape without changing code.

## Repository Contents

```text
src/transformer/        Pipeline implementation
tests/                  Unit and integration tests
samples/                Sample inputs, custom config, and produced outputs
data/skills_taxonomy.csv Curated skill taxonomy used by skill canonicalization
streamlit_app.py        Thin UI wrapper around the same pipeline
scripts/threshold_probe.py Characterization script for the fuzzy skill threshold
```

## Setup

Requires Python 3.12+ and `uv`.

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

Recruiter-note skill extraction uses spaCy's small English model. Install it
once after syncing dependencies:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m spacy download en_core_web_sm
```

## Run The CLI

Run all four sample sources through the default projection:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m src.transformer.cli \
  --csv samples/recruiter_csv/kelsey_hightower.csv \
  --ats samples/ats_json/kelsey_hightower.json \
  --github samples/github/kelsey_hightower.json \
  --notes samples/recruiter_notes/kelsey_hightower.txt
```

Write the default output to a file:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m src.transformer.cli \
  --csv samples/recruiter_csv/kelsey_hightower.csv \
  --ats samples/ats_json/kelsey_hightower.json \
  --github samples/github/kelsey_hightower.json \
  --notes samples/recruiter_notes/kelsey_hightower.txt \
  --output samples/output/default_output.json
```

Run with a custom projection config:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m src.transformer.cli \
  --csv samples/recruiter_csv/kelsey_hightower.csv \
  --ats samples/ats_json/kelsey_hightower.json \
  --github samples/github/kelsey_hightower.json \
  --notes samples/recruiter_notes/kelsey_hightower.txt \
  --config samples/custom_config.json \
  --output samples/output/custom_output.json
```

Run a multi-candidate CSV batch:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m src.transformer.cli \
  --csv samples/recruiter_csv/multi_candidate.csv \
  --output samples/output/multi_candidate_output.json
```

Add `--include-audit` to include structured extract/normalize/merge audit events
in the returned payload.

## Produced Outputs

The repository includes generated outputs for review without rerunning the
pipeline:

- `samples/output/default_output.json`
- `samples/output/custom_output.json`
- `samples/output/multi_candidate_output.json`

The custom output uses `samples/custom_config.json`, which selects a subset of
canonical fields, renames several fields, and includes a confidence sidecar.

## Run The Streamlit Demo

The Streamlit UI is a thin wrapper over the same `run_pipeline` function used by
the CLI. It supports built-in sample bundles and custom uploads.

```bash
UV_CACHE_DIR=.uv-cache uv run streamlit run streamlit_app.py
```

## Run Tests

```bash
UV_CACHE_DIR=.uv-cache uv run pytest
```

Current suite size: 228 tests.

Coverage includes:

- Source parsing and graceful handling of malformed inputs
- Phone, date, email, country, URL, and location normalization
- Skill canonicalization and guarded fuzzy matching
- Merge conflict resolution, provenance, and confidence behavior
- Runtime projection config and schema validation
- Multi-candidate batches, including duplicate-name records that must not merge
- GitHub public email preservation when the API exposes one

## Design Notes

- Candidate grouping uses normalized email and normalized phone as the only strong
  pre-merge identity anchors. Name alone never merges two records.
- Sparse records without email or phone attach only when the input bundle is
  unambiguous; same-source unanchored duplicate rows stay separate.
- GitHub fixture replay is used for deterministic sample runs. Live capture code
  exists in `src/transformer/ingest.py`, but committed samples are local JSON.
- Recruiter notes are treated as supplementary unstructured evidence. The notes
  adapter extracts contacts, URLs, and skills; free-text name, education, and work
  history extraction are intentionally out of scope for this MVP.
- Unknown or low-confidence values are left empty or lowered in confidence rather
  than invented.
