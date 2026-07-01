# candidate-data-transformer

## Setup

Install dependencies with `uv`:

```bash
UV_CACHE_DIR=.uv-cache uv sync
```

The skills linker uses spaCy's small English POS tagger for recruiter-note
candidate filtering. Runtime code expects `spacy==3.8.14` and
`en-core-web-sm==3.8.0`; the model is installed with spaCy's downloader:

```bash
UV_CACHE_DIR=.uv-cache uv run python -m spacy download en_core_web_sm
```
