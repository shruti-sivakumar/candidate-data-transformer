"""Module 1 — Ingest. The only module allowed to touch the filesystem or network.

Reads raw bytes from local files, or fetches from the GitHub API. Returns
strings (or raw JSON text) — no parsing, no normalization. Downstream
modules consume this output and never see I/O themselves.
"""

from __future__ import annotations

import json
from pathlib import Path


class IngestError(Exception):
    """Raised when a source cannot be read.

    Callers (source adapters) should catch this and degrade gracefully — log
    the failure, return an empty result, let the run continue on other sources.
    """


def read_file(path: Path) -> str:
    """Read a local file as UTF-8 text and return its contents as a string.

    Source-agnostic: the file's *content* (CSV vs JSON vs prose) is the
    adapter's concern in Module 2, not this function's.

    Raises:
        IngestError: if the file doesn't exist, isn't readable, or isn't
            valid UTF-8. The caller decides whether to skip the source or
            propagate the failure.
    """
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as e:
        raise IngestError(f"File not found: {path}") from e
    except PermissionError as e:
        raise IngestError(f"Permission denied: {path}") from e
    except UnicodeDecodeError as e:
        raise IngestError(f"File is not valid UTF-8: {path}") from e


GITHUB_API_BASE = "https://api.github.com"


def fetch_github_profile(
    username: str,
    fixture_path: Path,
    *,
    live: bool = False,
    timeout_seconds: float = 10.0,
) -> str:
    """Fetch a GitHub user profile + their public repos as a single JSON string.

    Two modes:

    - ``live=True``: hits the GitHub REST API, combines /users/{u} and
      /users/{u}/repos into one JSON document, writes it to ``fixture_path``,
      and returns it.
    - ``live=False`` (default): reads and returns the contents of
      ``fixture_path``. This is the mode used in tests, the demo, and the
      reproducible run path.

    The fixture-replay design is what makes the GitHub source deterministic:
    once the fixture is captured, every subsequent run produces byte-identical
    output regardless of network state or upstream profile drift.

    Raises:
        IngestError: in live mode, on network/HTTP failure. In replay mode,
            if the fixture file is missing or unreadable.
    """
    if live:
        # Lazy import — replay mode doesn't need `requests` installed.
        import requests

        try:
            profile_resp = requests.get(
                f"{GITHUB_API_BASE}/users/{username}", timeout=timeout_seconds
            )
            profile_resp.raise_for_status()

            repos_resp = requests.get(
                f"{GITHUB_API_BASE}/users/{username}/repos", timeout=timeout_seconds
            )
            repos_resp.raise_for_status()
        except requests.RequestException as e:
            raise IngestError(f"GitHub API call failed for {username}: {e}") from e

        combined = {
            "profile": profile_resp.json(),
            "repos": repos_resp.json(),
        }

        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(combined, indent=2, sort_keys=True)
        fixture_path.write_text(payload, encoding="utf-8")
        return payload

    # Replay mode
    return read_file(fixture_path)