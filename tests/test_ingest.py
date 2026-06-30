"""Tests for the ingest layer (Module 1)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.transformer.ingest import (
    IngestError,
    fetch_github_profile,
    read_file,
)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


def test_read_file_returns_contents(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("hello world", encoding="utf-8")
    assert read_file(f) == "hello world"


def test_read_file_raises_ingest_error_when_missing(tmp_path: Path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(IngestError) as info:
        read_file(missing)
    assert "File not found" in str(info.value)


def test_read_file_raises_ingest_error_on_non_utf8(tmp_path: Path):
    """A file with non-UTF-8 bytes should fail loudly with IngestError, not silently."""
    f = tmp_path / "bad.txt"
    # 0xFF is not valid UTF-8 in any position; write_bytes bypasses encoding.
    f.write_bytes(b"\xff\xfe\xfd")
    with pytest.raises(IngestError) as info:
        read_file(f)
    assert "not valid UTF-8" in str(info.value)


def test_read_file_preserves_original_exception_as_cause(tmp_path: Path):
    """Original exception chains via `from e` — critical for debugging."""
    missing = tmp_path / "nope.txt"
    with pytest.raises(IngestError) as info:
        read_file(missing)
    assert isinstance(info.value.__cause__, FileNotFoundError)


# ---------------------------------------------------------------------------
# fetch_github_profile — replay mode only
# ---------------------------------------------------------------------------


def test_github_replay_returns_fixture_contents(tmp_path: Path):
    fixture = tmp_path / "user.json"
    fixture.write_text('{"profile": {"login": "test"}, "repos": []}', encoding="utf-8")
    result = fetch_github_profile("test", fixture)  # live=False by default
    assert "test" in result
    assert "repos" in result


def test_github_replay_raises_when_fixture_missing(tmp_path: Path):
    missing = tmp_path / "no_such_fixture.json"
    with pytest.raises(IngestError):
        fetch_github_profile("anyone", missing)