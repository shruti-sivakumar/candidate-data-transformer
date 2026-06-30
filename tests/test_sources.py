"""Regression tests for Module 2 source adapters.

Targeted pins for the two fixes applied in this session:
  1. CSV restkey — over-wide rows must survive (not be dropped).
  2. Skill punctuation lookaround — C++, C#, .NET must be recognized;
     "Go" must not fire inside "Google".
"""
from __future__ import annotations

import pytest

from src.transformer.sources.csv_source import CSVSource
from src.transformer.sources.notes_source import NotesSource


# ---------------------------------------------------------------------------
# CSVSource — restkey regression
# ---------------------------------------------------------------------------


def test_csv_over_wide_row_is_preserved_not_dropped():
    """A row with more fields than the header must produce a RawRecord, not []."""
    src = CSVSource()
    payload = "name,email\nKelsey,k@x.com,EXTRA_VALUE"
    records = src.extract(payload)
    assert len(records) == 1, "whole-row drop regression: expected 1 record, got 0"
    assert records[0].raw_fields["name"] == "Kelsey"
    assert records[0].raw_fields["email"] == "k@x.com"


def test_csv_over_wide_row_overflow_lands_in_extra():
    """Overflow values must bucket into '_extra', not silently vanish."""
    src = CSVSource()
    payload = "name,email\nKelsey,k@x.com,EXTRA_VALUE"
    rf = src.extract(payload)[0].raw_fields
    assert "_extra" in rf
    assert rf["_extra"] == ["EXTRA_VALUE"]


def test_csv_normal_row_has_no_extra_key():
    """When column count matches header, '_extra' must not appear in raw_fields."""
    src = CSVSource()
    payload = "name,email\nKelsey,k@x.com"
    rf = src.extract(payload)[0].raw_fields
    assert "_extra" not in rf


# ---------------------------------------------------------------------------
# NotesSource — skill punctuation lookaround regression
# ---------------------------------------------------------------------------

_PUNCT_VOCAB = ["C++", "C#", ".NET", "Node.js", "Go", "Python"]
_PUNCT_TEXT = (
    "Strong in C++ and C#, did some .NET and Node.js. Python and Go too. Works at Google."
)


def test_skills_cxx_recognized():
    src = NotesSource(skill_vocabulary=_PUNCT_VOCAB)
    assert "C++" in src._extract_skills(_PUNCT_TEXT)


def test_skills_csharp_recognized():
    src = NotesSource(skill_vocabulary=_PUNCT_VOCAB)
    assert "C#" in src._extract_skills(_PUNCT_TEXT)


def test_skills_dotnet_recognized():
    src = NotesSource(skill_vocabulary=_PUNCT_VOCAB)
    assert ".NET" in src._extract_skills(_PUNCT_TEXT)


def test_skills_go_does_not_match_inside_google():
    """'Go' in the vocabulary must not fire on 'Google'."""
    src = NotesSource(skill_vocabulary=["Go"])
    assert src._extract_skills("Works at Google.") == []


def test_skills_unknown_skill_not_recognized():
    """A term absent from the vocabulary must never appear in output."""
    src = NotesSource(skill_vocabulary=["Python"])
    assert "Rust" not in src._extract_skills("Expert in Rust and Python.")
