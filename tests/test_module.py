"""Tests for the canonical data models (Module 0)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.transformer.models import (
    AggregatedValue,
    CanonicalProfile,
    Links,
    Location,
    MergedEducationEntry,
    MergedExperienceEntry,
    ProjectEntry,
    RawRecord,
    SkillEntry,
    Source,
    TrackedValue,
)


# ---------------------------------------------------------------------------
# TrackedValue
# ---------------------------------------------------------------------------


def test_tracked_value_constructs_and_exposes_fields():
    tv = TrackedValue[str](
        value="Priya Sharma", source="ats_json", method="direct", confidence=0.9
    )
    assert tv.value == "Priya Sharma"
    assert tv.source == "ats_json"
    assert tv.method == "direct"
    assert tv.confidence == 0.9


def test_tracked_value_is_frozen():
    """Once created, fields can't be reassigned — guards canonical-record immutability."""
    tv = TrackedValue[str](value="x", source="ats_json", method="direct", confidence=0.5)
    with pytest.raises(ValidationError):
        tv.confidence = 0.9  # type: ignore[misc]


def test_tracked_value_rejects_invalid_method():
    """The Literal type should reject any method outside the allowed set."""
    with pytest.raises(ValidationError):
        TrackedValue[str](
            value="x", source="ats_json", method="infered", confidence=0.5  # type: ignore[arg-type]
        )


def test_tracked_value_holds_complex_types():
    """Generic should accept nested BaseModels, not just primitives."""
    loc = Location(city="Bangalore", country="IN")
    tv = TrackedValue[Location](
        value=loc, source="ats_json", method="direct", confidence=0.9
    )
    assert tv.value.city == "Bangalore"
    assert tv.value.country == "IN"


# ---------------------------------------------------------------------------
# RawRecord
# ---------------------------------------------------------------------------


def test_raw_record_keeps_source_vocabulary():
    """raw_fields stays loose — adapters don't normalize during extraction."""
    rec = RawRecord(
        source="ats_json",
        raw_fields={"givenName": "Priya", "familyName": "Sharma", "yrs": 5},
    )
    assert rec.source == "ats_json"
    assert rec.raw_fields["givenName"] == "Priya"
    assert rec.raw_fields["yrs"] == 5  # heterogeneous types allowed


# ---------------------------------------------------------------------------
# Source protocol
# ---------------------------------------------------------------------------


def test_source_protocol_matches_structurally():
    """A class with the right shape satisfies Source without inheriting from it."""

    class FakeAdapter:
        name = "fake"
        trust = 0.5

        def extract(self, payload: str) -> list[RawRecord]:
            return [RawRecord(source="fake", raw_fields={})]

    adapter = FakeAdapter()
    assert isinstance(adapter, Source)


def test_source_protocol_rejects_wrong_shape():
    """A class missing required attributes is not a Source."""

    class Incomplete:
        name = "bad"
        # missing trust, missing extract

    assert not isinstance(Incomplete(), Source)


# ---------------------------------------------------------------------------
# CanonicalProfile
# ---------------------------------------------------------------------------


def _make_profile(**overrides) -> CanonicalProfile:
    """Construct a minimal valid CanonicalProfile, with optional overrides."""
    defaults = dict(
        candidate_id="cand_001",
        full_name=TrackedValue[str](
            value="Priya Sharma", source="ats_json", method="direct", confidence=0.9
        ),
        emails=[
            AggregatedValue[str](
                value="priya@x.com", confidence=0.97, sources=["ats_json", "recruiter_csv"]
            )
        ],
        phones=[],
        location=TrackedValue[Location](
            value=Location(city="Bangalore", country="IN"),
            source="ats_json",
            method="direct",
            confidence=0.9,
        ),
        links=TrackedValue[Links](
            value=Links(), source="ats_json", method="direct", confidence=0.5
        ),
        headline=TrackedValue[str | None](
            value=None, source="ats_json", method="direct", confidence=0.5
        ),
        years_experience=TrackedValue[float | None](
            value=None, source="ats_json", method="direct", confidence=0.5
        ),
        skills=[
            SkillEntry(name="Python", confidence=0.97, sources=["ats_json", "github"])
        ],
        experience=[
            MergedExperienceEntry(
                company="Stripe",
                title="Engineer",
                start="2021-03",
                confidence=0.91,
                sources=["ats_json", "recruiter_csv"],
            )
        ],
        education=[
            MergedEducationEntry(
                institution="Georgia Tech",
                degree="B.S.",
                field="CS",
                end_year=2007,
                confidence=0.9,
                sources=["ats_json"],
            )
        ],
        projects=[
            ProjectEntry(
                name="edgemind",
                primary_language="Python",
                confidence=0.7,
                sources=["github"],
            )
        ],
        overall_confidence=0.85,
    )
    defaults.update(overrides)
    return CanonicalProfile(**defaults)


def test_canonical_profile_constructs():
    p = _make_profile()
    assert p.candidate_id == "cand_001"
    assert p.full_name.value == "Priya Sharma"
    assert p.skills[0].name == "Python"
    assert p.projects[0].primary_language == "Python"


def test_canonical_profile_is_frozen():
    p = _make_profile()
    with pytest.raises(ValidationError):
        p.overall_confidence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# get_provenance
# ---------------------------------------------------------------------------


def test_provenance_includes_all_populated_fields():
    p = _make_profile()
    prov = p.get_provenance()
    fields_seen = {row["field"] for row in prov}
    # Every field that has a value should appear at least once
    expected = {
        "full_name", "emails", "location", "links", "headline",
        "years_experience", "skills", "experience", "education", "projects",
    }
    assert expected.issubset(fields_seen)


def test_provenance_emits_one_row_per_source_for_aggregated_fields():
    """A skill from two sources produces two provenance rows."""
    p = _make_profile()
    skill_rows = [row for row in p.get_provenance() if row["field"] == "skills"]
    assert len(skill_rows) == 2
    sources = {row["source"] for row in skill_rows}
    assert sources == {"ats_json", "github"}
    # Aggregated fields use method='merged'
    assert all(row["method"] == "merged" for row in skill_rows)


def test_provenance_is_sorted_deterministically():
    """Two calls on the same profile produce byte-identical output."""
    p = _make_profile()
    assert p.get_provenance() == p.get_provenance()
    # Also: it's actually sorted
    prov = p.get_provenance()
    keys = [(row["field"], row["source"]) for row in prov]
    assert keys == sorted(keys)
