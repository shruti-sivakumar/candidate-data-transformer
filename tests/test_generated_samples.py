"""Generalization tests over synthetic candidate bundles.

The shipped fixtures only ever exercise two candidates (Kelsey, Andrej). These
tests run several NEW synthetic bundles — different in shape — through the exact
runner the CLI delegates to (`run_pipeline`, see cli.py), to prove the pipeline
generalizes to other valid inputs *within the current taxonomy's coverage*.

Design rules (deliberate):
  * Every skill mentioned in every sample is an in-taxonomy term, so the taxonomy
    itself is never the thing under test — generalization is.
  * Assertions are GENERAL properties (no crash; recognized skills are canonical
    in-taxonomy names; no adjective/stray-initial false positives; missing/empty
    sources degrade to empties, not exceptions). We do NOT hand-copy exact output
    skill sets, because the post-A3 notes path is corroboration- and POS-gated:
    a skill in loose prose ("fluent in Go") may be deliberately dropped while the
    same skill in a corroborated list ("Skills: ...") is kept. Baking exact sets
    would re-encode fixtures rather than prove generalization.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.transformer.pipeline import PipelineInputs, read_text, run_pipeline

SAMPLES = Path("samples")
TAXONOMY_PATH = Path("data/skills_taxonomy.csv")


def _canonical_skill_names() -> frozenset[str]:
    """The set of canonical skill names (column 1) the taxonomy can emit."""
    with TAXONOMY_PATH.open(newline="", encoding="utf-8-sig") as handle:
        names = {
            (row.get("canonical_skill") or "").strip()
            for row in csv.DictReader(handle)
        }
    return frozenset(name for name in names if name)


CANONICAL_SKILLS = _canonical_skill_names()

# Words that describe a candidate but are NOT skills. None are in the taxonomy, so
# recognizing any of them would be an unambiguous false positive (the A3 class).
FORBIDDEN_SKILL_WORDS = {
    "fluent",
    "proficient",
    "expert",
    "senior",
    "backend",
    "frontend",
    "pragmatic",
}


def _read(rel: str | None) -> str | None:
    if rel is None:
        return None
    return read_text(SAMPLES / rel)


def _inputs(*, csv=None, ats=None, github=None, notes=None) -> PipelineInputs:
    return PipelineInputs(
        csv_payload=_read(csv),
        ats_payload=_read(ats),
        github_payload=_read(github),
        notes_payload=_read(notes),
    )


# id -> (PipelineInputs, expects_structured_name)
# expects_structured_name is True when a structured source supplies a name, so we
# can assert the name generalizes; False for the notes-only bundle, where prose
# name extraction is descoped (A2) and an empty name is the documented behavior.
BUNDLES: dict[str, tuple[PipelineInputs, bool]] = {
    "all_four_sources": (
        _inputs(
            csv="recruiter_csv/priya_nair.csv",
            ats="ats_json/priya_nair.json",
            github="github/priya_nair.json",
            notes="recruiter_notes/priya_nair.txt",
        ),
        True,
    ),
    "structured_plus_notes": (
        _inputs(
            csv="recruiter_csv/marcus_bell.csv",
            notes="recruiter_notes/marcus_bell.txt",
        ),
        True,
    ),
    "notes_only": (
        _inputs(notes="recruiter_notes/dana_okoro.txt"),
        False,
    ),
    "missing_and_malformed": (
        _inputs(
            csv="recruiter_csv/sam_lindqvist.csv",
            ats="ats_json/sam_lindqvist.json",
        ),
        True,
    ),
}


def _skill_names(output: dict) -> list[str]:
    return [entry["name"] for entry in (output.get("skills") or [])]


@pytest.fixture(params=list(BUNDLES), ids=list(BUNDLES))
def bundle(request):
    inputs, expects_name = BUNDLES[request.param]
    # Running here means "does not crash on this bundle": a raising pipeline fails
    # the fixture, and thus every test parametrized on it, with the real traceback.
    output = run_pipeline(inputs)
    return request.param, output, expects_name


def test_bundle_runs_and_has_core_shape(bundle):
    """Every bundle produces a well-formed projection, never an exception."""
    _name, output, _expects_name = bundle
    assert isinstance(output, dict)
    for key in ("candidate_id", "full_name", "skills", "overall_confidence"):
        assert key in output, key
    assert isinstance(output["candidate_id"], str) and output["candidate_id"].startswith("cand_")
    assert isinstance(output["skills"], list)


def test_recognized_skills_are_in_taxonomy_canonical(bundle):
    """Skills that survive recognition are canonical taxonomy names, nothing else."""
    _name, output, _expects_name = bundle
    names = _skill_names(output)
    unknown = [n for n in names if n not in CANONICAL_SKILLS]
    assert not unknown, f"non-canonical skills emitted: {unknown}"
    # No duplicate canonical skills after merge.
    assert len(names) == len(set(names)), f"duplicate skills: {names}"


def test_no_adjective_or_stray_initial_false_positives(bundle):
    """Adjectives ('fluent in X') and stray initials must not become skills."""
    _name, output, _expects_name = bundle
    names = _skill_names(output)
    lowered = {n.lower() for n in names}
    assert not (lowered & FORBIDDEN_SKILL_WORDS), (
        f"adjective/context word recognized as skill: {lowered & FORBIDDEN_SKILL_WORDS}"
    )
    # No lone single-letter skill: guards against the A3 stray-initial class
    # ("R" mined from a sign-off). None of these bundles legitimately claim a
    # single-letter skill, so any that appears is a false positive.
    single_letters = [n for n in names if len(n) == 1 and n.isalpha()]
    assert not single_letters, f"single-letter skill(s) from prose: {single_letters}"


def test_corroborated_notes_skills_are_recognized(bundle):
    """A corroborated 'Skills:'/'Stack:' list yields at least one canonical skill.

    Every bundle here contains a corroborated skill list (all four also carry
    structured skills), so recognition should never silently produce nothing.
    """
    _name, output, _expects_name = bundle
    assert _skill_names(output), "expected at least one recognized skill"


def test_missing_and_empty_sources_degrade_gracefully(bundle):
    """Absent/empty/malformed inputs project to empties, not exceptions."""
    name, output, expects_name = bundle

    # Contact lists are always lists (possibly empty), never null/exception.
    assert isinstance(output.get("emails"), list)
    assert isinstance(output.get("phones"), list)

    if expects_name:
        # A structured source supplied a name: it should generalize beyond the
        # two shipped fixtures (non-empty, and not a Kelsey/Andrej value).
        assert isinstance(output["full_name"], str) and output["full_name"].strip()
        assert output["full_name"] not in {"Kelsey Hightower", "Andrej Karpathy"}

    if name == "notes_only":
        # Key stress case. Prose name/experience extraction is descoped (A2), so
        # graceful degradation means an EMPTY name — not a crash, and not a
        # hallucinated one from prose.
        assert not output["full_name"], repr(output["full_name"])
        assert output.get("experience") in (None, [])
        assert output.get("education") in (None, [])

    if name == "missing_and_malformed":
        # CSV has an empty email + 'N/A' phone + 'twelve' years + junk dates; ATS
        # has an empty email list + invalid month. All must be dropped quietly.
        assert output.get("emails") == []
        assert output.get("years_experience") is None
