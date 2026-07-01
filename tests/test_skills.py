"""Tests for skill taxonomy loading, candidate generation, and linking."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.transformer.normalize.skills import (
    FUZZY_MATCH_THRESHOLD,
    SkillTaxonomy,
    canonicalize_skill,
    default_skill_taxonomy,
    generate_candidates,
    normalize_alias,
)


def _taxonomy(tmp_path: Path, rows: list[tuple[str, str]]) -> SkillTaxonomy:
    path = tmp_path / "skills.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_skill", "aliases"])
        writer.writeheader()
        for canonical, aliases in rows:
            writer.writerow({"canonical_skill": canonical, "aliases": aliases})
    return SkillTaxonomy.from_csv(path)


def test_normalize_alias_is_single_source_of_truth():
    assert normalize_alias("  K8S ") == normalize_alias("k8s")
    assert normalize_alias("React JS") == normalize_alias("react-js")


def test_generate_candidates_emits_deterministic_ngrams():
    candidates = generate_candidates("Built platform engineering with k8s.", max_n=2)
    pairs = [(candidate.text, candidate.normalized) for candidate in candidates[:4]]
    assert pairs == [
        ("Built platform", "built platform"),
        ("Built", "built"),
        ("platform engineering", "platform engineering"),
        ("platform", "platform"),
    ]
    assert candidates == generate_candidates("Built platform engineering with k8s.", max_n=2)


def test_default_taxonomy_canonicalizes_exact_aliases():
    assert canonicalize_skill("k8s") == "Kubernetes"
    assert canonicalize_skill("golang") == "Go"
    assert canonicalize_skill("torch") == "PyTorch"


def test_structured_short_skills_still_canonicalize_without_pos_gate():
    assert canonicalize_skill("C") == "C"
    assert canonicalize_skill("R") == "R"
    assert canonicalize_skill("Go") == "Go"


def test_structured_canonicalization_uses_guarded_fuzzy_fallback():
    assert canonicalize_skill("Kubernets") == "Kubernetes"


def test_default_taxonomy_keeps_unmatched_clean_original():
    assert canonicalize_skill("Very Specific Internal Tool") == "Very Specific Internal Tool"


def test_link_candidate_reports_exact_and_fuzzy_methods(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Kubernetes", "k8s"), ("Python", "")])

    assert taxonomy.link_candidate("k8s") == ("Kubernetes", 100, "exact")
    canonical, score, method = taxonomy.link_candidate("Kubernets")
    assert (canonical, method) == ("Kubernetes", "fuzzy")
    assert score >= FUZZY_MATCH_THRESHOLD


def test_near_miss_below_threshold_is_rejected(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Kubernetes", "k8s"), ("Python", "")])
    canonical, _, method = taxonomy.link_candidate("Karaoke")
    assert canonical is None
    assert method is None


def test_two_character_candidate_never_fuzzy_matches(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Go", "golang"), ("Machine Learning", "ML")])
    assert taxonomy.link_candidate("go") == ("Go", 100, "exact")
    assert taxonomy.link_candidate("ml") == ("Machine Learning", 100, "exact")
    assert taxonomy.link_candidate("gl")[2] is None


def test_extract_from_text_is_deterministic_and_dedupes(tmp_path: Path):
    taxonomy = _taxonomy(
        tmp_path,
        [("Kubernetes", "k8s"), ("Platform Engineering", "platform eng")],
    )
    first = taxonomy.extract_from_text("k8s and platform engineering; k8s again.")
    second = taxonomy.extract_from_text("k8s and platform engineering; k8s again.")
    assert first == second
    assert [match[0] for match in first] == ["Kubernetes", "Platform Engineering"]


def test_unmatched_prose_candidate_is_honest_miss(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Python", "")])
    assert taxonomy.extract_from_text("Expert in an internal platform called BlueNova.") == []


def test_prose_pos_gate_rejects_adjective_before_fuzzy(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Fluentd", ""), ("Rust", "")])
    matches = taxonomy.extract_from_text("Super fluent in Rust.")
    # "fluent" (adjective) must not fuzzy-match Fluentd; Rust is still recognized.
    assert [match[0] for match in matches] == ["Rust"]


def test_prose_pos_gate_generalizes_beyond_observed_words(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Rust", ""), ("Python", "")])
    assert [match[0] for match in taxonomy.extract_from_text("Seasoned in Rust.")] == ["Rust"]
    assert [match[0] for match in taxonomy.extract_from_text("Strong in Python.")] == ["Python"]


def test_prose_short_exact_needs_corroboration(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("C", ""), ("R", ""), ("Go", "golang")])
    # Bare one/two-letter tokens in ordinary prose carry no skill-claim evidence.
    assert taxonomy.extract_from_text("Strong in C and R. Also Go.") == []


def test_prose_short_exact_emitted_when_corroborated(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("C", ""), ("R", ""), ("Go", "golang")])
    # A delimited list / skills-context cue corroborates them as deliberate mentions.
    matches = taxonomy.extract_from_text("Stack: Go, R, C.")
    assert [match[0] for match in matches] == ["Go", "R", "C"]


def test_prose_sign_off_initial_not_mined_as_skill():
    taxonomy = default_skill_taxonomy()
    # "— Riya R." must not invent skill "R" from the recruiter's initial.
    assert "R" not in [match[0] for match in taxonomy.extract_from_text("— Riya R.")]


def test_prose_skills_list_keeps_short_skill():
    taxonomy = default_skill_taxonomy()
    matches = [match[0] for match in taxonomy.extract_from_text("skills: Python, R, SQL")]
    assert "R" in matches and "Python" in matches and "SQL" in matches


def test_prose_longer_skills_unaffected_by_corroboration_gate():
    taxonomy = default_skill_taxonomy()
    assert "Kubernetes" in [m[0] for m in taxonomy.extract_from_text("Deployed Kubernetes in prod.")]
    assert "Go" in [m[0] for m in taxonomy.extract_from_text("Stack: Go, Rust.")]


def test_structured_short_skills_bypass_prose_corroboration():
    # Structured sources declare skills explicitly and must keep exact-linking.
    assert canonicalize_skill("R") == "R"
    assert canonicalize_skill("C") == "C"
    assert canonicalize_skill("Go") == "Go"


def test_prose_fuzzy_typo_still_works_after_pos_gate(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Kubernetes", "k8s")])
    matches = taxonomy.extract_from_text("Built platform on Kubernets.")
    assert [match[0] for match in matches] == ["Kubernetes"]


def test_prose_extraction_is_repeatable_with_pos_tagger(tmp_path: Path):
    taxonomy = _taxonomy(tmp_path, [("Rust", ""), ("Python", ""), ("Kubernetes", "k8s")])
    text = "Seasoned in Rust, strong in Python, and deployed Kubernets."
    assert taxonomy.extract_from_text(text) == taxonomy.extract_from_text(text)


def test_alias_collision_load_raises(tmp_path: Path):
    path = tmp_path / "skills.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["canonical_skill", "aliases"])
        writer.writeheader()
        writer.writerow({"canonical_skill": "One", "aliases": "shared"})
        writer.writerow({"canonical_skill": "Two", "aliases": "shared"})

    with pytest.raises(ValueError, match="shared"):
        SkillTaxonomy.from_csv(path)


def test_default_taxonomy_keeps_exact_short_forms_without_fuzzy_guessing():
    taxonomy = default_skill_taxonomy()
    assert taxonomy.link_candidate("go") == ("Go", 100, "exact")
    assert taxonomy.link_candidate("r") == ("R", 100, "exact")
    assert taxonomy.link_candidate("zz")[2] is None
