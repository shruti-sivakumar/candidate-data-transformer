"""Characterization probe for FUZZY_MATCH_THRESHOLD (skills.py).

This is a TRANSPARENCY / CHARACTERIZATION probe, NOT threshold tuning against a
labeled dataset. We do not have credible labels and we deliberately avoid
overfitting the threshold to hand-picked examples. The goal is to document how
the fuzzy skill-matching decision boundary actually behaves so the chosen
threshold (94) can be justified from what the scores show.

It reuses the REAL scoring path from src.transformer.normalize.skills:
  - the same taxonomy snapshot (default_skill_taxonomy)
  - the same normalization (normalize_alias)
  - the same scorer (rapidfuzz fuzz.token_sort_ratio via process.extractOne)
  - the same candidate/choice guards (_is_dangerous_candidate length guard)
matching link_candidate's behavior. The only difference: we run extractOne
WITHOUT score_cutoff so we can observe scores on both sides of the boundary.

Run:  python scripts/threshold_probe.py
Writes CSV tables under docs/ (docs/threshold_probe_*.csv) and prints a report
to stdout.
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

from rapidfuzz import fuzz, process

from src.transformer.normalize.skills import (
    FUZZY_MATCH_THRESHOLD,
    _is_dangerous_candidate,
    default_skill_taxonomy,
    normalize_alias,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "docs"
TAX = default_skill_taxonomy()

# The exact choice list link_candidate builds before calling process.extractOne.
_CHOICES = [s for s in TAX.fuzzy_surfaces if not _is_dangerous_candidate(s)]


def best_match(surface: str) -> tuple[str | None, float]:
    """Mirror link_candidate's fuzzy step but WITHOUT the score_cutoff.

    Returns (best matching normalized taxonomy surface, score). Returns
    (None, 0) when the input normalizes away or is caught by the same length
    guard the real code applies before fuzzy matching.
    """
    normalized = normalize_alias(surface)
    if not normalized or _is_dangerous_candidate(normalized):
        return None, 0.0
    match = process.extractOne(normalized, _CHOICES, scorer=fuzz.token_sort_ratio)
    if match is None:
        return None, 0.0
    choice, score, _ = match
    return choice, float(score)


def score_against(surface: str, target_surface: str) -> float:
    """token_sort_ratio between two surfaces after the real normalization."""
    a, b = normalize_alias(surface), normalize_alias(target_surface)
    if not a or not b:
        return 0.0
    return float(fuzz.token_sort_ratio(a, b))


# --------------------------------------------------------------------------- #
# Category 1: TRUE TYPOS — generated programmatically via single mechanical
# edits on a spread of real canonical skills. No cherry-picking.
# --------------------------------------------------------------------------- #

def _canonical_skills() -> list[str]:
    """Real canonical skills from the taxonomy, deduped, in taxonomy order."""
    seen: dict[str, None] = {}
    for canonical in TAX.alias_to_canonical.values():
        seen.setdefault(canonical, None)
    return list(seen)


def _sample_skills(n: int = 28) -> list[str]:
    """Pick a deterministic spread of skills across name lengths.

    Only skills whose single-edit perturbation survives the length guard are
    useful (very short names get rejected before fuzzy runs, which is a
    separate defensive layer, not a threshold behavior). We sort by length and
    take an even stride so short and long names are both represented.
    """
    usable = sorted(
        {s for s in _canonical_skills() if len(normalize_alias(s) or "") >= 5},
        key=lambda s: (len(s), s.casefold()),
    )
    if len(usable) <= n:
        return usable
    stride = len(usable) / n
    return [usable[int(i * stride)] for i in range(n)]


def _perturbations(word: str) -> list[tuple[str, str]]:
    """Deterministic single-edit typos: (edit_type, perturbed_string).

    Edits are applied to the lowercased alphanumeric core so the result reads
    like a realistic slip, not a normalization artifact. One deletion, one
    substitution, one adjacent transposition, one insertion, each at a stable
    interior position so re-runs reproduce exactly.
    """
    w = word
    edits: list[tuple[str, str]] = []
    # Interior indices (avoid index 0 so we perturb mid-word like a real typo).
    letters = [i for i, ch in enumerate(w) if ch.isalpha()]
    interior = [i for i in letters if i != letters[0]] if letters else []
    if not interior:
        return edits

    mid = interior[len(interior) // 2]

    # Deletion: drop one interior character (kubernetes -> kubernetes w/o a char).
    edits.append(("deletion", w[:mid] + w[mid + 1 :]))

    # Substitution: replace one interior letter with a keyboard-plausible other.
    sub = "x" if w[mid].lower() != "x" else "z"
    edits.append(("substitution", w[:mid] + sub + w[mid + 1 :]))

    # Adjacent transposition: swap two neighboring interior letters.
    swap_pos = next((i for i in interior if i + 1 < len(w) and w[i + 1].isalpha()), None)
    if swap_pos is not None:
        j = swap_pos
        edits.append(("transposition", w[:j] + w[j + 1] + w[j] + w[j + 2 :]))

    # Insertion: duplicate one interior character (postgresql -> postgressql).
    edits.append(("insertion", w[:mid] + w[mid] + w[mid:]))

    return edits


def build_typos() -> list[dict]:
    rows: list[dict] = []
    for skill in _sample_skills():
        for edit_type, perturbed in _perturbations(skill):
            if perturbed == skill:
                continue
            choice, score = best_match(perturbed)
            # Score of the perturbed string specifically vs its OWN origin skill,
            # which is what we care about for recall (the intended match).
            self_score = score_against(perturbed, skill)
            rows.append(
                {
                    "original": skill,
                    "perturbed": perturbed,
                    "edit_type": edit_type,
                    "best_match": TAX.alias_to_canonical.get(choice or "", choice),
                    "score": self_score,
                    "matches_origin": bool(
                        choice and normalize_alias(skill) == choice
                    ),
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# Category 2: DANGEROUS NEAR-MISSES — real, named boundary cases. These are
# ILLUSTRATIVE, not exhaustive: each input is a DIFFERENT real thing that looks
# similar to a taxonomy skill. Cannot be generated mechanically.
# --------------------------------------------------------------------------- #

NEAR_MISSES: list[tuple[str, str]] = [
    ("fluent", "Fluentd"),     # adjective / different tool
    ("react", "Redux"),        # sibling libs, distinct
    ("java", "JavaScript"),    # classic collision
    ("next", "Nuxt"),          # different frameworks
    ("vue", "Vite"),           # different tools, same ecosystem
    ("go", "Golang"),          # fuzzy view only; exact lookup really catches Go
]


def build_near_misses() -> list[dict]:
    """Characterize whether each input WRONGLY fuzzes to its named look-alike.

    Honest caveat baked into the columns: several of these inputs (react, java,
    vue, go) are THEMSELVES real taxonomy entries, so their overall best match
    is their own correct skill at ~100 via exact lookup. That is correct, not a
    leak. The dangerous-collapse signal we actually care about is
    `score_vs_lookalike`: does the input score high against the DIFFERENT skill
    it superficially resembles? That is the number that must stay LOW.
    """
    rows: list[dict] = []
    for input_str, lookalike in NEAR_MISSES:
        choice, best = best_match(input_str)
        target_score = score_against(input_str, lookalike)
        norm = normalize_alias(input_str)
        closest = TAX.alias_to_canonical.get(choice or "", choice)
        has_own = input_str in TAX.alias_to_canonical or (
            norm in TAX.alias_to_canonical
        )
        rows.append(
            {
                "input": input_str,
                "wrong_lookalike": lookalike,
                # THE signal that must be LOW: input vs the wrong look-alike.
                "score_vs_lookalike": target_score,
                # Context: the input's actual best taxonomy match.
                "own_best_match": closest,
                "own_best_score": best,
                "has_own_exact_entry": bool(has_own),
                "length_guarded": bool(not norm or _is_dangerous_candidate(norm)),
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Category 3: DISTINCT-BUT-SIMILAR real taxonomy pairs that must stay separate.
# Pulled as actual canonical skills already in the taxonomy.
# --------------------------------------------------------------------------- #

DISTINCT_PAIRS: list[tuple[str, str]] = [
    ("React", "Redux"),
    ("Next.js", "Nuxt"),
    ("Vue.js", "Vite"),
    ("Apache Spark", "Spark SQL"),
    ("Spring Framework", "Spring Boot"),
    ("React", "React Native"),
]


def build_distinct_pairs() -> list[dict]:
    rows: list[dict] = []
    for a, b in DISTINCT_PAIRS:
        in_tax = a in _canonical_skills() and b in _canonical_skills()
        rows.append(
            {
                "skill_a": a,
                "skill_b": b,
                "cross_score": score_against(a, b),
                "both_in_taxonomy": in_tax,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _save_csv(name: str, rows: list[dict]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    if not rows:
        return path
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _summary(scores: list[float]) -> dict:
    return {
        "n": len(scores),
        "min": min(scores) if scores else None,
        "median": statistics.median(scores) if scores else None,
        "max": max(scores) if scores else None,
    }


def _print_table(title: str, rows: list[dict], score_key: str, cols: list[str]) -> None:
    print(f"\n=== {title} (sorted by {score_key}) ===")
    ordered = sorted(rows, key=lambda r: r.get(score_key, 0), reverse=True)
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in ordered)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in ordered:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))


def main() -> None:
    print("Skill fuzzy-match threshold characterization probe")
    print(f"Taxonomy canonical skills: {len(set(_canonical_skills()))}")
    print(f"Fuzzy choice surfaces (post length-guard): {len(_CHOICES)}")
    print(f"Scorer: rapidfuzz.fuzz.token_sort_ratio (same as link_candidate)")
    print(f"Current FUZZY_MATCH_THRESHOLD = {FUZZY_MATCH_THRESHOLD}")

    typos = build_typos()
    near = build_near_misses()
    distinct = build_distinct_pairs()

    _print_table(
        "1. TRUE TYPOS (want HIGH)", typos, "score",
        ["original", "perturbed", "edit_type", "score", "matches_origin"],
    )
    _print_table(
        "2. DANGEROUS NEAR-MISSES (score_vs_lookalike want LOW)", near,
        "score_vs_lookalike",
        ["input", "wrong_lookalike", "score_vs_lookalike",
         "own_best_match", "own_best_score", "has_own_exact_entry",
         "length_guarded"],
    )
    _print_table(
        "3. DISTINCT-BUT-SIMILAR PAIRS (must stay separate)", distinct, "cross_score",
        ["skill_a", "skill_b", "cross_score", "both_in_taxonomy"],
    )

    typo_scores = [r["score"] for r in typos]
    # "should be low" signal = collapse onto the WRONG look-alike skill.
    near_scores = [r["score_vs_lookalike"] for r in near]
    t = _summary(typo_scores)
    n = _summary(near_scores)

    print("\n=== SUMMARY STATS ===")
    print(f"TRUE TYPOS                n={t['n']:>3}  min={t['min']}  median={t['median']}  max={t['max']}")
    print(f"NEAR-MISS (vs lookalike)  n={n['n']:>3}  min={n['min']}  median={n['median']}  max={n['max']}")

    thr = FUZZY_MATCH_THRESHOLD
    typos_kept = sum(1 for s in typo_scores if s >= thr)
    typos_lost = sum(1 for s in typo_scores if s < thr)
    near_rejected = sum(1 for s in near_scores if s < thr)
    near_leaked = sum(1 for s in near_scores if s >= thr)

    # The separation: highest wrong-collapse near-miss vs lowest typo.
    print("\n=== SEPARATION @ threshold =", thr, "===")
    print(f"TRUE TYPOS kept  (score >= {thr}): {typos_kept}/{len(typo_scores)}")
    print(f"TRUE TYPOS lost  (score <  {thr}): {typos_lost}/{len(typo_scores)}  (recall cost)")
    print(f"NEAR-MISS correctly NOT collapsed (vs-lookalike < {thr}): {near_rejected}/{len(near_scores)}")
    print(f"NEAR-MISS wrongly collapsed        (vs-lookalike >= {thr}): {near_leaked}/{len(near_scores)}  (precision risk)")
    if near_scores and typo_scores:
        gap = min(typo_scores) - max(near_scores)
        print(f"\nGap (min typo - max near-miss) = {min(typo_scores)} - {max(near_scores)} = {gap}")
        # Interpret the two sides of the boundary separately: the precision side
        # (do near-misses reach the threshold?) vs the recall side (do typos
        # fall below it?). These are not symmetric.
        print(f"Highest near-miss (vs-lookalike) score = {max(near_scores)}  (< {thr} => precision-side clean)")
        print(f"Precision-side headroom = {thr} - {max(near_scores)} = {thr - max(near_scores):.2f}")
        if max(near_scores) < thr:
            print("=> On the PRECISION side the boundary is clean: no named near-miss")
            print(f"   reaches {thr}. The negative overall 'gap' comes from the RECALL side:")
            print("   many single-char typos (esp. substitution/transposition on short")
            print("   names) also score below {}, so raising precision costs typo recall.".format(thr))
        else:
            print("=> OVERLAP on the precision side: a near-miss reaches the threshold.")
            print("   Threshold alone is insufficient; POS gate + length guard carry it.")

    p1 = _save_csv("threshold_probe_typos.csv", typos)
    p2 = _save_csv("threshold_probe_near_misses.csv", near)
    p3 = _save_csv("threshold_probe_distinct_pairs.csv", distinct)
    print(f"\nSaved: {p1.name}, {p2.name}, {p3.name}")


if __name__ == "__main__":
    main()
