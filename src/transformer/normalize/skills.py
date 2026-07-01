"""Skill taxonomy loading, n-gram recognition, and guarded fuzzy linking."""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from rapidfuzz import fuzz, process

from src.transformer.normalize.formats import clean_string

DEFAULT_TAXONOMY_PATH = Path("data/skills_taxonomy.csv")
FUZZY_MATCH_THRESHOLD = 94
MAX_CANDIDATE_NGRAM = 3
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_VERSION = "3.8.0"

_TOKEN_RE = re.compile(r"\.?[A-Za-z0-9][A-Za-z0-9.+#-]*")
LinkMethod = Literal["exact", "fuzzy"]


@dataclass(frozen=True)
class SkillCandidate:
    """One deterministic n-gram candidate extracted from prose."""

    text: str
    normalized: str
    start: int
    end: int


def normalize_alias(raw: str | None) -> str | None:
    """Normalize a skill surface form for exact alias lookup and fuzzy matching."""
    cleaned = clean_string(raw)
    if not cleaned:
        return None
    text = unicodedata.normalize("NFC", cleaned).casefold()
    text = re.sub(r"[\s_/-]+", " ", text)
    text = re.sub(r"[^\w+#. ]+", "", text)
    return " ".join(text.split()) or None


def generate_candidates(text: str, *, max_n: int = MAX_CANDIDATE_NGRAM) -> list[SkillCandidate]:
    """Generate contiguous normalized n-gram candidates from prose."""
    tokens = list(_TOKEN_RE.finditer(text))
    candidates: list[SkillCandidate] = []
    for start_index in range(len(tokens)):
        max_end = min(len(tokens), start_index + max_n)
        for end_index in range(max_end, start_index, -1):
            start = tokens[start_index].start()
            end = tokens[end_index - 1].end()
            span = text[start:end].rstrip(".,;:!?")
            normalized = normalize_alias(span)
            if normalized:
                candidates.append(SkillCandidate(text=span, normalized=normalized, start=start, end=end))
    return candidates


def _split_aliases(raw: str | None) -> list[str]:
    """Split the CSV aliases cell into individual surface forms."""
    if not raw or not raw.strip():
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _is_dangerous_candidate(normalized: str) -> bool:
    """Return whether a candidate is too short to fuzzy match safely."""
    compact = normalized.replace(" ", "")
    return len(compact) <= 2


@lru_cache(maxsize=1)
def _spacy_nlp():
    """Load the pinned spaCy tagger once for notes-only POS gating."""
    import spacy

    nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner", "lemmatizer"])
    version = nlp.meta.get("version")
    if version != SPACY_MODEL_VERSION:
        raise RuntimeError(
            f"Expected {SPACY_MODEL} {SPACY_MODEL_VERSION}, got {version!r}"
        )
    return nlp


def _candidate_head_pos(text: str) -> str | None:
    """Return the rightmost non-punctuation token's coarse POS tag."""
    doc = _spacy_nlp()(text)
    for token in reversed(doc):
        if not token.is_punct and token.text.strip():
            return token.pos_
    return None


def _is_prose_safe(surface: str) -> bool:
    """Return whether a taxonomy surface form is safe enough for prose matching."""
    normalized = normalize_alias(surface)
    if not normalized or _is_dangerous_candidate(normalized):
        return False
    compact = normalized.replace(" ", "")
    if any(ch.isdigit() for ch in surface) and len(compact) >= 3:
        return True
    if len(compact) <= 4:
        return False
    has_boundary_signal = " " in normalized or any(ch in surface for ch in ".+#")
    if has_boundary_signal:
        return True
    return len(compact) >= 6 and not surface.isupper()


@dataclass(frozen=True)
class SkillTaxonomy:
    """A local, deterministic skill taxonomy snapshot."""

    alias_to_canonical: dict[str, str]
    prose_forms: tuple[str, ...]
    fuzzy_surfaces: tuple[str, ...]

    @classmethod
    def from_csv(cls, path: Path) -> "SkillTaxonomy":
        """Load canonical skills and aliases from a curated CSV snapshot."""
        alias_to_canonicals: dict[str, set[str]] = {}
        surface_by_alias: dict[str, str] = {}
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                canonical = clean_string(row.get("canonical_skill"))
                if not canonical:
                    continue
                for surface in [canonical, *_split_aliases(row.get("aliases"))]:
                    normalized = normalize_alias(surface)
                    if not normalized:
                        continue
                    alias_to_canonicals.setdefault(normalized, set()).add(canonical)
                    surface_by_alias.setdefault(normalized, surface)

        collisions = {
            alias: tuple(sorted(canonicals))
            for alias, canonicals in sorted(alias_to_canonicals.items())
            if len(canonicals) > 1
        }
        if collisions:
            details = "; ".join(
                f"{alias}: {', '.join(canonicals)}"
                for alias, canonicals in collisions.items()
            )
            raise ValueError(f"Skill taxonomy alias collisions: {details}")

        alias_to_canonical = {
            alias: next(iter(canonicals))
            for alias, canonicals in sorted(alias_to_canonicals.items())
        }
        prose_forms = tuple(
            sorted(
                {
                    surface_by_alias[alias]
                    for alias in alias_to_canonical
                    if _is_prose_safe(surface_by_alias[alias])
                },
                key=lambda item: (-len(item), item.casefold()),
            )
        )
        fuzzy_surfaces = tuple(sorted(alias_to_canonical))
        return cls(
            alias_to_canonical=alias_to_canonical,
            prose_forms=prose_forms,
            fuzzy_surfaces=fuzzy_surfaces,
        )

    def canonicalize(self, raw: str | None) -> str | None:
        """Return the canonical skill if known; otherwise keep the cleaned literal."""
        cleaned = clean_string(raw)
        if not cleaned:
            return None
        canonical, _, _ = self.link_candidate(cleaned)
        return canonical or cleaned

    def link_candidate(self, text_span: str) -> tuple[str | None, int, LinkMethod | None]:
        """Link one candidate span to the taxonomy by exact lookup then fuzzy match."""
        normalized = normalize_alias(text_span)
        if not normalized:
            return None, 0, None
        exact = self.alias_to_canonical.get(normalized)
        if exact is not None:
            return exact, 100, "exact"
        if _is_dangerous_candidate(normalized):
            return None, 0, None

        best_score = 0
        choices = [surface for surface in self.fuzzy_surfaces if not _is_dangerous_candidate(surface)]
        match = process.extractOne(
            normalized,
            choices,
            scorer=fuzz.token_sort_ratio,
            score_cutoff=FUZZY_MATCH_THRESHOLD,
        )
        if match is None:
            return None, best_score, None
        surface, score, _ = match
        return self.alias_to_canonical[surface], int(round(score)), "fuzzy"

    def extract_from_text(self, text: str) -> list[tuple[str, SkillCandidate, int, LinkMethod]]:
        """Recognize and link skill mentions from prose candidates."""
        found: list[tuple[str, SkillCandidate, int, LinkMethod]] = []
        seen: set[str] = set()
        occupied: list[tuple[int, int]] = []
        for candidate in generate_candidates(text):
            if any(candidate.start < end and candidate.end > start for start, end in occupied):
                continue
            exact = self.alias_to_canonical.get(candidate.normalized)
            if exact is not None:
                canonical, score, method = exact, 100, "exact"
            else:
                if _candidate_head_pos(candidate.text) not in {"NOUN", "PROPN"}:
                    continue
                canonical, score, method = self.link_candidate(candidate.text)
                if canonical is None or method is None:
                    continue
            key = canonical.casefold()
            if key in seen:
                continue
            seen.add(key)
            occupied.append((candidate.start, candidate.end))
            found.append((canonical, candidate, score, method))
        return found


@lru_cache(maxsize=1)
def default_skill_taxonomy() -> SkillTaxonomy:
    """Load the default committed skill taxonomy snapshot."""
    return SkillTaxonomy.from_csv(DEFAULT_TAXONOMY_PATH)


def canonicalize_skill(raw: str | None) -> str | None:
    """Map a raw skill surface form to a canonical taxonomy name when possible."""
    return default_skill_taxonomy().canonicalize(raw)
