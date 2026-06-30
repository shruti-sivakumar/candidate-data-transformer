"""Recruiter notes source adapter.

Stage-1 recognition over free prose. Recognizes contact values (emails, phones,
URLs) by pattern, and skill surface-forms by gazetteer membership. Emits raw
recognized substrings — no normalization (that is Module 3) and no canonicalization
(skill linking is Module 3's fuzzy-match stage). Returns [] on failure, never raises.

Scope note: name / education / experience are NOT extracted from prose in this MVP.
Those fields are owned by the structured sources (CSV, ATS), which carry them
schema-labeled and at higher trust. Prose extraction of those fields is a planned
low-confidence supplementary layer (see _extract_* extension points below); it is
deliberately descoped here because the provided fixtures always pair notes with
structured sources, so the notes-only case does not occur in test data.
"""

import logging
import re
from collections.abc import Iterable

from src.transformer.models import RawRecord

logger = logging.getLogger(__name__)

# Contact recognition patterns. These recognize *shape*; validity (for phones)
# is confirmed separately via the phonenumbers library.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s,<>\"')]+", re.IGNORECASE)
# Phone candidates: a leading + and country code, then 7-14 more digits possibly
# separated by spaces/dashes/dots/parens. Intentionally requires the leading + so
# bare numbers, comp figures, and dates are not candidates.
_PHONE_CANDIDATE_RE = re.compile(r"\+\d[\d\s().-]{7,16}\d")


class NotesSource:
    """Adapter for recruiter notes (.txt free text).

    Consumes the raw note string and emits a single RawRecord whose raw_fields
    hold lists of recognized contacts and skill surface-forms, each preserved as
    they appeared in the text. One notes file -> one RawRecord.
    """

    name: str = "recruiter_notes"
    trust: float = 0.55

    def __init__(self, skill_vocabulary: Iterable[str] | None = None) -> None:
        """skill_vocabulary: surface forms to recognize (gazetteer keys).

        Injected, not hardcoded — the same vocabulary feeds every source's skill
        recognition, and keeping it a constructor dependency means this adapter
        pre-commits nothing about how that vocabulary is sourced (Lightcast snapshot)
        or how matches are later canonicalized (Module 3). An empty/None vocabulary
        means no skills are recognized, which is a valid degraded mode.
        """
        # Stored as-given (not lowercased) — the re.IGNORECASE flag handles
        # case-insensitive matching. Recognition emits the vocabulary form, not
        # the as-found substring; canonicalization is Module 3's job.
        self._skill_forms: list[str] = sorted(
            {s.strip() for s in (skill_vocabulary or ()) if s and s.strip()}
        )

    def extract(self, payload: str) -> list[RawRecord]:
        """Recognize contacts and skills in a note. Returns [] on empty/failure."""
        try:
            if not payload or not payload.strip():
                return []  # empty-but-valid: silent, per adapter contract

            fields: dict[str, object] = {
                "emails": self._extract_emails(payload),
                "phones": self._extract_phones(payload),
                "urls": self._extract_urls(payload),
                "skills": self._extract_skills(payload),
            }
            return [RawRecord(source=self.name, raw_fields=fields)]
        except Exception as e:  # recognition must never abort the pipeline
            logger.warning("NotesSource failed to recognize payload: %s", e)
            return []

    # --- contact recognition (pattern-based) ---------------------------------

    def _extract_emails(self, text: str) -> list[str]:
        # Deduplicate preserving first-seen order.
        return _dedupe(_EMAIL_RE.findall(text))

    def _extract_urls(self, text: str) -> list[str]:
        # Strip trailing sentence-ending punctuation that prose wraps URLs in.
        return _dedupe(u.rstrip(".,;:!?") for u in _URL_RE.findall(text))

    def _extract_phones(self, text: str) -> list[str]:
        """Recognize phone candidates by shape, then keep only those the
        phonenumbers library confirms are valid. The raw matched substring is
        emitted as-found; E.164 normalization is Module 3's job, not the adapter's.
        """
        try:
            import phonenumbers  # lazy: only needed when a candidate appears
        except ImportError:
            logger.warning("phonenumbers unavailable; skipping phone recognition")
            return []

        kept: list[str] = []
        for raw in _PHONE_CANDIDATE_RE.findall(text):
            candidate = raw.strip()
            try:
                parsed = phonenumbers.parse(candidate, None)  # None: must be E.164-ish (has +)
            except phonenumbers.NumberParseException:
                continue
            if phonenumbers.is_valid_number(parsed):
                kept.append(candidate)  # emit as-found, un-normalized
        return _dedupe(kept)

    # --- skill recognition (gazetteer-based) ---------------------------------

    def _extract_skills(self, text: str) -> list[str]:
        """Recognize skill surface-forms present in the text by gazetteer
        membership. Emits the matched form as it appears in the vocabulary;
        canonicalization / fuzzy-linking is Module 3. Lookaround anchors so
        "Go" does not match inside "Google", and punctuation-bearing skills
        like "C++", "C#", ".NET" are correctly matched.
        """
        if not self._skill_forms:
            return []
        found: list[str] = []
        for form in self._skill_forms:
            # (?<!\w)/(?!\w): adjacent char must not be a word character.
            # Handles trailing/leading punctuation (\b fails there).
            # token-based matching is the upgrade path once the vocabulary's punctuation profile is known.
            pattern = re.compile(rf"(?<!\w){re.escape(form)}(?!\w)", re.IGNORECASE)
            if pattern.search(text):
                found.append(form)
        return _dedupe(found)

    # --- extension points (planned, descoped for MVP) ------------------------
    # Prose name/education/experience extraction will slot in here as separate
    # methods (e.g. _extract_names, _extract_education) emitting low-confidence
    # recognized values. Adding them does not disturb the contact/skill paths.


def _dedupe(items: Iterable[str]) -> list[str]:
    """Order-preserving dedupe (determinism: stable output for stable input)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out