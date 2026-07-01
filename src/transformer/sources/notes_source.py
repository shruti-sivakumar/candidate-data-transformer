"""Recruiter notes source adapter.

Recognizes contact values by pattern and skill mentions by two-stage linking:
n-gram candidate generation followed by exact/fuzzy taxonomy linking. Emits
canonical skill names because the source stage is where prose recognition happens.
Returns [] on failure, never raises.

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

from src.transformer.audit import make_event
from src.transformer.models import AuditEvent, RawRecord
from src.transformer.normalize.skills import SkillTaxonomy, default_skill_taxonomy, normalize_alias

logger = logging.getLogger(__name__)

# Contact recognition patterns. These recognize *shape*; validity (for phones)
# is confirmed separately via the phonenumbers library.
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_URL_RE = re.compile(r"\bhttps?://[^\s,<>\"')]+", re.IGNORECASE)
# Phone candidates: a leading + and country code, then 7-14 more digits possibly
# separated by spaces/dashes/dots/parens. Intentionally requires the leading + so
# bare numbers, comp figures, and dates are not candidates.
_PHONE_CANDIDATE_RE = re.compile(r"\+\d[\d\s().-]{7,16}\d")
_BARE_PHONE_CANDIDATE_RE = re.compile(r"(?<![\w+])\d{10}(?!\w)")


class NotesSource:
    """Adapter for recruiter notes (.txt free text).

    Consumes the raw note string and emits a single RawRecord whose raw_fields
    hold lists of recognized contacts and skill surface-forms, each preserved as
    they appeared in the text. One notes file -> one RawRecord.
    """

    name: str = "recruiter_notes"
    trust: float = 0.55

    def __init__(
        self,
        skill_vocabulary: Iterable[str] | None = None,
        taxonomy: SkillTaxonomy | None = None,
        default_region: str = "IN",
    ) -> None:
        """Create a notes adapter.

        skill_vocabulary is kept for small tests/backward compatibility. In the
        real pipeline, taxonomy defaults to the committed skill snapshot.
        """
        self.default_region = default_region
        if taxonomy is not None:
            self._taxonomy = taxonomy
        elif skill_vocabulary is None:
            self._taxonomy = default_skill_taxonomy()
        else:
            self._taxonomy = _taxonomy_from_vocabulary(skill_vocabulary)

    def extract(self, payload: str) -> list[RawRecord]:
        """Backward-compatible wrapper returning only records."""
        records, _ = self.extract_with_audit(payload)
        return records

    def extract_with_audit(self, payload: str) -> tuple[list[RawRecord], list[AuditEvent]]:
        """Recognize contacts and skills in a note. Returns [] on empty/failure."""
        audit_log: list[AuditEvent] = []
        try:
            if not payload or not payload.strip():
                audit_log.append(
                    make_event("extract", "payload", "source_empty", "empty_payload", source=self.name)
                )
                return [], audit_log

            skill_matches = self._extract_skill_matches(payload)
            phone_fields = self._extract_phone_fields(payload)
            fields: dict[str, object] = {
                "emails": self._extract_emails(payload),
                "phones": phone_fields["phones"],
                "phone_default_region": self.default_region,
                "phone_recognition_methods": phone_fields["recognition_methods"],
                "urls": self._extract_urls(payload),
                "skills": [match["canonical"] for match in skill_matches],
                "skill_matches": skill_matches,
            }
            for match in skill_matches:
                audit_log.append(
                    make_event(
                        "extract",
                        "skills",
                        "value_recognized",
                        "skill_candidate_linked",
                        source=self.name,
                        raw_value=match["surface"],
                        canonical=match["canonical"],
                        score=match["score"],
                        match_method=match["method"],
                    )
                )
            for field_name, values in fields.items():
                if field_name in {"skill_matches", "phone_default_region", "phone_recognition_methods"}:
                    continue
                if not values:
                    audit_log.append(
                        make_event(
                            "extract",
                            field_name,
                            "field_missing",
                            "no_values_recognized",
                            source=self.name,
                        )
                    )
            return [RawRecord(source=self.name, raw_fields=fields)], audit_log
        except Exception as e:  # recognition must never abort the pipeline
            logger.warning("NotesSource failed to recognize payload: %s", e)
            audit_log.append(
                make_event("extract", "payload", "source_failed", "recognition_failed", source=self.name, error=str(e))
            )
            return [], audit_log

    # --- contact recognition (pattern-based) ---------------------------------

    def _extract_emails(self, text: str) -> list[str]:
        # Deduplicate preserving first-seen order.
        return _dedupe(_EMAIL_RE.findall(text))

    def _extract_urls(self, text: str) -> list[str]:
        # Strip trailing sentence-ending punctuation that prose wraps URLs in.
        return _dedupe(u.rstrip(".,;:!?") for u in _URL_RE.findall(text))

    def _extract_phones(self, text: str) -> list[str]:
        """Backward-compatible wrapper returning just the recognized values."""
        return self._extract_phone_fields(text)["phones"]

    def _extract_phone_fields(self, text: str) -> dict[str, object]:
        """Recognize phone candidates by shape, then keep only those the
        phonenumbers library confirms are valid. The raw matched substring is
        emitted as-found; E.164 normalization is Module 3's job, not the adapter's.
        """
        try:
            import phonenumbers  # lazy: only needed when a candidate appears
        except ImportError:
            logger.warning("phonenumbers unavailable; skipping phone recognition")
            return {"phones": [], "recognition_methods": {}}

        kept: list[str] = []
        recognition_methods: dict[str, str] = {}
        for raw in _PHONE_CANDIDATE_RE.findall(text):
            candidate = raw.strip()
            try:
                parsed = phonenumbers.parse(candidate, None)  # None: must be E.164-ish (has +)
            except phonenumbers.NumberParseException:
                continue
            if phonenumbers.is_valid_number(parsed):
                kept.append(candidate)  # emit as-found, un-normalized
                recognition_methods.setdefault(candidate, "regex")

        for raw in _BARE_PHONE_CANDIDATE_RE.findall(text):
            candidate = raw.strip()
            try:
                parsed = phonenumbers.parse(candidate, self.default_region)
            except phonenumbers.NumberParseException:
                continue
            if phonenumbers.is_valid_number(parsed):
                kept.append(candidate)  # emit as-found, un-normalized
                recognition_methods.setdefault(candidate, "region_bare")

        phones = _dedupe(kept)
        return {
            "phones": phones,
            "recognition_methods": {
                phone: recognition_methods[phone]
                for phone in phones
                if phone in recognition_methods
            },
        }

    # --- skill recognition (n-gram candidates + taxonomy linking) ------------

    def _extract_skills(self, text: str) -> list[str]:
        """Recognize and link canonical skill names from note prose."""
        return [match["canonical"] for match in self._extract_skill_matches(text)]

    def _extract_skill_matches(self, text: str) -> list[dict[str, object]]:
        """Return canonical skill matches with span and match-method details."""
        matches: list[dict[str, object]] = []
        for canonical, candidate, score, method in self._taxonomy.extract_from_text(text):
            matches.append(
                {
                    "canonical": canonical,
                    "surface": candidate.text,
                    "score": score,
                    "method": method,
                }
            )
        return matches

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


def _taxonomy_from_vocabulary(skill_vocabulary: Iterable[str]) -> SkillTaxonomy:
    """Build a tiny exact-only taxonomy from injected test vocabulary."""
    alias_to_canonical: dict[str, str] = {}
    forms: list[str] = []
    for form in skill_vocabulary:
        cleaned = str(form).strip()
        normalized = normalize_alias(cleaned)
        if cleaned and normalized:
            alias_to_canonical[normalized] = cleaned
            forms.append(cleaned)
    return SkillTaxonomy(
        alias_to_canonical=dict(sorted(alias_to_canonical.items())),
        prose_forms=tuple(sorted(set(forms), key=lambda item: (-len(item), item.casefold()))),
        fuzzy_surfaces=tuple(sorted(alias_to_canonical)),
    )
