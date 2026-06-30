"""Notes source normalizer: RawRecord ({emails, phones, urls, skills}) ->
NormalizedRecord.

The notes adapter already recognized values from prose (regex contacts, gazetteer
skills). This normalizer maps those found values to canonical fields. Everything
is method 'regex' (pattern/gazetteer recognized, not a direct field read), giving
the lower 0.7 method trust. URLs are classified into Links by domain.
"""
from __future__ import annotations

from src.transformer.audit import make_event
from src.transformer.models import AuditEvent, Links, NormalizedRecord, RawRecord, TrackedValue
from src.transformer.normalize.formats import (
    normalize_email,
    normalize_phone,
    normalize_url,
)
from src.transformer.normalize.skills import canonicalize_skill

_SOURCE = "recruiter_notes"
_TRUST = 0.55
_METHOD_TRUST = {"direct": 1.0, "regex": 0.7, "inferred": 0.5}


def _tracked(value: object, method: str, format_validity: float) -> TrackedValue:
    return TrackedValue(
        value=value, source=_SOURCE, method=method,
        confidence=_TRUST * _METHOD_TRUST[method] * format_validity,
    )


def normalize_notes(record: RawRecord) -> NormalizedRecord:
    """Backward-compatible wrapper returning only the normalized record."""
    normalized, _ = normalize_notes_with_audit(record)
    return normalized


def normalize_notes_with_audit(record: RawRecord) -> tuple[NormalizedRecord, list[AuditEvent]]:
    """Map and normalize one notes RawRecord into a NormalizedRecord."""
    f = record.raw_fields
    audit_log: list[AuditEvent] = []

    emails: list[TrackedValue] = []
    raw_emails = f.get("emails") or []
    for e in raw_emails:
        val, valid = normalize_email(e)
        if val is not None:
            emails.append(_tracked(val, "regex", valid))
        elif e:
            audit_log.append(make_event("normalize", "emails", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=e))
    if not raw_emails:
        audit_log.append(make_event("normalize", "emails", "field_missing", "source_field_absent", source=_SOURCE))

    phones: list[TrackedValue] = []
    raw_phones = f.get("phones") or []
    for p in raw_phones:
        val, valid = normalize_phone(p if isinstance(p, str) else None)
        if val is not None:
            phones.append(_tracked(val, "regex", valid))
        elif p:
            audit_log.append(make_event("normalize", "phones", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=p))
    if not raw_phones:
        audit_log.append(make_event("normalize", "phones", "field_missing", "source_field_absent", source=_SOURCE))

    # urls -> links, classified by domain
    links = None
    linkedin = github = portfolio = None
    other: list[str] = []
    raw_urls = f.get("urls") or []
    for u in raw_urls:
        url_val, _ = normalize_url(u)
        if url_val is None:
            if u:
                audit_log.append(make_event("normalize", "links", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=u))
            continue
        low = url_val.lower()
        if "linkedin.com" in low and linkedin is None:
            linkedin = url_val
        elif "github.com" in low and github is None:
            github = url_val
        else:
            other.append(url_val)
    if any((linkedin, github, portfolio, other)):
        links = _tracked(
            Links(linkedin=linkedin, github=github, portfolio=portfolio, other=other),
            "regex", 1.0,
        )
    elif not raw_urls:
        audit_log.append(make_event("normalize", "links", "field_missing", "source_field_absent", source=_SOURCE))

    skills: list[TrackedValue] = []
    seen: set[str] = set()
    raw_skills = f.get("skills") or []
    for s in raw_skills:
        canon = canonicalize_skill(s)
        if canon and canon.lower() not in seen:
            seen.add(canon.lower())
            skills.append(_tracked(canon, "regex", 1.0))
        elif canon:
            audit_log.append(make_event("normalize", "skills", "value_dropped", "duplicate_canonical_skill", source=_SOURCE, raw_value=canon))
        else:
            audit_log.append(make_event("normalize", "skills", "value_dropped", "failed_canonicalization", source=_SOURCE, raw_value=s))
    if not raw_skills:
        audit_log.append(make_event("normalize", "skills", "field_missing", "source_field_absent", source=_SOURCE))

    return NormalizedRecord(
        source=_SOURCE,
        full_name=None,          # prose name extraction descoped (see handoff)
        emails=emails,
        phones=phones,
        location=None,
        links=links,
        headline=None,
        years_experience=None,
        skills=skills,
        experience=[],
        education=[],
        projects=[],
    ), audit_log
