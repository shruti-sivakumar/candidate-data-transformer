"""Notes source normalizer: RawRecord ({emails, phones, urls, skills}) ->
NormalizedRecord.

The notes adapter already recognized values from prose (regex contacts, gazetteer
skills). This normalizer maps those found values to canonical fields. Everything
is method 'regex' (pattern/gazetteer recognized, not a direct field read), giving
the lower 0.7 method trust. URLs are classified into Links by domain.
"""
from __future__ import annotations

from src.transformer.models import Links, NormalizedRecord, RawRecord, TrackedValue
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
    """Map and normalize one notes RawRecord into a NormalizedRecord."""
    f = record.raw_fields

    emails: list[TrackedValue] = []
    for e in f.get("emails") or []:
        val, valid = normalize_email(e)
        if val is not None:
            emails.append(_tracked(val, "regex", valid))

    phones: list[TrackedValue] = []
    for p in f.get("phones") or []:
        val, valid = normalize_phone(p if isinstance(p, str) else None)
        if val is not None:
            phones.append(_tracked(val, "regex", valid))

    # urls -> links, classified by domain
    links = None
    linkedin = github = portfolio = None
    other: list[str] = []
    for u in f.get("urls") or []:
        url_val, _ = normalize_url(u)
        if url_val is None:
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

    skills: list[TrackedValue] = []
    seen: set[str] = set()
    for s in f.get("skills") or []:
        canon = canonicalize_skill(s)
        if canon and canon.lower() not in seen:
            seen.add(canon.lower())
            skills.append(_tracked(canon, "regex", 1.0))

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
    )