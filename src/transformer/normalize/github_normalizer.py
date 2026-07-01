"""GitHub source normalizer: RawRecord ({profile, repos}) -> NormalizedRecord.

profile -> full_name (direct), location (direct), links (direct), headline from
bio (direct). repos -> projects (direct: the repo is a stated fact) and skills
from repo language (inferred: 'owns a Go repo' -> 'knows Go' is an unstated leap).
Forks are dropped (the API cannot attest authorship); null-language repos
contribute no skill.
"""
from __future__ import annotations

from src.transformer.audit import make_event
from src.transformer.models import (
    AuditEvent,
    Links,
    NormalizedProject,
    NormalizedRecord,
    RawRecord,
    TrackedValue,
)
from src.transformer.normalize.formats import classify_location, clean_string, normalize_email, normalize_url
from src.transformer.normalize.skills import canonicalize_skill

_SOURCE = "github"
_TRUST = 0.70
_METHOD_TRUST = {"direct": 1.0, "regex": 0.7, "inferred": 0.5}


def _tracked(value: object, method: str, format_validity: float) -> TrackedValue:
    return TrackedValue(
        value=value, source=_SOURCE, method=method,
        confidence=_TRUST * _METHOD_TRUST[method] * format_validity,
    )


def normalize_github(record: RawRecord) -> NormalizedRecord:
    """Backward-compatible wrapper returning only the normalized record."""
    normalized, _ = normalize_github_with_audit(record)
    return normalized


def normalize_github_with_audit(record: RawRecord) -> tuple[NormalizedRecord, list[AuditEvent]]:
    """Map and normalize one GitHub RawRecord into a NormalizedRecord."""
    f = record.raw_fields
    audit_log: list[AuditEvent] = []
    profile = f.get("profile") or {}
    repos = f.get("repos") or []
    if not isinstance(profile, dict):
        profile = {}

    # full_name
    full_name = None
    name_str = clean_string(profile.get("name"))
    if name_str:
        full_name = _tracked(name_str, "direct", 1.0)
    else:
        audit_log.append(make_event("normalize", "full_name", "field_missing", "source_field_absent", source=_SOURCE))

    # headline from bio
    headline = None
    bio = clean_string(profile.get("bio"))
    if bio:
        headline = _tracked(bio, "direct", 1.0)
    else:
        audit_log.append(make_event("normalize", "headline", "field_missing", "source_field_absent", source=_SOURCE))

    # public profile email, when GitHub exposes one
    emails: list[TrackedValue] = []
    email_val, email_valid = normalize_email(profile.get("email") if isinstance(profile.get("email"), str) else None)
    if email_val is not None:
        emails.append(_tracked(email_val, "direct", email_valid))
    elif profile.get("email"):
        audit_log.append(make_event("normalize", "emails", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=profile.get("email")))
    else:
        audit_log.append(make_event("normalize", "emails", "field_missing", "source_field_absent", source=_SOURCE))

    # location: GitHub location is usually a bare city (positional parse handles it)
    location = None
    raw_location = profile.get("location")
    loc_val, loc_valid = classify_location(raw_location)
    if loc_val is not None:
        location = _tracked(loc_val, "direct", loc_valid)
    elif raw_location:
        audit_log.append(
            make_event("normalize", "location", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=raw_location)
        )
    else:
        audit_log.append(make_event("normalize", "location", "field_missing", "source_field_absent", source=_SOURCE))

    # links: profile html_url (github), blog (portfolio)
    links = None
    github_url, _ = normalize_url(profile.get("html_url"))
    blog_url, _ = normalize_url(profile.get("blog"))
    if github_url or blog_url:
        links = _tracked(
            Links(github=github_url, portfolio=blog_url, other=[]),
            "direct", 1.0,
        )
    else:
        audit_log.append(make_event("normalize", "links", "field_missing", "source_field_absent", source=_SOURCE))

    # projects (non-fork repos, direct) + skills (repo language, inferred)
    projects: list[TrackedValue] = []
    skills: list[TrackedValue] = []
    seen_langs: set[str] = set()
    for repo in repos:
        if not isinstance(repo, dict):
            audit_log.append(make_event("normalize", "repos[]", "entry_dropped", "non_object_array_entry", source=_SOURCE))
            continue
        if repo.get("fork"):
            audit_log.append(
                make_event(
                    "normalize",
                    "projects",
                    "entry_dropped",
                    "fork_repo_excluded",
                    source=_SOURCE,
                    repo_name=repo.get("name"),
                )
            )
            continue  # drop forks: API cannot attest authorship
        # project
        pname = clean_string(repo.get("name"))
        if pname:
            url_val, _ = normalize_url(repo.get("html_url"))
            entry = NormalizedProject(
                name=pname,
                description=clean_string(repo.get("description")),
                url=url_val,
                primary_language=clean_string(repo.get("language")),
            )
            projects.append(_tracked(entry, "direct", 1.0))
        else:
            audit_log.append(make_event("normalize", "projects", "entry_dropped", "missing_project_name", source=_SOURCE))
        # skill from language (inferred), deduped across repos
        lang = clean_string(repo.get("language"))
        if lang:
            canon = canonicalize_skill(lang)
            if canon and canon.lower() not in seen_langs:
                seen_langs.add(canon.lower())
                skills.append(_tracked(canon, "inferred", 1.0))
            elif canon:
                audit_log.append(
                    make_event("normalize", "skills", "value_dropped", "duplicate_canonical_skill", source=_SOURCE, raw_value=canon)
                )
            else:
                audit_log.append(
                    make_event("normalize", "skills", "value_dropped", "failed_canonicalization", source=_SOURCE, raw_value=lang)
                )
        else:
            audit_log.append(
                make_event("normalize", "skills", "value_dropped", "missing_language_for_inference", source=_SOURCE, repo_name=repo.get("name"))
            )

    if not repos:
        audit_log.append(make_event("normalize", "projects", "field_missing", "source_field_absent", source=_SOURCE))
        audit_log.append(make_event("normalize", "skills", "field_missing", "source_field_absent", source=_SOURCE))

    return NormalizedRecord(
        source=_SOURCE,
        full_name=full_name,
        emails=emails,
        phones=[],
        location=location,
        links=links,
        headline=headline,
        years_experience=None,
        skills=skills,
        experience=[],
        education=[],
        projects=projects,
    ), audit_log
