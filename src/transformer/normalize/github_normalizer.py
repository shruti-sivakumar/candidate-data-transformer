"""GitHub source normalizer: RawRecord ({profile, repos}) -> NormalizedRecord.

profile -> full_name (direct), location (direct), links (direct), headline from
bio (direct). repos -> projects (direct: the repo is a stated fact) and skills
from repo language (inferred: 'owns a Go repo' -> 'knows Go' is an unstated leap).
Forks are dropped (the API cannot attest authorship); null-language repos
contribute no skill.
"""
from __future__ import annotations

from src.transformer.models import (
    Links,
    NormalizedProject,
    NormalizedRecord,
    RawRecord,
    TrackedValue,
)
from src.transformer.normalize.formats import classify_location, clean_string, normalize_url
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
    """Map and normalize one GitHub RawRecord into a NormalizedRecord."""
    f = record.raw_fields
    profile = f.get("profile") or {}
    repos = f.get("repos") or []
    if not isinstance(profile, dict):
        profile = {}

    # full_name
    full_name = None
    name_str = clean_string(profile.get("name"))
    if name_str:
        full_name = _tracked(name_str, "direct", 1.0)

    # headline from bio
    headline = None
    bio = clean_string(profile.get("bio"))
    if bio:
        headline = _tracked(bio, "direct", 1.0)

    # location: GitHub location is usually a bare city (positional parse handles it)
    location = None
    loc_val, loc_valid = classify_location(profile.get("location"))
    if loc_val is not None:
        location = _tracked(loc_val, "direct", loc_valid)

    # links: profile html_url (github), blog (portfolio)
    links = None
    github_url, _ = normalize_url(profile.get("html_url"))
    blog_url, _ = normalize_url(profile.get("blog"))
    if github_url or blog_url:
        links = _tracked(
            Links(github=github_url, portfolio=blog_url, other=[]),
            "direct", 1.0,
        )

    # projects (non-fork repos, direct) + skills (repo language, inferred)
    projects: list[TrackedValue] = []
    skills: list[TrackedValue] = []
    seen_langs: set[str] = set()
    for repo in repos:
        if not isinstance(repo, dict):
            continue
        if repo.get("fork"):
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
        # skill from language (inferred), deduped across repos
        lang = clean_string(repo.get("language"))
        if lang:
            canon = canonicalize_skill(lang)
            if canon and canon.lower() not in seen_langs:
                seen_langs.add(canon.lower())
                skills.append(_tracked(canon, "inferred", 1.0))

    return NormalizedRecord(
        source=_SOURCE,
        full_name=full_name,
        emails=[],               # GitHub profile email is usually null in fixtures
        phones=[],
        location=location,
        links=links,
        headline=headline,
        years_experience=None,
        skills=skills,
        experience=[],
        education=[],
        projects=projects,
    )