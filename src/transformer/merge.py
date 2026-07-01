"""Module 4: merge normalized per-source records into one canonical profile."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import re
from typing import TypeVar

from pydantic import BaseModel

from src.transformer.models import (
    AggregatedValue,
    AuditEvent,
    CanonicalProfile,
    EducationEntry,
    ExperienceEntry,
    Links,
    Location,
    MergedEducationEntry,
    MergedExperienceEntry,
    NormalizedProject,
    NormalizedRecord,
    ProjectEntry,
    SkillEntry,
    TrackedValue,
)

T = TypeVar("T")

_SINGLE_VALUE_THRESHOLD = 0.70
_CONFLICT_FLOOR = 0.30
_CONFLICT_STEP = 0.15
_COMPANY_SUFFIXES = {
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "company",
}


@dataclass(frozen=True)
class _Group:
    key: str
    values: list[TrackedValue]
    sources: list[str]
    confidence: float
    winner: TrackedValue


def _serialize(value: object) -> str:
    """Return a deterministic string key for grouping values."""

    def convert(obj: object) -> object:
        if isinstance(obj, BaseModel):
            return {k: convert(v) for k, v in obj.model_dump(mode="python").items()}
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in sorted(obj.items())}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    return json.dumps(convert(value), sort_keys=True, ensure_ascii=True, default=str)


def _sorted_unique(values: list[str]) -> list[str]:
    """Return sorted unique strings."""
    return sorted(set(values))


def _winner(values: list[TrackedValue[T]]) -> TrackedValue[T]:
    """Pick the deterministic winner inside one agreeing group."""
    return sorted(values, key=lambda tv: (-tv.confidence, tv.source))[0]


def _noisy_or(confidences: list[float]) -> float:
    """Combine independent agreeing confidences with diminishing returns."""
    if not confidences:
        return 0.0
    product = 1.0
    for conf in confidences:
        product *= (1.0 - conf)
    return max(0.0, min(1.0, 1.0 - product))


def _group_values(values: list[TrackedValue[T]]) -> list[_Group]:
    """Group tracked values by canonical equality."""
    buckets: dict[str, list[TrackedValue[T]]] = defaultdict(list)
    for value in values:
        buckets[_serialize(value.value)].append(value)
    groups: list[_Group] = []
    for key, bucket in buckets.items():
        winner = _winner(bucket)
        groups.append(
            _Group(
                key=key,
                values=bucket,
                sources=_sorted_unique([tv.source for tv in bucket]),
                confidence=_noisy_or([tv.confidence for tv in bucket]),
                winner=winner,
            )
        )
    return sorted(groups, key=lambda group: (-group.confidence, group.winner.source, group.key))


def _conflict_penalty(losing_sources: int) -> float:
    """Penalize conflicts while keeping a non-zero confidence floor."""
    return max(_CONFLICT_FLOOR, 1.0 - (_CONFLICT_STEP * losing_sources))


def _event(
    stage: str,
    field: str,
    kind: str,
    reason: str,
    **details: object,
) -> AuditEvent:
    """Build one audit event with a small, typed details payload."""
    return AuditEvent(stage=stage, field=field, kind=kind, reason=reason, details=details)


def _merge_single_value(
    field: str,
    values: list[TrackedValue[T]],
    audit_log: list[AuditEvent],
) -> TrackedValue[T] | None:
    """Resolve one canonical single-value field."""
    if not values:
        audit_log.append(
            _event("merge", field, "field_missing", "no_source_provided_value")
        )
        return None

    groups = _group_values(values)
    best = groups[0]
    merged_confidence = best.confidence
    merged_method = "merged" if len(best.sources) > 1 else best.winner.method

    if len(groups) > 1:
        losing_sources = sum(len(group.sources) for group in groups[1:])
        merged_confidence *= _conflict_penalty(losing_sources)
        audit_log.append(
            _event(
                "merge",
                field,
                "conflict_resolved",
                "highest_confidence_group_won",
                winner_source=best.winner.source,
                winner_value=best.winner.value,
                winner_confidence=round(merged_confidence, 6),
                losing_sources=_sorted_unique(
                    [src for group in groups[1:] for src in group.sources]
                ),
                losing_values=[group.values[0].value for group in groups[1:]],
            )
        )
    elif len(best.sources) > 1:
        audit_log.append(
            _event(
                "merge",
                field,
                "agreement_merged",
                "multiple_sources_agreed",
                sources=best.sources,
                confidence=round(merged_confidence, 6),
            )
        )

    if merged_confidence < _SINGLE_VALUE_THRESHOLD:
        audit_log.append(
            _event(
                "merge",
                field,
                "field_dropped",
                "winner_below_threshold",
                threshold=_SINGLE_VALUE_THRESHOLD,
                winner_source=best.winner.source,
                winner_value=best.winner.value,
                winner_confidence=round(merged_confidence, 6),
            )
        )
        return None

    return TrackedValue(
        value=best.winner.value,
        source=best.winner.source,
        method=merged_method,
        confidence=merged_confidence,
    )


def _merge_multi_values(
    field: str,
    values: list[TrackedValue[T]],
    audit_log: list[AuditEvent],
) -> list[AggregatedValue[T]]:
    """Deduplicate and aggregate exact list values across sources."""
    groups = _group_values(values)
    merged: list[AggregatedValue[T]] = []
    for group in groups:
        merged.append(
            AggregatedValue(
                value=group.winner.value,
                confidence=group.confidence,
                sources=group.sources,
            )
        )
        if len(group.sources) > 1:
            audit_log.append(
                _event(
                    "merge",
                    field,
                    "agreement_merged",
                    "duplicate_values_collapsed",
                    value=group.winner.value,
                    sources=group.sources,
                    confidence=round(group.confidence, 6),
                )
            )
    return merged


def _normalize_company(name: str | None) -> str:
    """Normalize a company name just enough for suffix-insensitive matching."""
    if not name:
        return ""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", name.casefold())
    tokens = [token for token in cleaned.split() if token]
    while tokens and tokens[-1] in _COMPANY_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _normalize_institution(name: str | None) -> str:
    """Normalize an institution name for exact matching."""
    if not name:
        return ""
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", name.casefold())
    return " ".join(cleaned.split())


def _normalize_project_name(name: str | None) -> str:
    """Normalize a project name for exact matching."""
    if not name:
        return ""
    return re.sub(r"\s+", " ", name.casefold()).strip()


def _dates_compatible(
    left_start: str | None,
    left_end: str | None,
    right_start: str | None,
    right_end: str | None,
) -> bool:
    """Treat dates as supporting evidence when both sides actually have them."""
    if left_start and right_start and left_start != right_start:
        return False
    if left_end and right_end and left_end != right_end:
        return False
    return True


def _experience_groups(values: list[TrackedValue[ExperienceEntry]]) -> list[list[TrackedValue[ExperienceEntry]]]:
    """Cluster experience entries by company, with dates as supporting evidence."""
    groups: list[list[TrackedValue[ExperienceEntry]]] = []
    for value in values:
        matched = False
        norm_company = _normalize_company(value.value.company)
        for group in groups:
            exemplar = group[0].value
            if _normalize_company(exemplar.company) != norm_company:
                continue
            if _dates_compatible(
                exemplar.start,
                exemplar.end,
                value.value.start,
                value.value.end,
            ):
                group.append(value)
                matched = True
                break
        if not matched:
            groups.append([value])
    return groups


def _education_groups(values: list[TrackedValue[EducationEntry]]) -> list[list[TrackedValue[EducationEntry]]]:
    """Cluster education entries by institution, with year as supporting evidence."""
    groups: list[list[TrackedValue[EducationEntry]]] = []
    for value in values:
        matched = False
        norm_institution = _normalize_institution(value.value.institution)
        for group in groups:
            exemplar = group[0].value
            if _normalize_institution(exemplar.institution) != norm_institution:
                continue
            if exemplar.end_year and value.value.end_year and exemplar.end_year != value.value.end_year:
                continue
            group.append(value)
            matched = True
            break
        if not matched:
            groups.append([value])
    return groups


def _project_groups(values: list[TrackedValue[NormalizedProject]]) -> list[list[TrackedValue[NormalizedProject]]]:
    """Cluster projects by normalized name, using URL as a supporting signal."""
    groups: list[list[TrackedValue[NormalizedProject]]] = []
    for value in values:
        matched = False
        norm_name = _normalize_project_name(value.value.name)
        for group in groups:
            exemplar = group[0].value
            if _normalize_project_name(exemplar.name) != norm_name:
                continue
            if exemplar.url and value.value.url and exemplar.url != value.value.url:
                continue
            group.append(value)
            matched = True
            break
        if not matched:
            groups.append([value])
    return groups


def _merge_experience(
    values: list[TrackedValue[ExperienceEntry]],
    audit_log: list[AuditEvent],
) -> list[MergedExperienceEntry]:
    """Merge matched experience entries into aggregated canonical entries."""
    merged: list[MergedExperienceEntry] = []
    for group in _experience_groups(values):
        company = _merge_single_value(
            "experience.company",
            [TrackedValue(value=item.value.company, source=item.source, method=item.method, confidence=item.confidence) for item in group],
            audit_log,
        )
        if company is None:
            audit_log.append(
                _event(
                    "merge",
                    "experience",
                    "entry_dropped",
                    "company_anchor_missing_after_merge",
                    sources=_sorted_unique([item.source for item in group]),
                )
            )
            continue
        title = _merge_single_value(
            "experience.title",
            [
                TrackedValue(value=item.value.title, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.title
            ],
            audit_log,
        )
        start = _merge_single_value(
            "experience.start",
            [
                TrackedValue(value=item.value.start, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.start
            ],
            audit_log,
        )
        end = _merge_single_value(
            "experience.end",
            [
                TrackedValue(value=item.value.end, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.end
            ],
            audit_log,
        )
        summary = _merge_single_value(
            "experience.summary",
            [
                TrackedValue(value=item.value.summary, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.summary
            ],
            audit_log,
        )
        sources = _sorted_unique([item.source for item in group])
        confidence = _noisy_or([item.confidence for item in group])
        merged.append(
            MergedExperienceEntry(
                company=company.value,
                title=title.value if title else None,
                start=start.value if start else None,
                end=end.value if end else None,
                summary=summary.value if summary else None,
                confidence=confidence,
                sources=sources,
            )
        )
        if len(sources) > 1:
            audit_log.append(
                _event(
                    "merge",
                    "experience",
                    "entry_merged",
                    "matching_company_entries_collapsed",
                    sources=sources,
                    company=company.value,
                )
            )
    return sorted(merged, key=lambda entry: (entry.company.casefold(), entry.start or "", entry.end or ""))


def _merge_education(
    values: list[TrackedValue[EducationEntry]],
    audit_log: list[AuditEvent],
) -> list[MergedEducationEntry]:
    """Merge matched education entries into aggregated canonical entries."""
    merged: list[MergedEducationEntry] = []
    for group in _education_groups(values):
        institution = _merge_single_value(
            "education.institution",
            [TrackedValue(value=item.value.institution, source=item.source, method=item.method, confidence=item.confidence) for item in group],
            audit_log,
        )
        if institution is None:
            continue
        degree = _merge_single_value(
            "education.degree",
            [
                TrackedValue(value=item.value.degree, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.degree
            ],
            audit_log,
        )
        field = _merge_single_value(
            "education.field",
            [
                TrackedValue(value=item.value.field, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.field
            ],
            audit_log,
        )
        end_year = _merge_single_value(
            "education.end_year",
            [
                TrackedValue(value=item.value.end_year, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.end_year is not None
            ],
            audit_log,
        )
        sources = _sorted_unique([item.source for item in group])
        confidence = _noisy_or([item.confidence for item in group])
        merged.append(
            MergedEducationEntry(
                institution=institution.value,
                degree=degree.value if degree else None,
                field=field.value if field else None,
                end_year=end_year.value if end_year else None,
                confidence=confidence,
                sources=sources,
            )
        )
    return sorted(merged, key=lambda entry: (entry.institution.casefold(), entry.end_year or 0))


def _merge_projects(
    values: list[TrackedValue[NormalizedProject]],
    audit_log: list[AuditEvent],
) -> list[ProjectEntry]:
    """Merge matched project entries into aggregated canonical entries."""
    merged: list[ProjectEntry] = []
    for group in _project_groups(values):
        name = _merge_single_value(
            "projects.name",
            [TrackedValue(value=item.value.name, source=item.source, method=item.method, confidence=item.confidence) for item in group],
            audit_log,
        )
        if name is None:
            continue
        description = _merge_single_value(
            "projects.description",
            [
                TrackedValue(value=item.value.description, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.description
            ],
            audit_log,
        )
        url = _merge_single_value(
            "projects.url",
            [
                TrackedValue(value=item.value.url, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.url
            ],
            audit_log,
        )
        language = _merge_single_value(
            "projects.primary_language",
            [
                TrackedValue(value=item.value.primary_language, source=item.source, method=item.method, confidence=item.confidence)
                for item in group
                if item.value.primary_language
            ],
            audit_log,
        )
        sources = _sorted_unique([item.source for item in group])
        confidence = _noisy_or([item.confidence for item in group])
        merged.append(
            ProjectEntry(
                name=name.value,
                description=description.value if description else None,
                url=url.value if url else None,
                primary_language=language.value if language else None,
                confidence=confidence,
                sources=sources,
            )
        )
    return sorted(merged, key=lambda entry: entry.name.casefold())


def _merge_skills(values: list[TrackedValue[str]], audit_log: list[AuditEvent]) -> list[SkillEntry]:
    """Aggregate canonical skill strings into merged skill entries."""
    merged = _merge_multi_values("skills", values, audit_log)
    return [
        SkillEntry(name=item.value, confidence=item.confidence, sources=item.sources)
        for item in merged
    ]


def _merge_links(
    values: list[TrackedValue[Links]],
    audit_log: list[AuditEvent],
) -> TrackedValue[Links] | None:
    """Merge the Links object per subfield rather than atomically."""
    if not values:
        audit_log.append(
            _event("merge", "links", "field_missing", "no_source_provided_value")
        )
        return None

    linkedin_values: list[TrackedValue[str]] = []
    github_values: list[TrackedValue[str]] = []
    portfolio_values: list[TrackedValue[str]] = []
    other_values: list[TrackedValue[str]] = []
    for value in values:
        if value.value.linkedin:
            linkedin_values.append(
                TrackedValue(
                    value=value.value.linkedin,
                    source=value.source,
                    method=value.method,
                    confidence=value.confidence,
                )
            )
        if value.value.github:
            github_values.append(
                TrackedValue(
                    value=value.value.github,
                    source=value.source,
                    method=value.method,
                    confidence=value.confidence,
                )
            )
        if value.value.portfolio:
            portfolio_values.append(
                TrackedValue(
                    value=value.value.portfolio,
                    source=value.source,
                    method=value.method,
                    confidence=value.confidence,
                )
            )
        for other in value.value.other:
            other_values.append(
                TrackedValue(
                    value=other,
                    source=value.source,
                    method=value.method,
                    confidence=value.confidence,
                )
            )

    linkedin = _merge_single_value("links.linkedin", linkedin_values, audit_log)
    github = _merge_single_value("links.github", github_values, audit_log)
    portfolio = _merge_single_value("links.portfolio", portfolio_values, audit_log)
    other = _merge_multi_values("links.other", other_values, audit_log)
    if not any((linkedin, github, portfolio, other)):
        audit_log.append(
            _event("merge", "links", "field_missing", "all_subfields_empty_after_merge")
        )
        return None

    contributing_sources = _sorted_unique(
        ([linkedin.source] if linkedin else [])
        + ([github.source] if github else [])
        + ([portfolio.source] if portfolio else [])
        + [src for item in other for src in item.sources]
    )
    confidences = [link.confidence for link in (linkedin, github, portfolio) if link is not None]
    if other:
        confidences.append(sum(item.confidence for item in other) / len(other))
    best_source = contributing_sources[0]
    best_confidence = sum(confidences) / len(confidences)
    return TrackedValue(
        value=Links(
            linkedin=linkedin.value if linkedin else None,
            github=github.value if github else None,
            portfolio=portfolio.value if portfolio else None,
            other=[item.value for item in other],
        ),
        source=best_source,
        method="merged" if len(contributing_sources) > 1 else "direct",
        confidence=best_confidence,
    )


def _candidate_id(
    full_name: TrackedValue[str] | None,
    emails: list[AggregatedValue[str]],
    phones: list[AggregatedValue[str]],
    location: TrackedValue[Location] | None,
) -> str:
    """Generate a deterministic, non-PII candidate ID from stable identity signals."""
    if emails:
        base = sorted(emails, key=lambda item: (-item.confidence, item.value))[0].value
    elif full_name and phones:
        best_phone = sorted(phones, key=lambda item: (-item.confidence, item.value))[0].value
        base = f"{full_name.value}|{best_phone}"
    elif full_name and location:
        base = f"{full_name.value}|{_serialize(location.value)}"
    elif full_name:
        base = full_name.value
    else:
        base = "candidate"
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()[:8]
    return f"cand_{digest}"


def _identity_tokens(record: NormalizedRecord) -> set[tuple[str, str]]:
    """Return the atomic identity anchors a single record exposes.

    These are exactly the signals ``_candidate_id`` keys on independently: an
    email (its top fallback tier) and a full name (its weakest tier). Phone and
    location never anchor identity on their own in that chain — they only ever
    refine a name — so they are not emitted as standalone tokens here. Two
    records that share any token are taken to describe the same candidate.
    """
    tokens: set[tuple[str, str]] = set()
    for email in record.emails:
        if email.value:
            tokens.add(("email", email.value.strip().casefold()))
    if record.full_name and record.full_name.value:
        tokens.add(("name", record.full_name.value.strip().casefold()))
    return tokens


def group_records_by_candidate(records: list[NormalizedRecord]) -> list[list[int]]:
    """Partition record indices into per-candidate groups before merge.

    Records are transitively linked when they share an identity token (see
    ``_identity_tokens``): e.g. an ATS row and a notes blob sharing an email, or
    a GitHub profile sharing only a name with the CSV row that also carries the
    email. A record with no identity signal at all becomes its own group.

    Groups (and the records within them) preserve first-appearance order so the
    downstream single-candidate pipeline runs deterministically. This is the
    ONLY new identity logic: merge/score/project are unchanged and each group is
    fed to them exactly as a lone candidate always has been.
    """
    parent = list(range(len(records)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    token_owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(records):
        for token in _identity_tokens(record):
            if token in token_owner:
                union(index, token_owner[token])
            else:
                token_owner[token] = index

    groups: dict[int, list[int]] = {}
    order: list[int] = []
    for index in range(len(records)):
        root = find(index)
        if root not in groups:
            groups[root] = []
            order.append(root)
        groups[root].append(index)
    return [groups[root] for root in order]


def merge_records(records: list[NormalizedRecord]) -> tuple[CanonicalProfile, list[AuditEvent]]:
    """Merge one candidate's normalized source records into one canonical profile."""
    if not records:
        raise ValueError("merge_records() requires at least one NormalizedRecord")

    audit_log: list[AuditEvent] = []

    full_name = _merge_single_value(
        "full_name",
        [record.full_name for record in records if record.full_name is not None],
        audit_log,
    )
    emails = _merge_multi_values(
        "emails",
        [email for record in records for email in record.emails],
        audit_log,
    )
    phones = _merge_multi_values(
        "phones",
        [phone for record in records for phone in record.phones],
        audit_log,
    )
    location = _merge_single_value(
        "location",
        [record.location for record in records if record.location is not None],
        audit_log,
    )
    links = _merge_links(
        [record.links for record in records if record.links is not None],
        audit_log,
    )
    headline = _merge_single_value(
        "headline",
        [record.headline for record in records if record.headline is not None],
        audit_log,
    )
    years_experience = _merge_single_value(
        "years_experience",
        [record.years_experience for record in records if record.years_experience is not None],
        audit_log,
    )
    skills = _merge_skills(
        [skill for record in records for skill in record.skills],
        audit_log,
    )
    experience = _merge_experience(
        [entry for record in records for entry in record.experience],
        audit_log,
    )
    education = _merge_education(
        [entry for record in records for entry in record.education],
        audit_log,
    )
    projects = _merge_projects(
        [project for record in records for project in record.projects],
        audit_log,
    )

    candidate_id = _candidate_id(full_name, emails, phones, location)

    # The module model still requires these core single-value fields; fall back
    # to honestly-empty placeholders only when the field is truly absent.
    full_name = full_name or TrackedValue(
        value="",
        source="merge",
        method="merged",
        confidence=0.0,
    )
    location = location or TrackedValue(
        value=Location(),
        source="merge",
        method="merged",
        confidence=0.0,
    )
    links = links or TrackedValue(
        value=Links(),
        source="merge",
        method="merged",
        confidence=0.0,
    )
    headline = headline or TrackedValue(
        value=None,
        source="merge",
        method="merged",
        confidence=0.0,
    )
    years_experience = years_experience or TrackedValue(
        value=None,
        source="merge",
        method="merged",
        confidence=0.0,
    )

    profile = CanonicalProfile(
        candidate_id=candidate_id,
        full_name=full_name,
        emails=emails,
        phones=phones,
        location=location,
        links=links,
        headline=headline,
        years_experience=years_experience,
        skills=skills,
        experience=experience,
        education=education,
        projects=projects,
        overall_confidence=0.0,
    )
    return profile, audit_log
