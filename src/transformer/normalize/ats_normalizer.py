"""ATS source normalizer: RawRecord (Greenhouse nested JSON) -> NormalizedRecord.

Maps the ATS's nested arrays (email_addresses[], phone_numbers[], employments[],
educations[], addresses[]) to canonical fields. All values are direct reads.
Email type tags (work/personal) are not preserved on the TrackedValue — merge
matches emails on normalized value, so the tag is not needed downstream.
"""
from __future__ import annotations

import re

from src.transformer.audit import make_event
from src.transformer.models import (
    AuditEvent,
    EducationEntry,
    ExperienceEntry,
    Links,
    Location,
    NormalizedRecord,
    RawRecord,
    TrackedValue,
)
from src.transformer.normalize.formats import (
    clean_string,
    normalize_country,
    normalize_date,
    normalize_email,
    normalize_phone,
    normalize_url,
)
from src.transformer.normalize.skills import canonicalize_skill

_SOURCE = "ats_json"
_TRUST = 0.90
_METHOD_TRUST = {"direct": 1.0, "regex": 0.7, "inferred": 0.5}


def _tracked(value: object, method: str, format_validity: float) -> TrackedValue:
    return TrackedValue(
        value=value, source=_SOURCE, method=method,
        confidence=_TRUST * _METHOD_TRUST[method] * format_validity,
    )


def _classify_location(raw: str | None) -> tuple[Location | None, float]:
    """Positional location parse (same convention as CSV: city, region, country)."""
    if not raw or not str(raw).strip():
        return None, 0.0
    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        return None, 0.0
    city = region = country = None
    validities: list[float] = []
    if len(tokens) == 1:
        city = clean_string(tokens[0]); validities.append(1.0)
    elif len(tokens) == 2:
        city = clean_string(tokens[0]); validities.append(1.0)
        c_val, c_valid = normalize_country(tokens[1])
        if c_val is not None:
            country = c_val; validities.append(c_valid)
        else:
            region = clean_string(tokens[1]); validities.append(1.0)
    else:
        city = clean_string(tokens[0]); validities.append(1.0)
        c_val, c_valid = normalize_country(tokens[-1])
        if c_val is not None:
            country = c_val; validities.append(c_valid); middle = tokens[1:-1]
        else:
            middle = tokens[1:]
        if middle:
            region = clean_string(", ".join(middle)); validities.append(1.0)
    if city is None and region is None and country is None:
        return None, 0.0
    return Location(city=city, region=region, country=country), min(validities)


def _experience_entry(emp: dict) -> TrackedValue | None:
    """Build experience from a Greenhouse employment object (company is anchor)."""
    company_c = clean_string(str(emp.get("company_name") or "") or None)
    if not company_c:
        return None
    start_raw = emp.get("start_date")
    end_raw = emp.get("end_date")
    start_val, start_valid = normalize_date(str(start_raw)) if start_raw else (None, 1.0)
    end_val, end_valid = normalize_date(str(end_raw)) if end_raw else (None, 1.0)
    entry = ExperienceEntry(
        company=company_c,
        title=clean_string(str(emp.get("title") or "") or None) or "",
        start=start_val, end=end_val, summary=None,
    )
    date_valids = [v for v, a in ((start_valid, start_raw), (end_valid, end_raw)) if a]
    return _tracked(entry, "direct", min(date_valids) if date_valids else 1.0)


def _education_entry(edu: dict) -> TrackedValue | None:
    """Build education from a Greenhouse education object (school is anchor)."""
    inst_c = clean_string(str(edu.get("school_name") or "") or None)
    if not inst_c:
        return None
    year_raw = edu.get("end_date")
    year_val, year_valid = normalize_date(str(year_raw)) if year_raw else (None, 1.0)
    end_year_int = None
    if year_val is not None:
        try:
            end_year_int = int(year_val[:4])
        except (ValueError, TypeError):
            end_year_int = None
    entry = EducationEntry(
        institution=inst_c,
        degree=clean_string(str(edu.get("degree") or "") or None),
        field=clean_string(str(edu.get("discipline") or "") or None),
        end_year=end_year_int,
    )
    return _tracked(entry, "direct", year_valid if year_raw else 1.0)


def _split_skill_values(raw: object) -> list[str]:
    """Extract candidate skill strings from common structured ATS shapes."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item for item in re.split(r"[;,|]", raw) if item.strip()]
    if isinstance(raw, list):
        out: list[str] = []
        for item in raw:
            out.extend(_split_skill_values(item))
        return out
    if isinstance(raw, dict):
        for key in ("name", "skill", "value", "label"):
            if key in raw:
                return _split_skill_values(raw.get(key))
    return []


def _field_name_suggests_skill(name: object) -> bool:
    """Return whether an ATS/custom field name is likely to hold skill values."""
    text = str(name).casefold()
    return any(token in text for token in ("skill", "technolog", "tech stack", "stack"))


def _collect_skill_candidates(fields: dict[str, object]) -> list[str]:
    """Collect skills from explicit ATS skill fields and skill-like custom fields."""
    candidates: list[str] = []
    for key in ("skills", "skill_names", "technologies", "technology_names", "tech_stack"):
        candidates.extend(_split_skill_values(fields.get(key)))

    custom_fields = fields.get("custom_fields") or fields.get("custom_fields_values")
    if isinstance(custom_fields, dict):
        for name, value in custom_fields.items():
            if _field_name_suggests_skill(name):
                candidates.extend(_split_skill_values(value))
    elif isinstance(custom_fields, list):
        for item in custom_fields:
            if not isinstance(item, dict):
                continue
            name = item.get("name") or item.get("field_name") or item.get("label")
            if _field_name_suggests_skill(name):
                candidates.extend(_split_skill_values(item.get("value") or item.get("values")))

    out: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        cleaned = clean_string(candidate)
        if cleaned and cleaned.casefold() not in seen:
            seen.add(cleaned.casefold())
            out.append(cleaned)
    return out


def normalize_ats(record: RawRecord) -> NormalizedRecord:
    """Backward-compatible wrapper returning only the normalized record."""
    normalized, _ = normalize_ats_with_audit(record)
    return normalized


def normalize_ats_with_audit(record: RawRecord) -> tuple[NormalizedRecord, list[AuditEvent]]:
    """Map and normalize one ATS RawRecord into a NormalizedRecord."""
    f = record.raw_fields
    audit_log: list[AuditEvent] = []

    # full_name: first + last
    full_name = None
    first = clean_string(str(f.get("first_name") or "") or None)
    last = clean_string(str(f.get("last_name") or "") or None)
    name_str = " ".join(p for p in (first, last) if p) or None
    if name_str:
        full_name = _tracked(name_str, "direct", 1.0)
    else:
        audit_log.append(make_event("normalize", "full_name", "field_missing", "no_name_parts", source=_SOURCE))

    # emails: email_addresses[].value (type tag dropped; merge matches on value)
    emails: list[TrackedValue] = []
    email_entries = f.get("email_addresses") or []
    for e in email_entries:
        if isinstance(e, dict):
            val, valid = normalize_email(e.get("value"))
            if val is not None:
                emails.append(_tracked(val, "direct", valid))
            elif e.get("value"):
                audit_log.append(
                    make_event("normalize", "emails", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=e.get("value"))
                )
        else:
            audit_log.append(make_event("normalize", "emails", "entry_dropped", "non_object_array_entry", source=_SOURCE))
    if not email_entries:
        audit_log.append(make_event("normalize", "emails", "field_missing", "source_field_absent", source=_SOURCE))

    # phones: phone_numbers[].value
    phones: list[TrackedValue] = []
    phone_entries = f.get("phone_numbers") or []
    for p in phone_entries:
        if isinstance(p, dict):
            val, valid = normalize_phone(p.get("value") if isinstance(p.get("value"), str) else None)
            if val is not None:
                phones.append(_tracked(val, "direct", valid))
            elif p.get("value"):
                audit_log.append(
                    make_event("normalize", "phones", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=p.get("value"))
                )
        else:
            audit_log.append(make_event("normalize", "phones", "entry_dropped", "non_object_array_entry", source=_SOURCE))
    if not phone_entries:
        audit_log.append(make_event("normalize", "phones", "field_missing", "source_field_absent", source=_SOURCE))

    # location: first address value
    location = None
    addresses = f.get("addresses") or []
    if addresses and isinstance(addresses[0], dict):
        loc_val, loc_valid = _classify_location(addresses[0].get("value"))
        if loc_val is not None:
            location = _tracked(loc_val, "direct", loc_valid)
        elif addresses[0].get("value"):
            audit_log.append(
                make_event("normalize", "location", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=addresses[0].get("value"))
            )
    elif addresses:
        audit_log.append(make_event("normalize", "location", "entry_dropped", "non_object_array_entry", source=_SOURCE))
    else:
        audit_log.append(make_event("normalize", "location", "field_missing", "source_field_absent", source=_SOURCE))

    # links: website + social addresses, classified by domain
    links = None
    linkedin = github = portfolio = None
    other: list[str] = []
    link_inputs_present = False
    for key in ("website_addresses", "social_media_addresses"):
        entries = f.get(key) or []
        if entries:
            link_inputs_present = True
        for w in entries:
            if not isinstance(w, dict):
                audit_log.append(make_event("normalize", "links", "entry_dropped", "non_object_array_entry", source=_SOURCE))
                continue
            url_val, _ = normalize_url(w.get("value"))
            if url_val is None:
                if w.get("value"):
                    audit_log.append(
                        make_event("normalize", "links", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=w.get("value"))
                    )
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
            "direct", 1.0,
        )
    elif not link_inputs_present:
        audit_log.append(make_event("normalize", "links", "field_missing", "source_field_absent", source=_SOURCE))

    # experience: employments[]
    experience: list[TrackedValue] = []
    employments = f.get("employments") or []
    for emp in employments:
        if isinstance(emp, dict):
            entry = _experience_entry(emp)
            if entry is not None:
                experience.append(entry)
            elif any(emp.get(key) for key in ("company_name", "title", "start_date", "end_date")):
                audit_log.append(make_event("normalize", "experience", "entry_dropped", "missing_company_anchor", source=_SOURCE))
        else:
            audit_log.append(make_event("normalize", "experience", "entry_dropped", "non_object_array_entry", source=_SOURCE))
    if not employments:
        audit_log.append(make_event("normalize", "experience", "field_missing", "source_field_absent", source=_SOURCE))

    # education: educations[]
    education: list[TrackedValue] = []
    educations = f.get("educations") or []
    for edu in educations:
        if isinstance(edu, dict):
            entry = _education_entry(edu)
            if entry is not None:
                education.append(entry)
            elif any(edu.get(key) for key in ("school_name", "degree", "discipline", "end_date")):
                audit_log.append(make_event("normalize", "education", "entry_dropped", "missing_institution_anchor", source=_SOURCE))
        else:
            audit_log.append(make_event("normalize", "education", "entry_dropped", "non_object_array_entry", source=_SOURCE))
    if not educations:
        audit_log.append(make_event("normalize", "education", "field_missing", "source_field_absent", source=_SOURCE))

    # skills: explicit ATS fields or skill-like custom fields only.
    skills: list[TrackedValue] = []
    for raw_skill in _collect_skill_candidates(f):
        canon = canonicalize_skill(raw_skill)
        if canon and canon.casefold() not in {skill.value.casefold() for skill in skills}:
            skills.append(_tracked(canon, "direct", 1.0))
        elif canon:
            audit_log.append(make_event("normalize", "skills", "value_dropped", "duplicate_canonical_skill", source=_SOURCE, raw_value=canon))
        else:
            audit_log.append(make_event("normalize", "skills", "value_dropped", "failed_canonicalization", source=_SOURCE, raw_value=raw_skill))
    if not skills:
        audit_log.append(make_event("normalize", "skills", "field_missing", "source_field_absent", source=_SOURCE))

    return NormalizedRecord(
        source=_SOURCE,
        full_name=full_name,
        emails=emails,
        phones=phones,
        location=location,
        links=links,
        headline=None,           # ATS fixtures carry no headline field
        years_experience=None,   # not in ATS fixtures
        skills=skills,
        experience=experience,
        education=education,
        projects=[],
    ), audit_log
