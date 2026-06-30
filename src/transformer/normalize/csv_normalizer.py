"""CSV source normalizer: RawRecord (CSV columns) -> NormalizedRecord.

Maps the recruiter CSV's columns to canonical fields, normalizes each value via
the shared format functions, and wraps results in TrackedValues carrying base
confidence (source_trust x method_trust x format_validity). CSV values are direct
column reads, so method is "direct" throughout.

Failed-normalization singles become absent (field stays None); failed list items
are dropped from their list. Field-level issues never raise.
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
    normalize_years,
)

_SOURCE = "recruiter_csv"
_TRUST = 0.80
_METHOD_TRUST = {"direct": 1.0, "regex": 0.7, "inferred": 0.5}


def _confidence(method: str, format_validity: float) -> float:
    """base = source_trust x method_trust x format_validity."""
    return _TRUST * _METHOD_TRUST[method] * format_validity


def _tracked(value: object, method: str, format_validity: float) -> TrackedValue:
    """Wrap a normalized value in a TrackedValue with computed base confidence."""
    return TrackedValue(
        value=value,
        source=_SOURCE,
        method=method,
        confidence=_confidence(method, format_validity),
    )


def _classify_location(raw: str | None) -> tuple[Location | None, float]:
    """Parse a location string into Location, positionally.

    Structured sources order location as 'city, region, country'. We trust that
    positional convention rather than validating each token against a gazetteer —
    gazetteer lookup cannot disambiguate tokens that legitimately belong to
    multiple categories (e.g. 'Washington' is both a city and a state, 'CA' is
    both California and Canada's code). Only the country token is resolved to its
    ISO code; city and region are free-text, cleaned but not validated.

      1 part  -> city
      2 parts -> city, then country if the 2nd token resolves as a country, else region
      3+ parts-> first = city, last = country (if it resolves), middle = region

    ASSUMPTION: positional order for structured sources; validating city/region
    against a geocoding database (e.g. geonamescache) is the named upgrade for
    freeform locations. Validity is the MIN of present sub-field validities.
    """
    if not raw or not str(raw).strip():
        return None, 0.0

    tokens = [t.strip() for t in str(raw).split(",") if t.strip()]
    if not tokens:
        return None, 0.0

    city: str | None = None
    region: str | None = None
    country: str | None = None
    validities: list[float] = []

    if len(tokens) == 1:
        city = clean_string(tokens[0])
        validities.append(1.0)
    elif len(tokens) == 2:
        city = clean_string(tokens[0])
        validities.append(1.0)
        c_val, c_valid = normalize_country(tokens[1])
        if c_val is not None:
            country = c_val
            validities.append(c_valid)
        else:
            region = clean_string(tokens[1])
            validities.append(1.0)
    else:  # 3 or more
        city = clean_string(tokens[0])
        validities.append(1.0)
        c_val, c_valid = normalize_country(tokens[-1])
        if c_val is not None:
            country = c_val
            validities.append(c_valid)
            middle = tokens[1:-1]
        else:
            # last token isn't a country; treat it as part of region, no country
            middle = tokens[1:]
        if middle:
            region = clean_string(", ".join(middle))
            validities.append(1.0)

    if city is None and region is None and country is None:
        return None, 0.0
    location = Location(city=city, region=region, country=country)
    return location, (min(validities) if validities else 0.0)


def _split_multi(raw: str) -> list[str]:
    """Split a multi-value cell on commas or semicolons."""
    return [p for p in re.split(r"[;,]", raw) if p.strip()]


def _experience_entry(
    company: str | None, title: str | None,
    start: str | None, end: str | None,
) -> TrackedValue | None:
    """Build a TrackedValue[ExperienceEntry], or None if no company (the anchor)."""
    company_c = clean_string(company)
    if not company_c:
        return None
    start_val, start_valid = normalize_date(start) if start else (None, 1.0)
    end_val, end_valid = normalize_date(end) if end else (None, 1.0)
    entry = ExperienceEntry(
        company=company_c,
        title=clean_string(title) or "",
        start=start_val,
        end=end_val,
        summary=None,
    )
    date_valids = [
        v for v, attempted in ((start_valid, start), (end_valid, end)) if attempted
    ]
    validity = min(date_valids) if date_valids else 1.0
    return _tracked(entry, "direct", validity)


def _education_entry(
    institution: str | None, degree: str | None,
    field: str | None, end_year: str | None,
) -> TrackedValue | None:
    """Build a TrackedValue[EducationEntry], or None if no institution (the anchor)."""
    inst_c = clean_string(institution)
    if not inst_c:
        return None
    year_val, year_valid = normalize_date(end_year) if end_year else (None, 1.0)
    end_year_int = None
    if year_val is not None:
        try:
            end_year_int = int(year_val[:4])
        except (ValueError, TypeError):
            end_year_int = None
    entry = EducationEntry(
        institution=inst_c,
        degree=clean_string(degree),
        field=clean_string(field),
        end_year=end_year_int,
    )
    return _tracked(entry, "direct", year_valid if end_year else 1.0)


def normalize_csv(record: RawRecord) -> NormalizedRecord:
    """Backward-compatible wrapper returning only the normalized record."""
    normalized, _ = normalize_csv_with_audit(record)
    return normalized


def normalize_csv_with_audit(record: RawRecord) -> tuple[NormalizedRecord, list[AuditEvent]]:
    """Map and normalize one CSV RawRecord into a NormalizedRecord."""
    f = record.raw_fields
    audit_log: list[AuditEvent] = []

    def get(col: str) -> str | None:
        v = f.get(col)
        return str(v) if v is not None else None

    full_name = None
    first = clean_string(get("first_name"))
    last = clean_string(get("last_name"))
    name_str = " ".join(p for p in (first, last) if p) or None
    if name_str:
        full_name = _tracked(name_str, "direct", 1.0)
    else:
        audit_log.append(make_event("normalize", "full_name", "field_missing", "no_name_parts", source=_SOURCE))

    emails: list[TrackedValue] = []
    raw_email = get("email")
    email_val, email_valid = normalize_email(raw_email)
    if email_val is not None:
        emails.append(_tracked(email_val, "direct", email_valid))
    elif raw_email:
        audit_log.append(
            make_event("normalize", "emails", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=raw_email)
        )
    else:
        audit_log.append(make_event("normalize", "emails", "field_missing", "source_field_absent", source=_SOURCE))

    phones: list[TrackedValue] = []
    raw_phone = get("phone")
    phone_val, phone_valid = normalize_phone(raw_phone)
    if phone_val is not None:
        phones.append(_tracked(phone_val, "direct", phone_valid))
    elif raw_phone:
        audit_log.append(
            make_event("normalize", "phones", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=raw_phone)
        )
    else:
        audit_log.append(make_event("normalize", "phones", "field_missing", "source_field_absent", source=_SOURCE))

    location = None
    raw_location = get("location")
    loc_val, loc_valid = _classify_location(raw_location)
    if loc_val is not None:
        location = _tracked(loc_val, "direct", loc_valid)
    elif raw_location:
        audit_log.append(
            make_event("normalize", "location", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=raw_location)
        )
    else:
        audit_log.append(make_event("normalize", "location", "field_missing", "source_field_absent", source=_SOURCE))

    links = None
    raw_linkedin = get("linkedin_url")
    linkedin_val, linkedin_valid = normalize_url(raw_linkedin)
    if linkedin_val is not None:
        links = _tracked(Links(linkedin=linkedin_val), "direct", linkedin_valid)
    elif raw_linkedin:
        audit_log.append(
            make_event("normalize", "links.linkedin", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=raw_linkedin)
        )
    else:
        audit_log.append(make_event("normalize", "links.linkedin", "field_missing", "source_field_absent", source=_SOURCE))

    headline = None
    raw_headline = get("headline")
    headline_str = clean_string(raw_headline)
    if headline_str:
        headline = _tracked(headline_str, "direct", 1.0)
    else:
        audit_log.append(make_event("normalize", "headline", "field_missing", "source_field_absent", source=_SOURCE))

    years_experience = None
    raw_years = get("years_experience")
    years_val, years_valid = normalize_years(raw_years)
    if years_val is not None:
        years_experience = _tracked(years_val, "direct", years_valid)
    elif raw_years:
        audit_log.append(
            make_event("normalize", "years_experience", "value_dropped", "failed_normalization", source=_SOURCE, raw_value=raw_years)
        )
    else:
        audit_log.append(make_event("normalize", "years_experience", "field_missing", "source_field_absent", source=_SOURCE))

    skills: list[TrackedValue] = []
    raw_skills = get("top_skills")
    if raw_skills:
        for piece in _split_multi(raw_skills):
            s = clean_string(piece)
            if s:
                skills.append(_tracked(s, "direct", 1.0))
        if not skills:
            audit_log.append(
                make_event("normalize", "skills", "field_dropped", "no_clean_skill_values", source=_SOURCE, raw_value=raw_skills)
            )
    else:
        audit_log.append(make_event("normalize", "skills", "field_missing", "source_field_absent", source=_SOURCE))

    experience: list[TrackedValue] = []
    cur_company = get("current_company")
    cur_title = get("current_title")
    cur_start = get("current_title_start")
    cur = _experience_entry(
        cur_company, cur_title,
        cur_start, end=None,
    )
    if cur is not None:
        experience.append(cur)
    elif any((cur_company, cur_title, cur_start)):
        audit_log.append(
            make_event("normalize", "experience.current", "entry_dropped", "missing_company_anchor", source=_SOURCE)
        )
    prev = _experience_entry(
        get("prev_company"), get("prev_title"),
        get("prev_start"), get("prev_end"),
    )
    if prev is not None:
        experience.append(prev)
    elif any((get("prev_company"), get("prev_title"), get("prev_start"), get("prev_end"))):
        audit_log.append(
            make_event("normalize", "experience.previous", "entry_dropped", "missing_company_anchor", source=_SOURCE)
        )
    if not experience and not any(
        (
            cur_company, cur_title, cur_start,
            get("prev_company"), get("prev_title"), get("prev_start"), get("prev_end"),
        )
    ):
        audit_log.append(make_event("normalize", "experience", "field_missing", "source_field_absent", source=_SOURCE))

    education: list[TrackedValue] = []
    raw_inst = get("education_institution")
    edu = _education_entry(
        raw_inst, get("education_degree"),
        get("education_field"), get("education_end_year"),
    )
    if edu is not None:
        education.append(edu)
    elif any((raw_inst, get("education_degree"), get("education_field"), get("education_end_year"))):
        audit_log.append(
            make_event("normalize", "education", "entry_dropped", "missing_institution_anchor", source=_SOURCE)
        )
    else:
        audit_log.append(make_event("normalize", "education", "field_missing", "source_field_absent", source=_SOURCE))

    return NormalizedRecord(
        source=_SOURCE,
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
        projects=[],
    ), audit_log
