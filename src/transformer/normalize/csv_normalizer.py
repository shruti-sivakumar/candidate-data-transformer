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

from src.transformer.models import (
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
    """Map and normalize one CSV RawRecord into a NormalizedRecord."""
    f = record.raw_fields

    def get(col: str) -> str | None:
        v = f.get(col)
        return str(v) if v is not None else None

    full_name = None
    first = clean_string(get("first_name"))
    last = clean_string(get("last_name"))
    name_str = " ".join(p for p in (first, last) if p) or None
    if name_str:
        full_name = _tracked(name_str, "direct", 1.0)

    emails: list[TrackedValue] = []
    email_val, email_valid = normalize_email(get("email"))
    if email_val is not None:
        emails.append(_tracked(email_val, "direct", email_valid))

    phones: list[TrackedValue] = []
    phone_val, phone_valid = normalize_phone(get("phone"))
    if phone_val is not None:
        phones.append(_tracked(phone_val, "direct", phone_valid))

    location = None
    loc_val, loc_valid = _classify_location(get("location"))
    if loc_val is not None:
        location = _tracked(loc_val, "direct", loc_valid)

    links = None
    linkedin_val, linkedin_valid = normalize_url(get("linkedin_url"))
    if linkedin_val is not None:
        links = _tracked(Links(linkedin=linkedin_val), "direct", linkedin_valid)

    headline = None
    headline_str = clean_string(get("headline"))
    if headline_str:
        headline = _tracked(headline_str, "direct", 1.0)

    years_experience = None
    years_val, years_valid = normalize_years(get("years_experience"))
    if years_val is not None:
        years_experience = _tracked(years_val, "direct", years_valid)

    skills: list[TrackedValue] = []
    raw_skills = get("top_skills")
    if raw_skills:
        for piece in _split_multi(raw_skills):
            s = clean_string(piece)
            if s:
                skills.append(_tracked(s, "direct", 1.0))

    experience: list[TrackedValue] = []
    cur = _experience_entry(
        get("current_company"), get("current_title"),
        get("current_title_start"), end=None,
    )
    if cur is not None:
        experience.append(cur)
    prev = _experience_entry(
        get("prev_company"), get("prev_title"),
        get("prev_start"), get("prev_end"),
    )
    if prev is not None:
        experience.append(prev)

    education: list[TrackedValue] = []
    edu = _education_entry(
        get("education_institution"), get("education_degree"),
        get("education_field"), get("education_end_year"),
    )
    if edu is not None:
        education.append(edu)

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
    )