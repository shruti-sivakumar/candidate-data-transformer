"""ATS source normalizer: RawRecord (Greenhouse nested JSON) -> NormalizedRecord.

Maps the ATS's nested arrays (email_addresses[], phone_numbers[], employments[],
educations[], addresses[]) to canonical fields. All values are direct reads.
Email type tags (work/personal) are not preserved on the TrackedValue — merge
matches emails on normalized value, so the tag is not needed downstream.
"""
from __future__ import annotations

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
)

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


def normalize_ats(record: RawRecord) -> NormalizedRecord:
    """Map and normalize one ATS RawRecord into a NormalizedRecord."""
    f = record.raw_fields

    # full_name: first + last
    full_name = None
    first = clean_string(str(f.get("first_name") or "") or None)
    last = clean_string(str(f.get("last_name") or "") or None)
    name_str = " ".join(p for p in (first, last) if p) or None
    if name_str:
        full_name = _tracked(name_str, "direct", 1.0)

    # emails: email_addresses[].value (type tag dropped; merge matches on value)
    emails: list[TrackedValue] = []
    for e in f.get("email_addresses") or []:
        if isinstance(e, dict):
            val, valid = normalize_email(e.get("value"))
            if val is not None:
                emails.append(_tracked(val, "direct", valid))

    # phones: phone_numbers[].value
    phones: list[TrackedValue] = []
    for p in f.get("phone_numbers") or []:
        if isinstance(p, dict):
            val, valid = normalize_phone(p.get("value") if isinstance(p.get("value"), str) else None)
            if val is not None:
                phones.append(_tracked(val, "direct", valid))

    # location: first address value
    location = None
    addresses = f.get("addresses") or []
    if addresses and isinstance(addresses[0], dict):
        loc_val, loc_valid = _classify_location(addresses[0].get("value"))
        if loc_val is not None:
            location = _tracked(loc_val, "direct", loc_valid)

    # links: website + social addresses, classified by domain
    links = None
    linkedin = github = portfolio = None
    other: list[str] = []
    for key in ("website_addresses", "social_media_addresses"):
        for w in f.get(key) or []:
            if not isinstance(w, dict):
                continue
            url_val, _ = normalize_url(w.get("value"))
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
            "direct", 1.0,
        )

    # experience: employments[]
    experience: list[TrackedValue] = []
    for emp in f.get("employments") or []:
        if isinstance(emp, dict):
            entry = _experience_entry(emp)
            if entry is not None:
                experience.append(entry)

    # education: educations[]
    education: list[TrackedValue] = []
    for edu in f.get("educations") or []:
        if isinstance(edu, dict):
            entry = _education_entry(edu)
            if entry is not None:
                education.append(entry)

    return NormalizedRecord(
        source=_SOURCE,
        full_name=full_name,
        emails=emails,
        phones=phones,
        location=location,
        links=links,
        headline=None,           # ATS fixtures carry no headline field
        years_experience=None,   # not in ATS fixtures
        skills=[],               # ATS fixtures carry no skills
        experience=experience,
        education=education,
        projects=[],
    )