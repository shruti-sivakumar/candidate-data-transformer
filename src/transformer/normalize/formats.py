"""Shared format normalizers. Pure functions: (raw value) -> (canonical value, format_validity).

Each returns the normalized value (or None if it could not be normalized) plus a
format_validity score in [0.0, 1.0] that feeds base confidence. Validity is
computed here because normalization is where 'did this parse into valid canonical
form' is actually determined.
"""
from __future__ import annotations

import unicodedata

from src.transformer.models import Location

def clean_string(raw: str | None) -> str | None:
    """Canonicalize a free-text string: Unicode NFC + collapse whitespace.

    Returns the cleaned string, or None if empty after cleaning. No validity
    tuple — cleaning cannot "fail", it either yields text or nothing. NFC
    normalization ensures two visually-identical strings in different Unicode
    encodings (e.g. composed vs decomposed 'é') compare equal in merge.
    """
    if raw is None:
        return None
    # NFC: canonical composition, so 'café' (e + combining accent) == 'café' (é).
    text = unicodedata.normalize("NFC", str(raw))
    # Collapse internal whitespace runs to single spaces, strip ends.
    text = " ".join(text.split())
    return text or None


def normalize_phone(raw: str, default_region: str | None = None) -> tuple[str | None, float]:
    """Normalize a phone string to E.164.

    Returns (e164_string, validity). Validity is:
      1.0  — parsed and is_valid_number (a real, dialable number)
      0.6  — parsed and is_possible_number but not is_valid (right shape, not confirmed)
      0.0  — could not parse, or not even possibly valid → value is None

    default_region: ISO-3166 alpha-2 (e.g. 'US') used to interpret numbers that
    lack a '+' country code. None means the number must already be international.
    """
    if not raw or not raw.strip():
        return None, 0.0

    try:
        import phonenumbers
    except ImportError:
        # phonenumbers is in the locked stack; if it's somehow absent, fail honest.
        return None, 0.0

    try:
        parsed = phonenumbers.parse(raw, default_region)
    except phonenumbers.NumberParseException:
        return None, 0.0

    if phonenumbers.is_valid_number(parsed):
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164, 1.0

    if phonenumbers.is_possible_number(parsed):
        # Right length/shape but not a confirmed valid number — keep it, flag lower.
        e164 = phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
        return e164, 0.6

    return None, 0.0


def normalize_date(raw: str) -> tuple[str | None, float]:
    """Normalize a date string to canonical YYYY-MM, or YYYY when only year
    precision is available.

    Returns (canonical, validity):
      ("YYYY-MM", 1.0) — full month precision parsed cleanly
      ("YYYY",    0.8) — only year precision available (e.g. an education end year)
      (None,      0.0) — could not parse a usable date

    Year-only input is accepted, not rejected: it is real, lower-precision
    information, and the reduced validity reflects the missing month rather than
    inventing one or discarding the year.
    """
    if not raw or not str(raw).strip():
        return None, 0.0

    text = str(raw).strip()

    # Year-only fast path: a bare 4-digit year in a sensible range. Handled
    # explicitly because dateutil would fill in a default month/day, masking
    # that the source only gave a year.
    if text.isdigit() and len(text) == 4:
        year = int(text)
        if 1900 <= year <= 2100:
            return text, 0.8
        return None, 0.0

    try:
        from dateutil import parser as date_parser
    except ImportError:
        return None, 0.0

    try:
        # default with day=1 so a YYYY-MM input doesn't get today's day grafted on;
        # we only ever read year and month from the result anyway.
        import datetime
        parsed = date_parser.parse(text, default=datetime.datetime(2000, 1, 1))
    except (ValueError, OverflowError):
        return None, 0.0

    return f"{parsed.year:04d}-{parsed.month:02d}", 1.0


def normalize_email(raw: str | None) -> tuple[str | None, float]:
    """Normalize an email: strip, lowercase, basic shape check.

    Returns (email, validity):
      1.0 — has exactly one '@' with non-empty local and domain parts
      0.0 — empty or fails the basic shape check → None

    Lowercasing is safe canonicalization: the domain is case-insensitive, and
    while the local part is technically case-sensitive per spec, in practice
    providers treat it case-insensitively, so lowercasing aids cross-source
    matching (the merge agreement case) with negligible real risk.
    """
    if not raw or not str(raw).strip():
        return None, 0.0
    email = str(raw).strip().lower()
    # Minimal structural check — not full RFC validation (deliberately).
    parts = email.split("@")
    if len(parts) == 2 and parts[0] and parts[1] and "." in parts[1]:
        return email, 1.0
    return None, 0.0


def normalize_url(raw: str | None) -> tuple[str | None, float]:
    """Normalize a URL: strip, lowercase scheme+host, trim trailing junk.

    Returns (url, validity):
      1.0 — has a scheme (http/https) and a non-empty host
      0.6 — looks like a bare host/path with no scheme (prepend https://)
      0.0 — empty or unusable → None

    Only scheme and host are lowercased; the path is left as-is (paths can be
    case-sensitive). Trailing sentence punctuation is stripped.
    """
    if not raw or not str(raw).strip():
        return None, 0.0
    text = str(raw).strip().rstrip(".,;:!?)\"'")

    try:
        from urllib.parse import urlsplit, urlunsplit
    except ImportError:
        return None, 0.0

    has_scheme = "://" in text
    candidate = text if has_scheme else f"https://{text}"

    try:
        parts = urlsplit(candidate)
    except ValueError:
        return None, 0.0

    if not parts.netloc:
        return None, 0.0

    # Lowercase scheme + host only; preserve path/query case.
    normalized = urlunsplit((
        parts.scheme.lower(),
        parts.netloc.lower(),
        parts.path,
        parts.query,
        parts.fragment,
    ))
    return normalized, (1.0 if has_scheme else 0.6)


def classify_location(raw: str | None) -> tuple["Location | None", float]:
    """Parse a location string into Location, positionally.

    Structured sources order location as 'city, region, country'; we trust that
    convention rather than validating tokens against a gazetteer (which cannot
    disambiguate tokens belonging to multiple categories, e.g. 'Washington' city
    vs state, 'CA' California vs Canada). Only the country token is resolved to
    its ISO code; city and region are cleaned but not validated.

      1 part  -> city
      2 parts -> city, then country if the 2nd token resolves, else region
      3+ parts-> first = city, last = country (if it resolves), middle = region

    Returns (Location, validity) where validity is MIN of present sub-validities.
    Validating city/region needs a geocoding database (named upgrade).
    """
    from src.transformer.models import Location  # local import avoids cycle

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


def normalize_country(raw: str | None) -> tuple[str | None, float]:
    """Resolve a country reference to ISO-3166 alpha-2 via the countrynames
    database (handles ISO codes, official names, and informal forms like 'UK').

    Returns (alpha2, validity): (code, 1.0) if it resolves to a current alpha-2
    country, else (None, 0.0). Exact database lookup (fuzzy off) — deterministic,
    and unlike a generic fuzzy matcher it resolves 'Republic of Korea' to KR, not
    KP. Sub-region/dissolved codes (e.g. GB-SCT, SUHH) are rejected so a region or
    defunct entity never masquerades as a country.
    """
    if not raw or not str(raw).strip():
        return None, 0.0
    try:
        import countrynames
    except ImportError:
        return None, 0.0
    code = countrynames.to_code(str(raw).strip())
    if code and len(code) == 2 and code.isalpha():
        return code.upper(), 1.0
    return None, 0.0


def normalize_years(raw: object) -> tuple[float | None, float]:
    """Normalize years-of-experience to a float.

    Returns (years, validity):
      1.0 — parsed to a number in a plausible range [0, 70]
      0.0 — unparseable or out of range → None
    """
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None, 0.0
    try:
        years = float(str(raw).strip())
    except (ValueError, TypeError):
        return None, 0.0
    if 0.0 <= years <= 70.0:
        return years, 1.0
    return None, 0.0