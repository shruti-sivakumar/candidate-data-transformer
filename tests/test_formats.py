"""Tests for the pure format-normalization functions in normalize/formats.py.

These are pure (input -> output, no I/O, no state), so each smoke-tested case
is pinned as a one-line assertion. Organized one test class per function; the
dangerous edge cases (city-not-country, vanity-not-phone, year-only precision,
URL path-case preservation) get explicit tests since they encode real decisions.
"""

from src.transformer.normalize.formats import (
    clean_string,
    normalize_country,
    normalize_date,
    normalize_email,
    normalize_phone,
    normalize_url,
)


class TestNormalizePhone:
    def test_real_fixture_spaces_and_dashes_collapse_to_same_e164(self):
        # The merge agreement case depends on these being identical.
        assert normalize_phone("+1 202 555 0142") == ("+12025550142", 1.0)
        assert normalize_phone("+1-202-555-0142") == ("+12025550142", 1.0)

    def test_andrej_number(self):
        assert normalize_phone("+1 650 555 0287") == ("+16505550287", 1.0)

    def test_international_numbers(self):
        assert normalize_phone("+91 98765 43210") == ("+919876543210", 1.0)
        assert normalize_phone("+44 20 7946 0958") == ("+442079460958", 1.0)

    def test_formatting_variants_normalize(self):
        assert normalize_phone("+1 (202) 555-0142") == ("+12025550142", 1.0)
        assert normalize_phone("+1.202.555.0142") == ("+12025550142", 1.0)
        assert normalize_phone("  +1  202  555  0142  ") == ("+12025550142", 1.0)

    def test_region_hint_for_schemeless_number(self):
        assert normalize_phone("202 555 0142", "US") == ("+12025550142", 1.0)

    def test_empty_and_garbage_return_none(self):
        assert normalize_phone("") == (None, 0.0)
        assert normalize_phone("not a phone") == (None, 0.0)

    def test_comp_figure_not_treated_as_phone(self):
        # "+650,000" is phone-shaped junk; must not become a valid number.
        assert normalize_phone("+650,000") == (None, 0.0)


class TestNormalizeDate:
    def test_full_precision_year_month(self):
        assert normalize_date("2021-03") == ("2021-03", 1.0)
        assert normalize_date("2024-07") == ("2024-07", 1.0)

    def test_year_only_is_lower_precision_not_rejected(self):
        # Education dates are legitimately year-only: keep the year, flag 0.8.
        assert normalize_date("2007") == ("2007", 0.8)
        assert normalize_date("2015") == ("2015", 0.8)

    def test_int_input_coerced(self):
        # education_end_year arrives as an int, not a string.
        assert normalize_date(2007) == ("2007", 0.8)

    def test_other_formats_absorbed_to_year_month(self):
        assert normalize_date("03/2021") == ("2021-03", 1.0)
        assert normalize_date("March 2021") == ("2021-03", 1.0)
        assert normalize_date("2021-03-15") == ("2021-03", 1.0)

    def test_determinism_year_month_independent_of_today(self):
        # Must not graft today's day; result is stable across runs.
        assert normalize_date("2021-03") == ("2021-03", 1.0)

    def test_empty_garbage_and_out_of_range_year(self):
        assert normalize_date("") == (None, 0.0)
        assert normalize_date("not a date") == (None, 0.0)
        assert normalize_date("0142") == (None, 0.0)  # out-of-range 4-digit


class TestCleanString:
    def test_collapses_whitespace(self):
        assert clean_string("  Kelsey   Hightower  ") == "Kelsey Hightower"

    def test_nfc_preserves_accented_text(self):
        # NFC must not mangle; two encodings of the same glyph compare equal.
        composed = clean_string("caf\u00e9")          # café (precomposed é)
        decomposed = clean_string("cafe\u0301")       # café (e + combining accent)
        assert composed == decomposed                  # the determinism guarantee

    def test_empty_and_none_return_none(self):
        assert clean_string("") is None
        assert clean_string(None) is None
        assert clean_string("   ") is None


class TestNormalizeEmail:
    def test_lowercases_and_strips(self):
        assert normalize_email("Kelsey.Hightower@Gmail.com") == (
            "kelsey.hightower@gmail.com", 1.0,
        )

    def test_valid_work_email(self):
        assert normalize_email("kelsey@stripe.com") == ("kelsey@stripe.com", 1.0)

    def test_rejects_malformed(self):
        assert normalize_email("not-an-email") == (None, 0.0)
        assert normalize_email("a@b") == (None, 0.0)        # no dot in domain
        assert normalize_email("") == (None, 0.0)


class TestNormalizeUrl:
    def test_full_url_kept(self):
        assert normalize_url("https://linkedin.com/in/kelsey-hightower") == (
            "https://linkedin.com/in/kelsey-hightower", 1.0,
        )

    def test_host_lowercased_path_case_preserved(self):
        # Host is case-insensitive; path is NOT — capital K must survive.
        assert normalize_url("https://GitHub.com/Kelsey") == (
            "https://github.com/Kelsey", 1.0,
        )

    def test_schemeless_gets_https_at_lower_validity(self):
        assert normalize_url("linkedin.com/in/foo") == (
            "https://linkedin.com/in/foo", 0.6,
        )

    def test_trailing_punctuation_stripped(self):
        assert normalize_url("https://kelsey.dev.") == ("https://kelsey.dev", 1.0)

    def test_empty_returns_none(self):
        assert normalize_url("") == (None, 0.0)


class TestNormalizeCountry:
    def test_iso_codes_and_names(self):
        assert normalize_country("US") == ("US", 1.0)
        assert normalize_country("USA") == ("US", 1.0)
        assert normalize_country("United States") == ("US", 1.0)
        assert normalize_country("India") == ("IN", 1.0)

    def test_informal_names_resolve(self):
        assert normalize_country("UK") == ("GB", 1.0)
        assert normalize_country("Britain") == ("GB", 1.0)

    def test_ambiguous_korea_resolves_correctly(self):
        # Regression guard: generic fuzzy matched this to KP (North Korea);
        # the countrynames database resolves correctly to KR (South).
        assert normalize_country("Republic of Korea") == ("KR", 1.0)

    def test_subregion_and_city_are_not_countries(self):
        assert normalize_country("Scotland") == (None, 0.0)
        assert normalize_country("Stanford") == (None, 0.0)

    def test_empty_and_garbage(self):
        assert normalize_country("") == (None, 0.0)
        assert normalize_country("xyzzy") == (None, 0.0)