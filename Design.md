# Multi-Source Candidate Data Transformer — Technical Design

**Eightfold Engineering Intern Assignment · Shruti Sivakumar**

## Problem Framing

The transformer has to turn messy, partially overlapping candidate inputs into a
single trustworthy profile per candidate. The main risk is not missing a value;
it is confidently emitting a wrong value that pollutes a hiring workflow. My
design therefore favors precision, explicit provenance, and validated output over
aggressive inference.

## Pipeline

The implementation is split into deterministic stages:

```text
extract -> normalize -> merge -> score -> project -> validate
```

`extract` reads each source into its own raw vocabulary. `normalize` maps those
source-specific fields into canonical values and assigns base confidence.
`merge` groups records by identity and resolves field conflicts. `score` computes
overall profile confidence after merge. `project` reshapes the canonical record
using a runtime config, then `validate` checks the result against a schema derived
from that same config.

The code implements two structured sources, recruiter CSV and ATS JSON, and two
unstructured sources, GitHub JSON and recruiter notes. Each source is isolated
behind the same adapter contract so another source can be added without changing
the core merge/project stages.

## Canonical Schema And Normalization

Internally, every value carries:

```text
value, source, method, confidence
```

The default projected schema includes candidate identity, contacts, location,
links, headline, years of experience, skills, experience, education, projects,
provenance, and overall confidence.

Key normalizations:

- Phones use `phonenumbers` and normalize to E.164.
- Dates normalize to `YYYY-MM`, with `YYYY` preserved for year-only education
  dates.
- Countries normalize to ISO-3166 alpha-2 using exact country lookup.
- Emails and URLs are normalized before merge.
- Skills use a curated taxonomy: exact alias lookup first, then a guarded fuzzy
  fallback for typo recovery.

Skill matching is deliberately conservative. Short ambiguous prose tokens such as
`R` or `Go` need skill-list context before being accepted from recruiter notes,
while structured sources can keep those exact skills because the field label
already supplies context.

## Identity, Merge, And Confidence

Records are grouped before merge. The only strong identity anchors are normalized
email and normalized phone. Name alone never joins records, because two distinct
people can share the same name. Sparse records without email or phone attach only
when the bundle is unambiguous; multiple unanchored rows from the same source
remain separate.

Within one candidate group, values are merged field by field. Equal normalized
values reinforce each other with noisy-OR confidence, so independent agreement
raises confidence without simply averaging away signal. Conflicting single-value
fields pick the highest-confidence value with deterministic tie-breaks, apply a
conflict penalty, and keep losing values in provenance. Low-confidence
single-value winners are dropped to empty rather than emitted as false certainty.
List fields such as emails, phones, skills, experience, and projects are unioned
and deduplicated after normalization.

Overall confidence is computed in a separate scoring stage so field-level merge
logic and whole-profile scoring remain independent.

## Runtime Output Config

The canonical record is richer than any one customer-facing output. A runtime
projection config selects fields, renames output keys, chooses missing-value
behavior (`null`, `omit`, or `error`), and can include confidence and provenance.
The config also builds the JSON Schema used to validate the final output. This
keeps the core transformer stable while allowing downstream products to request
different shapes without code changes.

## Edge Cases And Scope

Handled edge cases:

- Missing, empty, or malformed sources degrade gracefully instead of crashing.
- Multi-row CSV input produces one profile per candidate.
- Duplicate names with different email/phone anchors remain separate.
- Public GitHub email is preserved when present.
- Bare domestic phone numbers from notes are accepted only after region-aware
  validation.
- Stray initials in notes are not treated as skills without skill context.

Deliberately out of scope:

- Free-text extraction of name, education, and work history from recruiter notes.
  A reliable version needs NER or a purpose-built parser; regex guesses would be
  less honest than leaving the fields empty.
- LinkedIn scraping, because there is no compliant public API path for this MVP.
- Resume PDF/DOCX parsing, because robust layout-aware extraction is a separate
  problem.
- Cross-bundle entity resolution at large scale. This implementation resolves
  identity within a submitted bundle/batch.
