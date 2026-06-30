"""Normalize orchestrator: dispatch a RawRecord to its source normalizer."""
from __future__ import annotations

import logging

from src.transformer.models import NormalizedRecord, RawRecord
from src.transformer.normalize.ats_normalizer import normalize_ats
from src.transformer.normalize.csv_normalizer import normalize_csv
from src.transformer.normalize.github_normalizer import normalize_github
from src.transformer.normalize.notes_normalizer import normalize_notes

logger = logging.getLogger(__name__)

_NORMALIZERS = {
    "recruiter_csv": normalize_csv,
    "ats_json": normalize_ats,
    "github": normalize_github,
    "recruiter_notes": normalize_notes,
}


def normalize_record(record: RawRecord) -> NormalizedRecord | None:
    """Normalize one RawRecord via the normalizer matching its source.

    Returns None for an unknown source (logged), so an unexpected source never
    crashes the pipeline.
    """
    fn = _NORMALIZERS.get(record.source)
    if fn is None:
        logger.warning("No normalizer for source %r; skipping", record.source)
        return None
    return fn(record)