"""Normalize orchestrator: dispatch a RawRecord to its source normalizer."""
from __future__ import annotations

import logging

from src.transformer.audit import make_event
from src.transformer.models import AuditEvent, NormalizedRecord, RawRecord
from src.transformer.normalize.ats_normalizer import normalize_ats, normalize_ats_with_audit
from src.transformer.normalize.csv_normalizer import normalize_csv, normalize_csv_with_audit
from src.transformer.normalize.github_normalizer import normalize_github, normalize_github_with_audit
from src.transformer.normalize.notes_normalizer import normalize_notes, normalize_notes_with_audit

logger = logging.getLogger(__name__)

_NORMALIZERS = {
    "recruiter_csv": normalize_csv,
    "ats_json": normalize_ats,
    "github": normalize_github,
    "recruiter_notes": normalize_notes,
}

_NORMALIZERS_WITH_AUDIT = {
    "recruiter_csv": normalize_csv_with_audit,
    "ats_json": normalize_ats_with_audit,
    "github": normalize_github_with_audit,
    "recruiter_notes": normalize_notes_with_audit,
}


def normalize_record(record: RawRecord) -> NormalizedRecord | None:
    """Backward-compatible wrapper returning only the normalized record."""
    normalized, _ = normalize_record_with_audit(record)
    return normalized


def normalize_record_with_audit(record: RawRecord) -> tuple[NormalizedRecord | None, list[AuditEvent]]:
    """Normalize one RawRecord via the normalizer matching its source.

    Returns None for an unknown source (logged), so an unexpected source never
    crashes the pipeline.
    """
    fn = _NORMALIZERS_WITH_AUDIT.get(record.source)
    if fn is None:
        logger.warning("No normalizer for source %r; skipping", record.source)
        return None, [
            make_event(
                "normalize",
                "source",
                "source_failed",
                "unknown_source",
                source=record.source,
            )
        ]
    return fn(record)
