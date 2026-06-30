"""Shared helpers for structured audit events."""
from __future__ import annotations

from src.transformer.models import AuditEvent


def make_event(
    stage: str,
    field: str,
    kind: str,
    reason: str,
    **details: object,
) -> AuditEvent:
    """Build one structured audit event."""
    return AuditEvent(stage=stage, field=field, kind=kind, reason=reason, details=details)
