"""Skill canonicalization — Stage 2 linking.

STUB: currently identity (cleaned passthrough). The real implementation will
fuzzy-match each raw skill surface-form against the Lightcast taxonomy snapshot
(rapidfuzz over canonical names + aliases) to map e.g. 'Kubernetis'/'k8s' ->
'Kubernetes'. Kept behind this interface so normalizers call canonicalize_skill()
unchanged when the matcher lands.
"""
from __future__ import annotations

from src.transformer.normalize.formats import clean_string


def canonicalize_skill(raw: str | None) -> str | None:
    """Map a raw skill surface-form to its canonical name.

    STUB: returns the cleaned surface form unchanged. Replace with Lightcast
    fuzzy-linking. Returns None if the input is empty.
    """
    return clean_string(raw)