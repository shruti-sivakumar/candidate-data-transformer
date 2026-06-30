"""ATS source adapter.

Parses Greenhouse Harvest-shape candidate JSON into RawRecords, preserving
the source's nested structure verbatim (arrays stay arrays, nested objects
stay nested). No normalization or flattening happens here — that is Module
3's job. The merge layer relies on the type-tagged arrays
(email_addresses[].type, etc.) staying intact.
"""

import json
import logging

from src.transformer.models import RawRecord

logger = logging.getLogger(__name__)


class ATSSource:
    """Adapter for ATS (Greenhouse Harvest) candidate exports.

    Consumes the raw JSON string (from ingest.read_file) and emits one
    RawRecord per candidate object, with the candidate's fields preserved
    one-to-one — including nested arrays like email_addresses[].
    """

    name: str = "ats_json"
    trust: float = 0.90

    def extract(self, payload: str) -> list[RawRecord]:
        """Parse an ATS JSON string into RawRecords.

        Accepts either a single candidate object or a JSON array of them.
        Each object becomes one RawRecord whose raw_fields mirror the parsed
        JSON exactly — nested arrays and objects are left untouched. Returns
        [] on any parse failure rather than raising, so one bad source never
        aborts the pipeline.
        """
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("ATSSource failed to parse payload: %s", e)
            return []

        # Normalize the container shape only: a single object becomes a
        # one-element list. This is container-level, not field-level — the
        # candidate fields themselves are never reshaped.
        if isinstance(data, dict):
            candidates = [data]
        elif isinstance(data, list):
            candidates = data
        else:
            logger.warning(
                "ATSSource expected object or array, got %s", type(data).__name__
            )
            return []

        records: list[RawRecord] = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                logger.warning(
                    "ATSSource skipping non-object candidate entry: %s",
                    type(candidate).__name__,
                )
                continue
            records.append(RawRecord(source=self.name, raw_fields=candidate))
        return records