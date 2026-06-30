"""CSV source adapter.

Parses recruiter CSV exports into RawRecords, preserving the source's own
column names. No normalization happens here.
"""

import csv
import io
import logging

from src.transformer.audit import make_event
from src.transformer.models import AuditEvent
from src.transformer.models import RawRecord

logger = logging.getLogger(__name__)


class CSVSource:
    """Adapter for recruiter CSV exports.

    Consumes the raw CSV string (from ingest.read_file) and emits one
    RawRecord per data row, with the CSV's own headers as field keys.
    """

    name: str = "recruiter_csv"
    trust: float = 0.80

    def extract(self, payload: str) -> list[RawRecord]:
        """Backward-compatible wrapper returning only records."""
        records, _ = self.extract_with_audit(payload)
        return records

    def extract_with_audit(self, payload: str) -> tuple[list[RawRecord], list[AuditEvent]]:
        """Parse a CSV string into RawRecords.

        Each row becomes one RawRecord whose raw_fields are the row's
        column→value pairs, untouched. Returns [] on any parse failure
        rather than raising, so one bad source never aborts the pipeline.
        """
        audit_log: list[AuditEvent] = []
        if not payload or not payload.strip():
            audit_log.append(
                make_event("extract", "payload", "source_empty", "empty_payload", source=self.name)
            )
            return [], audit_log
        try:
            # restkey="_extra" buckets overflow columns into a list[str] under "_extra" —
            # intentionally different shape from real fields (all str). The normalize layer
            # must skip "_extra" entirely; it is overflow junk, not a mappable field.
            reader = csv.DictReader(io.StringIO(payload), restkey="_extra")
            records: list[RawRecord] = []
            for row in reader:
                # row is a dict[str, str | None] keyed by header.
                # Preserve keys exactly; strip surrounding whitespace on values.
                fields: dict[str, object] = {
                    key: (value.strip() if isinstance(value, str) else value)
                    for key, value in row.items()
                }
                if fields.get("_extra"):
                    audit_log.append(
                        make_event(
                            "extract",
                            "row._extra",
                            "entry_flagged",
                            "overflow_columns_preserved",
                            source=self.name,
                            row_index=len(records),
                        )
                    )
                records.append(RawRecord(source=self.name, raw_fields=fields))
            if not records:
                audit_log.append(
                    make_event("extract", "records", "source_empty", "no_data_rows", source=self.name)
                )
            return records, audit_log
        except (csv.Error, ValueError) as e:
            logger.warning("CSVSource failed to parse payload: %s", e)
            audit_log.append(
                make_event(
                    "extract",
                    "payload",
                    "source_failed",
                    "parse_failed",
                    source=self.name,
                    error=str(e),
                )
            )
            return [], audit_log
