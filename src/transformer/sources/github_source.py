"""GitHub source adapter.

Parses the combined {profile, repos} document (built by ingest.fetch_github_profile)
into a single RawRecord. Selects the candidate-relevant fields using GitHub's own
field names — it does NOT rename, normalize, or reinterpret them, and it does NOT
filter repos (e.g. forks are carried with their flag, for the normalize layer to
judge). Field selection trims GitHub's large payload to what downstream consumes;
the dropped fields are API plumbing (URL templates, node ids, counts) that no part
of the canonical schema reads.
"""

import json
import logging

from src.transformer.audit import make_event
from src.transformer.models import AuditEvent
from src.transformer.models import RawRecord

logger = logging.getLogger(__name__)

# GitHub's own keys, carried forward verbatim. Selection only — no renaming.
_PROFILE_FIELDS = (
    "login",
    "name",
    "location",
    "bio",
    "blog",
    "company",
    "email",
    "html_url",
)
_REPO_FIELDS = (
    "name",
    "description",
    "html_url",
    "language",
    "fork",
    "topics",
)


class GitHubSource:
    """Adapter for the combined GitHub {profile, repos} document.

    Consumes the raw JSON string and emits exactly one RawRecord whose
    raw_fields hold a slimmed profile dict plus a slimmed repos list. The
    repos list is preserved (not aggregated) — turning repo languages into
    skills and repos into projects is the normalize layer's job, not the
    adapter's. Returns [] on any parse failure rather than raising.
    """

    name: str = "github"
    trust: float = 0.70

    def extract(self, payload: str) -> list[RawRecord]:
        """Backward-compatible wrapper returning only records."""
        records, _ = self.extract_with_audit(payload)
        return records

    def extract_with_audit(self, payload: str) -> tuple[list[RawRecord], list[AuditEvent]]:
        """Parse the combined GitHub document into a single RawRecord."""
        audit_log: list[AuditEvent] = []
        try:
            data = json.loads(payload)
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("GitHubSource failed to parse payload: %s", e)
            audit_log.append(
                make_event("extract", "payload", "source_failed", "parse_failed", source=self.name, error=str(e))
            )
            return [], audit_log

        if not isinstance(data, dict):
            logger.warning(
                "GitHubSource expected an object, got %s", type(data).__name__
            )
            audit_log.append(
                make_event(
                    "extract",
                    "payload",
                    "source_failed",
                    "unexpected_top_level_type",
                    source=self.name,
                    got=type(data).__name__,
                )
            )
            return [], audit_log

        profile_raw = data.get("profile")
        repos_raw = data.get("repos")

        # A document with neither half is empty-but-not-an-error: return [].
        if not isinstance(profile_raw, dict) and not isinstance(repos_raw, list):
            audit_log.append(
                make_event("extract", "records", "source_empty", "no_profile_or_repos", source=self.name)
            )
            return [], audit_log

        profile: dict[str, object] = {}
        if isinstance(profile_raw, dict):
            profile = {
                key: profile_raw.get(key) for key in _PROFILE_FIELDS
            }

        repos: list[dict[str, object]] = []
        if isinstance(repos_raw, list):
            for repo in repos_raw:
                if not isinstance(repo, dict):
                    audit_log.append(
                        make_event(
                            "extract",
                            "repos[]",
                            "entry_dropped",
                            "non_object_repo",
                            source=self.name,
                            got=type(repo).__name__,
                        )
                    )
                    continue  # skip malformed array entry, keep going
                repos.append({key: repo.get(key) for key in _REPO_FIELDS})

        fields: dict[str, object] = {"profile": profile, "repos": repos}
        return [RawRecord(source=self.name, raw_fields=fields)], audit_log
