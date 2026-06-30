from __future__ import annotations
from typing import Generic, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

T = TypeVar("T")

Method = Literal["direct", "regex", "inferred", "merged"]


class TrackedValue(BaseModel, Generic[T]):
    """Wraps a single field's value together with its provenance and confidence."""

    value: T
    source: str
    method: Method
    confidence: float

    model_config = {"frozen": True}


class RawRecord(BaseModel):
    """A raw record from a source, with the source's own vocabulary."""
    source: str
    raw_fields: dict[str, object]


@runtime_checkable
class Source(Protocol):
    """The protocol every adapter must satisfy (structural, no inheritance)."""
    name: str
    trust: float

    def extract(self, payload: str) -> list["RawRecord"]:
        ...


class Location(BaseModel):
    """A location with city, region, and country."""
    city: str | None = None
    region: str | None = None
    country: str | None = None  # ISO-3166 alpha-2


class Links(BaseModel):
    """A collection of links to the candidate's online presence."""
    linkedin: str | None = None
    github: str | None = None
    portfolio: str | None = None
    other: list[str] = []


class ExperienceEntry(BaseModel):
    """An entry in the candidate's work experience."""
    company: str
    title: str
    start: str | None = None  # YYYY-MM
    end: str | None = None    # YYYY-MM, or None if current
    summary: str | None = None


class EducationEntry(BaseModel):
    """An entry in the candidate's education."""
    institution: str
    degree: str | None = None
    field: str | None = None
    end_year: int | None = None


class SkillEntry(BaseModel):
    """An entry in the candidate's skills."""
    name: str
    confidence: float
    sources: list[str]


class ProjectEntry(BaseModel):
    """An entry in the candidate's projects."""
    name: str
    description: str | None = None
    url: str | None = None
    primary_language: str | None = None
    confidence: float
    sources: list[str]


class CanonicalProfile(BaseModel):
    """A canonical profile for a candidate."""
    candidate_id: str

    full_name: TrackedValue[str]
    emails: list[TrackedValue[str]]
    phones: list[TrackedValue[str]]
    location: TrackedValue[Location]
    links: TrackedValue[Links]
    headline: TrackedValue[str | None]
    years_experience: TrackedValue[float | None]
    skills: list[SkillEntry]
    experience: list[TrackedValue[ExperienceEntry]]
    education: list[TrackedValue[EducationEntry]]
    projects: list[ProjectEntry]

    overall_confidence: float

    model_config = {"frozen": True}

    def get_provenance(self) -> list[dict[str, str]]:
        """Walk every tracked value in this profile and return the flat provenance list."""
        out: list[dict[str, str]] = []

        def add_tv(field: str, tv: TrackedValue) -> None:
            out.append({"field": field, "source": tv.source, "method": tv.method})

        def add_aggregated(field: str, sources: list[str]) -> None:
            for src in sources:
                out.append({"field": field, "source": src, "method": "merged"})

        # Single TrackedValue fields
        add_tv("full_name", self.full_name)
        add_tv("location", self.location)
        add_tv("links", self.links)
        add_tv("headline", self.headline)
        add_tv("years_experience", self.years_experience)

        # Lists of TrackedValues
        for tv in self.emails:     add_tv("emails", tv)
        for tv in self.phones:     add_tv("phones", tv)
        for tv in self.experience: add_tv("experience", tv)
        for tv in self.education:  add_tv("education", tv)

        # Aggregated multi-source types (each entry already carries sources[])
        for s in self.skills:       add_aggregated("skills", s.sources)
        for p in self.projects:     add_aggregated("projects", p.sources)

        return sorted(out, key=lambda r: (r["field"], r["source"]))

    