"""Module 5: score a merged canonical profile."""
from __future__ import annotations

from src.transformer.models import CanonicalProfile

_OVERALL_WEIGHTS = {
    "emails": 0.22,
    "full_name": 0.18,
    "phones": 0.15,
    "experience": 0.12,
    "skills": 0.10,
    "location": 0.08,
    "education": 0.05,
    "links": 0.04,
    "headline": 0.03,
    "years_experience": 0.02,
    "projects": 0.01,
}


def field_family_confidence(profile: CanonicalProfile) -> dict[str, float]:
    """Compute one confidence score per populated canonical field family."""
    scores: dict[str, float] = {}
    if profile.full_name.confidence > 0.0:
        scores["full_name"] = profile.full_name.confidence
    if profile.emails:
        scores["emails"] = sum(item.confidence for item in profile.emails) / len(profile.emails)
    if profile.phones:
        scores["phones"] = sum(item.confidence for item in profile.phones) / len(profile.phones)
    if profile.location.confidence > 0.0:
        scores["location"] = profile.location.confidence
    if profile.links.confidence > 0.0:
        scores["links"] = profile.links.confidence
    if profile.headline.confidence > 0.0:
        scores["headline"] = profile.headline.confidence
    if profile.years_experience.confidence > 0.0:
        scores["years_experience"] = profile.years_experience.confidence
    if profile.skills:
        scores["skills"] = sum(item.confidence for item in profile.skills) / len(profile.skills)
    if profile.experience:
        scores["experience"] = sum(item.confidence for item in profile.experience) / len(profile.experience)
    if profile.education:
        scores["education"] = sum(item.confidence for item in profile.education) / len(profile.education)
    if profile.projects:
        scores["projects"] = sum(item.confidence for item in profile.projects) / len(profile.projects)
    return scores


def overall_confidence(profile: CanonicalProfile) -> float:
    """Compute the weighted overall confidence over populated field families only."""
    scores = field_family_confidence(profile)
    numerator = 0.0
    denominator = 0.0
    for field, score in scores.items():
        weight = _OVERALL_WEIGHTS[field]
        numerator += weight * score
        denominator += weight
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def score_profile(profile: CanonicalProfile) -> CanonicalProfile:
    """Return a copy of the merged profile with overall_confidence populated."""
    return profile.model_copy(update={"overall_confidence": overall_confidence(profile)})
