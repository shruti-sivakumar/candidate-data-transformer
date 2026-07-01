"""Module 6: project the canonical profile into a config-defined output shape."""
from __future__ import annotations

from typing import Any, Literal

import jmespath
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate
from pydantic import BaseModel, ConfigDict, model_validator

from src.transformer.models import CanonicalProfile

MissingPolicy = Literal["null", "omit", "error"]
FieldType = Literal["string", "number", "integer", "boolean", "object", "array"]


class ProjectionError(ValueError):
    """Raised when a projection config cannot be satisfied."""


class FieldConfig(BaseModel):
    """One output field requested by the runtime config.

    Two config vocabularies are accepted and normalized to the same internal
    shape (``name`` = output field, ``path`` = canonical source path):

    * Legacy: ``{"name": <output>, "path": <source>, "on_missing": ...}``.
    * PS example: ``{"path": <output>, "from": <source>, "required": true,
      "type": "string[]", "normalize": ...}``. In this vocabulary ``path`` is
      the OUTPUT name and ``from`` is the SOURCE path (``path`` doubles as the
      source when ``from`` is absent).

    Vocabularies are distinguished by the presence of an explicit ``name`` key:
    legacy configs always carry one, PS configs never do.
    """

    name: str
    path: str
    type: FieldType
    nullable: bool = False
    on_missing: MissingPolicy = "null"
    normalize: str | None = None
    description: str | None = None

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _accept_ps_vocabulary(cls, data: Any) -> Any:
        """Translate PS-example field keys into the internal name/path shape."""
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # PS vocabulary: no explicit output "name". "path" is the output field
        # name; "from" (when present) is the canonical source path, otherwise
        # "path" doubles as the source.
        if "name" not in data and "path" in data:
            output_name = data["path"]
            data["path"] = data.pop("from", output_name)
            data["name"] = output_name

        # PS "required: true" means "must be present" -> error on missing, but an
        # explicit per-field on_missing (legacy or PS) always wins.
        required = data.pop("required", None)
        if required and "on_missing" not in data:
            data["on_missing"] = "error"

        # PS "string[]" (and any "<type>[]") is an array for schema building.
        field_type = data.get("type")
        if isinstance(field_type, str) and field_type.endswith("[]"):
            data["type"] = "array"

        return data


class ProjectionConfig(BaseModel):
    """Config describing how to project the canonical record."""

    fields: list[FieldConfig]
    include_confidence: bool = False
    include_provenance: bool = False

    model_config = ConfigDict(frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _apply_default_on_missing(cls, data: Any) -> Any:
        """Let the PS top-level ``on_missing`` act as a per-field default."""
        if not isinstance(data, dict):
            return data
        data = dict(data)

        default_policy = data.pop("on_missing", None)
        fields = data.get("fields")
        if default_policy is None or not isinstance(fields, list):
            return data

        # A per-field on_missing (or required, which implies "error") overrides
        # the top-level default; only fields specifying neither inherit it.
        patched: list[Any] = []
        for field in fields:
            if (
                isinstance(field, dict)
                and "on_missing" not in field
                and not field.get("required")
            ):
                field = {**field, "on_missing": default_policy}
            patched.append(field)
        data["fields"] = patched
        return data


def _plain_view(profile: CanonicalProfile) -> dict[str, Any]:
    """Return the plain, projection-friendly canonical view."""
    return {
        "candidate_id": profile.candidate_id,
        "full_name": profile.full_name.value,
        "emails": [item.value for item in profile.emails],
        "phones": [item.value for item in profile.phones],
        "location": profile.location.value.model_dump(mode="python"),
        "links": profile.links.value.model_dump(mode="python"),
        "headline": profile.headline.value,
        "years_experience": profile.years_experience.value,
        "skills": [item.model_dump(mode="python") for item in profile.skills],
        "experience": [item.model_dump(mode="python") for item in profile.experience],
        "education": [item.model_dump(mode="python") for item in profile.education],
        "projects": [item.model_dump(mode="python") for item in profile.projects],
        "overall_confidence": profile.overall_confidence,
    }


def _confidence_view(profile: CanonicalProfile) -> dict[str, Any]:
    """Return a confidence sidecar aligned with the plain canonical view."""
    return {
        "candidate_id": 1.0,
        "full_name": profile.full_name.confidence,
        "emails": [item.confidence for item in profile.emails],
        "phones": [item.confidence for item in profile.phones],
        "location": profile.location.confidence,
        "links": profile.links.confidence,
        "headline": profile.headline.confidence,
        "years_experience": profile.years_experience.confidence,
        "skills": [item.confidence for item in profile.skills],
        "experience": [item.confidence for item in profile.experience],
        "education": [item.confidence for item in profile.education],
        "projects": [item.confidence for item in profile.projects],
        "overall_confidence": profile.overall_confidence,
    }


def _confidence_path(path: str) -> str:
    """Map an output path to the closest aligned confidence path."""
    # A projected subfield of list entries, e.g. skills[].name, is supported by
    # the confidence of each parent list entry.
    if "[]." in path:
        return path.split(".", 1)[0]
    return path


def _project_field_confidence(
    field: FieldConfig,
    confidence_view: dict[str, Any],
) -> Any:
    """Return the confidence value aligned with one projected output field."""
    direct = jmespath.search(_confidence_path(field.path), confidence_view)
    if direct is not None:
        return direct

    # Subfields of a tracked object, e.g. location.city, inherit the parent
    # field's confidence because the object was normalized as one value.
    root = field.path.split(".", 1)[0].split("[", 1)[0]
    return confidence_view.get(root)


def build_json_schema(config: ProjectionConfig) -> dict[str, Any]:
    """Build a JSON Schema from the projection config."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    for field in config.fields:
        field_type: str | list[str] = [field.type, "null"] if field.nullable else field.type
        properties[field.name] = {"type": field_type}
        if field.description:
            properties[field.name]["description"] = field.description
        if field.on_missing == "error":
            required.append(field.name)

    if config.include_confidence:
        properties["confidence"] = {"type": "object"}
    if config.include_provenance:
        properties["provenance"] = {"type": "array"}

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = sorted(required)
    return schema


def validate_projection(output: dict[str, Any], config: ProjectionConfig) -> None:
    """Validate a projected output against the config-derived schema."""
    schema = build_json_schema(config)
    try:
        validate(output, schema)
    except JsonSchemaValidationError as exc:
        raise ProjectionError(str(exc)) from exc


def project_profile(
    profile: CanonicalProfile,
    config: ProjectionConfig | dict[str, Any],
) -> dict[str, Any]:
    """Project the canonical profile into the config-defined output shape."""
    parsed = config if isinstance(config, ProjectionConfig) else ProjectionConfig.model_validate(config)
    view = _plain_view(profile)
    confidence_view = _confidence_view(profile)
    output: dict[str, Any] = {}
    projected_confidence: dict[str, Any] = {}

    for field in parsed.fields:
        value = jmespath.search(field.path, view)
        if value is None:
            if field.on_missing == "omit":
                continue
            if field.on_missing == "error":
                raise ProjectionError(
                    f"Required field {field.name!r} missing at path {field.path!r}"
                )
            output[field.name] = None
            projected_confidence[field.name] = None
            continue
        output[field.name] = value
        projected_confidence[field.name] = _project_field_confidence(field, confidence_view)

    if parsed.include_confidence:
        output["confidence"] = projected_confidence
    if parsed.include_provenance:
        output["provenance"] = profile.get_provenance()

    validate_projection(output, parsed)
    return output


def default_projection_config() -> ProjectionConfig:
    """Return the assignment-aligned default output projection."""
    return ProjectionConfig.model_validate(
        {
            "fields": [
                {"name": "candidate_id", "path": "candidate_id", "type": "string", "on_missing": "error"},
                {"name": "full_name", "path": "full_name", "type": "string", "on_missing": "error"},
                {"name": "emails", "path": "emails", "type": "array", "on_missing": "null"},
                {"name": "phones", "path": "phones", "type": "array", "on_missing": "null"},
                {"name": "location", "path": "location", "type": "object", "on_missing": "null"},
                {"name": "links", "path": "links", "type": "object", "on_missing": "null"},
                {"name": "headline", "path": "headline", "type": "string", "nullable": True, "on_missing": "null"},
                {"name": "years_experience", "path": "years_experience", "type": "number", "nullable": True, "on_missing": "null"},
                {"name": "skills", "path": "skills[].{name: name, confidence: confidence, sources: sources}", "type": "array", "on_missing": "null"},
                {"name": "experience", "path": "experience[].{company: company, title: title, start: start, end: end, summary: summary}", "type": "array", "on_missing": "null"},
                {"name": "education", "path": "education[].{institution: institution, degree: degree, field: field, end_year: end_year}", "type": "array", "on_missing": "null"},
                {"name": "projects", "path": "projects[].{name: name, description: description, url: url, primary_language: primary_language}", "type": "array", "on_missing": "null"},
                {"name": "overall_confidence", "path": "overall_confidence", "type": "number", "on_missing": "error"},
            ],
            "include_provenance": True,
        }
    )
