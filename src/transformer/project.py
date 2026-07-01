"""Module 6: project the canonical profile into a config-defined output shape."""
from __future__ import annotations

from typing import Any, Callable, Literal

import jmespath
from jsonschema import ValidationError as JsonSchemaValidationError
from jsonschema import validate
from pydantic import BaseModel, ConfigDict, model_validator

from src.transformer.models import CanonicalProfile
from src.transformer.normalize.formats import normalize_country, normalize_phone
from src.transformer.normalize.skills import canonicalize_skill

MissingPolicy = Literal["null", "omit", "error"]
FieldType = Literal["string", "number", "integer", "boolean", "object", "array"]

# Per-field normalizers the projection can apply, keyed by the config identifier.
# Each reuses an existing canonical-layer normalizer (no new normalization logic
# here) and declares which declared field types it may be applied to. The tuple
# normalizers return (value, validity); the projection only needs the value.
_NORMALIZERS: dict[str, tuple[Callable[[str], Any], frozenset[str]]] = {
    "E164": (lambda value: normalize_phone(value)[0], frozenset({"string"})),
    "ISO3166": (lambda value: normalize_country(value)[0], frozenset({"string"})),
    "canonical": (canonicalize_skill, frozenset({"string", "array"})),
}


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

    ``normalize`` (a PS-example per-field key) applies one of the canonical
    normalizers to the projected value at projection time — ``"E164"`` (phones),
    ``"canonical"`` (skills, scalar or array), ``"ISO3166"`` (countries). It is
    validated against the field's declared ``type`` (see ``_NORMALIZERS``); an
    unknown identifier or a type mismatch raises ``ProjectionError``. Because the
    canonical layer already normalizes upstream, the call is usually idempotent,
    but it is applied for real so a raw value would still be normalized.
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
        """Translate PS-example field keys into the internal name/path shape.

        This is the single place field-level ``on_missing`` precedence is
        resolved. ``ProjectionConfig`` only forwards the config-level default
        (via ``_default_on_missing``); it never decides precedence itself.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # Config-level default forwarded by ProjectionConfig; lowest precedence.
        default_on_missing = data.pop("_default_on_missing", None)

        # PS vocabulary: no explicit output "name". "path" is the output field
        # name; "from" (when present) is the canonical source path, otherwise
        # "path" doubles as the source.
        if "name" not in data and "path" in data:
            output_name = data["path"]
            data["path"] = data.pop("from", output_name)
            data["name"] = output_name

        # on_missing precedence, highest first:
        #   explicit per-field on_missing > required:true (=> "error")
        #   > config-level default > the model default ("null").
        required = data.pop("required", None)
        if "on_missing" not in data:
            if required:
                data["on_missing"] = "error"
            elif default_on_missing is not None:
                data["on_missing"] = default_on_missing

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
    def _distribute_on_missing_default(cls, data: Any) -> Any:
        """Forward the PS top-level ``on_missing`` down as a per-field default.

        Precedence between this default, a per-field ``on_missing``, and
        ``required`` is resolved entirely in ``FieldConfig``; this method only
        plumbs the value through so the two validators can never disagree.
        """
        if not isinstance(data, dict):
            return data
        data = dict(data)

        default_policy = data.pop("on_missing", None)
        fields = data.get("fields")
        if default_policy is None or not isinstance(fields, list):
            return data

        data["fields"] = [
            {**field, "_default_on_missing": default_policy}
            if isinstance(field, dict) and "_default_on_missing" not in field
            else field
            for field in fields
        ]
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


def _resolve_normalizer(field: FieldConfig) -> Callable[[str], Any] | None:
    """Return the scalar normalizer for a field's ``normalize`` key, or None.

    Raises ProjectionError for an unknown identifier or a type mismatch (e.g.
    ``"E164"`` on a non-string field), rather than silently skipping it.
    """
    if field.normalize is None:
        return None
    spec = _NORMALIZERS.get(field.normalize)
    if spec is None:
        raise ProjectionError(
            f"Field {field.name!r}: unknown normalize {field.normalize!r} "
            f"(supported: {sorted(_NORMALIZERS)})"
        )
    normalizer, allowed_types = spec
    if field.type not in allowed_types:
        raise ProjectionError(
            f"Field {field.name!r}: normalize {field.normalize!r} cannot apply to "
            f"type {field.type!r} (expected one of {sorted(allowed_types)})"
        )
    return normalizer


def _apply_normalizer(
    normalizer: Callable[[str], Any],
    field_type: str,
    value: Any,
) -> Any:
    """Apply a scalar normalizer to a projected value or each of its array items."""
    if field_type == "array" and isinstance(value, list):
        normalized = [normalizer(item) for item in value]
        return [item for item in normalized if item is not None]
    return normalizer(value)


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

    # Resolve (and validate) normalizers up front so a bad "normalize" key fails
    # fast, even for a field whose source value turns out to be missing.
    normalizers = {field.name: _resolve_normalizer(field) for field in parsed.fields}

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
        normalizer = normalizers[field.name]
        if normalizer is not None:
            value = _apply_normalizer(normalizer, field.type, value)
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
