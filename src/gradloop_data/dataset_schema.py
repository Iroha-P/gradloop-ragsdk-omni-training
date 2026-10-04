"""Canonical Stage A training-sample records."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass

SCENARIOS = frozenset(
    {"course_learning", "technical_interview", "research_defense"}
)
MODALITIES = frozenset({"text", "document", "image", "audio", "video"})
TASK_TYPES = frozenset(
    {
        "evidence_answer",
        "follow_up",
        "response_grading",
        "weakness_summary",
        "multimodal_interpretation",
        "refusal",
    }
)
_ID = re.compile(r"[a-z0-9][a-z0-9._:-]{2,127}")
_MAX_TEXT_LENGTH = 32_768
_MAX_COLLECTION_ITEMS = 64
_SAMPLE_FIELDS = frozenset(
    {
        "sample_id",
        "scenario",
        "task_type",
        "modalities",
        "source_ids",
        "evidence_ids",
        "prompt",
        "reference_answer",
        "follow_ups",
        "rubric",
        "refusal_reason",
        "provenance_ids",
        "generation_method",
        "quality_status",
        "privacy_status",
        "asset_sidecars",
    }
)
_RUBRIC_FIELDS = frozenset({"dimension", "description", "max_score"})
_ASSET_SIDECAR_FIELDS = frozenset(
    {"asset_id", "modality", "ocr_text", "transcript", "metadata_text"}
)


class DatasetSchemaError(ValueError):
    """A stable, non-sensitive reason why a sample was rejected."""


def _invalid_sample() -> None:
    raise DatasetSchemaError("invalid_sample")


def _is_text(value: object, *, allow_empty: bool = True) -> bool:
    return (
        type(value) is str
        and len(value) <= _MAX_TEXT_LENGTH
        and (allow_empty or bool(value))
    )


def _is_identifier(value: object) -> bool:
    return type(value) is str and _ID.fullmatch(value) is not None


def _is_string_tuple(value: object) -> bool:
    return (
        type(value) is tuple
        and len(value) <= _MAX_COLLECTION_ITEMS
        and all(_is_text(item) for item in value)
    )


def _is_identifier_tuple(value: object, *, non_empty: bool) -> bool:
    return (
        type(value) is tuple
        and len(value) <= _MAX_COLLECTION_ITEMS
        and (bool(value) or not non_empty)
        and all(_is_identifier(item) for item in value)
        and len(set(value)) == len(value)
    )


@dataclass(frozen=True)
class RubricDimension:
    dimension: str
    description: str
    max_score: int

    def __post_init__(self) -> None:
        if (
            not _is_identifier(self.dimension)
            or not _is_text(self.description, allow_empty=False)
            or type(self.max_score) is not int
            or self.max_score < 1
        ):
            _invalid_sample()


@dataclass(frozen=True)
class AssetSidecar:
    asset_id: str
    modality: str
    ocr_text: str
    transcript: str
    metadata_text: str

    def __post_init__(self) -> None:
        if (
            not _is_identifier(self.asset_id)
            or type(self.modality) is not str
            or self.modality not in MODALITIES
            or any(
                not _is_text(value)
                for value in (self.ocr_text, self.transcript, self.metadata_text)
            )
        ):
            _invalid_sample()


@dataclass(frozen=True)
class TrainingSample:
    sample_id: str
    scenario: str
    task_type: str
    modalities: tuple[str, ...]
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    prompt: str
    reference_answer: str
    follow_ups: tuple[str, ...]
    rubric: tuple[RubricDimension, ...]
    refusal_reason: str
    provenance_ids: tuple[str, ...]
    generation_method: str
    quality_status: str
    privacy_status: str
    asset_sidecars: tuple[AssetSidecar, ...]

    def __post_init__(self) -> None:
        if (
            not _is_identifier(self.sample_id)
            or type(self.scenario) is not str
            or self.scenario not in SCENARIOS
            or type(self.task_type) is not str
            or self.task_type not in TASK_TYPES
            or not _is_string_tuple(self.modalities)
            or not self.modalities
            or any(modality not in MODALITIES for modality in self.modalities)
            or len(set(self.modalities)) != len(self.modalities)
            or not _is_identifier_tuple(self.source_ids, non_empty=True)
            or not _is_identifier_tuple(self.evidence_ids, non_empty=True)
            or not _is_text(self.prompt, allow_empty=False)
            or not _is_text(self.reference_answer)
            or not _is_string_tuple(self.follow_ups)
            or type(self.rubric) is not tuple
            or len(self.rubric) > _MAX_COLLECTION_ITEMS
            or any(type(item) is not RubricDimension for item in self.rubric)
            or not _is_text(self.refusal_reason)
            or not _is_identifier_tuple(self.provenance_ids, non_empty=True)
            or not _is_text(self.generation_method, allow_empty=False)
            or self.quality_status != "candidate"
            or self.privacy_status != "unscanned"
            or type(self.asset_sidecars) is not tuple
            or len(self.asset_sidecars) > _MAX_COLLECTION_ITEMS
            or any(type(item) is not AssetSidecar for item in self.asset_sidecars)
            or len({item.asset_id for item in self.asset_sidecars})
            != len(self.asset_sidecars)
            or any(item.modality not in self.modalities for item in self.asset_sidecars)
        ):
            _invalid_sample()
        if self.task_type == "refusal":
            if self.reference_answer or not self.refusal_reason:
                _invalid_sample()
        elif self.refusal_reason or not self.reference_answer:
            _invalid_sample()


def _list_to_string_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not list or len(value) > _MAX_COLLECTION_ITEMS:
        _invalid_sample()
    if any(not _is_text(item) for item in value):
        _invalid_sample()
    return tuple(value)


def _list_to_identifier_tuple(value: object) -> tuple[str, ...]:
    if type(value) is not list or len(value) > _MAX_COLLECTION_ITEMS:
        _invalid_sample()
    if any(not _is_identifier(item) for item in value) or len(set(value)) != len(value):
        _invalid_sample()
    return tuple(value)


def _rubric_from_mapping(value: object) -> RubricDimension:
    if not isinstance(value, Mapping) or set(value) != _RUBRIC_FIELDS:
        _invalid_sample()
    return RubricDimension(
        dimension=value["dimension"],
        description=value["description"],
        max_score=value["max_score"],
    )


def _sidecar_from_mapping(value: object) -> AssetSidecar:
    if not isinstance(value, Mapping) or set(value) != _ASSET_SIDECAR_FIELDS:
        _invalid_sample()
    return AssetSidecar(
        asset_id=value["asset_id"],
        modality=value["modality"],
        ocr_text=value["ocr_text"],
        transcript=value["transcript"],
        metadata_text=value["metadata_text"],
    )


def _list_to_rubric(value: object) -> tuple[RubricDimension, ...]:
    if type(value) is not list or len(value) > _MAX_COLLECTION_ITEMS:
        _invalid_sample()
    return tuple(_rubric_from_mapping(item) for item in value)


def _list_to_sidecars(value: object) -> tuple[AssetSidecar, ...]:
    if type(value) is not list or len(value) > _MAX_COLLECTION_ITEMS:
        _invalid_sample()
    return tuple(_sidecar_from_mapping(item) for item in value)


def sample_from_mapping(values: Mapping[str, object]) -> TrainingSample:
    """Build an immutable sample from a complete JSON-compatible mapping."""

    try:
        if not isinstance(values, Mapping) or set(values) != _SAMPLE_FIELDS:
            _invalid_sample()
        return TrainingSample(
            sample_id=values["sample_id"],
            scenario=values["scenario"],
            task_type=values["task_type"],
            modalities=_list_to_string_tuple(values["modalities"]),
            source_ids=_list_to_identifier_tuple(values["source_ids"]),
            evidence_ids=_list_to_identifier_tuple(values["evidence_ids"]),
            prompt=values["prompt"],
            reference_answer=values["reference_answer"],
            follow_ups=_list_to_string_tuple(values["follow_ups"]),
            rubric=_list_to_rubric(values["rubric"]),
            refusal_reason=values["refusal_reason"],
            provenance_ids=_list_to_identifier_tuple(values["provenance_ids"]),
            generation_method=values["generation_method"],
            quality_status=values["quality_status"],
            privacy_status=values["privacy_status"],
            asset_sidecars=_list_to_sidecars(values["asset_sidecars"]),
        )
    except (DatasetSchemaError, KeyError, TypeError, ValueError):
        _invalid_sample()


def sample_to_mapping(sample: TrainingSample) -> dict[str, object]:
    """Return a JSON-compatible representation of a canonical sample."""

    if type(sample) is not TrainingSample:
        _invalid_sample()
    values = asdict(sample)
    for field in (
        "modalities",
        "source_ids",
        "evidence_ids",
        "follow_ups",
        "rubric",
        "provenance_ids",
        "asset_sidecars",
    ):
        values[field] = list(values[field])
    return values
