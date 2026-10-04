from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.dataset_schema import (
    DatasetSchemaError,
    TrainingSample,
    sample_from_mapping,
    sample_to_mapping,
)

VALID = {
    "sample_id": "course-001",
    "scenario": "course_learning",
    "task_type": "evidence_answer",
    "modalities": ["text"],
    "source_ids": ["public-source-001"],
    "evidence_ids": ["evidence-001"],
    "prompt": "Explain the relationship using the supplied evidence.",
    "reference_answer": "The supplied evidence describes a direct relationship.",
    "follow_ups": ["What assumption supports that conclusion?"],
    "rubric": [
        {
            "dimension": "evidence_use",
            "description": "Uses the supplied evidence.",
            "max_score": 5,
        }
    ],
    "refusal_reason": "",
    "provenance_ids": ["provenance-001"],
    "generation_method": "deterministic_template_v1",
    "quality_status": "candidate",
    "privacy_status": "unscanned",
    "asset_sidecars": [],
}


def test_valid_sample_round_trips_without_type_drift() -> None:
    sample = sample_from_mapping(VALID)

    assert isinstance(sample, TrainingSample)
    assert sample.scenario == "course_learning"
    assert sample_to_mapping(sample) == VALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scenario", "general_chat"),
        ("modalities", ["text", "unknown"]),
        ("source_ids", []),
        ("quality_status", "approved"),
        ("privacy_status", "clean"),
    ],
)
def test_invalid_or_premature_values_fail_closed(field: str, value: object) -> None:
    payload = {**VALID, field: value}

    with pytest.raises(DatasetSchemaError, match="invalid_sample"):
        sample_from_mapping(payload)


def test_refusal_requires_reason_and_no_unsupported_answer() -> None:
    payload = {
        **VALID,
        "task_type": "refusal",
        "reference_answer": "",
        "refusal_reason": "insufficient_evidence",
    }

    assert sample_from_mapping(payload).refusal_reason == "insufficient_evidence"


def test_asset_modality_must_appear_in_sample_modalities() -> None:
    payload = {
        **VALID,
        "asset_sidecars": [
            {
                "asset_id": "asset-001",
                "modality": "image",
                "ocr_text": "synthetic chart",
                "transcript": "",
                "metadata_text": "",
            }
        ],
    }

    with pytest.raises(DatasetSchemaError, match="invalid_sample"):
        sample_from_mapping(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("modalities", "text"),
        ("source_ids", ["public-source-001", "public-source-001"]),
        ("evidence_ids", ["evidence-001", "evidence-001"]),
        ("provenance_ids", ["provenance-001", "provenance-001"]),
        ("rubric", [{"dimension": "evidence_use", "description": "uses evidence", "max_score": True}]),
    ],
)
def test_noncanonical_collections_and_identifiers_fail_closed(
    field: str, value: object
) -> None:
    payload = {**VALID, field: value}

    with pytest.raises(DatasetSchemaError, match="invalid_sample"):
        sample_from_mapping(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {**VALID, "task_type": "refusal"},
        {**VALID, "task_type": "refusal", "reference_answer": "unsupported"},
        {**VALID, "refusal_reason": "insufficient_evidence"},
    ],
)
def test_refusal_fields_must_match_task_type(payload: dict[str, object]) -> None:
    with pytest.raises(DatasetSchemaError, match="invalid_sample"):
        sample_from_mapping(payload)


def test_direct_constructor_rejects_unhashable_enum_values_as_invalid_samples() -> None:
    sample = sample_from_mapping(VALID)

    with pytest.raises(DatasetSchemaError, match="invalid_sample"):
        replace(sample, scenario=[])
