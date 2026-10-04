from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.dataset_schema import sample_from_mapping
from gradloop_data.splits import (
    SplitAssignment,
    SplitError,
    assert_no_source_leakage,
    assign_source_connected_splits,
)

VALID_SAMPLE_MAPPING = {
    "sample_id": "sample-000",
    "scenario": "course_learning",
    "task_type": "evidence_answer",
    "modalities": ["text"],
    "source_ids": ["source-a"],
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


def make_sample(sample_id: str, source_ids: list[str]) -> object:
    payload = {
        **VALID_SAMPLE_MAPPING,
        "sample_id": sample_id,
        "source_ids": source_ids,
    }
    return sample_from_mapping(payload)


def test_shared_sources_force_samples_into_same_split() -> None:
    samples = [
        make_sample("sample-001", ["source-a", "source-b"]),
        make_sample("sample-002", ["source-b", "source-c"]),
        make_sample("sample-003", ["source-z"]),
    ]

    assignments = assign_source_connected_splits(samples, salt="stage-a-v1")

    by_sample = {item.sample_id: item.split for item in assignments}
    assert by_sample["sample-001"] == by_sample["sample-002"]
    assert_no_source_leakage(assignments)


def test_assignment_is_input_order_independent() -> None:
    samples = [
        make_sample("sample-001", ["source-a"]),
        make_sample("sample-002", ["source-b"]),
    ]

    first = assign_source_connected_splits(samples, salt="stage-a-v1")
    second = assign_source_connected_splits(list(reversed(samples)), salt="stage-a-v1")

    assert sorted(first, key=lambda item: item.sample_id) == sorted(
        second, key=lambda item: item.sample_id
    )


def test_leakage_assertion_rejects_shared_source_in_different_splits() -> None:
    assignments = [
        SplitAssignment("sample-001", ("source-a",), "train"),
        SplitAssignment("sample-002", ("source-a",), "test"),
    ]

    with pytest.raises(SplitError, match="source_leakage"):
        assert_no_source_leakage(assignments)


@pytest.mark.parametrize(
    ("samples", "salt"),
    [
        ([], "stage-a-v1"),
        ([make_sample("sample-001", ["source-a"])] * 2, "stage-a-v1"),
        ([make_sample("sample-001", ["source-a"])], 1),
    ],
)
def test_assignment_rejects_invalid_split_input(
    samples: list[object], salt: object
) -> None:
    with pytest.raises(SplitError, match="invalid_split_input"):
        assign_source_connected_splits(samples, salt=salt)
