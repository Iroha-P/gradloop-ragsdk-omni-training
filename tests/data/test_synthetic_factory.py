from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.release_gate import (
    assert_bundle_releasable,
    evaluate_release_candidate,
)
from gradloop_data.synthetic import (
    build_synthetic_samples,
    synthetic_evidence,
)


def test_factory_balances_all_three_scenarios() -> None:
    samples = build_synthetic_samples(per_scenario=4)
    counts = {
        scenario: sum(sample.scenario == scenario for sample in samples)
        for scenario in (
            "course_learning",
            "technical_interview",
            "research_defense",
        )
    }

    assert counts == {
        "course_learning": 4,
        "technical_interview": 4,
        "research_defense": 4,
    }


def test_factory_is_byte_stable_for_same_recipe() -> None:
    first = build_synthetic_samples(3, recipe_version="stage-a-v1")
    second = build_synthetic_samples(3, recipe_version="stage-a-v1")

    assert first == second
    assert synthetic_evidence(first) == synthetic_evidence(second)


def test_factory_evidence_matches_the_sample_recipe_index() -> None:
    sample = build_synthetic_samples(1)[0]

    assert synthetic_evidence((sample,)) == {
        sample.evidence_ids[0]: (
            "A synthetic process has input 1, applies transformation 2, and "
            "produces output 2."
        )
    }


def test_factory_contains_no_person_or_institution_identity_fields() -> None:
    rendered = repr(build_synthetic_samples(5)).casefold()

    for forbidden in (
        "email",
        "phone",
        "student_id",
        "identity_card",
        "school_name",
    ):
        assert forbidden not in rendered


def test_factory_includes_refusal_and_document_image_audio_sidecars() -> None:
    samples = build_synthetic_samples(2)

    assert {sample.task_type == "refusal" for sample in samples} == {False, True}
    assert {
        sidecar.modality
        for sample in samples
        for sidecar in sample.asset_sidecars
    } == {"document", "image", "audio"}
    assert all(
        set(sample.modalities) == {"text", sample.asset_sidecars[0].modality}
        for sample in samples
    )
    assert all(
        sample.refusal_reason == "insufficient_evidence"
        and sample.reference_answer == ""
        for sample in samples
        if sample.task_type == "refusal"
    )


def test_factory_rejects_counts_outside_the_supported_range() -> None:
    for invalid_count in (0, 101, True, "2"):
        with pytest.raises(ValueError, match="invalid_per_scenario"):
            build_synthetic_samples(invalid_count)  # type: ignore[arg-type]


def test_factory_rejects_an_unknown_recipe_version() -> None:
    with pytest.raises(ValueError, match="unsupported_recipe_version"):
        build_synthetic_samples(1, recipe_version="other-recipe")


def test_generated_samples_pass_release_gate() -> None:
    samples = build_synthetic_samples(10)
    evidence = synthetic_evidence(samples)
    approved = {source_id for sample in samples for source_id in sample.source_ids}
    reports = [
        evaluate_release_candidate(sample, evidence, approved)
        for sample in samples
    ]

    assert all(report.allowed for report in reports)
    assert_bundle_releasable(reports)
