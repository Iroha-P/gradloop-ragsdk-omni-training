from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_eval.metrics import compute_metrics
from gradloop_eval.schema import (
    EvaluationCase,
    EvaluationPrediction,
    EvaluationSchemaError,
)


def test_metrics_report_retrieval_citation_refusal_and_structure() -> None:
    cases = (
        EvaluationCase(
            case_id="case-001",
            scenario="course_learning",
            relevant_evidence_ids=("evidence-a",),
            requires_refusal=False,
        ),
        EvaluationCase(
            case_id="case-002",
            scenario="research_defense",
            relevant_evidence_ids=(),
            requires_refusal=True,
        ),
    )
    predictions = (
        EvaluationPrediction(
            case_id="case-001",
            retrieved_evidence_ids=("evidence-a", "evidence-x"),
            cited_evidence_ids=("evidence-a",),
            refused=False,
            structure_valid=True,
            latency_ms=120,
        ),
        EvaluationPrediction(
            case_id="case-002",
            retrieved_evidence_ids=(),
            cited_evidence_ids=(),
            refused=True,
            structure_valid=True,
            latency_ms=80,
        ),
    )

    metrics = compute_metrics(cases, predictions, recall_k=5)

    assert metrics.case_count == 2
    assert metrics.recall_support == 1
    assert metrics.citation_support == 2
    assert metrics.refusal_positive_support == 1
    assert metrics.refusal_negative_support == 1
    assert metrics.recall_at_k == 1.0
    assert metrics.citation_precision == 1.0
    assert metrics.refusal_accuracy == 1.0
    assert metrics.structure_valid_rate == 1.0
    assert metrics.mean_latency_ms == 100.0
    assert metrics.by_scenario == {
        "course_learning": {
            "case_count": 1.0,
            "recall_at_k": 1.0,
            "recall_support": 1.0,
            "citation_precision": 1.0,
            "citation_support": 1.0,
            "refusal_accuracy": 0.0,
            "refusal_positive_support": 0.0,
            "refusal_negative_support": 1.0,
            "structure_valid_rate": 1.0,
            "mean_latency_ms": 120.0,
        },
        "technical_interview": {
            "case_count": 0.0,
            "recall_at_k": 0.0,
            "recall_support": 0.0,
            "citation_precision": 0.0,
            "citation_support": 0.0,
            "refusal_accuracy": 0.0,
            "refusal_positive_support": 0.0,
            "refusal_negative_support": 0.0,
            "structure_valid_rate": 0.0,
            "mean_latency_ms": 0.0,
        },
        "research_defense": {
            "case_count": 1.0,
            "recall_at_k": 0.0,
            "recall_support": 0.0,
            "citation_precision": 1.0,
            "citation_support": 1.0,
            "refusal_accuracy": 1.0,
            "refusal_positive_support": 1.0,
            "refusal_negative_support": 0.0,
            "structure_valid_rate": 1.0,
            "mean_latency_ms": 80.0,
        },
    }


@pytest.mark.parametrize(
    ("factory", "updates"),
    [
        (EvaluationCase, {"case_id": "", "scenario": "course_learning"}),
        (EvaluationCase, {"scenario": "general_chat"}),
        (EvaluationCase, {"relevant_evidence_ids": ["evidence-a"]}),
        (EvaluationPrediction, {"case_id": ""}),
        (EvaluationPrediction, {"retrieved_evidence_ids": ["evidence-a"]}),
        (EvaluationPrediction, {"refused": 1}),
        (EvaluationPrediction, {"latency_ms": -1}),
        (EvaluationPrediction, {"latency_ms": True}),
    ],
)
def test_records_reject_invalid_schema_values(
    factory: type[EvaluationCase | EvaluationPrediction],
    updates: dict[str, object],
) -> None:
    case = {
        "case_id": "case-001",
        "scenario": "course_learning",
        "relevant_evidence_ids": ("evidence-a",),
        "requires_refusal": False,
    }
    prediction = {
        "case_id": "case-001",
        "retrieved_evidence_ids": ("evidence-a",),
        "cited_evidence_ids": ("evidence-a",),
        "refused": False,
        "structure_valid": True,
        "latency_ms": 20,
    }
    values = {**(case if factory is EvaluationCase else prediction), **updates}

    with pytest.raises(EvaluationSchemaError):
        factory(**values)


def test_compute_metrics_rejects_duplicate_case_ids() -> None:
    cases = (
        EvaluationCase("case-001", "course_learning", (), False),
        EvaluationCase("case-001", "technical_interview", (), False),
    )
    predictions = (
        EvaluationPrediction("case-001", (), (), False, True, 20),
        EvaluationPrediction("case-001", (), (), False, True, 30),
    )

    with pytest.raises(EvaluationSchemaError, match="invalid_evaluation_set"):
        compute_metrics(cases, predictions)


@pytest.mark.parametrize(
    "predictions",
    [
        (),
        (EvaluationPrediction("case-002", (), (), False, True, 20),),
    ],
)
def test_compute_metrics_rejects_missing_or_extra_predictions(
    predictions: tuple[EvaluationPrediction, ...],
) -> None:
    cases = (EvaluationCase("case-001", "course_learning", (), False),)

    with pytest.raises(EvaluationSchemaError, match="invalid_evaluation_set"):
        compute_metrics(cases, predictions)


def test_compute_metrics_rejects_recall_k_less_than_one() -> None:
    cases = (EvaluationCase("case-001", "course_learning", (), False),)
    predictions = (EvaluationPrediction("case-001", (), (), False, True, 20),)

    with pytest.raises(EvaluationSchemaError, match="invalid_recall_k"):
        compute_metrics(cases, predictions, recall_k=0)


def test_scenario_metrics_expose_hidden_retrieval_and_citation_regression() -> None:
    cases = (
        EvaluationCase("case-001", "course_learning", ("evidence-a",), False),
        EvaluationCase("case-002", "technical_interview", ("evidence-b",), False),
        EvaluationCase("case-003", "research_defense", (), True),
    )
    predictions = (
        EvaluationPrediction(
            "case-001", ("evidence-a",), ("evidence-a",), False, True, 10
        ),
        EvaluationPrediction(
            "case-002", ("evidence-x",), ("evidence-x",), False, True, 30
        ),
        EvaluationPrediction("case-003", (), (), True, True, 20),
    )

    metrics = compute_metrics(cases, predictions)

    assert metrics.recall_at_k == 0.5
    assert metrics.citation_precision == pytest.approx(2 / 3)
    assert metrics.by_scenario["technical_interview"] == {
        "case_count": 1.0,
        "recall_at_k": 0.0,
        "recall_support": 1.0,
        "citation_precision": 0.0,
        "citation_support": 1.0,
        "refusal_accuracy": 0.0,
        "refusal_positive_support": 0.0,
        "refusal_negative_support": 1.0,
        "structure_valid_rate": 1.0,
        "mean_latency_ms": 30.0,
    }


def test_zero_positive_refusal_support_does_not_report_perfect_accuracy() -> None:
    cases = (EvaluationCase("case-001", "course_learning", ("evidence-a",), False),)
    predictions = (
        EvaluationPrediction(
            "case-001", ("evidence-a",), ("evidence-a",), False, True, 10
        ),
    )

    metrics = compute_metrics(cases, predictions)

    assert metrics.refusal_positive_support == 0
    assert metrics.refusal_accuracy == 0.0
