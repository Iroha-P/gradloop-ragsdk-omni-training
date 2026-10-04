"""Deterministic aggregate metrics for frozen evaluation records."""

from __future__ import annotations

from dataclasses import dataclass

from gradloop_eval.schema import (
    EvaluationCase,
    EvaluationPrediction,
    EvaluationSchemaError,
)

_SCENARIO_ORDER = (
    "course_learning",
    "technical_interview",
    "research_defense",
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate scores for a complete, one-to-one evaluation run."""

    case_count: int
    recall_at_k: float
    recall_support: int
    citation_precision: float
    citation_support: int
    refusal_accuracy: float
    refusal_positive_support: int
    refusal_negative_support: int
    structure_valid_rate: float
    mean_latency_ms: float
    by_scenario: dict[str, dict[str, float]]


def _invalid_evaluation_set() -> None:
    raise EvaluationSchemaError("invalid_evaluation_set")


def _validate_evaluation_set(
    cases: tuple[EvaluationCase, ...],
    predictions: tuple[EvaluationPrediction, ...],
) -> dict[str, EvaluationPrediction]:
    if (
        type(cases) is not tuple
        or type(predictions) is not tuple
        or not cases
        or len(cases) != len(predictions)
        or any(type(case) is not EvaluationCase for case in cases)
        or any(type(prediction) is not EvaluationPrediction for prediction in predictions)
    ):
        _invalid_evaluation_set()

    case_ids = tuple(case.case_id for case in cases)
    prediction_ids = tuple(prediction.case_id for prediction in predictions)
    if (
        len(set(case_ids)) != len(case_ids)
        or len(set(prediction_ids)) != len(prediction_ids)
        or set(case_ids) != set(prediction_ids)
    ):
        _invalid_evaluation_set()

    return {prediction.case_id: prediction for prediction in predictions}


def _citation_score(case: EvaluationCase, prediction: EvaluationPrediction) -> float:
    if not prediction.cited_evidence_ids:
        return float(case.requires_refusal and prediction.refused)
    supported_count = len(
        set(case.relevant_evidence_ids).intersection(prediction.cited_evidence_ids)
    )
    return supported_count / len(prediction.cited_evidence_ids)


def _scenario_metrics(
    cases: tuple[EvaluationCase, ...],
    predictions_by_id: dict[str, EvaluationPrediction],
    recall_k: int,
) -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {}
    for scenario in _SCENARIO_ORDER:
        scenario_cases = tuple(case for case in cases if case.scenario == scenario)
        scenario_predictions = tuple(
            predictions_by_id[case.case_id] for case in scenario_cases
        )
        metrics[scenario] = _metric_values(
            scenario_cases, scenario_predictions, recall_k=recall_k
        )
    return metrics


def _metric_values(
    cases: tuple[EvaluationCase, ...],
    predictions: tuple[EvaluationPrediction, ...],
    *,
    recall_k: int,
) -> dict[str, float]:
    count = len(cases)
    relevant_pairs = tuple(
        (case, prediction)
        for case, prediction in zip(cases, predictions)
        if case.relevant_evidence_ids
    )
    positive_support = sum(case.requires_refusal for case in cases)
    negative_support = count - positive_support
    recall_at_k = (
        sum(
            len(
                set(case.relevant_evidence_ids).intersection(
                    prediction.retrieved_evidence_ids[:recall_k]
                )
            )
            / len(case.relevant_evidence_ids)
            for case, prediction in relevant_pairs
        )
        / len(relevant_pairs)
        if relevant_pairs
        else 0.0
    )
    return {
        "case_count": float(count),
        "recall_at_k": recall_at_k,
        "recall_support": float(len(relevant_pairs)),
        "citation_precision": (
            sum(
                _citation_score(case, prediction)
                for case, prediction in zip(cases, predictions)
            )
            / count
            if count
            else 0.0
        ),
        "citation_support": float(count),
        "refusal_accuracy": (
            sum(
                case.requires_refusal is prediction.refused
                for case, prediction in zip(cases, predictions)
            )
            / count
            if count and positive_support
            else 0.0
        ),
        "refusal_positive_support": float(positive_support),
        "refusal_negative_support": float(negative_support),
        "structure_valid_rate": (
            sum(prediction.structure_valid for prediction in predictions) / count
            if count
            else 0.0
        ),
        "mean_latency_ms": (
            sum(prediction.latency_ms for prediction in predictions) / count
            if count
            else 0.0
        ),
    }


def compute_metrics(
    cases: tuple[EvaluationCase, ...],
    predictions: tuple[EvaluationPrediction, ...],
    *,
    recall_k: int = 5,
) -> EvaluationMetrics:
    """Compute deterministic metrics for a complete, validated evaluation run."""

    if type(recall_k) is not int or recall_k < 1:
        raise EvaluationSchemaError("invalid_recall_k")
    predictions_by_id = _validate_evaluation_set(cases, predictions)
    ordered_predictions = tuple(predictions_by_id[case.case_id] for case in cases)
    values = _metric_values(cases, ordered_predictions, recall_k=recall_k)

    return EvaluationMetrics(
        case_count=len(cases),
        recall_at_k=values["recall_at_k"],
        recall_support=int(values["recall_support"]),
        citation_precision=values["citation_precision"],
        citation_support=int(values["citation_support"]),
        refusal_accuracy=values["refusal_accuracy"],
        refusal_positive_support=int(values["refusal_positive_support"]),
        refusal_negative_support=int(values["refusal_negative_support"]),
        structure_valid_rate=values["structure_valid_rate"],
        mean_latency_ms=values["mean_latency_ms"],
        by_scenario=_scenario_metrics(cases, predictions_by_id, recall_k),
    )
