"""Immutable, public-safe records for evaluation inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass

from gradloop_data.dataset_schema import SCENARIOS


class EvaluationSchemaError(ValueError):
    """A stable, non-sensitive reason why evaluation data was rejected."""


def _is_non_empty_id(value: object) -> bool:
    return type(value) is str and bool(value)


def _is_unique_id_tuple(value: object) -> bool:
    return (
        type(value) is tuple
        and all(_is_non_empty_id(item) for item in value)
        and len(set(value)) == len(value)
    )


@dataclass(frozen=True)
class EvaluationCase:
    """The expected evidence and refusal behavior for one evaluation case."""

    case_id: str
    scenario: str
    relevant_evidence_ids: tuple[str, ...]
    requires_refusal: bool

    def __post_init__(self) -> None:
        if (
            not _is_non_empty_id(self.case_id)
            or type(self.scenario) is not str
            or self.scenario not in SCENARIOS
            or not _is_unique_id_tuple(self.relevant_evidence_ids)
            or type(self.requires_refusal) is not bool
        ):
            raise EvaluationSchemaError("invalid_evaluation_case")


@dataclass(frozen=True)
class EvaluationPrediction:
    """One model prediction paired to an :class:`EvaluationCase`."""

    case_id: str
    retrieved_evidence_ids: tuple[str, ...]
    cited_evidence_ids: tuple[str, ...]
    refused: bool
    structure_valid: bool
    latency_ms: int

    def __post_init__(self) -> None:
        if (
            not _is_non_empty_id(self.case_id)
            or not _is_unique_id_tuple(self.retrieved_evidence_ids)
            or not _is_unique_id_tuple(self.cited_evidence_ids)
            or type(self.refused) is not bool
            or type(self.structure_valid) is not bool
            or type(self.latency_ms) is not int
            or self.latency_ms < 0
        ):
            raise EvaluationSchemaError("invalid_evaluation_prediction")
