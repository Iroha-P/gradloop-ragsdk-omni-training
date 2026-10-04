"""Compute aggregate metrics for a frozen, public-safe evaluation set."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_eval.metrics import compute_metrics
from gradloop_eval.schema import (
    EvaluationCase,
    EvaluationPrediction,
    EvaluationSchemaError,
)

MAX_INPUT_BYTES = 16 * 1024 * 1024


class ArgumentParseFailure(ValueError):
    """Internal fixed-output signal for unsafe command-line input."""


class InputFailure(ValueError):
    """Internal fixed-output signal for unsafe evaluation input."""


class SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that never reflects raw argument values on errors."""

    def error(self, message: str) -> None:
        del message
        raise ArgumentParseFailure


def _reject_json_constant(value: str) -> None:
    del value
    raise InputFailure


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise InputFailure
        value[key] = item
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def _read_lines(path: Path) -> tuple[dict[str, object], ...]:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise InputFailure
        raw = path.read_bytes()
    except (OSError, ValueError):
        raise InputFailure from None
    if len(raw) > MAX_INPUT_BYTES:
        raise InputFailure
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        raise InputFailure from None
    records: list[dict[str, object]] = []
    for line in text.splitlines():
        if not line:
            raise InputFailure
        try:
            value = json.loads(
                line,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
            raise InputFailure from None
        if type(value) is not dict:
            raise InputFailure
        records.append(value)
    if not records:
        raise InputFailure
    return tuple(records)


def _case_from_mapping(value: Mapping[str, object]) -> EvaluationCase:
    if set(value) != {
        "case_id",
        "scenario",
        "relevant_evidence_ids",
        "requires_refusal",
    } or type(value["relevant_evidence_ids"]) is not list:
        raise InputFailure
    try:
        return EvaluationCase(
            case_id=value["case_id"],
            scenario=value["scenario"],
            relevant_evidence_ids=tuple(value["relevant_evidence_ids"]),
            requires_refusal=value["requires_refusal"],
        )
    except (EvaluationSchemaError, TypeError, ValueError):
        raise InputFailure from None


def _prediction_from_mapping(value: Mapping[str, object]) -> EvaluationPrediction:
    if set(value) != {
        "case_id",
        "retrieved_evidence_ids",
        "cited_evidence_ids",
        "refused",
        "structure_valid",
        "latency_ms",
    } or any(
        type(value[field]) is not list
        for field in ("retrieved_evidence_ids", "cited_evidence_ids")
    ):
        raise InputFailure
    try:
        return EvaluationPrediction(
            case_id=value["case_id"],
            retrieved_evidence_ids=tuple(value["retrieved_evidence_ids"]),
            cited_evidence_ids=tuple(value["cited_evidence_ids"]),
            refused=value["refused"],
            structure_valid=value["structure_valid"],
            latency_ms=value["latency_ms"],
        )
    except (EvaluationSchemaError, TypeError, ValueError):
        raise InputFailure from None


def _write_metrics(
    output: Path,
    metrics: object,
    *,
    allowed_output_roots: tuple[Path, ...],
) -> None:
    if not allowed_output_roots:
        raise ArgumentParseFailure
    destination = (
        output.resolve()
        if output.is_absolute()
        else (REPOSITORY_ROOT / output).resolve()
    )
    allowed_roots = tuple(root.resolve() for root in allowed_output_roots)
    try:
        if not any(destination.is_relative_to(root) for root in allowed_roots):
            raise ArgumentParseFailure
    except ValueError:
        raise ArgumentParseFailure from None
    if destination.exists():
        raise ArgumentParseFailure
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".frozen-evaluation-", suffix=".json", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(asdict(metrics), stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _emit(reason: str, status: int) -> int:
    print(json.dumps({"reason": reason}, sort_keys=True, separators=(",", ":")))
    return status


def main(
    argv: Sequence[str] | None = None,
    *,
    allowed_output_roots: tuple[Path, ...] | None = None,
) -> int:
    try:
        args = _build_parser().parse_args(argv)
        if allowed_output_roots is None and args.output.is_absolute():
            raise ArgumentParseFailure
        roots = allowed_output_roots or (
            REPOSITORY_ROOT / "data",
            REPOSITORY_ROOT / "outputs",
            REPOSITORY_ROOT / "dist",
        )
        cases = tuple(_case_from_mapping(value) for value in _read_lines(args.cases))
        predictions = tuple(
            _prediction_from_mapping(value) for value in _read_lines(args.predictions)
        )
        metrics = compute_metrics(cases, predictions)
        _write_metrics(args.output, metrics, allowed_output_roots=roots)
    except (InputFailure, EvaluationSchemaError):
        return _emit("invalid_input", 1)
    except (ArgumentParseFailure, OSError, ValueError):
        return _emit("invalid_arguments", 2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
