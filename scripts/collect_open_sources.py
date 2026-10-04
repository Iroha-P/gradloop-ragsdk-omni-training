"""Evaluate public-source license policy in dry-run mode only."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_data.web_sources import (
    SourceDecision,
    SourcePolicyError,
    dry_run_source,
    load_policy,
    source_license_from_mapping,
)

MAX_INPUT_BYTES = 64 * 1024
MAX_LINE_BYTES = 16 * 1024
MAX_JSON_NESTING = 128


class ArgumentParseFailure(ValueError):
    """Internal fixed-output signal for unsafe command-line input."""


class SafeArgumentParser(argparse.ArgumentParser):
    """Argument parser that never reflects raw argument values on errors."""

    def error(self, message: str) -> None:
        del message
        raise ArgumentParseFailure


class InputFailure(ValueError):
    """Internal fixed-output signal for unsafe declaration input."""


def _build_parser() -> argparse.ArgumentParser:
    parser = SafeArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="evaluate category and license policy without fetching (default)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON Lines declarations; omit to read standard input",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="authoritative JSON-compatible YAML policy",
    )
    return parser


def _read_input_bytes(path: Path | None) -> bytes:
    if path is None:
        raw = sys.stdin.buffer.read(MAX_INPUT_BYTES + 1)
    else:
        with path.open("rb") as stream:
            raw = stream.read(MAX_INPUT_BYTES + 1)
    if len(raw) > MAX_INPUT_BYTES:
        raise InputFailure
    if any(len(line) > MAX_LINE_BYTES for line in raw.splitlines()):
        raise InputFailure
    return raw


def _safe_denial(reason: str = "invalid_spec") -> SourceDecision:
    return SourceDecision(
        allowed=False,
        reason=reason,
        category="unclassified",
        license_id="unknown",
    )


def _validate_json_nesting(line: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in line:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING:
                raise InputFailure
        elif character in "]}":
            depth -= 1
            if depth < 0:
                raise InputFailure


def _evaluate_line(
    line: str,
    policy_path: Path | None,
) -> SourceDecision:
    try:
        _validate_json_nesting(line)
        raw = json.loads(line)
        if not isinstance(raw, Mapping):
            return _safe_denial()
        spec = source_license_from_mapping(raw)
        return dry_run_source(spec, policy_path=policy_path)
    except SourcePolicyError:
        return _safe_denial()


def _safe_output(decision: SourceDecision) -> str:
    return json.dumps(
        {
            "category": decision.category,
            "decision": "allowed" if decision.allowed else "denied",
            "license_id": decision.license_id,
            "reason": decision.reason,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


def _emit_safe_denial(reason: str) -> int:
    print(_safe_output(_safe_denial(reason)))
    return 2 if reason == "invalid_arguments" else 1


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except ArgumentParseFailure:
        return _emit_safe_denial("invalid_arguments")
    try:
        load_policy(args.config)
    except SourcePolicyError:
        return _emit_safe_denial("invalid_policy")
    try:
        raw_input = _read_input_bytes(args.input)
        text = raw_input.decode("utf-8")
        decisions: list[SourceDecision] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            decisions.append(_evaluate_line(line, args.config))
    except (
        InputFailure,
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        return _emit_safe_denial("invalid_input")

    denied = any(not decision.allowed for decision in decisions)
    for decision in decisions:
        print(_safe_output(decision))
    return 1 if denied else 0


if __name__ == "__main__":
    raise SystemExit(main())
