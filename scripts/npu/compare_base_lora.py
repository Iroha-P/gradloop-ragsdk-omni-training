from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.comparison import ComparisonError, compare_base_lora


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-report", required=True)
    parser.add_argument("--lora-report", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        output = Path(args.output)
        if output.exists():
            raise ComparisonError("output_exists")
        report = compare_base_lora(
            base_report_path=Path(args.base_report),
            lora_report_path=Path(args.lora_report),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except (ComparisonError, OSError, ValueError):
        print(json.dumps({"reason": "comparison_blocked"}, separators=(",", ":")))
        return 1
    print(
        json.dumps(
            {"status": "evaluated", "decision": report["promotion_decision"]},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
