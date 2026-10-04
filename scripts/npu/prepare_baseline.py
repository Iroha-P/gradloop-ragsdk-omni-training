from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.preflight import PreflightError, prepare_preflight


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        output = Path(args.output)
        if output.exists():
            raise PreflightError("output_exists")
        report = prepare_preflight(
            bundle_root=Path(args.bundle),
            model_cache_root=Path(args.model_cache),
            model_revision=args.model_revision,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except (PreflightError, OSError, ValueError):
        print(json.dumps({"reason": "preflight_blocked"}, separators=(",", ":")))
        return 1
    print(json.dumps({"status": "ready"}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
