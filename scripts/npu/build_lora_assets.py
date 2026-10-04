from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.lora_assets import LoraAssetError, build_lora_assets


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        manifest = build_lora_assets(
            smoke_root=Path(args.smoke_root),
            output_root=Path(args.output),
        )
    except (LoraAssetError, OSError, ValueError):
        print(json.dumps({"reason": "lora_assets_blocked"}, separators=(",", ":")))
        return 1
    print(
        json.dumps(
            {"status": "ready", "sample_count": manifest["sample_count"]},
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
