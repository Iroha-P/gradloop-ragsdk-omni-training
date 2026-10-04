from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.lora_run import LoraRunError, run_lora_feasibility


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--swift-executable", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--adapter-manifest", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--device-index", type=int, default=0)
    return parser


def _write_new(path: Path, value: dict[str, object]) -> None:
    if path.exists():
        raise LoraRunError("output_exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        report_path = Path(args.report)
        manifest_path = Path(args.adapter_manifest)
        if report_path.exists() or manifest_path.exists():
            raise LoraRunError("output_exists")
        report, adapter_manifest = run_lora_feasibility(
            swift_executable=Path(args.swift_executable),
            config_path=Path(args.config),
            model_cache_root=Path(args.model_cache),
            model_revision=args.model_revision,
            dataset_root=Path(args.dataset_root),
            output_root=Path(args.output_root),
            log_path=Path(args.log),
            code_commit=args.code_commit,
            device_index=args.device_index,
        )
        _write_new(manifest_path, adapter_manifest)
        _write_new(report_path, report)
    except (LoraRunError, OSError, ValueError):
        print(json.dumps({"reason": "lora_feasibility_blocked"}, separators=(",", ":")))
        return 1
    print(json.dumps({"status": "trained"}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
