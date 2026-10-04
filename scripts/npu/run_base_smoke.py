from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.base_smoke import (
    BaseSmokeError,
    TorchNpuMiniCPMBackend,
    run_base_smoke,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-manifest")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        output = Path(args.output)
        if output.exists():
            raise BaseSmokeError("output_exists")
        adapter_root = Path(args.adapter) if args.adapter else None
        adapter_manifest_path = (
            Path(args.adapter_manifest) if args.adapter_manifest else None
        )
        if (adapter_root is None) is not (adapter_manifest_path is None):
            raise BaseSmokeError("adapter_config_invalid")
        report = run_base_smoke(
            smoke_root=Path(args.smoke_root),
            model_revision=args.model_revision,
            code_commit=args.code_commit,
            config_sha256=_sha256(Path(args.config)),
            backend_factory=lambda route: TorchNpuMiniCPMBackend(
                route=route,
                model_cache_root=Path(args.model_cache),
                device_index=args.device_index,
                max_new_tokens=args.max_new_tokens,
                adapter_root=adapter_root,
            ),
            adapter_manifest_sha256=(
                _sha256(adapter_manifest_path)
                if adapter_manifest_path is not None
                else None
            ),
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except (BaseSmokeError, OSError, ValueError):
        print(json.dumps({"reason": "base_smoke_blocked"}, separators=(",", ":")))
        return 1
    print(json.dumps({"status": "passed"}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
