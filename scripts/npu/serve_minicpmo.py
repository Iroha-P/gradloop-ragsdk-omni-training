"""Start the public-safe, local-cache-only MiniCPM-o service on TorchNPU."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.service import ServiceError, build_torchnpu_service, create_app


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-cache", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--device-index", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.host not in {"127.0.0.1", "0.0.0.0"} or not 1 <= args.port <= 65535:
        print('{"reason":"service_config_invalid"}')
        return 1
    try:
        import uvicorn

        app = create_app(
            build_torchnpu_service(
                model_cache=args.model_cache, device_index=args.device_index
            )
        )
        uvicorn.run(app, host=args.host, port=args.port, access_log=False, log_level="warning")
    except ServiceError as exc:
        print('{"reason":"' + str(exc) + '"}')
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
