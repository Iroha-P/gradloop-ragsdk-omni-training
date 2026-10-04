from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.base_smoke import (
    BackendResult,
    BaseSmokeError,
    SmokeCase,
    load_smoke_cases,
    run_base_smoke,
)
from gradloop_npu.smoke_assets import build_smoke_assets


class _FakeBackend:
    def __init__(self, route: str, closed: list[str]) -> None:
        self.route = route
        self.closed = closed

    def infer(
        self, case: SmokeCase, assets: dict[str, Path]
    ) -> BackendResult:
        assert case.route == self.route
        assert all(path.is_file() for path in assets.values())
        return BackendResult(
            text=f"synthetic response for {case.case_id}",
            first_chunk_ms=10,
            end_to_end_ms=25,
            response_start_method="stream_observed",
            peak_memory_bytes=1024,
        )

    def close(self) -> None:
        self.closed.append(self.route)


def test_base_smoke_covers_all_routes_without_retaining_content(
    tmp_path: Path,
) -> None:
    smoke_root = tmp_path / "smoke"
    build_smoke_assets(smoke_root)
    closed: list[str] = []

    report = run_base_smoke(
        smoke_root=smoke_root,
        model_revision="a" * 40,
        code_commit="test-commit",
        config_sha256="b" * 64,
        backend_factory=lambda route: _FakeBackend(route, closed),
    )

    assert report["status"] == "passed"
    assert report["model"]["adapter_manifest_sha256"] is None
    assert report["case_count"] == 10
    assert report["route_counts"] == {
        "text": 2,
        "vision": 3,
        "audio": 3,
        "omni": 2,
    }
    assert report["success_rate"] == 1.0
    assert report["privacy"] == {
        "private_data_allowed": False,
        "prompts_retained": False,
        "outputs_retained": False,
        "paths_retained": False,
    }
    assert closed == ["text", "vision", "audio", "omni"]
    rendered = json.dumps(report)
    assert "Describe the supplied synthetic signal" not in rendered
    assert "synthetic response for" not in rendered
    assert str(tmp_path) not in rendered
    assert all(len(record["output_sha256"]) == 64 for record in report["records"])


def test_smoke_bundle_integrity_fails_closed(tmp_path: Path) -> None:
    smoke_root = tmp_path / "smoke"
    build_smoke_assets(smoke_root)
    (smoke_root / "synthetic-tone.wav").write_bytes(b"changed")

    with pytest.raises(BaseSmokeError, match="smoke_bundle_invalid"):
        load_smoke_cases(smoke_root)


def test_backend_failure_is_redacted(tmp_path: Path) -> None:
    smoke_root = tmp_path / "smoke"
    build_smoke_assets(smoke_root)

    class _FailingBackend(_FakeBackend):
        def infer(
            self, case: SmokeCase, assets: dict[str, Path]
        ) -> BackendResult:
            raise RuntimeError("sensitive path and prompt")

    with pytest.raises(BaseSmokeError, match="backend_failed") as captured:
        run_base_smoke(
            smoke_root=smoke_root,
            model_revision="c" * 40,
            code_commit="test-commit",
            config_sha256="d" * 64,
            backend_factory=lambda route: _FailingBackend(route, []),
        )
    assert "sensitive path" not in str(captured.value)
