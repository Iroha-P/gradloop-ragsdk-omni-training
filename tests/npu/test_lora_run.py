from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from gradloop_npu.lora_assets import build_lora_assets
from gradloop_npu.lora_run import LoraRunError, run_lora_feasibility
from gradloop_npu.smoke_assets import build_smoke_assets


def _write_runtime(root: Path) -> tuple[Path, Path, Path]:
    swift = root / "swift"
    config = root / "config.yaml"
    model = root / "model"
    swift.write_text("fixture", encoding="utf-8")
    config.write_text("tuner_type: lora\n", encoding="utf-8")
    model.mkdir()
    return swift, config, model


def test_lora_run_proves_update_and_redacts_paths(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    log = tmp_path / "run.log"
    build_smoke_assets(smoke)
    build_lora_assets(smoke_root=smoke, output_root=dataset)
    swift, config, model = _write_runtime(tmp_path)

    def fake_runner(
        command: tuple[str, ...],
        environment: dict[str, str],
        working_directory: Path,
        log_path: Path,
    ) -> int:
        assert command[1] == "sft"
        assert Path(command[2]).is_absolute()
        assert Path(command[4]).is_absolute()
        assert Path(command[6]).is_absolute()
        assert Path(command[8]).is_absolute()
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert working_directory == dataset
        log_path.write_text("synthetic log", encoding="utf-8")
        checkpoint = output / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 1}) + "\n", encoding="utf-8"
        )
        return 0

    report, manifest = run_lora_feasibility(
        swift_executable=swift,
        config_path=config,
        model_cache_root=model,
        model_revision="a" * 40,
        dataset_root=dataset,
        output_root=output,
        log_path=log,
        code_commit="test-commit",
        device_index=0,
        process_runner=fake_runner,
    )

    assert report["status"] == "trained"
    assert report["training"]["optimizer_updates"] == 1
    assert manifest["full_model_weights_included"] is False
    assert manifest["artifacts"] == {
        "adapter_model.safetensors": hashlib.sha256(b"adapter").hexdigest(),
        "adapter_config.json": hashlib.sha256(
            (output / "checkpoint-1" / "adapter_config.json").read_bytes()
        ).hexdigest(),
    }
    assert str(tmp_path) not in json.dumps(report)


def test_lora_run_rejects_full_model_artifact(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke"
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    build_smoke_assets(smoke)
    build_lora_assets(smoke_root=smoke, output_root=dataset)
    swift, config, model = _write_runtime(tmp_path)

    def fake_runner(*args: object) -> int:
        checkpoint = output / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        (checkpoint / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"forbidden")
        (checkpoint / "trainer_state.json").write_text(
            '{"global_step":1}\n', encoding="utf-8"
        )
        return 0

    with pytest.raises(LoraRunError, match="full_model_artifact_forbidden"):
        run_lora_feasibility(
            swift_executable=swift,
            config_path=config,
            model_cache_root=model,
            model_revision="a" * 40,
            dataset_root=dataset,
            output_root=output,
            log_path=tmp_path / "log",
            code_commit="test-commit",
            device_index=0,
            process_runner=fake_runner,
        )
