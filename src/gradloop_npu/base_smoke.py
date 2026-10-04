"""Privacy-preserving MiniCPM-o Base smoke execution on TorchNPU."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

_PINNED_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ROUTE_ORDER = ("text", "vision", "audio", "omni")
_ALLOWED_MODALITIES = frozenset({"text", "image", "audio"})
_ALLOWED_ASSETS = frozenset({"synthetic-grid.png", "synthetic-tone.wav"})


class BaseSmokeError(ValueError):
    """Stable, non-sensitive failure from the Base smoke gate."""


@dataclass(frozen=True)
class SmokeCase:
    """Validated synthetic smoke input."""

    case_id: str
    modalities: tuple[str, ...]
    assets: tuple[str, ...]
    prompt: str

    @property
    def route(self) -> str:
        has_image = "image" in self.modalities
        has_audio = "audio" in self.modalities
        if has_image and has_audio:
            return "omni"
        if has_image:
            return "vision"
        if has_audio:
            return "audio"
        return "text"


@dataclass(frozen=True)
class BackendResult:
    """One inference result before redaction."""

    text: str
    first_chunk_ms: int
    end_to_end_ms: int
    response_start_method: str
    peak_memory_bytes: int


class SmokeBackend(Protocol):
    """Minimal backend contract used by the deterministic smoke runner."""

    def infer(self, case: SmokeCase, assets: Mapping[str, Path]) -> BackendResult:
        """Run one case."""

    def close(self) -> None:
        """Release route-specific model state."""


BackendFactory = Callable[[str], SmokeBackend]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_manifest(smoke_root: Path) -> dict[str, object]:
    try:
        manifest = json.loads(
            (smoke_root / "smoke_manifest.json").read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise BaseSmokeError("smoke_bundle_invalid") from exc
    if (
        type(manifest) is not dict
        or manifest.get("schema_version") != "1.0"
        or manifest.get("sample_count") != 10
        or type(manifest.get("artifacts")) is not dict
    ):
        raise BaseSmokeError("smoke_bundle_invalid")
    expected = manifest["artifacts"]
    if set(expected) != {"cases.jsonl", *_ALLOWED_ASSETS}:
        raise BaseSmokeError("smoke_bundle_invalid")
    for name, expected_hash in expected.items():
        path = smoke_root / name
        if (
            type(expected_hash) is not str
            or _SHA256.fullmatch(expected_hash) is None
            or not path.is_file()
            or _sha256(path) != expected_hash
        ):
            raise BaseSmokeError("smoke_bundle_invalid")
    return manifest


def _parse_case(value: object) -> SmokeCase:
    if type(value) is not dict:
        raise BaseSmokeError("smoke_bundle_invalid")
    case_id = value.get("case_id")
    modalities = value.get("modalities")
    assets = value.get("assets")
    prompt = value.get("prompt")
    if (
        type(case_id) is not str
        or not case_id
        or type(modalities) is not list
        or not modalities
        or any(type(item) is not str for item in modalities)
        or set(modalities) - _ALLOWED_MODALITIES
        or "text" not in modalities
        or len(modalities) != len(set(modalities))
        or type(assets) is not list
        or any(type(item) is not str for item in assets)
        or set(assets) - _ALLOWED_ASSETS
        or len(assets) != len(set(assets))
        or type(prompt) is not str
        or not prompt
    ):
        raise BaseSmokeError("smoke_bundle_invalid")
    if ("image" in modalities) is not ("synthetic-grid.png" in assets):
        raise BaseSmokeError("smoke_bundle_invalid")
    if ("audio" in modalities) is not ("synthetic-tone.wav" in assets):
        raise BaseSmokeError("smoke_bundle_invalid")
    return SmokeCase(
        case_id=case_id,
        modalities=tuple(modalities),
        assets=tuple(assets),
        prompt=prompt,
    )


def load_smoke_cases(smoke_root: Path) -> tuple[dict[str, object], tuple[SmokeCase, ...]]:
    """Load a deterministic, integrity-checked synthetic bundle."""

    if not smoke_root.is_dir():
        raise BaseSmokeError("smoke_bundle_invalid")
    manifest = _validate_manifest(smoke_root)
    try:
        values = tuple(
            json.loads(line)
            for line in (smoke_root / "cases.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        )
    except Exception as exc:
        raise BaseSmokeError("smoke_bundle_invalid") from exc
    cases = tuple(_parse_case(value) for value in values)
    case_ids = tuple(case.case_id for case in cases)
    routes = {case.route for case in cases}
    if (
        len(cases) != 10
        or len(case_ids) != len(set(case_ids))
        or routes != set(_ROUTE_ORDER)
    ):
        raise BaseSmokeError("smoke_bundle_invalid")
    return manifest, cases


def _redacted_record(case: SmokeCase, result: BackendResult) -> dict[str, object]:
    if (
        type(result.text) is not str
        or not result.text.strip()
        or type(result.first_chunk_ms) is not int
        or result.first_chunk_ms < 0
        or type(result.end_to_end_ms) is not int
        or result.end_to_end_ms < result.first_chunk_ms
        or result.response_start_method
        not in {"stream_observed", "non_streaming_upper_bound"}
        or type(result.peak_memory_bytes) is not int
        or result.peak_memory_bytes < 0
    ):
        raise BaseSmokeError("backend_result_invalid")
    return {
        "case_id": case.case_id,
        "route": case.route,
        "modalities": list(case.modalities),
        "status": "passed",
        "output_sha256": hashlib.sha256(result.text.encode("utf-8")).hexdigest(),
        "output_char_count": len(result.text),
        "time_to_first_chunk_ms": result.first_chunk_ms,
        "response_start_method": result.response_start_method,
        "end_to_end_ms": result.end_to_end_ms,
        "peak_memory_bytes": result.peak_memory_bytes,
    }


def run_base_smoke(
    *,
    smoke_root: Path,
    model_revision: str,
    code_commit: str,
    config_sha256: str,
    backend_factory: BackendFactory,
    adapter_manifest_sha256: str | None = None,
) -> dict[str, object]:
    """Run all four modality routes without retaining prompts or outputs."""

    if (
        type(model_revision) is not str
        or _PINNED_REVISION.fullmatch(model_revision) is None
        or type(code_commit) is not str
        or not code_commit
        or type(config_sha256) is not str
        or _SHA256.fullmatch(config_sha256) is None
        or (
            adapter_manifest_sha256 is not None
            and (
                type(adapter_manifest_sha256) is not str
                or _SHA256.fullmatch(adapter_manifest_sha256) is None
            )
        )
    ):
        raise BaseSmokeError("run_identity_invalid")
    _, cases = load_smoke_cases(smoke_root)
    assets = {name: smoke_root / name for name in _ALLOWED_ASSETS}
    records: list[dict[str, object]] = []
    route_counts: dict[str, int] = {}
    for route in _ROUTE_ORDER:
        route_cases = tuple(case for case in cases if case.route == route)
        route_counts[route] = len(route_cases)
        backend: SmokeBackend | None = None
        try:
            backend = backend_factory(route)
            for case in route_cases:
                records.append(_redacted_record(case, backend.infer(case, assets)))
        except BaseSmokeError:
            raise
        except Exception as exc:
            raise BaseSmokeError("backend_failed") from exc
        finally:
            if backend is not None:
                try:
                    backend.close()
                except Exception as exc:
                    raise BaseSmokeError("backend_close_failed") from exc

    latency = tuple(int(record["end_to_end_ms"]) for record in records)
    response_start = tuple(
        int(record["time_to_first_chunk_ms"]) for record in records
    )
    peak_memory = max(int(record["peak_memory_bytes"]) for record in records)
    return {
        "schema_version": "1.0",
        "status": "passed",
        "model": {
            "id": "openbmb/MiniCPM-o-4_5",
            "revision": model_revision,
            "dtype": "bfloat16",
            "local_files_only": True,
            "trust_remote_code": True,
            "adapter_manifest_sha256": adapter_manifest_sha256,
        },
        "run_identity": {
            "code_commit": code_commit,
            "config_sha256": config_sha256,
            "smoke_manifest_sha256": _sha256(smoke_root / "smoke_manifest.json"),
        },
        "privacy": {
            "private_data_allowed": False,
            "prompts_retained": False,
            "outputs_retained": False,
            "paths_retained": False,
        },
        "case_count": len(records),
        "route_counts": route_counts,
        "success_rate": 1.0,
        "mean_time_to_first_chunk_ms": sum(response_start) / len(response_start),
        "mean_end_to_end_ms": sum(latency) / len(latency),
        "peak_memory_bytes": peak_memory,
        "records": records,
    }


class TorchNpuMiniCPMBackend:
    """Official Transformers chat route backed by a local-only model cache."""

    def __init__(
        self,
        *,
        route: str,
        model_cache_root: Path,
        device_index: int,
        max_new_tokens: int,
        adapter_root: Path | None = None,
    ) -> None:
        if (
            route not in _ROUTE_ORDER
            or type(device_index) is not int
            or device_index < 0
            or type(max_new_tokens) is not int
            or max_new_tokens < 1
        ):
            raise BaseSmokeError("backend_config_invalid")
        try:
            import torch
            import torch_npu  # noqa: F401
            from transformers import AutoModel
        except Exception as exc:
            raise BaseSmokeError("backend_dependency_unavailable") from exc

        self._torch = torch
        self._route = route
        self._device_index = device_index
        self._max_new_tokens = max_new_tokens
        init_vision = route in {"vision", "omni"}
        init_audio = route in {"audio", "omni"}
        try:
            torch.npu.set_device(device_index)
            model = AutoModel.from_pretrained(
                str(model_cache_root),
                trust_remote_code=True,
                local_files_only=True,
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16,
                init_vision=init_vision,
                init_audio=init_audio,
                init_tts=False,
            )
            if adapter_root is not None:
                from peft import PeftModel

                model = PeftModel.from_pretrained(
                    model,
                    str(adapter_root),
                    is_trainable=False,
                )
            self._model = model.eval().to(f"npu:{device_index}")
        except Exception as exc:
            raise BaseSmokeError("model_load_failed") from exc

    def _content(self, case: SmokeCase, assets: Mapping[str, Path]) -> list[object]:
        content: list[object] = []
        try:
            if "image" in case.modalities:
                from PIL import Image

                content.append(Image.open(assets["synthetic-grid.png"]).convert("RGB"))
            if "audio" in case.modalities:
                import librosa

                audio, _ = librosa.load(
                    str(assets["synthetic-tone.wav"]), sr=16_000, mono=True
                )
                content.append(audio)
        except Exception as exc:
            raise BaseSmokeError("asset_decode_failed") from exc
        content.append(case.prompt)
        return content

    def infer(self, case: SmokeCase, assets: Mapping[str, Path]) -> BackendResult:
        torch = self._torch
        try:
            torch.npu.synchronize(self._device_index)
            torch.npu.reset_peak_memory_stats(self._device_index)
            started = time.monotonic_ns()
            response = self._model.chat(
                msgs=[{"role": "user", "content": self._content(case, assets)}],
                use_tts_template=False,
                enable_thinking=False,
                stream=True,
                max_new_tokens=self._max_new_tokens,
            )
            if isinstance(response, str):
                text = response
                torch.npu.synchronize(self._device_index)
                ended = time.monotonic_ns()
                elapsed = max(0, (ended - started) // 1_000_000)
                first_chunk_ms = elapsed
                response_start_method = "non_streaming_upper_bound"
            else:
                chunks: list[str] = []
                first_chunk_ns: int | None = None
                for chunk in response:
                    if first_chunk_ns is None:
                        first_chunk_ns = time.monotonic_ns()
                    chunks.append(str(chunk))
                torch.npu.synchronize(self._device_index)
                ended = time.monotonic_ns()
                text = "".join(chunks)
                if first_chunk_ns is None:
                    raise BaseSmokeError("empty_stream")
                first_chunk_ms = max(0, (first_chunk_ns - started) // 1_000_000)
                elapsed = max(first_chunk_ms, (ended - started) // 1_000_000)
                response_start_method = "stream_observed"
            peak_memory = int(torch.npu.max_memory_allocated(self._device_index))
            return BackendResult(
                text=text,
                first_chunk_ms=int(first_chunk_ms),
                end_to_end_ms=int(elapsed),
                response_start_method=response_start_method,
                peak_memory_bytes=peak_memory,
            )
        except BaseSmokeError:
            raise
        except Exception as exc:
            raise BaseSmokeError("inference_failed") from exc

    def close(self) -> None:
        model = getattr(self, "_model", None)
        if model is not None:
            del self._model
        self._torch.npu.empty_cache()
