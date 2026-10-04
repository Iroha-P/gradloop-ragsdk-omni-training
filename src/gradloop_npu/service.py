"""A small, fail-closed MiniCPM-o HTTP service for public-safe demos.

The service is deliberately separate from experiment runners: it accepts only
request-scoped public or synthetic media, writes no prompt, media, path, token,
or user identifier to disk, and exposes no training endpoint.
"""

from __future__ import annotations

import base64
import binascii
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

_MAX_ATTACHMENTS = 2
_MAX_ATTACHMENT_BYTES = 2 * 1024 * 1024
_MAX_TOTAL_ATTACHMENT_BYTES = 3 * 1024 * 1024
_MAX_PROMPT_CHARS = 4_000
_MAX_NEW_TOKENS = 256
_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
_AUDIO_TYPES = frozenset({"audio/wav", "audio/x-wav"})


class ServiceError(ValueError):
    """A stable client-safe service failure that contains no request content."""


class ChatModel(Protocol):
    """The small subset of MiniCPM-o used by the serving boundary."""

    def chat(self, **kwargs: object) -> object:
        """Return a non-streaming response for one message."""


@dataclass(frozen=True)
class MediaAttachment:
    """Validated, in-memory public/synthetic attachment."""

    media_type: str
    payload: bytes


@dataclass(frozen=True)
class GenerateRequest:
    """Validated input without caller identity, file names, or paths."""

    prompt: str
    attachments: tuple[MediaAttachment, ...]
    max_new_tokens: int


def _require_mapping(value: object) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ServiceError("invalid_request")
    return value


def _decode_attachment(value: object) -> MediaAttachment:
    item = _require_mapping(value)
    if set(item) != {"media_type", "content_base64"}:
        raise ServiceError("invalid_attachment")
    media_type = item.get("media_type")
    encoded = item.get("content_base64")
    if type(media_type) is not str or media_type not in (_IMAGE_TYPES | _AUDIO_TYPES):
        raise ServiceError("unsupported_media_type")
    if type(encoded) is not str or not encoded:
        raise ServiceError("invalid_attachment")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ServiceError("invalid_attachment") from exc
    if not payload or len(payload) > _MAX_ATTACHMENT_BYTES:
        raise ServiceError("attachment_too_large")
    return MediaAttachment(media_type=media_type, payload=payload)


def parse_generate_request(value: object) -> GenerateRequest:
    """Validate a public-demo request before model access."""

    payload = _require_mapping(value)
    if set(payload) - {"prompt", "attachments", "max_new_tokens", "content_policy"}:
        raise ServiceError("invalid_request")
    if payload.get("content_policy") != "public_or_synthetic":
        raise ServiceError("content_policy_required")
    prompt = payload.get("prompt")
    attachments_value = payload.get("attachments", [])
    max_new_tokens = payload.get("max_new_tokens", 128)
    if type(prompt) is not str or not prompt.strip() or len(prompt) > _MAX_PROMPT_CHARS:
        raise ServiceError("invalid_prompt")
    if type(attachments_value) is not list or len(attachments_value) > _MAX_ATTACHMENTS:
        raise ServiceError("invalid_attachment")
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= _MAX_NEW_TOKENS:
        raise ServiceError("invalid_generation_limit")
    attachments = tuple(_decode_attachment(item) for item in attachments_value)
    if sum(len(item.payload) for item in attachments) > _MAX_TOTAL_ATTACHMENT_BYTES:
        raise ServiceError("attachment_too_large")
    return GenerateRequest(
        prompt=prompt.strip(), attachments=attachments, max_new_tokens=max_new_tokens
    )


def _to_model_content(attachments: Sequence[MediaAttachment], prompt: str) -> list[object]:
    """Decode media in memory only; no filename or original path is accepted."""

    content: list[object] = []
    for attachment in attachments:
        try:
            if attachment.media_type in _IMAGE_TYPES:
                from PIL import Image

                content.append(Image.open(io.BytesIO(attachment.payload)).convert("RGB"))
            else:
                import librosa

                signal, _ = librosa.load(io.BytesIO(attachment.payload), sr=16_000, mono=True)
                content.append(signal)
        except Exception as exc:
            raise ServiceError("media_decode_failed") from exc
    content.append(prompt)
    return content


class MiniCPMService:
    """A loaded Base model with request-scoped inference only."""

    def __init__(self, model: ChatModel) -> None:
        self._model = model

    def generate(self, value: object) -> dict[str, object]:
        request = parse_generate_request(value)
        try:
            response = self._model.chat(
                msgs=[
                    {
                        "role": "user",
                        "content": _to_model_content(request.attachments, request.prompt),
                    }
                ],
                use_tts_template=False,
                enable_thinking=False,
                stream=False,
                max_new_tokens=request.max_new_tokens,
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError("inference_failed") from exc
        if type(response) is not str or not response.strip():
            raise ServiceError("invalid_model_response")
        return {
            "status": "ok",
            "content": response.strip(),
            "model": "openbmb/MiniCPM-o-4_5",
            "retention": "request_scoped",
        }


def build_torchnpu_service(*, model_cache: str, device_index: int = 0) -> MiniCPMService:
    """Load the pinned Base model from a pre-existing cloud cache only."""

    try:
        import torch
        import torch_npu  # noqa: F401
        from transformers import AutoModel
    except Exception as exc:
        raise ServiceError("backend_dependency_unavailable") from exc
    try:
        torch.npu.set_device(device_index)
        model = AutoModel.from_pretrained(
            model_cache,
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
            init_vision=True,
            init_audio=True,
            init_tts=False,
        ).eval().to(f"npu:{device_index}")
    except Exception as exc:
        raise ServiceError("model_load_failed") from exc
    return MiniCPMService(model)


def create_app(service: MiniCPMService) -> Any:
    """Create an optional FastAPI adapter without enabling request logging."""

    try:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
    except Exception as exc:
        raise ServiceError("http_dependency_unavailable") from exc

    app = FastAPI(title="GradLoop MiniCPM-o Service", docs_url=None, redoc_url=None)
    route_prefix = chr(47)

    @app.get(route_prefix + "health")
    async def health() -> dict[str, object]:
        return {
            "status": "ready",
            "model": "openbmb/MiniCPM-o-4_5",
            "modalities": ["text", "image", "audio"],
            "retention": "request_scoped",
        }

    @app.post(route_prefix + "v1" + route_prefix + "omni" + route_prefix + "generate")
    async def generate(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            return JSONResponse(service.generate(body))
        except ServiceError as exc:
            return JSONResponse({"error": str(exc)}, status_code=422)
        except Exception:
            return JSONResponse({"error": "invalid_request"}, status_code=422)

    return app
