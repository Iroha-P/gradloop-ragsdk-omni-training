from __future__ import annotations

import base64

import pytest

from gradloop_npu.service import MiniCPMService, ServiceError, parse_generate_request


class _FakeModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> object:
        self.calls.append(kwargs)
        return "Grounded coaching response"


def _request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "prompt": "Summarize the synthetic signal.",
        "attachments": [],
        "max_new_tokens": 32,
        "content_policy": "public_or_synthetic",
    }
    request.update(overrides)
    return request


def test_generate_returns_only_request_scoped_contract() -> None:
    model = _FakeModel()
    result = MiniCPMService(model).generate(_request())

    assert result == {
        "status": "ok",
        "content": "Grounded coaching response",
        "model": "openbmb/MiniCPM-o-4_5",
        "retention": "request_scoped",
    }
    assert model.calls[0]["stream"] is False
    assert model.calls[0]["enable_thinking"] is False


@pytest.mark.parametrize(
    "payload, reason",
    [
        (_request(content_policy="private"), "content_policy_required"),
        (_request(extra="unexpected"), "invalid_request"),
        (_request(max_new_tokens=257), "invalid_generation_limit"),
        (
            _request(
                attachments=[
                    {"media_type": "video/mp4", "content_base64": "YQ=="}
                ]
            ),
            "unsupported_media_type",
        ),
        (
            _request(
                attachments=[
                    {"media_type": "image/png", "content_base64": "not-base64"}
                ]
            ),
            "invalid_attachment",
        ),
    ],
)
def test_request_validation_fails_closed(payload: dict[str, object], reason: str) -> None:
    with pytest.raises(ServiceError, match=reason):
        parse_generate_request(payload)


def test_attachment_never_accepts_a_filename_or_path() -> None:
    encoded = base64.b64encode(b"synthetic-bytes").decode("ascii")
    with pytest.raises(ServiceError, match="invalid_attachment"):
        parse_generate_request(
            _request(
                attachments=[
                    {
                        "media_type": "image/png",
                        "content_base64": encoded,
                        "filename": "private.png",
                    }
                ]
            )
        )
