from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.turnstile import TurnstileSettings, TurnstileVerifier


def verifier() -> TurnstileVerifier:
    return TurnstileVerifier(TurnstileSettings("secret", {"bixprint.mx"}, 1.0))


def mock_response(payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "https://example.test"))


@pytest.mark.asyncio
async def test_accepts_expected_hostname_and_action():
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response({
        "success": True, "hostname": "bixprint.mx", "action": "remove_background"
    }))):
        await verifier().verify("fresh-token", "remove_background")


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [
    {"success": False, "error-codes": ["timeout-or-duplicate"]},
    {"success": True, "hostname": "evil.example", "action": "remove_background"},
    {"success": True, "hostname": "bixprint.mx", "action": "upscale"},
])
async def test_rejects_invalid_result(payload):
    with patch("httpx.AsyncClient.post", new=AsyncMock(return_value=mock_response(payload))):
        with pytest.raises(HTTPException) as raised:
            await verifier().verify("token", "remove_background")
        assert raised.value.status_code == 403


@pytest.mark.asyncio
async def test_fails_closed_without_secret():
    instance = TurnstileVerifier(TurnstileSettings("", {"bixprint.mx"}, 1.0))
    with pytest.raises(HTTPException) as raised:
        await instance.verify("token", "remove_background")
    assert raised.value.status_code == 503

