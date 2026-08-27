from __future__ import annotations

import os
from dataclasses import dataclass

import httpx
from fastapi import HTTPException

SITEVERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"


def _csv_env(name: str, default: str) -> set[str]:
    return {value.strip().lower() for value in os.getenv(name, default).split(",") if value.strip()}


@dataclass(frozen=True)
class TurnstileSettings:
    secret: str
    allowed_hostnames: set[str]
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> "TurnstileSettings":
        return cls(
            secret=os.getenv("TURNSTILE_SECRET", "").strip(),
            allowed_hostnames=_csv_env(
                "TURNSTILE_ALLOWED_HOSTNAMES",
                "bixprint.mx,bixstudio-builder.pages.dev,"
                "manuelsmxjna-crypto.github.io,localhost,127.0.0.1",
            ),
            timeout_seconds=float(os.getenv("TURNSTILE_TIMEOUT_SECONDS", "10")),
        )


class TurnstileVerifier:
    def __init__(self, settings: TurnstileSettings | None = None):
        self.settings = settings or TurnstileSettings.from_env()

    @property
    def configured(self) -> bool:
        return bool(self.settings.secret)

    async def verify(self, token: str, expected_action: str) -> None:
        if not self.configured:
            raise HTTPException(status_code=503, detail="Turnstile no está configurado en el servidor.")
        if not token or len(token) > 2048:
            raise HTTPException(status_code=403, detail="Verificación de seguridad requerida.")

        try:
            async with httpx.AsyncClient(timeout=self.settings.timeout_seconds) as client:
                response = await client.post(
                    SITEVERIFY_URL,
                    data={"secret": self.settings.secret, "response": token},
                )
                response.raise_for_status()
                result = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise HTTPException(
                status_code=502,
                detail="No fue posible validar la verificación de seguridad.",
            ) from exc

        hostname = str(result.get("hostname") or "").lower()
        action = str(result.get("action") or "")
        if result.get("success") is not True:
            raise HTTPException(status_code=403, detail="Verificación de seguridad inválida o vencida.")
        if action != expected_action:
            raise HTTPException(status_code=403, detail="La verificación no corresponde a esta operación.")
        if hostname not in self.settings.allowed_hostnames:
            raise HTTPException(status_code=403, detail="Origen no autorizado para BixAI.")
