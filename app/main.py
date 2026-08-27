from __future__ import annotations

import io
import os
import time
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError

from .background import BackgroundRemover
from .turnstile import TurnstileVerifier
from .upscaler import Upscaler

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "30"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "50000000"))
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}

app = FastAPI(title="BixAI Server", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://bixprint.mx"],
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

bg = BackgroundRemover()
up = Upscaler()
turnstile = TurnstileVerifier()


async def _read_upload(file: UploadFile) -> Image.Image:
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"Archivo demasiado grande. Máximo: {MAX_UPLOAD_MB} MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="El archivo no contiene una imagen válida.") from exc
    if image.format not in ALLOWED_IMAGE_FORMATS:
        raise HTTPException(status_code=415, detail="Formato no permitido. Usa PNG, JPG o WEBP.")
    if image.width < 1 or image.height < 1 or image.width * image.height > MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"Imagen demasiado grande. Máximo: {MAX_PIXELS:,} píxeles.")
    return image.convert("RGBA")


def _png_response(result: Image.Image, model: str, started: float) -> Response:
    output = io.BytesIO()
    result.save(output, format="PNG", optimize=False)
    return Response(
        content=output.getvalue(),
        media_type="image/png",
        headers={
            "X-BixAI-Model": model,
            "X-BixAI-Elapsed-Ms": str(round((time.perf_counter() - started) * 1000)),
        },
    )


@app.get("/")
def root():
    return {"service": "BixAI Server", "version": "0.2.0",
            "endpoints": ["/health", "/remove-background", "/upscale"]}


@app.get("/health")
def health():
    return {"ok": True, "turnstile": {"configured": turnstile.configured},
            "background": bg.status(), "upscaler": up.status()}


@app.post("/remove-background")
async def remove_background(
    image: UploadFile = File(...),
    turnstile_token: str = Form(..., alias="cf-turnstile-response"),
    alpha_mode: Literal["replace", "multiply"] = "multiply",
):
    await turnstile.verify(turnstile_token, "remove_background")
    source = await _read_upload(image)
    started = time.perf_counter()
    try:
        result = bg.run(source, alpha_mode=alpha_mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Background remover falló: {exc}") from exc
    return _png_response(result, "BiRefNet_lite_512", started)


@app.post("/upscale")
async def upscale(
    image: UploadFile = File(...),
    turnstile_token: str = Form(..., alias="cf-turnstile-response"),
    alpha_mode: Literal["bilinear", "binary", "opaque"] = "bilinear",
):
    await turnstile.verify(turnstile_token, "upscale")
    source = await _read_upload(image)
    started = time.perf_counter()
    try:
        result = up.run(source, alpha_mode=alpha_mode)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upscaler falló: {exc}") from exc
    return _png_response(result, "RealESRGAN_anime_x4", started)

