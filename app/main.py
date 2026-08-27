from __future__ import annotations

import io
import logging
import os
import time
from collections.abc import Iterator
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image, UnidentifiedImageError

from .background import BackgroundRemover
from .turnstile import TurnstileVerifier
from .upscaler import Upscaler

MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "30"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "50000000"))
MAX_OUTPUT_PIXELS = int(os.getenv("MAX_OUTPUT_PIXELS", "50000000"))
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}
STREAM_CHUNK_BYTES = 1024 * 1024
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "https://bixprint.mx,https://bixstudio-builder.pages.dev,"
        "https://manuelsmxjna-crypto.github.io",
    ).split(",")
    if origin.strip()
]

app = FastAPI(title="BixAI Server", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)

bg = BackgroundRemover()
up = Upscaler()
turnstile = TurnstileVerifier()
logger = logging.getLogger("bixai")


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


def _stream_bytes(buffer: io.BytesIO) -> Iterator[bytes]:
    try:
        while chunk := buffer.read(STREAM_CHUNK_BYTES):
            yield chunk
    finally:
        buffer.close()


def _png_response(result: Image.Image, model: str, started: float) -> StreamingResponse:
    output = io.BytesIO()
    result.save(output, format="PNG", optimize=False)
    output_bytes = output.tell()
    output.seek(0)
    elapsed_ms = round((time.perf_counter() - started) * 1000)
    logger.info(
        "PNG generado model=%s width=%s height=%s bytes=%s elapsed_ms=%s",
        model,
        result.width,
        result.height,
        output_bytes,
        elapsed_ms,
    )
    return StreamingResponse(
        _stream_bytes(output),
        media_type="image/png",
        headers={
            "X-BixAI-Model": model,
            "X-BixAI-Elapsed-Ms": str(elapsed_ms),
            "X-BixAI-Output-Bytes": str(output_bytes),
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
    target_width_px: int | None = Query(default=None, ge=1, le=30000),
    target_height_px: int | None = Query(default=None, ge=1, le=30000),
):
    await turnstile.verify(turnstile_token, "upscale")
    if (target_width_px is None) != (target_height_px is None):
        raise HTTPException(status_code=400, detail="El tamaño objetivo requiere ancho y alto.")
    if target_width_px and target_height_px and target_width_px * target_height_px > MAX_OUTPUT_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"La salida objetivo supera {MAX_OUTPUT_PIXELS:,} píxeles.",
        )
    source = await _read_upload(image)
    started = time.perf_counter()
    try:
        target_size = (
            (target_width_px, target_height_px)
            if target_width_px is not None and target_height_px is not None
            else None
        )
        result = up.run(source, alpha_mode=alpha_mode, max_size=target_size)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Upscaler falló: {exc}") from exc
    return _png_response(result, "RealESRGAN_anime_x4", started)
