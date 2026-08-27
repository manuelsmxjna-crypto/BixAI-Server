from __future__ import annotations

import os
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.getenv("UP_MODEL_PATH", "models/realesr-anime-x4.onnx")
TILE = int(os.getenv("UP_TILE", "256"))
PAD = int(os.getenv("UP_PAD", "24"))
SCALE = int(os.getenv("UP_SCALE", "4"))


def _providers() -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
        }), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class Upscaler:
    def __init__(self):
        self._session = None
        self._input_name = None
        self._output_name = None
        self._lock = threading.Lock()

    def _ensure_session(self):
        if self._session is not None:
            return
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"No existe {MODEL_PATH}.")
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=_providers())
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def status(self):
        try:
            self._ensure_session()
            return {"ready": True, "model": MODEL_PATH, "providers": self._session.get_providers(),
                    "input": self._input_name, "output": self._output_name,
                    "tile": TILE, "pad": PAD, "scale": SCALE}
        except Exception as exc:
            return {"ready": False, "error": str(exc), "model": MODEL_PATH}

    @staticmethod
    def _tile_to_chw(src: np.ndarray, tx: int, ty: int, size: int) -> np.ndarray:
        h, w, _ = src.shape
        tile = src[np.ix_(np.clip(np.arange(ty, ty + size), 0, h - 1),
                         np.clip(np.arange(tx, tx + size), 0, w - 1))]
        return np.ascontiguousarray(np.transpose(tile[..., :3].astype(np.float32) / 255.0, (2, 0, 1))[None, ...])

    def _infer_tile(self, tensor: np.ndarray) -> np.ndarray:
        with self._lock:
            out = self._session.run([self._output_name], {self._input_name: tensor})[0]
        return (np.clip(np.transpose(np.squeeze(out, axis=0), (1, 2, 0)), 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

    @staticmethod
    def _blend_tile(
        destination: np.ndarray,
        tile: np.ndarray,
        dx: int,
        dy: int,
        overlap: int,
        fade_left: bool,
        fade_top: bool,
    ) -> None:
        height, width, _ = tile.shape
        target = destination[dy:dy + height, dx:dx + width]
        alpha = np.ones((height, width), dtype=np.float32)

        if fade_left:
            blend_width = min(overlap, width)
            ramp = (np.arange(blend_width, dtype=np.float32) + 1.0) / (blend_width + 1.0)
            alpha[:, :blend_width] *= ramp[None, :]
        if fade_top:
            blend_height = min(overlap, height)
            ramp = (np.arange(blend_height, dtype=np.float32) + 1.0) / (blend_height + 1.0)
            alpha[:blend_height, :] *= ramp[:, None]

        mixed = target.astype(np.float32) * (1.0 - alpha[..., None])
        mixed += tile.astype(np.float32) * alpha[..., None]
        target[:] = np.clip(mixed + 0.5, 0, 255).astype(np.uint8)

    def run(self, image: Image.Image, alpha_mode: str = "bilinear") -> Image.Image:
        self._ensure_session()
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        h, w, _ = rgba.shape
        pad = max(0, min(TILE // 2, PAD))
        step = max(1, TILE - pad * 2)
        rgb_out = np.zeros((h * SCALE, w * SCALE, 3), dtype=np.uint8)
        for y in range(0, h, step):
            for x in range(0, w, step):
                source_x = x - pad
                source_y = y - pad
                tile_out = self._infer_tile(self._tile_to_chw(rgba, source_x, source_y, TILE))
                if tile_out.shape[0] // TILE != SCALE:
                    raise RuntimeError("El modelo devolvió una escala inesperada.")

                global_x0 = max(0, source_x)
                global_y0 = max(0, source_y)
                global_x1 = min(w, source_x + TILE)
                global_y1 = min(h, source_y + TILE)
                tile_x0 = (global_x0 - source_x) * SCALE
                tile_y0 = (global_y0 - source_y) * SCALE
                tile_x1 = tile_x0 + (global_x1 - global_x0) * SCALE
                tile_y1 = tile_y0 + (global_y1 - global_y0) * SCALE
                crop = tile_out[tile_y0:tile_y1, tile_x0:tile_x1]

                self._blend_tile(
                    rgb_out,
                    crop,
                    global_x0 * SCALE,
                    global_y0 * SCALE,
                    pad * 2 * SCALE,
                    fade_left=x > 0,
                    fade_top=y > 0,
                )
        alpha = Image.fromarray(rgba[..., 3], mode="L").resize((w*SCALE, h*SCALE), Image.Resampling.BILINEAR)
        if alpha_mode == "opaque":
            alpha = Image.new("L", (w*SCALE, h*SCALE), 255)
        elif alpha_mode == "binary":
            alpha = Image.fromarray(np.where(np.asarray(alpha) >= 128, 255, 0).astype(np.uint8), mode="L")
        result = Image.fromarray(rgb_out, mode="RGB").convert("RGBA")
        result.putalpha(alpha)
        return result
