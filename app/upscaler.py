from __future__ import annotations

import os
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.getenv("UP_MODEL_PATH", "models/realesr-anime-x4.onnx")
TILE = int(os.getenv("UP_TILE", "128"))
PAD = int(os.getenv("UP_PAD", "8"))
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

    def run(self, image: Image.Image, alpha_mode: str = "bilinear") -> Image.Image:
        self._ensure_session()
        rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        h, w, _ = rgba.shape
        pad = max(0, min(TILE // 2, PAD))
        step = max(1, TILE - pad * 2)
        rgb_out = np.zeros((h * SCALE, w * SCALE, 3), dtype=np.uint8)
        for y in range(0, h, step):
            for x in range(0, w, step):
                tile_out = self._infer_tile(self._tile_to_chw(rgba, x - pad, y - pad, TILE))
                if tile_out.shape[0] // TILE != SCALE:
                    raise RuntimeError("El modelo devolvió una escala inesperada.")
                sx0, sy0 = (pad if x > 0 else 0), (pad if y > 0 else 0)
                sx1 = TILE - pad if x + step < w else min(TILE, pad + (w - x))
                sy1 = TILE - pad if y + step < h else min(TILE, pad + (h - y))
                crop = tile_out[sy0*SCALE:sy1*SCALE, sx0*SCALE:sx1*SCALE]
                dx0, dy0 = x * SCALE, y * SCALE
                dx1, dy1 = min(w*SCALE, dx0 + crop.shape[1]), min(h*SCALE, dy0 + crop.shape[0])
                rgb_out[dy0:dy1, dx0:dx1] = crop[:dy1-dy0, :dx1-dx0]
        alpha = Image.fromarray(rgba[..., 3], mode="L").resize((w*SCALE, h*SCALE), Image.Resampling.BILINEAR)
        if alpha_mode == "opaque":
            alpha = Image.new("L", (w*SCALE, h*SCALE), 255)
        elif alpha_mode == "binary":
            alpha = Image.fromarray(np.where(np.asarray(alpha) >= 128, 255, 0).astype(np.uint8), mode="L")
        result = Image.fromarray(rgb_out, mode="RGB").convert("RGBA")
        result.putalpha(alpha)
        return result

