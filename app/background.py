from __future__ import annotations

import os
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_PATH = os.getenv("BG_MODEL_PATH", "models/bg-remove.onnx")
MODEL_SIZE = int(os.getenv("BG_MODEL_SIZE", "512"))
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _providers() -> list:
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return [("CUDAExecutionProvider", {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": "HEURISTIC",
        }), "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


class BackgroundRemover:
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
        self._output_name = self._session.get_outputs()[-1].name

    def status(self):
        try:
            self._ensure_session()
            return {"ready": True, "model": MODEL_PATH, "providers": self._session.get_providers(),
                    "input": self._input_name, "output": self._output_name}
        except Exception as exc:
            return {"ready": False, "error": str(exc), "model": MODEL_PATH}

    @staticmethod
    def _preprocess(image: Image.Image) -> np.ndarray:
        rgb = image.convert("RGB").resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.BILINEAR)
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        return np.ascontiguousarray(np.transpose(arr, (2, 0, 1))[None, ...], dtype=np.float32)

    @staticmethod
    def _normalize_mask(raw: np.ndarray) -> np.ndarray:
        value = np.squeeze(np.asarray(raw, dtype=np.float32))
        if np.nanmin(value) < 0.0 or np.nanmax(value) > 1.0:
            value = 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))
        return np.clip(value, 0.0, 1.0)

    def run(self, image: Image.Image, alpha_mode: str = "multiply") -> Image.Image:
        self._ensure_session()
        original = image.convert("RGBA")
        with self._lock:
            outputs = self._session.run(None, {self._input_name: self._preprocess(original)})
        mask = self._normalize_mask(outputs[-1])
        matte = Image.fromarray((mask * 255.0).astype(np.uint8), mode="L").resize(
            original.size, Image.Resampling.BILINEAR
        )
        rgba = np.asarray(original, dtype=np.uint8).copy()
        matte_array = np.asarray(matte, dtype=np.uint8)
        if alpha_mode == "multiply":
            rgba[..., 3] = ((rgba[..., 3].astype(np.uint16) * matte_array.astype(np.uint16)) // 255).astype(np.uint8)
        else:
            rgba[..., 3] = matte_array
        return Image.fromarray(rgba, mode="RGBA")

