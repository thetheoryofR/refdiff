"""Synthetic misregistration with exact ground truth.

DeepPCB pairs are only approximately aligned (integer template matching), so a test image cannot
serve as ground truth for sub-pixel scoring. Instead: start from the template, paste in the real
defect patches from the test image (inside the annotated boxes), so the content is a real defective
board with zero residual misalignment; then warp it by a known rotation + shift and degrade it like an
optical image (blur, gain/offset, noise).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import register as R


def defective_template(test: np.ndarray, temp: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    out = temp.copy()
    for x1, y1, x2, y2, _ in boxes:
        out[y1:y2, x1:x2] = test[y1:y2, x1:x2]
    return out


def degrade(img: np.ndarray, rng: np.random.Generator, blur: float = 1.2, noise: float = 6.0) -> np.ndarray:
    f = img.astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.5, 1.0) * blur)
    f = f * rng.uniform(0.85, 1.1) + rng.uniform(-15, 15)
    f += rng.normal(0, noise, f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def misregister(img: np.ndarray, rng: np.random.Generator, max_rot: float = 3.0, max_shift: float = 20.0,
                optics: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Returns (warped image, true 2x3 warp mapping the warped image back to the template frame)."""
    h, w = img.shape
    th = rng.uniform(-max_rot, max_rot)
    tx, ty = rng.uniform(-max_shift, max_shift, 2)
    Mt = R.euclid(th, tx, ty, (w / 2.0, h / 2.0))
    out = R.apply(img, Mt, (h, w))
    if optics:
        out = degrade(out, rng)
    return out, R.invert(Mt)
