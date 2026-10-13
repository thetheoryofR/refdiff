"""End to end: (test, template) -> register -> candidates -> crops. Scenarios for building datasets and
for evaluation share this so train and test see exactly the same preprocessing."""
from __future__ import annotations

import cv2
import numpy as np

from . import candidates as C
from . import register as R
from . import synth

CROP = 64       # classifier input side
CONTEXT = 1.5   # crop = candidate box grown by this factor (square), then resized to CROP


def prepare(test: np.ndarray, temp: np.ndarray, scenario: str, rng: np.random.Generator | None,
            reg_method: str = "guarded") -> tuple[np.ndarray, np.ndarray]:
    """Returns the test image in the template frame (or not, for the *_noreg scenarios).

    scenario: 'native' | 'native_noreg' | 'warp' (test misregistered by a known warp + optics, then
    registered back) | 'warp_noreg'.
    """
    if scenario.startswith("warp"):
        test, _ = synth.misregister(test, rng)
    if scenario.endswith("noreg"):
        return test, np.eye(2, 3)
    M = R.register(temp, test, reg_method).M
    return R.apply(test, M, temp.shape), M


def crop_stack(test: np.ndarray, temp: np.ndarray, boxes: np.ndarray, p: C.Params | None = None) -> np.ndarray:
    """(K, 3, CROP, CROP) uint8: test, template, signed diff (128 = same, 255 = extra, 0 = missing)."""
    p = p or C.Params()
    extra, missing = C.signed_diff(test, temp, p)
    diff = (128 + 127 * extra.astype(np.int16) - 128 * missing.astype(np.int16)).astype(np.uint8)
    out = np.zeros((len(boxes), 3, CROP, CROP), np.uint8)
    sides = (np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]) * CONTEXT).astype(int)
    pad = int(max(CROP, sides.max(initial=0))) + 2
    planes = [cv2.copyMakeBorder(x, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=v)
              for x, v in ((test, 255), (temp, 255), (diff, 128))]
    for i, (x1, y1, x2, y2) in enumerate(boxes[:, :4]):
        side = max(int(sides[i]), 8)
        cx, cy = (x1 + x2) // 2 + pad, (y1 + y2) // 2 + pad
        sx, sy = cx - side // 2, cy - side // 2
        for j, pl in enumerate(planes):
            out[i, j] = cv2.resize(pl[sy:sy + side, sx:sx + side], (CROP, CROP), interpolation=cv2.INTER_AREA)
    return out


def box_features(boxes: np.ndarray) -> np.ndarray:
    """Scalar side-features per candidate: sign, log area, log width, log height (the crop is resized,
    so absolute size would otherwise be lost)."""
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([boxes[:, 4].astype(np.float32), np.log1p(boxes[:, 5]), np.log1p(w), np.log1p(h)], 1).astype(
        np.float32)
