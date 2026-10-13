"""Reference diff -> candidate defect regions.

Tolerant signed XOR (the classic AOI rule): a pixel is *extra* if the test has copper there and the
template has none within `tol` px, *missing* if the template has copper there and the test has none
within `tol` px. The tolerance absorbs the sub-pixel residual and edge quantization that registration
cannot remove; morphology cleans the rest; nearby blobs of the same sign are merged into one box.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Params:
    tol: int = 0          # XOR tolerance radius (px)
    open_k: int = 3       # opening kernel; removes 1-px slivers
    min_area: int = 12    # px; smaller blobs are dropped
    merge: int = 7        # dilation used to merge blobs of one sign into one candidate
    pad: int = 8          # box padding, fitted on trainval to the GT box style
    min_side: int = 30    # grow boxes to at least this side (GT boxes are rarely under ~27 px)
    border: int = 4       # ignore this many px at the image edge (warp fill)


def binarize(img: np.ndarray) -> np.ndarray:
    """Copper mask. DeepPCB copper is dark on a light background."""
    g = cv2.GaussianBlur(img, (3, 3), 0)
    _, b = cv2.threshold(g, 0, 1, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return b.astype(np.uint8)


def signed_diff(test: np.ndarray, temp: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    bt, br = binarize(test), binarize(temp)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.tol + 1, 2 * p.tol + 1))
    extra = bt & (1 - cv2.dilate(br, k))
    missing = br & (1 - cv2.dilate(bt, k))
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (p.open_k, p.open_k))
    extra = cv2.morphologyEx(extra, cv2.MORPH_OPEN, ko)
    missing = cv2.morphologyEx(missing, cv2.MORPH_OPEN, ko)
    if p.border:
        for m in (extra, missing):
            m[: p.border] = m[-p.border:] = 0
            m[:, : p.border] = m[:, -p.border:] = 0
    return extra, missing


def boxes_from_mask(mask: np.ndarray, p: Params) -> list[tuple[int, int, int, int, int]]:
    """(x1, y1, x2, y2, area) per merged blob."""
    h, w = mask.shape
    km = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (p.merge, p.merge))
    merged = cv2.dilate(mask, km) if p.merge > 1 else mask
    n, lab, _, _ = cv2.connectedComponentsWithStats(merged, connectivity=8)
    out = []
    for i in range(1, n):
        area = int(mask[lab == i].sum())
        if area < p.min_area:
            continue
        ys, xs = np.nonzero((lab == i) & (mask > 0))
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
        x1, y1, x2, y2 = x1 - p.pad, y1 - p.pad, x2 + p.pad, y2 + p.pad
        if x2 - x1 < p.min_side:
            cx = (x1 + x2) / 2
            x1, x2 = round(cx - p.min_side / 2), round(cx + p.min_side / 2)
        if y2 - y1 < p.min_side:
            cy = (y1 + y2) / 2
            y1, y2 = round(cy - p.min_side / 2), round(cy + p.min_side / 2)
        out.append((max(0, x1), max(0, y1), min(w, x2), min(h, y2), area))
    return out


def candidates(test: np.ndarray, temp: np.ndarray, p: Params | None = None) -> np.ndarray:
    """(K, 6) int: x1, y1, x2, y2, sign (+1 extra copper / -1 missing copper), area."""
    p = p or Params()
    extra, missing = signed_diff(test, temp, p)
    rows = [(*b[:4], 1, b[4]) for b in boxes_from_mask(extra, p)]
    rows += [(*b[:4], -1, b[4]) for b in boxes_from_mask(missing, p)]
    return np.array(rows, dtype=np.int64).reshape(-1, 6)
