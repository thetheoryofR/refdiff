"""Die-to-database in miniature: the reference is a rendered layout, not a golden image.

A random Manhattan layout (traces, pads, vias) is built with gdstk and written as GDSII; it is
rasterized (anti-aliased) as the database image. The "optical" image is the same layout after etch
bias, a PSF blur, a known rotation + shift, an illumination gradient and sensor noise, with a few
defects (shorts, opens, spurious copper, pin holes) injected. Registration must bring the optical image back
onto the database, and the signed diff must find the injected defects.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import gdstk
import numpy as np

from . import register as R

SIZE_UM = 640.0      # die window side
PX_PER_UM = 1.0      # 640 x 640 px, same scale as a DeepPCB crop
SS = 4               # supersampling for the anti-aliased rasterization


def make_layout(rng: np.random.Generator, n_h: int = 9, n_v: int = 7) -> gdstk.Cell:
    cell = gdstk.Cell("DIE")
    width = rng.uniform(8, 14)
    ys = np.sort(rng.choice(np.arange(40, SIZE_UM - 40, 2 * width + 12), n_h, replace=False))
    xs = np.sort(rng.choice(np.arange(40, SIZE_UM - 40, 2 * width + 12), n_v, replace=False))
    for y in ys:  # horizontal traces, broken into segments
        x0 = rng.uniform(10, 120)
        while x0 < SIZE_UM - 60:
            x1 = min(SIZE_UM - 10, x0 + rng.uniform(80, 300))
            cell.add(gdstk.rectangle((x0, y - width / 2), (x1, y + width / 2), layer=1))
            x0 = x1 + rng.uniform(30, 120)
    for x in xs:  # vertical traces on the same layer, sparser
        if rng.random() < 0.6:
            y0 = rng.uniform(10, 200)
            y1 = min(SIZE_UM - 10, y0 + rng.uniform(100, 400))
            cell.add(gdstk.rectangle((x - width / 2, y0), (x + width / 2, y1), layer=1))
    for _ in range(int(rng.integers(8, 16))):  # pads with a via hole
        cx, cy = rng.uniform(40, SIZE_UM - 40, 2)
        s = rng.uniform(22, 36)
        cell.add(gdstk.rectangle((cx - s / 2, cy - s / 2), (cx + s / 2, cy + s / 2), layer=1))
        cell.add(gdstk.ellipse((cx, cy), s * 0.2, layer=2))  # drill: removes copper
    return cell


def write_gds(cell: gdstk.Cell, path: Path):
    lib = gdstk.Library(unit=1e-6, precision=1e-9)
    lib.add(cell)
    lib.write_gds(str(path))


def rasterize(cell: gdstk.Cell) -> np.ndarray:
    """uint8 image, copper dark (20) on a light background (230): DeepPCB polarity."""
    n = int(SIZE_UM * PX_PER_UM * SS)
    cu = np.zeros((n, n), np.uint8)
    holes = np.zeros((n, n), np.uint8)
    for poly in cell.polygons:
        pts = np.round(poly.points * PX_PER_UM * SS).astype(np.int32)
        pts[:, 1] = n - 1 - pts[:, 1]  # GDS y is up, image y is down
        cv2.fillPoly(cu if poly.layer == 1 else holes, [pts], 1)
    cu &= 1 - holes
    small = cv2.resize(cu.astype(np.float32), (n // SS, n // SS), interpolation=cv2.INTER_AREA)
    return (230 - 210 * small).astype(np.uint8)


DEFECTS = ["short", "open", "copper", "pin-hole"]


def inject(img: np.ndarray, rng: np.random.Generator, n: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Paint defects onto a (database-frame) image. Returns the image and (n, 5) boxes x1,y1,x2,y2,kind
    where kind indexes DEFECTS. Extra copper (short, copper) and missing copper (open, pin-hole).
    Each box is the bounding box of the copper that actually changed (a short painted over a trace
    only counts the bridge), which is what a diff can see."""
    out = img.copy()
    prev = out.copy()
    cu = img < 125
    boxes = []
    h, w = img.shape
    tries = 0
    while len(boxes) < n and tries < 500:
        tries += 1
        kind = int(rng.integers(0, 4))
        x, y = int(rng.integers(40, w - 40)), int(rng.integers(40, h - 40))
        s = int(rng.integers(8, 16))
        region = cu[y - s:y + s, x - s:x + s]
        if kind in (0, 2):     # extra copper: needs background around it
            if kind == 0 and not (0.15 < region.mean() < 0.6):   # a short bridges existing copper
                continue
            if kind == 2 and region.mean() > 0.02:              # spurious copper sits on bare board
                continue
            hh, ww = (s // 2, s) if rng.random() < 0.5 else (s, s // 2)
            cv2.rectangle(out, (x - ww, y - hh), (x + ww, y + hh), 20, -1)
            box = (x - ww, y - hh, x + ww, y + hh)
        else:                  # missing copper: needs copper under it
            if region.mean() < 0.6:
                continue
            r = s // 2 if kind == 3 else s // 2 + 2
            if kind == 1:
                cv2.rectangle(out, (x - r, y - s), (x + r, y + s), 230, -1)
                box = (x - r, y - s, x + r, y + s)
            else:
                cv2.circle(out, (x, y), r, 230, -1)
                box = (x - r, y - r, x + r, y + r)
        x1, y1, x2, y2 = (max(0, v) for v in box)
        changed = (out[y1:y2 + 1, x1:x2 + 1] < 125) != cu[y1:y2 + 1, x1:x2 + 1]
        if changed.sum() < 12:
            out = prev.copy()
            continue
        ys, xs = np.nonzero(changed)
        box = (x1 + xs.min(), y1 + ys.min(), x1 + xs.max() + 1, y1 + ys.max() + 1)
        if any(R_iou(box, b[:4]) > 0 for b in boxes):
            out = prev.copy()
            continue
        prev = out.copy()
        boxes.append((*box, kind))
    return out, np.array(boxes, np.int64).reshape(-1, 5)


def R_iou(a, b) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy


def optical(img: np.ndarray, rng: np.random.Generator, max_rot: float = 2.0, max_shift: float = 15.0):
    """Etch bias, PSF blur, rotation + shift, illumination gradient, noise. Returns (image, true warp
    optical -> database)."""
    h, w = img.shape
    cu = (img < 125).astype(np.uint8)
    bias = int(rng.integers(-1, 2))  # over/under-etch by a pixel
    if bias:
        k = np.ones((3, 3), np.uint8)
        cu = cv2.dilate(cu, k) if bias > 0 else cv2.erode(cu, k)
    f = (230 - 210 * cu).astype(np.float32)
    f = cv2.GaussianBlur(f, (0, 0), rng.uniform(1.0, 2.0))  # PSF: rounds corners, softens edges
    th = rng.uniform(-max_rot, max_rot)
    tx, ty = rng.uniform(-max_shift, max_shift, 2)
    Mt = R.euclid(th, tx, ty, (w / 2.0, h / 2.0))
    f = cv2.warpAffine(f, Mt, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=230)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = rng.uniform(-25, 25)
    f = f + g * (xx / w - 0.5) + rng.uniform(-10, 10) * (yy / h - 0.5)
    f = f + rng.normal(0, rng.uniform(3, 8), f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8), R.invert(Mt)
