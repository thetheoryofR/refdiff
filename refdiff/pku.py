"""The PKU synthetic PCB defect set (Huang & Wei 2019; Kaggle mirror akhatova/pcb-defects) as a second,
never-trained-on test of the DeepPCB pipeline.

693 colour photos of 10 boards with painted-in defects, one defect-free template photo per board
(PCB_USED/), and the same 693 images rotated (rotation/, no annotations). The defect images are
aligned with their templates to within about a pixel (most exactly; all of board 10 is 0.5 px off, and
some board-01 images 1 px), so their annotations are in template coordinates.

To reuse the DeepPCB pipeline unchanged, images are turned into what it was trained on: binary copper
(dark) on background (light), at a scale where defects are DeepPCB-sized (scripts/pku_transfer.py).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .data import _paths

# PKU class -> DeepPCB class id. "missing hole" (copper left where a hole should be drilled) has no
# DeepPCB counterpart; it is mapped to pin-hole for the class-aware score and reported on its own.
CLASS_MAP = {"open_circuit": 0, "short": 1, "mouse_bite": 2, "spur": 3, "spurious_copper": 4, "missing_hole": 5}


def root() -> Path:
    return Path(os.environ.get("REFDIFF_PKU") or _paths()["pku"])


@dataclass(frozen=True)
class Sample:
    name: str        # e.g. 01_short_01
    board: str       # 01
    kind: str        # PKU class name
    image: Path
    rotated: Path
    template: Path
    ann: Path

    def boxes(self) -> np.ndarray:
        """(N, 5) x1, y1, x2, y2, DeepPCB class id, in template pixels."""
        rows = []
        for o in ET.parse(self.ann).getroot().iter("object"):
            bb = o.find("bndbox")
            x1, y1, x2, y2 = (int(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
            rows.append((x1, y1, x2, y2, CLASS_MAP[o.find("name").text.strip().lower()]))
        return np.array(rows, np.int64).reshape(-1, 5)


def samples(r: Path | None = None) -> list[Sample]:
    r = r or root()
    out = []
    for ann in sorted((r / "Annotations").glob("*/[0-9]*.xml")):
        d = ann.parent.name                      # e.g. Mouse_bite
        out.append(Sample(ann.stem, ann.stem[:2], d.lower(), r / "images" / d / f"{ann.stem}.jpg",
                          r / "rotation" / f"{d}_rotation" / f"{ann.stem}.jpg", r / "PCB_USED" / f"{ann.stem[:2]}.JPG",
                          ann))
    return out


def binarize(bgr: np.ndarray) -> np.ndarray:
    """Copper mask from a colour photo: copper under solder mask is lighter green than bare substrate, so
    a local mean threshold on the green channel. Returned as uint8, 1 = copper."""
    g = cv2.GaussianBlur(bgr[:, :, 1], (0, 0), 1.5)
    return (cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 61, -6) > 0).astype(np.uint8)


def to_deeppcb(mask: np.ndarray, scale: float) -> np.ndarray:
    """Copper mask -> DeepPCB-style grey image (copper 0, background 255) at `scale`."""
    img = ((1 - mask) * 255).astype(np.uint8)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return np.where(img < 128, 0, 255).astype(np.uint8)


def fit_canvas(img: np.ndarray, shape: tuple[int, int], fill: int = 255) -> np.ndarray:
    """Centre-pad or centre-crop to `shape`. The rotated set rotates about the image centre on a larger
    canvas, so this keeps the rotation centre where the template's is."""
    h, w = shape
    H, W = img.shape
    out = np.full((h, w), fill, img.dtype)
    sy, sx = max(0, (H - h) // 2), max(0, (W - w) // 2)
    dy, dx = max(0, (h - H) // 2), max(0, (w - W) // 2)
    ch, cw = min(h, H), min(w, W)
    out[dy:dy + ch, dx:dx + cw] = img[sy:sy + ch, sx:sx + cw]
    return out
