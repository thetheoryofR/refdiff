"""DeepPCB loading: template/test pairs, boxes, the official trainval/test split."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

CLASSES = ["open", "short", "mousebite", "spur", "copper", "pin-hole"]  # DeepPCB ids 1..6
ROOT = Path(__file__).resolve().parent.parent


def _paths() -> dict:
    with open(ROOT / "config" / "paths.yaml") as f:
        return yaml.safe_load(f)


def data_root() -> Path:
    return Path(os.environ.get("REFDIFF_DATA") or _paths()["deeppcb"])


def out_root() -> Path:
    p = Path(os.environ.get("REFDIFF_OUT") or _paths()["out_root"])
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Pair:
    id: str            # e.g. "20085291"
    test: Path
    temp: Path
    ann: Path

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        """(test, template) as uint8 grayscale."""
        return (cv2.imread(str(self.test), cv2.IMREAD_GRAYSCALE),
                cv2.imread(str(self.temp), cv2.IMREAD_GRAYSCALE))

    def boxes(self) -> np.ndarray:
        """(N, 5) int array: x1, y1, x2, y2, cls with cls in 0..5 (DeepPCB id - 1)."""
        return read_boxes(self.ann)


def read_boxes(path: Path) -> np.ndarray:
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) >= 5:
            x1, y1, x2, y2, c = map(int, parts[:5])
            rows.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), c - 1))
    return np.array(rows, dtype=np.int64).reshape(-1, 5)


def split(name: str, root: Path | None = None) -> list[Pair]:
    """name: 'trainval' (1000 pairs) or 'test' (500 pairs; the list file has 499 lines + 1 unterminated)."""
    root = root or data_root()
    pairs = []
    for line in (root / f"{name}.txt").read_text().splitlines():
        if not line.strip():
            continue
        img, ann = line.split()
        stem = Path(img).stem
        d = root / Path(img).parent
        pairs.append(Pair(stem, d / f"{stem}_test.jpg", d / f"{stem}_temp.jpg", root / ann))
    return pairs
