"""Candidate classifier at inference: the committed ONNX model (models/candnet.onnx, no torch needed) or
a torch checkpoint. Turns candidates + crops into scored detections."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import pipeline
from .data import ROOT

ONNX_PATH = ROOT / "models" / "candnet.onnx"


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Classifier:
    def __init__(self, path: str | Path | None = None):
        path = Path(path or ONNX_PATH)
        if path.suffix == ".onnx":
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            self._torch = None
        else:
            import torch

            from .model import CandNet
            self._dev = "mps" if torch.backends.mps.is_available() else "cpu"
            self._torch = CandNet().to(self._dev).eval()
            self._torch.load_state_dict(torch.load(path, map_location=self._dev))

    def logits(self, crops: np.ndarray, side: np.ndarray) -> np.ndarray:
        if self._torch is not None:
            import torch
            with torch.no_grad():
                return self._torch(torch.from_numpy(crops).to(self._dev),
                                   torch.from_numpy(side).to(self._dev)).cpu().numpy()
        return self._sess.run(None, {"crops": crops, "side": side})[0]

    def detect(self, cands: np.ndarray, crops: np.ndarray) -> np.ndarray:
        """(K, 6) detections x1, y1, x2, y2, score, cls: the most probable defect class per candidate,
        scored by its probability, then class-wise NMS at IoU 0.5."""
        if not len(cands):
            return np.zeros((0, 6))
        prob = softmax(self.logits(crops, pipeline.box_features(cands)).astype(np.float64))[:, :6]
        cls = prob.argmax(1)
        d = np.concatenate([cands[:, :4].astype(float), prob[np.arange(len(cls)), cls][:, None], cls[:, None]], 1)
        return nms(d)


def nms(d: np.ndarray, thr: float = 0.5) -> np.ndarray:
    keep = []
    for c in np.unique(d[:, 5]):
        dc = d[d[:, 5] == c]
        boxes = dc[:, :4].copy()
        boxes[:, 2:] -= boxes[:, :2]
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), dc[:, 4].tolist(), 0.0, thr)
        keep.append(dc[np.array(idx).reshape(-1)])
    return np.concatenate(keep) if keep else d
