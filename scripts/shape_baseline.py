"""No-pixel baseline: the same diff-stage candidates as refdiff, classified from their geometry alone
(diff sign, area, width, height, aspect, fill) by gradient-boosted trees instead of the CNN on crops.

Same training data as the CNN (trainval candidates, as-shipped + warped-and-registered; the last 100
trainval images held out for early stopping), same NMS, same scorer. The gap to refdiff is what the CNN's
view of the pixels adds on top of the diff. Writes <out_root>/dets/shape_native.npz and
results/shape_baseline.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from refdiff import classify, data, metrics

FEATURES = ["sign", "log_area", "log_w", "log_h", "aspect", "fill"]


def features(c: np.ndarray) -> np.ndarray:
    """c: (K, 6) candidates x1, y1, x2, y2, sign, area (changed pixels)."""
    w = (c[:, 2] - c[:, 0]).astype(float)
    h = (c[:, 3] - c[:, 1]).astype(float)
    return np.stack([c[:, 4], np.log1p(c[:, 5]), np.log1p(w), np.log1p(h),
                     np.log(np.maximum(w, 1) / np.maximum(h, 1)), c[:, 5] / np.maximum(w * h, 1)], 1)


def load(split, sc):
    with np.load(data.out_root() / "cands" / f"{split}_{sc}.npz") as f:
        return {k: f[k] for k in ("ids", "cands", "labels", "img")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tr = [load("trainval", sc) for sc in ("native", "warp")]
    X = np.concatenate([features(z["cands"]) for z in tr])
    Y = np.concatenate([z["labels"] for z in tr])
    img = np.concatenate([z["img"] for z in tr])
    val = img >= img.max() + 1 - 100
    # model selection on the held-out trainval images, as for the CNN
    clf = HistGradientBoostingClassifier(learning_rate=0.05, early_stopping=False, random_state=args.seed)
    best, best_it = -1.0, 0
    for it in (50, 100, 200, 300, 500):
        clf.set_params(max_iter=it).fit(X[~val], Y[~val])
        acc = float((clf.predict(X[val]) == Y[val]).mean())
        if acc > best:
            best, best_it = acc, it
    clf.set_params(max_iter=best_it).fit(X[~val], Y[~val])
    print(f"val accuracy {best:.4f} at {best_it} iterations")

    te = load("test", "native")
    gts = {p.id: p.boxes() for p in data.split("test")}
    prob = clf.predict_proba(features(te["cands"]))[:, :len(data.CLASSES)]
    dets = {}
    for k, pid in enumerate(te["ids"]):
        sel = te["img"] == k
        c, p = te["cands"][sel], prob[sel]
        if not len(c):
            dets[str(pid)] = np.zeros((0, 6))
            continue
        cls = p.argmax(1)
        d = np.concatenate([c[:, :4].astype(float), p[np.arange(len(cls)), cls][:, None], cls[:, None]], 1)
        dets[str(pid)] = classify.nms(d)
    np.savez_compressed(data.out_root() / "dets" / "shape_native.npz", **dets)
    ev = metrics.evaluate(gts, dets, len(data.CLASSES))
    res = {"features": FEATURES, "val_accuracy": best, "iterations": best_it,
           **{k: ev[k] for k in ("mAP", "f1", "precision", "recall", "per_class")}}
    print(json.dumps({k: v for k, v in res.items() if k != "per_class"}, indent=1))
    Path("results/shape_baseline.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
