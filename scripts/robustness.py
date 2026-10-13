"""Where refdiff breaks: mAP on the 500 test images as one degradation is turned up at a time, with the
committed ONNX classifier and guarded registration (and, for rotation, without registration).

  rotation  test image rotated by ±level degrees (random sign) about its centre plus a random shift of up
            to ±20 px, no optics
  blur      Gaussian blur of sigma = level px on the as-shipped test image
  noise     Gaussian noise of sd = level grey levels on the as-shipped test image
The training distribution covers rotation up to 3°, blur sigma up to ~1.2 and noise sd 6 (refdiff/synth.py).
Writes results/robustness.json and docs/robustness.png.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from refdiff import candidates as C
from refdiff import classify, data, metrics, pipeline
from refdiff import register as R

AXES = {"rotation": [0, 1, 2, 3, 5, 8, 12], "blur": [0, 1, 2, 3, 4], "noise": [0, 15, 30, 45, 60]}


def perturb(test: np.ndarray, axis: str, level: float, rng: np.random.Generator) -> np.ndarray:
    h, w = test.shape
    if axis == "rotation":
        th = level * rng.choice([-1, 1])
        M = R.euclid(th, *rng.uniform(-20, 20, 2), (w / 2, h / 2))
        return R.apply(test, M, test.shape)
    f = test.astype(np.float32)
    if axis == "blur" and level:
        f = cv2.GaussianBlur(f, (0, 0), level)
    if axis == "noise" and level:
        f += rng.normal(0, level, f.shape).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def one(job):
    k, pair, axis, level, register = job
    rng = np.random.default_rng([11, k])
    test, temp = pair.load()
    x = perturb(test, axis, level, rng)
    if register:
        x = R.apply(x, R.register(temp, x, "guarded").M, temp.shape)
    cands = C.candidates(x, temp)
    return pair.id, cands, pipeline.crop_stack(x, temp, cands)


def plot(res: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
    xlabels = {"rotation": "rotation (deg), plus ±20 px shift", "blur": "blur sigma (px)", "noise": "noise sd (grey levels)"}
    train = {"rotation": 3, "blur": 1.2, "noise": 6}
    for ax, axis in zip(axs, AXES):
        for reg, style in (("guarded", "-o"), ("none", "--o")):
            pts = res[axis].get(reg)
            if pts:
                ax.plot(AXES[axis], [100 * p["mAP"] for p in pts], style, label=f"registration: {reg}")
        ax.axvline(train[axis], color="grey", lw=0.8, ls=":")
        ax.set_xlabel(xlabels[axis])
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("mAP (%)")
    axs[0].legend(fontsize=8)
    fig.suptitle("refdiff on the DeepPCB test split as one degradation grows (dotted: edge of the training range)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    pairs = data.split("test")
    gts = {p.id: p.boxes() for p in pairs}
    clf = classify.Classifier()
    res = {"levels": AXES}
    with ProcessPoolExecutor(args.workers) as ex:
        for axis, levels in AXES.items():
            res[axis] = {}
            for reg in ([True, False] if axis == "rotation" else [True]):
                pts = []
                for lv in levels:
                    dets = {}
                    for pid, c, cr in ex.map(one, [(k, p, axis, lv, reg) for k, p in enumerate(pairs)]):
                        dets[pid] = clf.detect(c, cr)
                    ev = metrics.evaluate(gts, dets, len(data.CLASSES))
                    pts.append({k: ev[k] for k in ("mAP", "f1", "precision", "recall")})
                    print(f"{axis:8s} {lv:>4} reg={reg!s:5s} mAP {ev['mAP']:.4f} F1 {ev['f1']:.4f}")
                res[axis]["guarded" if reg else "none"] = pts
    Path("results/robustness.json").write_text(json.dumps(res, indent=1) + "\n")
    plot(res, Path("docs/robustness.png"))


if __name__ == "__main__":
    main()
