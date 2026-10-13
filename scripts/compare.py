"""Paired image-level bootstrap: refdiff (native, registered) vs every baseline whose detections exist,
and registration on vs off. Same 500 test images, same resample indices for both sides of each comparison.
Writes results/compare.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from refdiff import data, metrics


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# baselines that sit in out_root when their scripts have run: name -> (dets file, label, sees template)
BASELINES = {
    "yolov8n": ("yolo/test_dets_yolov8n.npz", "YOLOv8n", "no"),
    "yolov8s": ("yolo/test_dets_yolov8s.npz", "YOLOv8s", "no"),
    "yolov8n_ttd": ("yolo_ttd/test_dets_yolov8n_ttd.npz", "YOLOv8n on test + template + diff", "yes"),
    "shape": ("dets/shape_native.npz", "refdiff candidates, trees on shape only (no CNN)", "yes"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    out = data.out_root()
    gts = {p.id: p.boxes() for p in data.split("test")}
    ref = {sc: load(out / "dets" / f"refdiff_{sc}.npz") for sc in ("native", "native_noreg")}
    res = {"registration_on_vs_off_native": metrics.paired_bootstrap(gts, ref["native"], ref["native_noreg"], args.n_boot),
           "baselines": {}}
    for name, (rel, label, sees) in BASELINES.items():
        if not (out / rel).exists():
            continue
        d = load(out / rel)
        d = {k: d.get(k, np.zeros((0, 6))) for k in gts}
        ev = metrics.evaluate(gts, d, len(data.CLASSES))
        res["baselines"][name] = {"label": label, "sees_template": sees,
                                  **{k: ev[k] for k in ("mAP", "f1", "precision", "recall", "per_class")},
                                  "refdiff_minus": metrics.paired_bootstrap(gts, ref["native"], d, args.n_boot)}
        r = res["baselines"][name]["refdiff_minus"]
        print(f"{name:12s} mAP {ev['mAP']:.4f} F1 {ev['f1']:.4f}  refdiff - it: mAP {r['delta_mAP']:+.4f} "
              f"{np.round(r['delta_mAP_ci95'], 4)}  F1 {r['delta_f1']:+.4f}")
    Path("results/compare.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
