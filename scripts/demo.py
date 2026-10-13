"""One DeepPCB test pair through the pipeline -> docs/demo.png (OpenCV only + the committed ONNX model).

Panels: misregistered test image | overlay before registration | overlay after | detections (green =
correct, red = wrong class or false alarm, yellow = missed ground truth). Overlays: template in magenta,
test in green, so aligned copper shows dark and any mismatch shows a coloured fringe.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from refdiff import candidates as C
from refdiff import classify, data, metrics, pipeline, synth
from refdiff import register as R


def overlay(test, temp):
    return cv2.merge([temp, test, temp])  # BGR: template -> B,R (magenta), test -> G


def label(img, text):
    out = cv2.copyMakeBorder(img, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(out, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=6, help="test-split index")
    ap.add_argument("--out", default="docs/demo.png")
    args = ap.parse_args()
    p = data.split("test")[args.index]
    test, temp = p.load()
    gt = p.boxes()
    mis, _ = synth.misregister(test, np.random.default_rng(args.index))
    r = R.register(temp, mis, "guarded")
    aligned = R.apply(mis, r.M, temp.shape)
    cands = C.candidates(aligned, temp)
    dets = classify.Classifier().detect(cands, pipeline.crop_stack(aligned, temp, cands))
    dets = dets[dets[:, 4] >= 0.5]

    vis = cv2.cvtColor(aligned, cv2.COLOR_GRAY2BGR)
    matched = np.zeros(len(gt), bool)
    for d in dets:
        I = metrics.iou(d[None, :4], gt[:, :4])[0] if len(gt) else np.zeros(0)
        ok = np.where((I > metrics.IOU_THR) & (gt[:, 4] == d[5]) & ~matched)[0]
        if len(ok):
            matched[ok[0]] = True
        color = (0, 170, 0) if len(ok) else (0, 0, 220)
        x1, y1, x2, y2 = d[:4].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"{data.CLASSES[int(d[5])]} {d[4]:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    for g, m in zip(gt, matched):
        if not m:
            cv2.rectangle(vis, tuple(g[:2]), tuple(g[2:4]), (0, 200, 230), 2)

    th = np.degrees(np.arctan2(r.M[1, 0], r.M[0, 0]))
    panels = [label(cv2.cvtColor(mis, cv2.COLOR_GRAY2BGR), f"test, misregistered ({p.id})"),
              label(overlay(mis, temp), "overlay, before"),
              label(overlay(aligned, temp), f"after: rot {th:+.2f} deg, NCC {r.ncc:.3f}"),
              label(vis, f"{len(dets)} detections, {int(matched.sum())}/{len(gt)} GT found")]
    row = lambda a, b: np.hstack([a, np.full((a.shape[0], 8, 3), 255, np.uint8), b])
    grid = np.vstack([row(panels[0], panels[1]), np.full((8, 2 * 640 + 8, 3), 255, np.uint8),
                      row(panels[2], panels[3])])
    Path(args.out).parent.mkdir(exist_ok=True)
    cv2.imwrite(args.out, grid)
    print(f"wrote {args.out}: {len(dets)} detections, {int(matched.sum())}/{len(gt)} ground-truth defects found")


if __name__ == "__main__":
    main()
