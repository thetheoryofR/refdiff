"""Sensitivity of the diff stage: XOR tolerance and minimum blob area, on the 500 test pairs (as shipped,
guarded registration), with the shipped classifier (trained at the defaults). Reports the candidate-stage
ceiling (GT boxes some candidate covers at IoU > 0.33), candidates per image and end-to-end mAP.
Registration runs once per pair; only the diff parameters change. Writes results/diff_sweep.json.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from refdiff import candidates as C
from refdiff import classify, data, metrics, pipeline
from refdiff import register as R

GRID = [{"tol": t, "min_area": a} for t in (0, 1, 2) for a in (6, 12, 24)]


def one(pair):
    test, temp = pair.load()
    aligned = R.apply(test, R.register(temp, test, "guarded").M, temp.shape)
    out = []
    for g in GRID:
        p = replace(C.Params(), **g)
        c = C.candidates(aligned, temp, p)
        out.append((c, pipeline.crop_stack(aligned, temp, c, p)))
    return pair.id, out


def main():
    pairs = data.split("test")
    gts = {p.id: p.boxes() for p in pairs}
    clf = classify.Classifier()
    per = [{"dets": {}, "hit": 0, "n": 0, "cands": 0} for _ in GRID]
    with ProcessPoolExecutor(8) as ex:
        for pid, outs in ex.map(one, pairs):
            g = gts[pid]
            for k, (c, crops) in enumerate(outs):
                per[k]["dets"][pid] = clf.detect(c, crops)
                per[k]["n"] += len(g)
                per[k]["cands"] += len(c)
                if len(c) and len(g):
                    per[k]["hit"] += int((metrics.iou(g[:, :4], c[:, :4]).max(1) > metrics.IOU_THR).sum())
    res = []
    for g, r in zip(GRID, per):
        ev = metrics.evaluate(gts, r["dets"], len(data.CLASSES))
        res.append({**g, "candidate_recall": r["hit"] / r["n"], "candidates_per_image": r["cands"] / len(pairs),
                    "mAP": ev["mAP"], "f1": ev["f1"]})
        print(res[-1])
    Path("results/diff_sweep.json").write_text(json.dumps({"defaults": {"tol": C.Params().tol,
                                                                        "min_area": C.Params().min_area},
                                                           "grid": res}, indent=1) + "\n")


if __name__ == "__main__":
    main()
