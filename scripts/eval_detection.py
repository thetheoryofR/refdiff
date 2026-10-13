"""Score the reference-diff detector on the 500-image DeepPCB test split.

Scenarios (see refdiff.pipeline.prepare):
  native        as-shipped pairs, guarded registration
  native_noreg  as-shipped pairs, no registration
  warp          test image misregistered (rot +-3 deg, shift +-20 px, blur/gain/noise), registered back
  warp_noreg    same misregistration, no registration

95% intervals: image-level bootstrap (1000 resamples). Seed spread: the same eval for every trained
classifier seed (models/candnet_s{seed}.pt), main numbers from seed 0.
Writes results/detection.json and the per-image detections to <out_root>/dets/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from refdiff import candidates as C
from refdiff import classify, data, metrics, pipeline
from refdiff import register as R

SCENARIOS = ["native", "native_noreg", "warp", "warp_noreg"]


def load_cands(sc: str) -> dict:
    with np.load(data.out_root() / "cands" / f"test_{sc}.npz") as f:
        return {k: f[k] for k in f.files}  # one decompression; NpzFile re-reads on every access


def run(clf: classify.Classifier, z: dict) -> dict[str, np.ndarray]:
    return {str(pid): clf.detect(z["cands"][z["img"] == k], z["crops"][z["img"] == k])
            for k, pid in enumerate(z["ids"])}


def bootstrap(gts: dict, dets: dict, n: int = 1000, seed: int = 0) -> dict:
    ids = sorted(gts)
    rng = np.random.default_rng(seed)
    vals = {"mAP": [], "f1": []}
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        g = {f"{i}_{j}": gts[ids[i]] for j, i in enumerate(pick)}
        d = {f"{i}_{j}": dets[ids[i]] for j, i in enumerate(pick)}
        ev = metrics.evaluate(g, d, len(data.CLASSES))
        vals["mAP"].append(ev["mAP"])
        vals["f1"].append(ev["f1"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    models = data.out_root() / "models"
    seeds = [int(s) for s in args.seeds.split(",") if (models / f"candnet_s{s}.pt").exists()]

    pairs = data.split("test")
    gts = {p.id: p.boxes() for p in pairs}
    res = {"n_images": len(pairs), "iou_thr": metrics.IOU_THR, "score_thr": 0.5, "seeds": seeds, "scenarios": {}}
    clfs = {s: classify.Classifier(models / f"candnet_s{s}.pt") for s in seeds}
    (data.out_root() / "dets").mkdir(exist_ok=True)
    for sc in args.scenarios.split(","):
        z = load_cands(sc)
        per_seed = {}
        for s, clf in clfs.items():
            dets = run(clf, z)
            per_seed[s] = (metrics.evaluate(gts, dets, len(data.CLASSES)), dets)
        ev, dets = per_seed[seeds[0]]
        np.savez_compressed(data.out_root() / "dets" / f"refdiff_{sc}.npz", **dets)
        hit = n = 0  # candidate-stage ceiling: GT boxes some candidate covers at IoU > 0.33
        for k, pid in enumerate(z["ids"]):
            g, c = gts[str(pid)], z["cands"][z["img"] == k]
            n += len(g)
            if len(c) and len(g):
                hit += int((metrics.iou(g[:, :4], c[:, :4]).max(1) > metrics.IOU_THR).sum())
        ev["candidate_recall"] = hit / n
        ev["candidates_per_image"] = float(len(z["cands"]) / len(pairs))
        # how well the diff sign alone predicts the class family, on candidates matched to a GT box
        lab, sign = z["labels"], z["cands"][:, 4]
        extra_cls = np.isin(lab, [1, 3, 4])      # short, spur, copper
        missing_cls = np.isin(lab, [0, 2, 5])    # open, mousebite, pin-hole
        ev["sign_agreement"] = {"extra_classes": [int((sign[extra_cls] > 0).sum()), int(extra_cls.sum())],
                                "missing_classes": [int((sign[missing_cls] < 0).sum()), int(missing_cls.sum())]}
        ev["ci95"] = bootstrap(gts, dets, args.n_boot)
        ev["seed_mAP"] = {str(s): v[0]["mAP"] for s, v in per_seed.items()}
        ev["seed_f1"] = {str(s): v[0]["f1"] for s, v in per_seed.items()}
        res["scenarios"][sc] = ev
        print(f"{sc:13s} mAP {ev['mAP']:.4f} {ev['ci95']['mAP']}  F1 {ev['f1']:.4f}  cand recall {hit / n:.4f}  "
              f"seeds {[round(v, 4) for v in ev['seed_mAP'].values()]}")

    # end-to-end latency, native scenario: register + diff + crops (CPU) + classify (committed ONNX, CPU)
    clf = classify.Classifier()
    t = {"register": 0.0, "candidates": 0.0, "classify": 0.0}
    for p in pairs[:100]:
        test, temp = p.load()
        t0 = time.perf_counter()
        aligned = R.apply(test, R.register(temp, test, "guarded").M, temp.shape)
        t1 = time.perf_counter()
        cands = C.candidates(aligned, temp)
        crops = pipeline.crop_stack(aligned, temp, cands)
        t2 = time.perf_counter()
        clf.detect(cands, crops)
        t3 = time.perf_counter()
        t["register"] += t1 - t0
        t["candidates"] += t2 - t1
        t["classify"] += t3 - t2
    res["ms_per_image"] = {k: round(10 * v, 1) for k, v in t.items()}
    res["ms_per_image"]["total"] = round(sum(res["ms_per_image"].values()), 1)
    print("ms/image", res["ms_per_image"])
    Path("results/detection.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
