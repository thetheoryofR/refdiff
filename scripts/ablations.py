"""What each part of refdiff's classifier input and training data buys, on the 500-image test split.

Each variant is the same CandNet, trained by scripts/train_classifier.py (seed 0) with something taken
away, and scored on the same as-shipped test candidates with the same thing taken away:
  full              test, template and signed-diff crops + side features (sign, sizes), as shipped
  no_template       template channel flat grey
  no_diff           signed-diff channel flat grey
  test_only         test crop alone (no template, no diff): the classifier is template-blind, but the
                    candidates still come from the diff
  no_side           side features zeroed
  no_warp_training  trained on as-shipped pairs only, without the warped-and-registered copies
Paired image-level bootstrap of full minus each variant. Trains any variant whose checkpoint is missing.
Writes results/ablations.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from refdiff import classify, data, metrics, pipeline

sys.path.insert(0, str(Path(__file__).parent))
from train_classifier import ablate

VARIANTS = {
    "full": ("", "native,warp"),
    "no_template": ("template", "native,warp"),
    "no_diff": ("diff", "native,warp"),
    "test_only": ("template,diff", "native,warp"),
    "no_side": ("side", "native,warp"),
    "no_warp_training": ("", "native"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    models = data.out_root() / "models"
    with np.load(data.out_root() / "cands" / "test_native.npz") as f:
        z = {k: f[k] for k in f.files}
    gts = {p.id: p.boxes() for p in data.split("test")}
    dets = {}
    res = {}
    for name, (drop, scen) in VARIANTS.items():
        ckpt = models / ("candnet_s0.pt" if name == "full" else f"candnet_ab_{name}.pt")
        if not ckpt.exists():
            subprocess.run([sys.executable, "-u", "scripts/train_classifier.py", "--seed", "0", "--scenarios", scen,
                            "--drop", drop, "--tag", f"ab_{name}"], check=True)
        clf = classify.Classifier(ckpt)
        crops, side = ablate(z["crops"], pipeline.box_features(z["cands"]), [d for d in drop.split(",") if d])
        d = {}
        for k, pid in enumerate(z["ids"]):
            sel = z["img"] == k
            c = z["cands"][sel]
            if not len(c):
                d[str(pid)] = np.zeros((0, 6))
                continue
            prob = classify.softmax(clf.logits(crops[sel], side[sel]).astype(np.float64))[:, :6]
            cls = prob.argmax(1)
            d[str(pid)] = classify.nms(np.concatenate(
                [c[:, :4].astype(float), prob[np.arange(len(cls)), cls][:, None], cls[:, None]], 1))
        dets[name] = d
        ev = metrics.evaluate(gts, d, len(data.CLASSES))
        res[name] = {"drop": drop, "train_scenarios": scen, **{k: ev[k] for k in ("mAP", "f1", "precision", "recall")},
                     "per_class_ap": [c["ap"] for c in ev["per_class"]]}
        if name != "full":
            res[name]["full_minus"] = metrics.paired_bootstrap(gts, dets["full"], d, args.n_boot)
        print(f"{name:17s} mAP {ev['mAP']:.4f}  F1 {ev['f1']:.4f}")
    Path("results/ablations.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
