"""Run the diff stage over a split and cache candidates + crops + labels as one .npz per scenario.

Label = class of the GT box the candidate overlaps best (IoU > 0.33), else 6 = background.
Writes <out_root>/cands/<split>_<scenario>.npz.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from refdiff import candidates as C
from refdiff import data, metrics, pipeline

BG = len(data.CLASSES)


def one(args):
    i, pair, scenario, seed = args
    rng = np.random.default_rng([seed, i])
    test, temp = pair.load()
    aligned, _ = pipeline.prepare(test, temp, scenario, rng)
    cands = C.candidates(aligned, temp)
    gt = pair.boxes()
    labels = np.full(len(cands), BG, np.int64)
    if len(cands) and len(gt):
        I = metrics.iou(cands[:, :4], gt[:, :4])
        j = I.argmax(1)
        ok = I[np.arange(len(cands)), j] > metrics.IOU_THR
        labels[ok] = gt[j[ok], 4]
    crops = pipeline.crop_stack(aligned, temp, cands)
    return pair.id, cands, labels, crops


def build(split: str, scenario: str, seed: int = 0, workers: int = 8):
    pairs = data.split(split)
    ids, cands, labels, crops, img = [], [], [], [], []
    with ProcessPoolExecutor(workers) as ex:
        for k, (pid, c, lab, cr) in enumerate(ex.map(one, [(i, p, scenario, seed) for i, p in enumerate(pairs)])):
            ids.append(pid)
            cands.append(c)
            labels.append(lab)
            crops.append(cr)
            img.append(np.full(len(c), k))
    out = data.out_root() / "cands"
    out.mkdir(exist_ok=True)
    path = out / f"{split}_{scenario}.npz"
    np.savez_compressed(path, ids=np.array(ids), cands=np.concatenate(cands), labels=np.concatenate(labels),
                        crops=np.concatenate(crops), img=np.concatenate(img))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="trainval")
    ap.add_argument("--scenarios", default="native,warp")
    args = ap.parse_args()
    for sc in args.scenarios.split(","):
        t0 = time.time()
        path = build(args.split, sc)
        z = np.load(path)
        lab = z["labels"]
        print(f"{args.split}/{sc}: {len(lab)} candidates, {np.mean(lab != BG):.2f} positive, "
              f"{time.time() - t0:.0f}s -> {path}")


if __name__ == "__main__":
    main()
