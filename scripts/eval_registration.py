"""Registration accuracy in pixels on the 500 DeepPCB test pairs.

Synthetic: defective template (real defect patches pasted into the template) warped by a known
rotation in [-3, 3] deg and shift in [-20, 20] px, then blurred / gain-shifted / noised. Error is the
mean displacement of the four image corners between the estimated and the true warp.

Native: the as-shipped pairs, scored by the fraction of binarized pixels that disagree outside the
(dilated) defect boxes, before and after registration.

Writes results/registration.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from refdiff import data, synth
from refdiff import register as R

METHODS = ["none", "orb", "sift", "phase", "fm", "ecc", "phase+ecc", "fm+ecc", "guarded"]


def offdefect_mismatch(img, temp, boxes, pad=4, margin=24):
    h, w = temp.shape
    m = np.zeros((h, w), bool)
    m[margin:-margin, margin:-margin] = True
    for x1, y1, x2, y2, _ in boxes:
        m[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = False
    return float(((img > 127) != (temp > 127))[m].mean())


def summarize(e):
    e = np.asarray(e)
    return {"median_px": float(np.median(e)), "mean_px": float(e.mean()), "p95_px": float(np.percentile(e, 95)),
            "max_px": float(e.max()), "frac_under_0.5px": float((e < 0.5).mean()),
            "frac_over_2px": float((e > 2).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit pairs (0 = all 500)")
    ap.add_argument("--out", default="results/registration.json")
    args = ap.parse_args()
    pairs = data.split("test")
    if args.n:
        pairs = pairs[: args.n]

    res = {"n_pairs": len(pairs), "protocol": {"rot_deg": 3.0, "shift_px": 20.0, "seed": 0}, "synthetic": {}}
    for optics in (False, True):
        rng = np.random.default_rng(0)
        cases = []
        for p in pairs:
            test, temp = p.load()
            img = synth.defective_template(test, temp, p.boxes())
            cases.append((temp, *synth.misregister(img, rng, optics=optics)))  # one rng: same warps per method
        key = "warp+optics" if optics else "warp_only"
        res["synthetic"][key] = {}
        for m in METHODS:
            t0 = time.perf_counter()
            regs = [R.register(temp, mis, m) for temp, mis, _ in cases]
            ms = 1000 * (time.perf_counter() - t0) / len(cases)
            errs = [R.corner_error(r.M, Mtrue, temp.shape) for r, (temp, _, Mtrue) in zip(regs, cases)]
            res["synthetic"][key][m] = {**summarize(errs), "ms_per_pair": round(ms, 1)}
            if m == "guarded":
                conf = np.array([r.confident for r in regs])
                e = np.array(errs)
                res["synthetic"][key][m]["flagged_for_review"] = int((~conf).sum())
                res["synthetic"][key][m]["max_px_confident"] = float(e[conf].max())
                res["synthetic"][key][m]["max_px_flagged"] = float(e[~conf].max()) if (~conf).any() else None
            print(f"{key:12s} {m:10s} median {np.median(errs):7.3f}  p95 {np.percentile(errs, 95):8.3f}  "
                  f"max {np.max(errs):8.2f}  {ms:5.0f} ms")

    before, after, moved = [], [], []
    for p in pairs:
        test, temp = p.load()
        b = p.boxes()
        r = R.register(temp, test, "guarded")
        before.append(offdefect_mismatch(test, temp, b))
        after.append(offdefect_mismatch(R.apply(test, r.M, temp.shape), temp, b))
        moved.append(R.corner_error(r.M, np.eye(2, 3), temp.shape))
    before, after, moved = map(np.array, (before, after, moved))
    res["native"] = {
        "offdefect_mismatch_identity_mean": float(before.mean()),
        "offdefect_mismatch_registered_mean": float(after.mean()),
        "pairs_moved_over_1px": int((moved > 1).sum()),
        "pairs_where_registration_cut_mismatch_by_half": int((after < 0.5 * before).sum()),
        "residual_px_p50_p95_max": [float(np.median(moved)), float(np.percentile(moved, 95)), float(moved.max())],
    }
    print(json.dumps(res["native"], indent=1))
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
