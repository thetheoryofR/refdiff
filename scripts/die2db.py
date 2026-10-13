"""Die-to-database stretch: register a simulated optical image to a rasterized GDSII layout and find
injected defects by signed diff (no classifier: the sign of the diff names extra vs missing copper).

N random layouts, seeds from --seed0 (1000): the diff parameters below were tuned on seeds 0-39, so the
reported layouts are ones the tuning never saw. Registration error in px against the known warp; defect recall / precision at
IoU > 0.33 with the right sign. Writes results/die2db.json and one example layout to examples/die.gds.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from refdiff import candidates as C
from refdiff import layout as L
from refdiff import metrics
from refdiff import register as R

SIGN = {0: 1, 1: -1, 2: 1, 3: -1}  # short, copper -> +1 extra copper; open, pin-hole -> -1 missing
PARAMS = C.Params(tol=1, open_k=3, min_area=12, pad=2, min_side=0)  # tol=1 absorbs etch bias + PSF


def one(seed: int):
    rng = np.random.default_rng(seed)
    cell = L.make_layout(rng)
    db = L.rasterize(cell)
    defective, gt = L.inject(db, rng)
    opt, Mtrue = L.optical(defective, rng)
    out = {}
    for m in ("none", "fm+ecc", "guarded"):
        r = R.register(db, opt, m)
        out[f"err_{m}"] = R.corner_error(r.M, Mtrue, db.shape)
        if m == "guarded":
            out["confident"] = r.confident
            aligned = R.apply(opt, r.M, db.shape, border=230)
    for name, img in (("registered", aligned), ("unregistered", opt)):
        c = C.candidates(img, db, PARAMS)
        tp = 0
        if len(c) and len(gt):
            I = metrics.iou(gt[:, :4], c[:, :4])
            used = np.zeros(len(c), bool)
            for i in range(len(gt)):
                ok = np.where(~used & (I[i] > metrics.IOU_THR) & (c[:, 4] == SIGN[int(gt[i, 4])]))[0]
                if len(ok):
                    used[ok[np.argmax(I[i, ok])]] = True
                    tp += 1
        out[f"{name}_tp"], out[f"{name}_ncand"], out["n_gt"] = tp, len(c), len(gt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, range(args.seed0, args.seed0 + args.n)))
    res = {"n_layouts": args.n, "seeds": [args.seed0, args.seed0 + args.n - 1], "protocol": {"rot_deg": 2.0, "shift_px": 15.0, "etch_bias_px": [-1, 0, 1],
                                             "psf_sigma_px": [1.0, 2.0], "defects_per_die": 4}}
    for m in ("none", "fm+ecc", "guarded"):
        e = np.array([r[f"err_{m}"] for r in rows])
        res[f"registration_{m}"] = {"median_px": float(np.median(e)), "p95_px": float(np.percentile(e, 95)),
                                    "max_px": float(e.max())}
    res["flagged_for_review"] = int(sum(not r["confident"] for r in rows))
    n_gt = sum(r["n_gt"] for r in rows)
    for name in ("registered", "unregistered"):
        tp = sum(r[f"{name}_tp"] for r in rows)
        nc = sum(r[f"{name}_ncand"] for r in rows)
        res[f"defects_{name}"] = {"recall": tp / n_gt, "precision": tp / max(nc, 1), "candidates_per_die": nc / args.n}
    print(json.dumps(res, indent=1))
    Path("results/die2db.json").write_text(json.dumps(res, indent=1) + "\n")
    Path("examples").mkdir(exist_ok=True)
    L.write_gds(L.make_layout(np.random.default_rng(0)), Path("examples/die.gds"))


if __name__ == "__main__":
    main()
