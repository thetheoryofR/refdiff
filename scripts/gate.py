"""CI quality gate. Needs only DeepPCB + the committed ONNX classifier (no torch, no GPU).

1. Registration: guarded method on the first --n-reg test pairs under warp + optics; corner error
   median / p95, the max over pairs the guard is confident about, and the fraction it flags for
   review must stay under eval/gate.json.
2. Detection: the full pipeline (guarded registration -> diff -> candidates -> ONNX classifier) on the
   500-image test split, native pairs; mAP and F1 must stay above eval/gate.json.
Exits 1 on any regression and prints the measured values either way.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from refdiff import candidates as C
from refdiff import classify, data, metrics, pipeline, synth
from refdiff import register as R


def reg_err(args):
    i, p = args
    rng = np.random.default_rng([0, i])
    test, temp = p.load()
    mis, Mtrue = synth.misregister(synth.defective_template(test, temp, p.boxes()), rng)
    r = R.register(temp, mis, "guarded")
    return R.corner_error(r.M, Mtrue, temp.shape), r.confident


def cands(p):
    test, temp = p.load()
    aligned, _ = pipeline.prepare(test, temp, "native", None)
    c = C.candidates(aligned, temp)
    return p.id, c, pipeline.crop_stack(aligned, temp, c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-reg", type=int, default=100)
    ap.add_argument("--n-det", type=int, default=0, help="0 = all 500")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    gate = json.loads(Path("eval/gate.json").read_text())
    pairs = data.split("test")
    fails, measured = [], {}

    with ProcessPoolExecutor(args.workers) as ex:
        out = list(ex.map(reg_err, enumerate(pairs[: args.n_reg])))
        errs = np.array([e for e, _ in out])
        conf = np.array([c for _, c in out])
        measured["reg_median_px"] = float(np.median(errs))
        measured["reg_p95_px"] = float(np.percentile(errs, 95))
        measured["reg_max_confident_px"] = float(errs[conf].max())  # flagged pairs go to review
        measured["reg_flagged_frac"] = float(1 - conf.mean())
        det_pairs = pairs[: args.n_det] if args.n_det else pairs
        clf = classify.Classifier()
        dets = {pid: clf.detect(c, cr) for pid, c, cr in ex.map(cands, det_pairs, chunksize=8)}
    ev = metrics.evaluate({p.id: p.boxes() for p in det_pairs}, dets, len(data.CLASSES))
    measured["det_mAP"], measured["det_f1"] = ev["mAP"], ev["f1"]

    for k, v in measured.items():
        lim = gate[k]
        ok = v <= lim["max"] if "max" in lim else v >= lim["min"]
        print(f"{'ok  ' if ok else 'FAIL'} {k:15s} {v:8.4f}   ({'<=' if 'max' in lim else '>='} "
              f"{lim.get('max', lim.get('min'))})")
        if not ok:
            fails.append(k)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
