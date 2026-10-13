"""Cross-dataset test: the DeepPCB-trained pipeline on the PKU synthetic PCB set, with no retraining and
no parameter changes. Every PKU image is binarized (refdiff/pku.py) and scaled by one global factor: the
ratio of the two datasets' median defect-box sides, since the pipeline's pixel parameters (box padding,
minimum side, crop context) were fitted to DeepPCB defect sizes. That ratio is dataset-level statistics,
not per-image labels; --scales sweeps it to show how much the result depends on it.

Scenarios (all 693 annotated images):
  aligned          the defect photo, already aligned with its template to within about a pixel; registered
  aligned_noreg    same, no registration
  rotated          PKU's own rotated copy of the photo (larger canvas, rotated about its centre), registered
                   back onto the template; the template-frame annotations are the ground truth
  rotated_noreg    same, no registration
Registration accuracy: the aligned photo under a known warp (rotation up to ±10°, shift up to ±40 px at
DeepPCB scale), scored by corner error, as on DeepPCB.

Baseline: the template-blind YOLOv8n trained on DeepPCB, run on the same binarized, scaled aligned images
in 640-px tiles. Scored class-aware (PKU classes mapped to DeepPCB's) and class-agnostic (any defect box
at IoU > 0.33). Writes results/pku.json.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from refdiff import candidates as C
from refdiff import classify, data, metrics, pipeline, pku
from refdiff import register as R

SCENARIOS = ["aligned", "aligned_noreg", "rotated", "rotated_noreg"]
SYN_ROT, SYN_SHIFT = 10.0, 40.0


def box_side(boxes: np.ndarray) -> np.ndarray:
    return np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])


def base_scale(ss: list[pku.Sample]) -> float:
    deep = np.median(box_side(np.concatenate([p.boxes() for p in data.split("trainval")])))
    return float(deep / np.median(box_side(np.concatenate([s.boxes() for s in ss]))))


_templates: dict = {}


def template(s: pku.Sample, scale: float) -> np.ndarray:
    """DeepPCB-style template image per (board, scale), cached per worker process."""
    if (s.board, scale) not in _templates:
        _templates[(s.board, scale)] = pku.to_deeppcb(pku.binarize(cv2.imread(str(s.template))), scale)
    return _templates[(s.board, scale)]


def one(job):
    s, scenario, scale, k = job
    T = template(s, scale)
    src = s.rotated if scenario.startswith("rotated") else s.image
    X = pku.to_deeppcb(pku.binarize(cv2.imread(str(src))), scale)
    X = pku.fit_canvas(X, T.shape)
    out = {"name": s.name, "scale": scale}
    if scenario == "synthetic":
        rng = np.random.default_rng([7, k])
        h, w = T.shape
        M = R.euclid(rng.uniform(-SYN_ROT, SYN_ROT), *rng.uniform(-SYN_SHIFT, SYN_SHIFT, 2), (w / 2, h / 2))
        X = R.apply(X, M, T.shape)
        t0 = time.perf_counter()
        reg = R.register(T, X, "guarded")
        out.update(err=R.corner_error(reg.M, R.invert(M), T.shape), confident=reg.confident,
                   ms=1000 * (time.perf_counter() - t0))
        return out
    if scenario.endswith("noreg"):
        A = X
    else:
        reg = R.register(T, X, "guarded")
        A = R.apply(X, reg.M, T.shape)
        out["confident"] = reg.confident
    cands = C.candidates(A, T)
    out["cands"] = cands
    out["crops"] = pipeline.crop_stack(A, T, cands)
    out["mismatch"] = float((C.binarize(A) != C.binarize(T)).mean())
    return out


def to_template_px(boxes: np.ndarray, scale: float) -> np.ndarray:
    b = boxes.astype(float).copy()
    b[:, :4] /= scale
    return b


def agnostic(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        v = v.copy()
        v[:, -1] = 0
        out[k] = v
    return out


def score(gts: dict, dets: dict) -> dict:
    ev = metrics.evaluate(gts, dets, len(data.CLASSES))
    ag = metrics.evaluate(agnostic(gts), agnostic(dets), 1)
    by_kind = {}
    for kind in pku.CLASS_MAP:
        ids = [k for k in gts if k[3:].rsplit("_", 1)[0] == kind]  # "spur" is a prefix of "spurious_copper"
        hit = n = 0
        for k in ids:
            g = agnostic({k: gts[k]})[k]
            gh, _ = metrics.match(g, agnostic({k: dets[k]})[k])
            hit += int(gh.sum())
            n += len(g)
        by_kind[kind] = {"recall_any_class": hit / max(n, 1), "n": n,
                         "ap_as_mapped_class": ev["per_class"][pku.CLASS_MAP[kind]]["ap"]}
    return {"mAP": ev["mAP"], "f1": ev["f1"], "precision": ev["precision"], "recall": ev["recall"],
            "agnostic": {k: ag[k] for k in ("mAP", "f1", "precision", "recall")}, "by_pku_class": by_kind}


def yolo_tiled(model, bgr: np.ndarray, tile: int = 640, stride: int = 480) -> np.ndarray:
    h, w = bgr.shape[:2]
    ys = sorted({*range(0, max(h - tile, 0) + 1, stride), max(h - tile, 0)})
    xs = sorted({*range(0, max(w - tile, 0) + 1, stride), max(w - tile, 0)})
    out = []
    for y in ys:
        for x in xs:
            crop = bgr[y:y + tile, x:x + tile]
            r = model.predict(crop, imgsz=tile, conf=0.001, iou=0.5, device="mps", verbose=False, max_det=100)[0]
            b = r.boxes
            if len(b):
                d = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None], b.cls.cpu().numpy()[:, None]], 1)
                d[:, [0, 2]] += x
                d[:, [1, 3]] += y
                out.append(d)
    return classify.nms(np.concatenate(out)) if out else np.zeros((0, 6))


def yolo_baselines(res: dict, ss, gts, base: float):
    """The DeepPCB-trained YOLOv8n models on the as-shipped PKU images, 640-px tiles: template-blind on the
    binarized test image, and template-aware on the registered test / template / signed-diff composite."""
    from ultralytics import YOLO
    runs = {"yolov8n_template_blind_aligned": (data.out_root() / "yolo/runs/yolov8n/weights/best.pt", "test"),
            "yolov8n_ttd_aligned": (data.out_root() / "yolo_ttd/runs/yolov8n_ttd/weights/best.pt", "ttd")}
    for key, (w, mode) in runs.items():
        if not w.exists():
            continue
        model = YOLO(str(w))
        dets = {}
        for s in ss:
            T = template(s, base)
            X = pku.fit_canvas(pku.to_deeppcb(pku.binarize(cv2.imread(str(s.image))), base), T.shape)
            if mode == "ttd":
                A = R.apply(X, R.register(T, X, "guarded").M, T.shape)
                extra, missing = C.signed_diff(A, T, C.Params())
                diff = (128 + 127 * extra.astype(np.int16) - 128 * missing.astype(np.int16)).astype(np.uint8)
                img = cv2.merge([diff, T, A])
            else:
                img = cv2.cvtColor(X, cv2.COLOR_GRAY2BGR)
            dets[s.name] = to_template_px(yolo_tiled(model, img), base)
        res[key] = score(gts, dets)
        y = res[key]
        print(f"{key}: mAP {y['mAP']:.4f} F1 {y['f1']:.4f} any-class mAP {y['agnostic']['mAP']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n", type=int, default=0, help="first n samples only (debugging)")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--scales", default="0.7,1,1.4", help="sweep, as multiples of the base scale")
    ap.add_argument("--yolo-only", action="store_true", help="add the YOLO baselines to an existing results/pku.json")
    args = ap.parse_args()
    ss = pku.samples()
    base = base_scale(ss)
    if args.n:
        ss = ss[:: max(1, len(ss) // args.n)][: args.n]  # spread over boards and classes
    gts = {s.name: s.boxes() for s in ss}
    clf = classify.Classifier()
    res = {"n_images": len(ss), "n_defects": int(sum(len(g) for g in gts.values())), "base_scale": base,
           "scenarios": {}, "scale_sweep": {}}

    if args.yolo_only:
        res = json.loads(Path("results/pku.json").read_text())
        yolo_baselines(res, ss, gts, base)
        Path("results/pku.json").write_text(json.dumps(res, indent=1) + "\n")
        return

    with ProcessPoolExecutor(args.workers) as ex:
        for m in (float(x) for x in args.scales.split(",")):
            if m == 1:
                continue
            dets = {}
            for r in ex.map(one, [(s, "aligned", base * m, k) for k, s in enumerate(ss)]):
                d = clf.detect(r["cands"], r["crops"])
                dets[r["name"]] = to_template_px(d, r["scale"]) if len(d) else np.zeros((0, 6))
            ev = score(gts, dets)
            res["scale_sweep"][str(m)] = {k: ev[k] for k in ("mAP", "f1", "agnostic")}
            print(f"scale x{m}: mAP {ev['mAP']:.4f} any-class mAP {ev['agnostic']['mAP']:.4f}")
        syn = list(ex.map(one, [(s, "synthetic", base, k) for k, s in enumerate(ss)]))
        e = np.array([r["err"] for r in syn])
        conf = np.array([r["confident"] for r in syn])
        res["registration"] = {"protocol": {"rot_deg": SYN_ROT, "shift_px": SYN_SHIFT}, "median_px": float(np.median(e)),
                               "p95_px": float(np.percentile(e, 95)), "max_px": float(e.max()),
                               "frac_over_2px": float((e > 2).mean()), "flagged_for_review": int((~conf).sum()),
                               "max_px_confident": float(e[conf].max()) if conf.any() else None,
                               "ms_per_pair": float(np.mean([r["ms"] for r in syn]))}
        print("registration", res["registration"])
        for sc in SCENARIOS:
            dets = {}
            rs = list(ex.map(one, [(s, sc, base, k) for k, s in enumerate(ss)]))
            for r in rs:
                d = clf.detect(r["cands"], r["crops"])
                dets[r["name"]] = to_template_px(d, r["scale"]) if len(d) else np.zeros((0, 6))
            ev = score(gts, dets)
            ev["mismatch_mean"] = float(np.mean([r["mismatch"] for r in rs]))
            if sc == "aligned":
                res["scale_sweep"]["1.0"] = {k: ev[k] for k in ("mAP", "f1", "agnostic")}
            if "confident" in rs[0]:
                ev["flagged_for_review"] = int(sum(not r["confident"] for r in rs))
            res["scenarios"][sc] = ev
            print(f"{sc:14s} mAP {ev['mAP']:.4f} F1 {ev['f1']:.4f}  any-class mAP {ev['agnostic']['mAP']:.4f} "
                  f"recall {ev['agnostic']['recall']:.4f} precision {ev['agnostic']['precision']:.4f}")

    if not args.no_yolo:
        yolo_baselines(res, ss, gts, base)
    if not args.n:
        Path("results/pku.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
