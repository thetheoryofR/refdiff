"""Every error at the operating point (score >= 0.5, the DeepPCB IoU rule), for refdiff (as-shipped
pairs, guarded registration) and the template-blind YOLOv8n, each tagged with a cause.

Missed defects (ground truth with no same-class hit):
  wrong class     a kept detection of another class covers it (IoU > 0.33)
  merged          a same-class detection covers it but was already spent on a neighbouring defect
  box off         a same-class detection overlaps it, but at IoU <= 0.33
  low score       refdiff: a candidate covers it but the classifier scored it < 0.5 or background;
                  YOLO: a same-class box at IoU > 0.33 exists below the 0.5 threshold
  diff box off    refdiff only: diff regions touch it, but none at IoU > 0.33 (split or merged blobs)
  diff empty      refdiff only: no diff region touches it (tolerance and opening erased it)
  not detected    YOLO only: nothing of the right class near it at any score
False alarms (kept detections with no same-class hit):
  wrong class     it covers a ground-truth defect of another class
  duplicate       it covers a same-class defect that another detection already claimed
  fragment        at least half of it lies inside a labelled defect, but IoU <= 0.33
  box off         it overlaps a same-class defect, at IoU <= 0.33
  burned-in label nothing labelled within it, on a test image with annotations drawn into the pixels
  no defect       nothing labelled within it

Writes results/failures.json, docs/failures/<method>_<kind>_<cause>.jpg galleries (template | test |
signed diff after registration, ground truth green, detection red) and the error list as
results/failures.csv, the input to the hand audit in eval/failure_audit.csv.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from refdiff import candidates as C
from refdiff import data, metrics
from refdiff import register as R

THR = 0.5
WIN = 112        # context window around an error, px
SCALE = 1.25     # gallery magnification
COLS = 3         # errors per gallery row
MAX_GALLERY = 36


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def first(mask: np.ndarray) -> int:
    i = np.flatnonzero(mask)
    return int(i[0]) if len(i) else -1


def ioa(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """(len(d), len(g)): fraction of each detection's area inside each GT box."""
    x1 = np.maximum(d[:, None, 0], g[None, :, 0])
    y1 = np.maximum(d[:, None, 1], g[None, :, 1])
    x2 = np.minimum(d[:, None, 2], g[None, :, 2])
    y2 = np.minimum(d[:, None, 3], g[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / np.maximum((d[:, 2] - d[:, 0]) * (d[:, 3] - d[:, 1]), 1e-9)[:, None]


def burned_in(pairs, min_px: int = 500) -> list[str]:
    """Test images with annotation boxes / text drawn into them: DeepPCB images are binary, the drawn
    labels are mid-grey. Clean images have at most a few dozen grey pixels."""
    out = []
    for p in pairs:
        t = cv2.imread(str(p.test), cv2.IMREAD_GRAYSCALE)
        if int(((t > 90) & (t < 200)).sum()) >= min_px:
            out.append(p.id)
    return out


def analyse(gts, dets, cands=None, flagged=()):
    """List of error dicts. cands: refdiff candidate boxes per image (enables the candidate-stage causes)."""
    errs = []
    for pid, g in gts.items():
        d = dets.get(pid, np.zeros((0, 6)))
        gt_hit, det_hit = metrics.match(g, d, score_thr=THR)
        kept = d[:, 4] >= THR
        ious = metrics.iou(g[:, :4], d[:, :4]) if len(g) and len(d) else np.zeros((len(g), len(d)))
        inside = ioa(d[:, :4], g[:, :4]).T if len(g) and len(d) else np.zeros((len(g), len(d)))
        for i in np.flatnonzero(~gt_hit):
            row, c = ious[i], g[i, 4]
            same, other = d[:, 5] == c, d[:, 5] != c
            if (j := first(kept & other & (row > metrics.IOU_THR))) >= 0:
                cause = "wrong class"
            elif (j := first(kept & same & det_hit & (row > metrics.IOU_THR))) >= 0:
                cause = "merged"
            elif (j := first(kept & same & (row > 0))) >= 0:
                cause = "box off"
            elif cands is None:
                j = first(~kept & same & (row > metrics.IOU_THR))
                cause = "low score" if j >= 0 else "not detected"
            else:
                cb = cands.get(pid, np.zeros((0, 6)))
                ci = metrics.iou(g[i:i + 1, :4], cb[:, :4])[0] if len(cb) else np.zeros(0)
                cause = ("low score" if (ci > metrics.IOU_THR).any() else
                         "diff box off" if (ci > 0).any() else "diff empty")
                j = -1
            errs.append({"id": pid, "kind": "missed", "cause": cause, "gt": g[i, :5].tolist(),
                         "det": d[j].tolist() if j >= 0 else None})
        for j in np.flatnonzero(kept & ~det_hit):
            col, c = ious[:, j], d[j, 5]
            over = col > metrics.IOU_THR
            same = g[:, 4] == c
            if (i := first(over & ~same)) >= 0:
                cause = "wrong class"
            elif (i := first(over & same)) >= 0:
                cause = "duplicate"
            elif (i := first(inside[:, j] >= 0.5)) >= 0:
                cause = "fragment"
            elif (i := first(same & (col > 0))) >= 0:
                cause = "box off"
            else:
                i, cause = -1, "burned-in label" if pid in flagged else "no defect"
            errs.append({"id": pid, "kind": "false alarm", "cause": cause,
                         "gt": g[i, :5].tolist() if i >= 0 else None, "det": d[j].tolist()})
    return errs


def summarise(errs, gts, dets) -> dict:
    n_gt = sum(len(g) for g in gts.values())
    n_kept = sum(int((d[:, 4] >= THR).sum()) for d in dets.values())
    out = {"n_gt": n_gt, "n_kept": n_kept}
    for kind in ("missed", "false alarm"):
        e = [x for x in errs if x["kind"] == kind]
        out[kind] = {"total": len(e), "by_cause": dict(Counter(x["cause"] for x in e).most_common()),
                     "by_class": dict(Counter(data.CLASSES[int((x["gt"] if kind == "missed" else x["det"])[
                         4 if kind == "missed" else 5])] for x in e).most_common())}
    conf = Counter(f"{data.CLASSES[int(x['gt'][4])]} -> {data.CLASSES[int(x['det'][5])]}"
                   for x in errs if x["kind"] == "missed" and x["cause"] == "wrong class")
    out["confusions"] = dict(conf.most_common())
    # DeepPCB comes from a handful of boards (one folder each); errors cluster by board
    grp = {p.id: p.test.parent.name for p in data.split("test")}
    n_img = Counter(grp.values())
    n_train = Counter(p.test.parent.name for p in data.split("trainval"))
    out["by_board"] = {g: {"images": n_img[g], "train_images": n_train[g], **{k: sum(1 for x in errs if x["kind"] == k and grp[x["id"]] == g)
                                                 for k in ("missed", "false alarm")}} for g in sorted(n_img)}
    # the official split holds some boards out of trainval entirely
    for name, keep in (("seen_boards", lambda g: n_train[g] > 0), ("unseen_boards", lambda g: n_train[g] == 0)):
        ids = [k for k in gts if keep(grp[k])]
        ev = metrics.evaluate({k: gts[k] for k in ids}, {k: dets[k] for k in ids}, len(data.CLASSES), score_thr=THR)
        out[name] = {"images": len(ids), "boards": sorted({grp[k] for k in ids}),
                     **{k: ev[k] for k in ("mAP", "f1", "precision", "recall")}}
    return out


class Images:
    """Template, test and the post-registration signed diff per pair, cached (registration is the slow part)."""

    def __init__(self):
        self.pairs = {p.id: p for p in data.split("test")}
        self._c = {}

    def __call__(self, pid):
        if pid not in self._c:
            test, temp = self.pairs[pid].load()
            aligned = R.apply(test, R.register(temp, test, "guarded").M, temp.shape)
            extra, missing = C.signed_diff(aligned, temp, C.Params())
            diff = np.full((*temp.shape, 3), 255, np.uint8)
            diff[extra > 0] = (40, 40, 220)    # extra copper: red
            diff[missing > 0] = (220, 120, 30)  # missing copper: blue
            self._c[pid] = (temp, test, diff)
        return self._c[pid]


def tile(imgs: Images, e: dict) -> np.ndarray:
    temp, test, diff = imgs(e["id"])
    boxes = [b for b in (e["gt"], e["det"]) if b is not None]
    cx = int(np.mean([(b[0] + b[2]) / 2 for b in boxes]))
    cy = int(np.mean([(b[1] + b[3]) / 2 for b in boxes]))
    h, w = temp.shape
    x0 = int(np.clip(cx - WIN // 2, 0, w - WIN))
    y0 = int(np.clip(cy - WIN // 2, 0, h - WIN))
    panels = []
    for p in (cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR), cv2.cvtColor(test, cv2.COLOR_GRAY2BGR), diff):
        p = cv2.resize(p[y0:y0 + WIN, x0:x0 + WIN], None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_NEAREST)
        for b, col in ((e["gt"], (0, 170, 0)), (e["det"], (0, 0, 230))):
            if b is not None:
                x1, y1, x2, y2 = (np.array(b[:4], float) - [x0, y0, x0, y0]) * SCALE
                cv2.rectangle(p, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        panels.append(cv2.copyMakeBorder(p, 0, 0, 0, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    t = np.hstack(panels)
    gt = data.CLASSES[int(e["gt"][4])] if e["gt"] is not None else "-"
    dt = f"{data.CLASSES[int(e['det'][5])]} {e['det'][4]:.2f}" if e["det"] is not None else "-"
    t = cv2.copyMakeBorder(t, 22, 8, 4, 4, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    cv2.putText(t, f"{e['id']}  gt: {gt}  det: {dt}", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return t


def gallery(imgs: Images, errs: list[dict], path: Path):
    tiles = [tile(imgs, e) for e in errs[:MAX_GALLERY]]
    rows = []
    for k in range(0, len(tiles), COLS):
        row = tiles[k:k + COLS]
        row += [np.full_like(tiles[0], 255)] * (COLS - len(row))
        rows.append(np.hstack(row))
    cv2.imwrite(str(path), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])


def main():
    out = data.out_root()
    gts = {p.id: p.boxes() for p in data.split("test")}
    z = load(out / "cands" / "test_native.npz")
    cands = {str(pid): z["cands"][z["img"] == k] for k, pid in enumerate(z["ids"])}
    methods = {"refdiff": (load(out / "dets" / "refdiff_native.npz"), cands),
               "yolov8n": (load(out / "yolo" / "test_dets_yolov8n.npz"), None)}

    res, rows = {}, []
    gdir = Path("docs/failures")
    gdir.mkdir(parents=True, exist_ok=True)
    for f in gdir.glob("*.jpg"):
        f.unlink()
    imgs = Images()
    flagged = burned_in(data.split("trainval") + data.split("test"))  # all 1,500 pairs
    res["burned_in_label_images"] = flagged
    for name, (dets, cb) in methods.items():
        dets = {k: dets.get(k, np.zeros((0, 6))) for k in gts}
        errs = analyse(gts, dets, cb, flagged)
        ev = metrics.evaluate(gts, dets, len(data.CLASSES), score_thr=THR)
        s = summarise(errs, gts, dets)
        # the per-class P/R in evaluate() and these counts must describe the same errors
        n_tp = sum(c["n_det"] * c["precision"] for c in ev["per_class"])
        assert round(n_tp) == s["n_kept"] - s["false alarm"]["total"], name
        res[name] = s
        for kind in ("missed", "false alarm"):
            for cause in s[kind]["by_cause"]:
                sel = [e for e in errs if e["kind"] == kind and e["cause"] == cause]
                gallery(imgs, sel, gdir / f"{name}_{kind.replace(' ', '-')}_{cause.replace(' ', '-')}.jpg")
        rows += [{"method": name, **{k: e[k] for k in ("id", "kind", "cause")},
                  "gt": " ".join(f"{v:g}" for v in e["gt"]) if e["gt"] is not None else "",
                  "det": " ".join(f"{v:.3g}" for v in e["det"]) if e["det"] is not None else ""} for e in errs]
        print(name, json.dumps({k: s[k] for k in ("missed", "false alarm", "confusions")}, indent=1))

    Path("results/failures.json").write_text(json.dumps(res, indent=1) + "\n")
    with open("results/failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
