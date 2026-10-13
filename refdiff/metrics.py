"""Detection scoring with the DeepPCB rule: a detection is a hit if IoU > 0.33 with an unmatched
ground-truth box of the same class. AP is VOC-style (all-point interpolated) per class; mAP is the mean.

The official script (evaluation/script.py) is Python 2 and needs the Polygon package; this
re-implements its matching rule so the numbers can come from a script that runs in CI.
"""
from __future__ import annotations

import numpy as np

IOU_THR = 0.33


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU of (N,4) and (M,4) boxes in x1,y1,x2,y2 pixel-edge coordinates."""
    a = np.asarray(a, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


def _ap(rec: np.ndarray, prec: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(gts: dict[str, np.ndarray], dets: dict[str, np.ndarray], n_cls: int,
             iou_thr: float = IOU_THR, score_thr: float = 0.5) -> dict:
    """gts[id]: (N,5) x1,y1,x2,y2,cls. dets[id]: (M,6) x1,y1,x2,y2,score,cls.

    Returns per-class AP, and precision / recall / F at score_thr, plus their means.
    """
    per_cls = []
    for c in range(n_cls):
        records = []  # (score, is_tp)
        n_gt = 0
        for k, g in gts.items():
            g = g[g[:, 4] == c]
            n_gt += len(g)
            d = dets.get(k, np.zeros((0, 6)))
            d = d[d[:, 5] == c]
            d = d[np.argsort(-d[:, 4], kind="stable")]
            used = np.zeros(len(g), bool)
            ious = iou(d[:, :4], g[:, :4]) if len(d) and len(g) else np.zeros((len(d), len(g)))
            for i in range(len(d)):
                hit = False
                if len(g):
                    cand = np.where(~used & (ious[i] > iou_thr))[0]
                    if len(cand):
                        j = cand[np.argmax(ious[i, cand])]
                        used[j] = True
                        hit = True
                records.append((d[i, 4], hit))
        records.sort(key=lambda r: -r[0])
        scores = np.array([r[0] for r in records])
        tp = np.array([r[1] for r in records], float)
        ctp = np.cumsum(tp)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.arange(1, len(tp) + 1) if len(tp) else np.zeros(0)
        keep = scores >= score_thr
        tp_t = tp[keep].sum()
        p = tp_t / max(keep.sum(), 1)
        r = tp_t / max(n_gt, 1)
        per_cls.append({"ap": _ap(rec, prec) if len(tp) else 0.0, "precision": float(p), "recall": float(r),
                        "f1": float(2 * p * r / (p + r)) if p + r else 0.0, "n_gt": int(n_gt),
                        "n_det": int(keep.sum())})
    out = {"per_class": per_cls,
           "mAP": float(np.mean([c["ap"] for c in per_cls])),
           "precision": float(np.mean([c["precision"] for c in per_cls])),
           "recall": float(np.mean([c["recall"] for c in per_cls]))}
    out["f1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-9)
    return out


def match(g: np.ndarray, d: np.ndarray, iou_thr: float = IOU_THR, score_thr: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """One image, the same greedy rule as evaluate() at score_thr. g: (N,5), d: (M,6).

    Returns (gt_hit (N,) bool, det_hit (M,) bool, over all d; detections below score_thr are never hits).
    """
    gt_hit = np.zeros(len(g), bool)
    det_hit = np.zeros(len(d), bool)
    for c in np.unique(np.concatenate([g[:, 4], d[:, 5]]).astype(int)):
        gi = np.where(g[:, 4] == c)[0]
        di = np.where((d[:, 5] == c) & (d[:, 4] >= score_thr))[0]
        di = di[np.argsort(-d[di, 4], kind="stable")]
        if not len(gi) or not len(di):
            continue
        ious = iou(d[di, :4], g[gi, :4])
        for k, i in enumerate(di):
            cand = np.where(~gt_hit[gi] & (ious[k] > iou_thr))[0]
            if len(cand):
                gt_hit[gi[cand[np.argmax(ious[k, cand])]]] = True
                det_hit[i] = True
    return gt_hit, det_hit


def paired_bootstrap(gts: dict, a: dict, b: dict, n: int = 1000, seed: int = 0, n_cls: int = 6) -> dict:
    """Image-level paired bootstrap of method a minus method b (same resampled images for both)."""
    ids = sorted(gts)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        g = {f"{j}": gts[ids[i]] for j, i in enumerate(pick)}
        ea = evaluate(g, {f"{j}": a[ids[i]] for j, i in enumerate(pick)}, n_cls)
        eb = evaluate(g, {f"{j}": b[ids[i]] for j, i in enumerate(pick)}, n_cls)
        diffs.append((ea["mAP"] - eb["mAP"], ea["f1"] - eb["f1"]))
    d = np.array(diffs)
    full_a = evaluate(gts, a, n_cls)
    full_b = evaluate(gts, b, n_cls)
    return {"delta_mAP": full_a["mAP"] - full_b["mAP"], "delta_mAP_ci95": np.percentile(d[:, 0], [2.5, 97.5]).tolist(),
            "delta_f1": full_a["f1"] - full_b["f1"], "delta_f1_ci95": np.percentile(d[:, 1], [2.5, 97.5]).tolist(),
            "p_a_not_better_mAP": float((d[:, 0] <= 0).mean())}
