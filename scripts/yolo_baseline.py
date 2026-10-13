"""YOLOv8 baselines, scored by refdiff.metrics like ours.

--input test  template-blind: the test image alone (it never sees a template).
--input ttd   template-aware: a 3-channel image of the registered test, the template and the signed diff
              (the same three planes refdiff's classifier crops), so YOLO gets the reference but none of
              refdiff's candidate stage. Boxes stay in the template frame, like refdiff's.

Same split as the classifier: trainval minus its last 100 images to train, those 100 to validate, the
500-image test split to score. Writes <out_root>/yolo[_ttd]/... and results/yolo_<model>[_ttd].json
(+ the raw test detections as .npz).

ultralytics is AGPL-3.0; it is an optional extra (`make setup-yolo`) used only for this baseline.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from refdiff import candidates as C
from refdiff import data, metrics, pipeline


def composite(pair: data.Pair) -> np.ndarray:
    """BGR planes: signed diff (128 same, 255 extra, 0 missing), template, registered test."""
    test, temp = pair.load()
    aligned, _ = pipeline.prepare(test, temp, "native", None)
    extra, missing = C.signed_diff(aligned, temp, C.Params())
    diff = (128 + 127 * extra.astype(np.int16) - 128 * missing.astype(np.int16)).astype(np.uint8)
    return cv2.merge([diff, temp, aligned])


def _write(job):
    pair, dst, mode = job
    if mode == "test":
        shutil.copy(pair.test, dst)
    else:
        cv2.imwrite(str(dst), composite(pair))


def export(out: Path, mode: str = "test"):
    ext = "jpg" if mode == "test" else "png"  # lossless: JPEG would smear the one-pixel diff plane
    jobs = []
    tv = data.split("trainval")
    splits = {"train": tv[:-100], "val": tv[-100:], "test": data.split("test")}
    for name, pairs in splits.items():
        (out / "images" / name).mkdir(parents=True, exist_ok=True)
        (out / "labels" / name).mkdir(parents=True, exist_ok=True)
        for p in pairs:
            dst = out / "images" / name / f"{p.id}.{ext}"
            if not dst.exists():
                jobs.append((p, dst, mode))
            h, w = 640, 640
            lines = [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                     for x1, y1, x2, y2, c in p.boxes()]
            (out / "labels" / name / f"{p.id}.txt").write_text("\n".join(lines) + "\n")
    with ProcessPoolExecutor(8) as ex:
        list(ex.map(_write, jobs))
    yaml = out / "deeppcb.yaml"
    yaml.write_text(f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    f"names: {dict(enumerate(data.CLASSES))}\n")
    return yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--input", choices=["test", "ttd"], default="test")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    from ultralytics import YOLO

    suffix = "" if args.input == "test" else f"_{args.input}"
    root = data.out_root() / f"yolo{suffix}"
    yaml = export(root, args.input)
    tag = Path(args.model).stem + suffix
    run_dir = root / "runs" / tag
    weights = run_dir / "weights" / "best.pt"
    t0 = time.time()
    if not args.skip_train:
        # the composite's channels mean test / template / diff: colour augmentation would scramble them
        hsv = {} if args.input == "test" else {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0}
        YOLO(args.model).train(data=str(yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                               device="mps", project=str(root / "runs"), name=tag, exist_ok=True, seed=0,
                               workers=4, plots=False, verbose=False, **hsv)
    train_s = time.time() - t0

    model = YOLO(str(weights))
    gts, dets = {}, {}
    pairs = data.split("test")
    t1 = time.time()
    for p in pairs:
        img = cv2.imread(str(p.test)) if args.input == "test" else composite(p)
        r = model.predict(img, imgsz=args.imgsz, conf=0.001, iou=0.5, device="mps",
                          verbose=False, max_det=100)[0]
        b = r.boxes
        dets[p.id] = np.concatenate([b.xyxy.cpu().numpy(), b.conf.cpu().numpy()[:, None],
                                     b.cls.cpu().numpy()[:, None]], 1) if len(b) else np.zeros((0, 6))
        gts[p.id] = p.boxes()
    ms = 1000 * (time.time() - t1) / len(pairs)
    np.savez_compressed(root / f"test_dets_{tag}.npz", **dets)
    res = {"model": args.model, "input": args.input, "epochs": args.epochs, "train_minutes": round(train_s / 60, 1),
           "ms_per_image": round(ms, 1)}
    res["by_threshold"] = {}
    for thr in (0.25, 0.5):
        res["by_threshold"][str(thr)] = metrics.evaluate(gts, dets, len(data.CLASSES), score_thr=thr)
    res["mAP"] = res["by_threshold"]["0.5"]["mAP"]
    print(json.dumps({k: v for k, v in res.items() if k != "by_threshold"}, indent=1))
    # the Makefile runs this from weights/ (ultralytics downloads base weights into the cwd)
    (data.ROOT / "results").mkdir(exist_ok=True)
    (data.ROOT / "results" / f"yolo_{tag}.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
