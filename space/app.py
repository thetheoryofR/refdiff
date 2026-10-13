"""refdiff interactive demo: warp a DeepPCB test image, watch registration put it back, see what the
diff + classifier find. CPU only: OpenCV + the committed ONNX classifier.

Run locally:  python space/app.py   (from the repo root, with `pip install -e '.[infer]' gradio`)
As a Hugging Face Space: `make space` assembles build/space/ with this file, the package and examples.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # repo checkout: the package sits next to space/

from refdiff import candidates as C
from refdiff import classify, data, metrics, pipeline
from refdiff import register as R

EXAMPLES = HERE / "examples"
IDS = sorted(p.name[: -len("_test.jpg")] for p in EXAMPLES.glob("*_test.jpg"))
CLF = classify.Classifier(next(p for p in (HERE / "models" / "candnet.onnx", classify.ONNX_PATH) if p.exists()))


def overlay(test, temp):
    return cv2.cvtColor(cv2.merge([temp, test, temp]), cv2.COLOR_BGR2RGB)  # template magenta, test green


def run(pair_id, rot, dx, dy, blur, noise, method):
    test = cv2.imread(str(EXAMPLES / f"{pair_id}_test.jpg"), cv2.IMREAD_GRAYSCALE)
    temp = cv2.imread(str(EXAMPLES / f"{pair_id}_temp.jpg"), cv2.IMREAD_GRAYSCALE)
    gt = data.read_boxes(EXAMPLES / f"{pair_id}.txt")
    h, w = temp.shape
    Mt = R.euclid(rot, dx, dy, (w / 2, h / 2))
    x = R.apply(test, Mt, temp.shape).astype(np.float32)
    if blur > 0:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    x += np.random.default_rng(0).normal(0, noise, x.shape).astype(np.float32)
    mis = np.clip(x, 0, 255).astype(np.uint8)

    reg = R.register(temp, mis, method)
    aligned = R.apply(mis, reg.M, temp.shape)
    err = R.corner_error(reg.M, R.invert(Mt), temp.shape)
    cands = C.candidates(aligned, temp)
    dets = CLF.detect(cands, pipeline.crop_stack(aligned, temp, cands))
    dets = dets[dets[:, 4] >= 0.5]

    vis = cv2.cvtColor(aligned, cv2.COLOR_GRAY2RGB)
    gh, dh = metrics.match(gt, dets)
    for d, ok in zip(dets, dh):
        col = (0, 160, 0) if ok else (220, 0, 0)
        x1, y1, x2, y2 = d[:4].astype(int)
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, f"{data.CLASSES[int(d[5])]} {d[4]:.2f}", (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, col, 1, cv2.LINE_AA)
    for g, ok in zip(gt, gh):
        if not ok:
            cv2.rectangle(vis, tuple(int(v) for v in g[:2]), tuple(int(v) for v in g[2:4]), (230, 180, 0), 2)
    flag = "" if reg.confident else "  ⚠ flagged for review (low overlap NCC)"
    summary = (f"Registration error {err:.2f} px (mean corner distance from the true warp){flag}.  "
               f"Found {int(gh.sum())} of {len(gt)} defects, {int((~dh).sum())} false alarms.")
    return overlay(mis, temp), overlay(aligned, temp), vis, summary


with gr.Blocks(title="refdiff") as demo:
    gr.Markdown("# refdiff: PCB defects by comparing to a reference\n"
                "Misregister a DeepPCB test image, then watch sub-pixel registration put it back and the signed "
                "diff + classifier find the defects. Overlays: template magenta, test green, so aligned copper is "
                "dark. Boxes: green = correct, red = wrong class or false alarm, amber = missed.")
    with gr.Row():
        with gr.Column(scale=1):
            pid = gr.Dropdown(IDS, value=IDS[0] if IDS else None, label="test pair")
            rot = gr.Slider(-10, 10, value=2.5, step=0.1, label="rotation (deg)")
            dx = gr.Slider(-40, 40, value=12, step=0.5, label="shift x (px)")
            dy = gr.Slider(-40, 40, value=-7, step=0.5, label="shift y (px)")
            blur = gr.Slider(0, 4, value=1.0, step=0.1, label="blur sigma (px)")
            noise = gr.Slider(0, 60, value=8, step=1, label="noise sd")
            method = gr.Radio(["guarded", "fm+ecc", "phase+ecc", "ecc", "sift", "none"], value="guarded",
                              label="registration")
            go = gr.Button("Run", variant="primary")
        with gr.Column(scale=3):
            summary = gr.Markdown()
            with gr.Row():
                before = gr.Image(label="before registration")
                after = gr.Image(label="after registration")
            dets = gr.Image(label="detections")
    inputs = [pid, rot, dx, dy, blur, noise, method]
    go.click(run, inputs, [before, after, dets, summary])
    demo.load(run, inputs, [before, after, dets, summary])

if __name__ == "__main__":
    demo.launch()
