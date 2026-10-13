---
title: refdiff
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
license: mit
---

# refdiff demo

Misregister a DeepPCB test image, then watch sub-pixel registration (Fourier–Mellin + phase correlation +
ECC, with a guard) put it back, and a signed diff + small CNN find the defects. CPU only: OpenCV and an
ONNX classifier. Code, evaluation and numbers: the refdiff repository.

The six example pairs are from [DeepPCB](https://github.com/tangsanli5201/DeepPCB) (MIT, see
`examples/DEEPPCB_LICENSE`).
