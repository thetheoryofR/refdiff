"""Export the trained classifier to models/candnet.onnx (committed; ~1 MB) and check it matches torch."""
from __future__ import annotations

import argparse

import numpy as np
import torch

from refdiff import classify, data
from refdiff.model import CandNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(data.out_root() / "models" / "candnet_s0.pt"))
    args = ap.parse_args()
    m = CandNet().eval()
    m.load_state_dict(torch.load(args.weights, map_location="cpu"))
    classify.ONNX_PATH.parent.mkdir(exist_ok=True)
    x = torch.randint(0, 255, (4, 3, 64, 64), dtype=torch.uint8)
    f = torch.randn(4, 4)
    torch.onnx.export(m, (x, f), str(classify.ONNX_PATH), input_names=["crops", "side"], output_names=["logits"],
                      dynamic_axes={"crops": {0: "n"}, "side": {0: "n"}, "logits": {0: "n"}}, opset_version=17,
                      dynamo=False)
    ref = m(x, f).detach().numpy()
    got = classify.Classifier(classify.ONNX_PATH).logits(x.numpy(), f.numpy())
    err = float(np.abs(ref - got).max())
    print(f"wrote {classify.ONNX_PATH} ({classify.ONNX_PATH.stat().st_size / 1e6:.2f} MB), max |torch - onnx| = {err:.2e}")
    assert err < 1e-3


if __name__ == "__main__":
    main()
