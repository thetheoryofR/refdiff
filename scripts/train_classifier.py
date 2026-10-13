"""Train the candidate classifier on cached trainval candidates (native + warp scenarios).

The last 100 trainval images are held out as validation (model selection only; the test split is
never touched here). Writes <out_root>/models/candnet_s<seed>.pt and results/train_classifier_s<seed>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from refdiff import data, pipeline
from refdiff.model import CandNet, dihedral

CHANNELS = {"test": 0, "template": 1, "diff": 2}


def ablate(crops: np.ndarray, side: np.ndarray, drop: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Ablation by input: a dropped crop channel becomes flat grey (128, i.e. 0 after normalization), dropped
    side-features become 0. Same network, less information."""
    crops, side = crops.copy(), side.copy()
    for name in drop:
        if name == "side":
            side[:] = 0
        else:
            crops[:, CHANNELS[name]] = 128
    return crops, side


def load(scenarios):
    xs, fs, ys, imgs = [], [], [], []
    for sc in scenarios:
        z = np.load(data.out_root() / "cands" / f"trainval_{sc}.npz")
        xs.append(z["crops"])
        fs.append(pipeline.box_features(z["cands"]))
        ys.append(z["labels"])
        imgs.append(z["img"])
    return np.concatenate(xs), np.concatenate(fs), np.concatenate(ys), np.concatenate(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", default="native,warp")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", default="", help="comma list of test,template,diff,side (ablation)")
    ap.add_argument("--tag", default="", help="checkpoint / results name (default s<seed>)")
    args = ap.parse_args()
    tag = args.tag or f"s{args.seed}"
    drop = [d for d in args.drop.split(",") if d]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    X, F, Y, IMG = load(args.scenarios.split(","))
    X, F = ablate(X, F, drop)
    n_img = IMG.max() + 1
    val = IMG >= n_img - 100
    Xt, Ft, Yt = (torch.from_numpy(a[~val]) for a in (X, F, Y))
    Xv, Fv, Yv = (torch.from_numpy(a[val]).to(dev) for a in (X, F, Y))
    print(f"train {len(Yt)}  val {len(Yv)}  class counts {np.bincount(Y[~val], minlength=7).tolist()}")

    model = CandNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(Yt) + args.bs - 1) // args.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    best, hist = (-1.0, None), []
    out = data.out_root() / "models"
    out.mkdir(exist_ok=True)
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Yt))
        tot = 0.0
        for i in range(0, len(perm), args.bs):
            idx = perm[i:i + args.bs]
            x, f = dihedral(Xt[idx], Ft[idx], int(np.random.randint(8)))
            x, f, y = x.to(dev), f.to(dev), Yt[idx].to(dev)
            loss = lossf(model(x, f), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xv[i:i + 1024], Fv[i:i + 1024]).argmax(1) for i in range(0, len(Yv), 1024)])
        acc = (pred == Yv).float().mean().item()
        hist.append({"epoch": ep + 1, "train_loss": tot / len(Yt), "val_acc": acc})
        print(f"ep {ep + 1:2d} loss {tot / len(Yt):.4f} val acc {acc:.4f}  {time.time() - t0:.0f}s")
        if acc > best[0]:
            best = (acc, ep + 1)
            torch.save(model.state_dict(), out / f"candnet_{tag}.pt")
    Path("results").mkdir(exist_ok=True)
    Path(f"results/train_classifier_{tag}.json").write_text(json.dumps(
        {"args": vars(args), "device": dev, "best_val_acc": best[0], "best_epoch": best[1], "history": hist},
        indent=1) + "\n")


if __name__ == "__main__":
    main()
