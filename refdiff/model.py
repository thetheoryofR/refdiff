"""Candidate classifier: 3-channel crop (test, template, signed diff) + 4 box side-features ->
6 defect classes + background."""
from __future__ import annotations

import torch
from torch import nn

N_OUT = 7  # 6 DeepPCB classes + background


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CandNet(nn.Module):
    def __init__(self, width: int = 32, n_side: int = 4):
        super().__init__()
        self.features = nn.Sequential(block(3, width), nn.MaxPool2d(2), block(width, 2 * width), nn.MaxPool2d(2),
                                      block(2 * width, 4 * width), nn.MaxPool2d(2), block(4 * width, 4 * width),
                                      nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(4 * width + n_side, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
                                  nn.Linear(128, N_OUT))

    def forward(self, crops: torch.Tensor, side: torch.Tensor) -> torch.Tensor:
        x = crops.float() / 127.5 - 1.0
        return self.head(torch.cat([self.features(x), side], 1))


def dihedral(crops: torch.Tensor, side: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """One of the 8 square symmetries (defect classes are orientation-free). Transposes swap the
    width/height side-features."""
    if k & 4:
        crops = crops.transpose(-1, -2)
        side = side[:, [0, 1, 3, 2]]
    if k & 3:
        crops = torch.rot90(crops, k & 3, (-2, -1))
        if (k & 3) % 2:
            side = side[:, [0, 1, 3, 2]]
    return crops, side
