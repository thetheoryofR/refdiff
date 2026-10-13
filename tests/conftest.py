import numpy as np
import pytest


@pytest.fixture
def board():
    """A PCB-like 256x256 image: dark copper traces and pads on a light background (DeepPCB polarity)."""
    rng = np.random.default_rng(1)
    img = np.full((256, 256), 230, np.uint8)
    for _ in range(14):
        x, y = rng.integers(10, 220, 2)
        if rng.random() < 0.5:
            img[y:y + 6, x:min(250, x + rng.integers(30, 120))] = 20
        else:
            img[y:min(250, y + rng.integers(30, 120)), x:x + 6] = 20
    for _ in range(10):
        x, y = rng.integers(10, 235, 2)
        img[y:y + 12, x:x + 12] = 20
    return img
