"""Register a test image to its reference: coarse rotation + translation by Fourier methods, then
sub-pixel ECC refinement of a Euclidean warp.

All warps here map *test* pixel coordinates to *template* coordinates as a 2x3 affine matrix M:
    template(x') = test(x) with x' = M @ [x, y, 1]
so `cv2.warpAffine(test, M, size)` puts the test image in the template's frame.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import cv2
import numpy as np


def euclid(theta_deg: float, tx: float, ty: float, center: tuple[float, float]) -> np.ndarray:
    """Rotation by theta about `center`, then translation by (tx, ty). 2x3 float64."""
    M = cv2.getRotationMatrix2D(center, theta_deg, 1.0)
    M[:, 2] += (tx, ty)
    return M


def invert(M: np.ndarray) -> np.ndarray:
    return cv2.invertAffineTransform(M)


def corner_error(M_est: np.ndarray, M_true: np.ndarray, shape: tuple[int, int]) -> float:
    """Mean distance (px) between where the two warps send the four image corners."""
    h, w = shape
    pts = np.array([[0, 0, 1], [w - 1, 0, 1], [0, h - 1, 1], [w - 1, h - 1, 1]], float).T
    return float(np.mean(np.linalg.norm(M_est @ pts - M_true @ pts, axis=0)))


def _prep(img: np.ndarray) -> np.ndarray:
    """Float32, zero-mean, lightly blurred: binarized PCB edges are too sharp for ECC gradients."""
    f = cv2.GaussianBlur(img.astype(np.float32) / 255.0, (0, 0), 1.0)
    return f - f.mean()


def phase_translation(ref: np.ndarray, mov: np.ndarray) -> tuple[float, float, float]:
    """Sub-pixel shift (dx, dy) such that mov shifted by (dx, dy) matches ref, and the peak response."""
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(_prep(mov), _prep(ref), win)
    return dx, dy, resp


@functools.lru_cache(maxsize=4)
def _highpass(h: int, w: int) -> np.ndarray:
    """High-pass emphasis on the FFT magnitude (Reddy & Chatterji 1996), so the DC blob does not dominate."""
    yy, xx = np.mgrid[-1:1:complex(h), -1:1:complex(w)]
    c = np.cos(np.pi * xx) * np.cos(np.pi * yy)
    return ((1 - c) * (2 - c)).astype(np.float32)


def logpolar_rotation(ref: np.ndarray, mov: np.ndarray) -> float:
    """Rotation (deg, cv2 sign convention) of mov relative to ref from the log-polar transform of the
    FFT magnitudes (Fourier-Mellin). Translation-invariant, so it runs before the shift estimate."""
    h, w = ref.shape
    win = cv2.createHanningWindow((w, h), cv2.CV_32F)

    hp = _highpass(h, w)

    def mag(img):
        F = np.fft.fftshift(np.fft.fft2(_prep(img) * win))
        return np.log1p(np.abs(F)).astype(np.float32) * hp

    c = (w / 2.0, h / 2.0)
    radius = min(h, w) / 2.0
    n_ang = 1440  # 0.25 deg per row over 360 deg
    lp_r = cv2.warpPolar(mag(ref), (w, n_ang), c, radius, cv2.WARP_POLAR_LOG)
    lp_m = cv2.warpPolar(mag(mov), (w, n_ang), c, radius, cv2.WARP_POLAR_LOG)
    # only the radial band with structure; inner rows are DC, outer are aliasing
    band = slice(w // 8, w * 7 // 8)
    (_, dang), _ = cv2.phaseCorrelate(lp_m[:, band].copy(), lp_r[:, band].copy())
    theta = -dang * 360.0 / n_ang
    # the magnitude spectrum is 180-deg symmetric: fold into (-90, 90]
    while theta > 90:
        theta -= 180
    while theta <= -90:
        theta += 180
    return theta


def keypoint_warp(template: np.ndarray, test: np.ndarray, kind: str = "sift") -> np.ndarray:
    """The textbook alternative, as a baseline: SIFT or ORB keypoints, ratio-test matches, and a RANSAC
    rotation + translation (+ uniform scale) fit. Identity if too few matches survive."""
    det = cv2.SIFT_create(4000) if kind == "sift" else cv2.ORB_create(4000)
    k1, d1 = det.detectAndCompute(test, None)
    k2, d2 = det.detectAndCompute(template, None)
    if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
        return np.eye(2, 3)
    norm = cv2.NORM_L2 if kind == "sift" else cv2.NORM_HAMMING
    pairs = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
    if len(good) < 4:
        return np.eye(2, 3)
    src = np.float32([k1[m.queryIdx].pt for m in good])
    dst = np.float32([k2[m.trainIdx].pt for m in good])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=5000,
                                       confidence=0.999)
    return np.eye(2, 3) if M is None else M.astype(np.float64)


MIN_COVER = 0.5   # a warp that leaves less than this fraction of the frame covered is implausible
CONFIDENT = 0.5   # guarded result with overlap NCC below this is flagged for review


@dataclass
class RegResult:
    M: np.ndarray        # 2x3, test -> template
    ecc: float | None    # final ECC correlation, None if ECC was skipped or failed
    coarse: np.ndarray   # 2x3 warp before ECC
    ncc: float | None = None  # guarded only: overlap NCC of the chosen warp

    @property
    def confident(self) -> bool:
        return self.ncc is None or self.ncc >= CONFIDENT


def valid_mask(test: np.ndarray, M: np.ndarray, shape: tuple[int, int], margin: int = 8) -> np.ndarray:
    """Template-frame pixels that the warped test actually covers, eroded by `margin`."""
    h, w = shape
    valid = cv2.warpAffine(np.ones_like(test, np.uint8), M, (w, h), flags=cv2.INTER_NEAREST)
    return cv2.erode(valid, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)) > 0


def overlap_ncc(template: np.ndarray, test: np.ndarray, M: np.ndarray, region: np.ndarray | None = None) -> float:
    """Normalized cross-correlation between template and the warped test over `region` (default: the
    pixels the warp covers). The guard's figure of merit."""
    h, w = template.shape
    valid = valid_mask(test, M, (h, w)) if region is None else region
    if valid.sum() < 0.25 * h * w:
        return -1.0
    a = template.astype(np.float32)[valid]
    b = cv2.warpAffine(test.astype(np.float32), M, (w, h), flags=cv2.INTER_LINEAR)[valid]
    a -= a.mean()
    b -= b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else -1.0


def register(template: np.ndarray, test: np.ndarray, method: str = "guarded",
             ecc_iters: int = 100, ecc_eps: float = 1e-5, levels: int = 3) -> RegResult:
    """method: 'none' | 'phase' (translation only) | 'fm' (rotation + translation) | 'ecc' (ECC from
    identity) | 'phase+ecc' | 'fm+ecc' | 'guarded' | 'sift' / 'orb' (keypoints + RANSAC, baselines).

    'guarded' runs identity, ECC-from-identity, phase+ECC and FM+ECC and keeps whichever has the highest overlap
    NCC on their common overlap. Periodic pad arrays can pull the Fourier estimate one pitch off, and near-blank crops give it
    nothing to lock onto; the guard catches the first, and `RegResult.confident` flags the second
    (best NCC < CONFIDENT) for review instead of returning a confident wrong warp.
    """
    h, w = template.shape
    if method == "guarded":
        return _guarded(template, test, ecc_iters, ecc_eps, levels)
    c = (w / 2.0, h / 2.0)
    M = np.eye(2, 3)
    if method == "none":
        return RegResult(M, None, M.copy())
    if method in ("sift", "orb"):
        M = keypoint_warp(template, test, method)
        return RegResult(M, None, M.copy())

    if method.startswith("fm"):
        theta = logpolar_rotation(template, test)
        R = euclid(theta, 0, 0, c)
        rot = cv2.warpAffine(test, R, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dx, dy, _ = phase_translation(template, rot)
        M = R.copy()
        M[:, 2] += (dx, dy)
    elif method.startswith("phase"):
        dx, dy, _ = phase_translation(template, test)
        M = np.array([[1, 0, dx], [0, 1, dy]], float)
    coarse = M.copy()

    if not method.endswith("ecc"):
        return RegResult(M, None, coarse)

    W, cc = _ecc(_pyramid(template, test, levels), invert(M).astype(np.float32), levels - 1, 0, ecc_iters, ecc_eps)
    return RegResult(invert(W.astype(np.float64)), cc, coarse)


def _pyramid(template: np.ndarray, test: np.ndarray, levels: int) -> list[tuple[np.ndarray, np.ndarray]]:
    pyr = [(_prep(template), _prep(test))]
    for _ in range(levels - 1):
        pyr.append((cv2.pyrDown(pyr[-1][0]), cv2.pyrDown(pyr[-1][1])))
    return pyr


def _ecc(pyr, W: np.ndarray, top: int, bottom: int, iters: int, eps: float) -> tuple[np.ndarray, float | None]:
    """ECC coarse-to-fine from pyramid level `top` down to `bottom`. W is the full-resolution warp
    template -> test (findTransformECC's direction, the inverse of ours) and is returned the same way: the
    loop rescales the translation after every level it runs, so stopping at `bottom` > 0 still returns
    full-resolution coordinates (a level-`bottom` estimate, upsampled)."""
    W = W.astype(np.float32).copy()
    W[:, 2] /= 2 ** top
    cc = None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)
    for lvl in range(top, bottom - 1, -1):
        r, m = pyr[lvl]
        try:
            cc, W = cv2.findTransformECC(r, m, W, cv2.MOTION_EUCLIDEAN, crit, None, 5)
        except cv2.error:
            cc = None  # did not converge at this level: keep the current estimate
        if lvl:
            W[:, 2] *= 2
    return W, cc


SAME_PX = 0.5  # two warps closer than this (mean corner distance) are the same candidate


def _distinct(warps: list[np.ndarray], shape: tuple[int, int]) -> list[int]:
    keep = []
    for i, M in enumerate(warps):
        if all(corner_error(M, warps[j], shape) >= SAME_PX for j in keep):
            keep.append(i)
    return keep


def _guarded(template: np.ndarray, test: np.ndarray, iters: int, eps: float, levels: int) -> RegResult:
    """Identity, plus ECC from identity, from the phase-correlation shift and from the Fourier-Mellin
    rotation + shift; keep the candidate with the highest NCC on the region all plausible candidates cover.

    The three ECC chains share one pyramid. They run through the coarse levels separately, then chains that
    have converged to the same warp (within SAME_PX) share one full-resolution pass, which is most of the cost.
    """
    h, w = template.shape
    pyr = _pyramid(template, test, levels)
    # Fourier estimates stay at full resolution: at half resolution the log-polar spectrum is too coarse
    # to pin the rotation, and 29% of ±3 deg / ±20 px test warps ended > 2 px off (vs 6%)
    starts = [np.eye(2, 3), register(template, test, "phase").M, register(template, test, "fm").M]
    mid = [invert(_ecc(pyr, invert(M).astype(np.float32), levels - 1, 1, iters, eps)[0].astype(np.float64))
           for M in starts]
    fine = {}
    for i in _distinct(mid, (h, w)):
        W, cc = _ecc(pyr, invert(mid[i]).astype(np.float32), 0, 0, iters, eps)
        fine[i] = RegResult(invert(W.astype(np.float64)), cc, starts[i])
    rep = {i: next(j for j in fine if corner_error(mid[i], mid[j], (h, w)) < SAME_PX) for i in range(len(mid))}
    cands = [RegResult(np.eye(2, 3), None, np.eye(2, 3))] + [fine[rep[i]] for i in range(len(starts))]
    # drop implausible warps, then score the rest on the same pixels: the region all of them cover.
    # Otherwise identity, which covers the whole frame, is compared over a different (larger) area
    # than a real warp
    masks = [valid_mask(test, r.M, (h, w)) for r in cands]
    keep = [i for i, m in enumerate(masks) if m.mean() >= MIN_COVER] or [0]
    common = np.logical_and.reduce([masks[i] for i in keep])
    scores = [overlap_ncc(template, test, cands[i].M, common) for i in keep]
    best = cands[keep[int(np.argmax(scores))]]
    return RegResult(best.M, best.ecc, best.coarse, float(max(scores)))


def apply(test: np.ndarray, M: np.ndarray, shape: tuple[int, int], border: int = 255) -> np.ndarray:
    """Warp test into the template frame. DeepPCB backgrounds are white (255)."""
    h, w = shape
    return cv2.warpAffine(test, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=border)
