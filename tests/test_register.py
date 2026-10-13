import numpy as np
import pytest

from refdiff import register as R
from refdiff import synth


def test_corner_error_identity_zero():
    assert R.corner_error(np.eye(2, 3), np.eye(2, 3), (64, 64)) == 0.0


@pytest.mark.parametrize("method", ["ecc", "phase+ecc", "fm+ecc", "guarded"])
def test_recovers_subpixel_shift(board, method):
    M = R.euclid(0.0, 3.4, -2.7, (128, 128))
    mov = R.apply(board, M, board.shape, border=230)
    est = R.register(board, mov, method).M
    assert R.corner_error(est, R.invert(M), board.shape) < 0.1


@pytest.mark.parametrize("method", ["fm+ecc", "guarded"])
def test_recovers_rotation_and_shift(board, method):
    M = R.euclid(2.2, -7.3, 5.1, (128, 128))
    mov = R.apply(board, M, board.shape, border=230)
    est = R.register(board, mov, method).M
    assert R.corner_error(est, R.invert(M), board.shape) < 0.2


def test_phase_only_misses_rotation(board):
    M = R.euclid(3.0, 0, 0, (128, 128))
    mov = R.apply(board, M, board.shape, border=230)
    assert R.corner_error(R.register(board, mov, "phase").M, R.invert(M), board.shape) > 2


def test_guard_keeps_identity_for_identical_images(board):
    assert R.corner_error(R.register(board, board.copy(), "guarded").M, np.eye(2, 3), board.shape) < 0.05


def test_misregister_returns_inverse(board):
    rng = np.random.default_rng(0)
    mov, Minv = synth.misregister(board, rng, optics=False)
    back = R.apply(mov, Minv, board.shape, border=230)
    inner = (slice(40, -40), slice(40, -40))
    assert np.mean(np.abs(back[inner].astype(int) - board[inner])) < 6


@pytest.mark.parametrize("method", ["sift", "orb"])
def test_keypoint_baselines_recover_a_warp(board, method):
    M = R.euclid(2.0, 6.0, -4.0, (128, 128))
    mov = R.apply(board, M, board.shape, border=230)
    assert R.corner_error(R.register(board, mov, method).M, R.invert(M), board.shape) < 1.5
