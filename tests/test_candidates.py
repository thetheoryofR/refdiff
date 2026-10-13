import numpy as np

from refdiff import candidates as C
from refdiff import synth


def test_identical_images_have_no_candidates(board):
    assert len(C.candidates(board, board.copy())) == 0


def test_extra_and_missing_copper_get_signs(board):
    test = board.copy()
    test[100:110, 30:40] = 20   # spurious copper on background
    ys, xs = np.nonzero(board[:, :] == 20)
    y, x = ys[len(ys) // 2], xs[len(xs) // 2]
    test[y - 3:y + 3, x - 3:x + 3] = 230  # hole punched through existing copper
    c = C.candidates(test, board)
    signs = sorted(c[:, 4].tolist())
    assert signs == [-1, 1]
    extra = c[c[:, 4] == 1][0]
    assert extra[0] <= 30 and extra[2] >= 40 and extra[1] <= 100 and extra[3] >= 110


def test_min_side_grows_small_boxes(board):
    test = board.copy()
    test[100:104, 30:34] = 20
    c = C.candidates(test, board, C.Params(min_area=4, open_k=1, min_side=30, pad=0))
    assert len(c) == 1 and c[0, 2] - c[0, 0] >= 30 and c[0, 3] - c[0, 1] >= 30


def test_defective_template_pastes_only_boxes(board):
    test = np.zeros_like(board)
    out = synth.defective_template(test, board, np.array([[10, 10, 20, 20, 0]]))
    assert (out[10:20, 10:20] == 0).all()
    assert (out[30:, 30:] == board[30:, 30:]).all()


def test_layout_rasterize_and_inject_labels_changed_pixels():
    from refdiff import layout as L
    rng = np.random.default_rng(3)
    db = L.rasterize(L.make_layout(rng))
    assert db.shape == (640, 640) and 0.03 < (db < 125).mean() < 0.5
    out, boxes = L.inject(db, rng)
    assert len(boxes) >= 1
    for x1, y1, x2, y2, _ in boxes:
        assert ((out[y1:y2, x1:x2] < 125) != (db[y1:y2, x1:x2] < 125)).any()
    outside = np.ones(db.shape, bool)
    for x1, y1, x2, y2, _ in boxes:
        outside[y1:y2, x1:x2] = False
    assert ((out < 125) == (db < 125))[outside].all()
