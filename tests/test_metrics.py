import numpy as np

from refdiff import metrics


def test_iou_basic():
    assert metrics.iou([[0, 0, 10, 10]], [[0, 0, 10, 10]])[0, 0] == 1.0
    assert metrics.iou([[0, 0, 10, 10]], [[20, 20, 30, 30]])[0, 0] == 0.0
    assert abs(metrics.iou([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0] - 1 / 3) < 1e-9


def test_perfect_detections_score_one():
    gts = {"a": np.array([[0, 0, 10, 10, 0], [20, 20, 40, 40, 1]])}
    dets = {"a": np.array([[0, 0, 10, 10, 0.9, 0], [20, 20, 40, 40, 0.8, 1]])}
    ev = metrics.evaluate(gts, dets, 2)
    assert ev["mAP"] == 1.0 and ev["precision"] == 1.0 and ev["recall"] == 1.0


def test_iou_threshold_is_strict_033():
    gts = {"a": np.array([[0, 0, 10, 10, 0]])}
    # IoU exactly 1/3 is not a hit (DeepPCB: "larger than 0.33" -> we use > 0.33)
    ev = metrics.evaluate(gts, {"a": np.array([[5, 0, 15, 10, 0.9, 0]])}, 1)
    assert ev["mAP"] == 1.0  # 0.3333 > 0.33
    ev = metrics.evaluate(gts, {"a": np.array([[6, 0, 16, 10, 0.9, 0]])}, 1)  # IoU 0.25
    assert ev["mAP"] == 0.0


def test_duplicate_and_wrong_class_are_false_positives():
    gts = {"a": np.array([[0, 0, 10, 10, 0]])}
    dets = {"a": np.array([[0, 0, 10, 10, 0.9, 0], [0, 0, 10, 10, 0.8, 0], [0, 0, 10, 10, 0.95, 1]])}
    ev = metrics.evaluate(gts, dets, 2)
    assert ev["per_class"][0]["precision"] == 0.5
    assert ev["per_class"][0]["ap"] == 1.0  # the higher-scored duplicate is the hit
    assert ev["per_class"][1]["ap"] == 0.0


def test_match_agrees_with_evaluate_counts():
    rng = np.random.default_rng(0)
    gts, dets = {}, {}
    for k in range(30):
        xy = rng.integers(0, 600, (5, 2))
        g = np.c_[xy, xy + rng.integers(8, 40, (5, 2)), rng.integers(0, 3, 5)]
        j = rng.normal(0, 4, (8, 4))
        src = g[rng.integers(0, 5, 8)]
        d = np.c_[src[:, :4] + j, rng.random(8), np.where(rng.random(8) < 0.8, src[:, 4], rng.integers(0, 3, 8))]
        gts[str(k)], dets[str(k)] = g, d
    ev = metrics.evaluate(gts, dets, 3)
    for c in range(3):
        tp = sum(int((metrics.match(gts[k], dets[k])[1] & (dets[k][:, 5] == c)).sum()) for k in gts)
        gt_tp = sum(int((metrics.match(gts[k], dets[k])[0] & (gts[k][:, 4] == c)).sum()) for k in gts)
        pc = ev["per_class"][c]
        assert tp == gt_tp
        assert abs(tp / pc["n_gt"] - pc["recall"]) < 1e-12
