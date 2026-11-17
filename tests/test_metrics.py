import numpy as np

from src.metrics.dsc_iou_hd95 import compute_dsc_iou_hd95, compute_normal_fpr
from src.metrics.bootstrap import bootstrap_ci, paired_bootstrap_ci


def test_identical_masks():
    m = np.zeros((256, 256), dtype=np.uint8)
    m[50:100, 50:100] = 255
    res = compute_dsc_iou_hd95(m, m)
    assert res["dsc"] == 1.0
    assert res["iou"] == 1.0
    assert res["hd95"] == 0.0
    assert res["is_empty_pred"] == 0.0


def test_disjoint_masks():
    m1 = np.zeros((256, 256), dtype=np.uint8)
    m2 = np.zeros((256, 256), dtype=np.uint8)
    m1[10:30, 10:30] = 255
    m2[100:120, 100:120] = 255
    res = compute_dsc_iou_hd95(m1, m2)
    assert res["dsc"] == 0.0
    assert res["iou"] == 0.0
    assert res["hd95"] > 50.0  # High distance


def test_empty_prediction_convention():
    m_empty = np.zeros((256, 256), dtype=np.uint8)
    m_gt = np.zeros((256, 256), dtype=np.uint8)
    m_gt[50:100, 50:100] = 255

    res = compute_dsc_iou_hd95(m_empty, m_gt)
    assert res["dsc"] == 0.0
    assert res["iou"] == 0.0
    assert res["is_empty_pred"] == 1.0
    # Fixed diagonal penalty convention: 256 * sqrt(2) ~ 362.0387
    expected_diag = 256.0 * np.sqrt(2.0)
    assert np.isclose(res["hd95"], expected_diag, atol=1e-3)


def test_both_empty_convention():
    m_empty1 = np.zeros((256, 256), dtype=np.uint8)
    m_empty2 = np.zeros((256, 256), dtype=np.uint8)

    res = compute_dsc_iou_hd95(m_empty1, m_empty2)
    assert res["dsc"] == 1.0
    assert res["iou"] == 1.0
    assert np.isnan(res["hd95"])  # Excluded from HD95 aggregation


def test_normal_image_fpr():
    m_clean = np.zeros((256, 256), dtype=np.uint8)
    assert compute_normal_fpr(m_clean)["fp_50px"] == 0.0
    assert compute_normal_fpr(m_clean)["fp_0px"] == 0.0

    m_small_noise = np.zeros((256, 256), dtype=np.uint8)
    m_small_noise[10:15, 10:15] = 255  # 25 px
    assert compute_normal_fpr(m_small_noise)["fp_50px"] == 0.0
    assert compute_normal_fpr(m_small_noise)["fp_0px"] == 1.0

    m_lesion_noise = np.zeros((256, 256), dtype=np.uint8)
    m_lesion_noise[10:20, 10:20] = 255  # 100 px
    assert compute_normal_fpr(m_lesion_noise)["fp_50px"] == 1.0
    assert compute_normal_fpr(m_lesion_noise)["fp_0px"] == 1.0


def test_bootstrap_ci():
    vals = np.array([0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.82])
    ci = bootstrap_ci(vals, n_resamples=500, seed=42)
    assert ci["n_samples"] == 7
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
    assert 0.70 <= ci["mean"] <= 0.82


def test_paired_bootstrap_ci():
    a = np.array([0.80, 0.82, 0.85, 0.78])
    b = np.array([0.75, 0.80, 0.82, 0.76])
    ci = paired_bootstrap_ci(a, b, n_resamples=500, seed=42)
    assert ci["mean"] > 0.0
    assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]
