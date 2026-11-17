import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import pytest


def test_udiat_dataset_integrity():
    splits_path = Path("data/udiat/splits.json")
    assert splits_path.exists(), "Missing data/udiat/splits.json"
    with open(splits_path) as f:
        d = json.load(f)
    assert len(d["metadata"]) == 163, f"Expected 163 scans in UDIAT, got {len(d['metadata'])}"
    assert len(d["train"]) == 100
    assert len(d["val"]) == 23
    assert len(d["test"]) == 40

    # Check a sample mask is strictly binary {0, 255}
    sample_mask_p = Path("data/udiat/masks") / d["metadata"][0]["filename"]
    assert sample_mask_p.exists()
    mask = cv2.imread(str(sample_mask_p), cv2.IMREAD_GRAYSCALE)
    unique_vals = set(np.unique(mask))
    assert unique_vals.issubset({0, 255}), f"Mask values not strictly binary: {unique_vals}"


def test_literature_benchmarks_exist():
    csv_p = Path("results/literature_benchmark_comparison.csv")
    assert csv_p.exists(), "Missing results/literature_benchmark_comparison.csv"
    df = pd.read_csv(csv_p)
    assert len(df) >= 7, f"Expected at least 7 models in benchmark, got {len(df)}"
    assert "Shifaa U-Net (Online Pretrained)" in df["Model"].values
    assert "Our MobileNetV3 v2 (Rescued Mixed k=13)" in df["Model"].values


def test_visual_deliverables_exist():
    assert Path("figures/benchmark_overlay_comparison.png").exists()
    assert Path("figures/benchmark_overlay_normal_cases.png").exists()
    assert Path("figures/model_size_latency_comparison.png").exists()
    assert Path("figures/cine_realtime_demo.mp4").exists()
    assert Path("figures/cine_realtime_demo.gif").exists()
    assert Path("results/udiat_cross_center_results.csv").exists()


def test_buvfm_groundtruth_data_loader():
    from src.metrics.buvfm_eval import load_buvfm_val_data
    try:
        samples = load_buvfm_val_data()
    except FileNotFoundError as e:
        pytest.skip(f"BUVFM dataset not cached locally: {e}")
    assert len(samples) == 500, f"Expected 500 BUVFM validation cases, got {len(samples)}"
    lesion_cases = [s for s in samples if not s["is_normal"]]
    normal_cases = [s for s in samples if s["is_normal"]]
    assert len(lesion_cases) == 399, f"Expected 399 lesion cases, got {len(lesion_cases)}"
    assert len(normal_cases) == 101, f"Expected 101 normal cases, got {len(normal_cases)}"


def test_buvfm_benchmark_integrity():
    bench_p = Path("results/buvfm_groundtruth_benchmark.csv")
    assert bench_p.exists(), "Missing results/buvfm_groundtruth_benchmark.csv"
    df = pd.read_csv(bench_p)
    assert len(df) == 6, f"Expected 6 model rows, got {len(df)}"
    assert "r34_fp32" in df["model"].values
    assert "r34_int8" in df["model"].values
    assert "mbv3s_fp32" in df["model"].values
    assert "mbv3s_int8" in df["model"].values
    assert "mbv3s_mixed_k13" in df["model"].values
    assert "cascade_mixed_r34" in df["model"].values

    # Check key columns have valid values and no NaNs
    for col in ["bbox_iou_mean", "bbox_dice_mean", "mask_in_bbox_prec_mean", "edge_fps", "normal_specificity_pct"]:
        assert not df[col].isna().any(), f"NaN detected in {col}"

    # Verify that mixed precision recovers significantly over naive INT8
    mbv3s_int8_row = df[df["model"] == "mbv3s_int8"].iloc[0]
    mbv3s_mixed_row = df[df["model"] == "mbv3s_mixed_k13"].iloc[0]
    assert mbv3s_mixed_row["bbox_iou_mean"] > mbv3s_int8_row["bbox_iou_mean"] * 3.0, "Mixed precision must outperform naive INT8 by >3x"
    assert mbv3s_mixed_row["normal_specificity_pct"] > mbv3s_int8_row["normal_specificity_pct"], "Mixed precision must improve normal specificity over naive INT8"


def test_buvfm_5sample_deliverables():
    gif_p = Path("figures/cine_5sample_tracking_demo.gif")
    mp4_p = Path("figures/cine_5sample_tracking_demo.mp4")
    csv_p = Path("results/buvfm_video_tracking_metrics.csv")

    assert gif_p.exists(), "Missing figures/cine_5sample_tracking_demo.gif"
    assert gif_p.stat().st_size > 1_000_000, f"GIF file too small: {gif_p.stat().st_size}"
    assert mp4_p.exists(), "Missing figures/cine_5sample_tracking_demo.mp4"
    assert mp4_p.stat().st_size > 500_000, f"MP4 file too small: {mp4_p.stat().st_size}"
    assert csv_p.exists(), "Missing results/buvfm_video_tracking_metrics.csv"

    # Also verify the new 5x3 Ground Truth Showcase and All-Models Quantization Demo
    gif_5x3 = Path("figures/cine_5sample_5x3_tracking_demo.gif")
    mp4_5x3 = Path("figures/cine_5sample_5x3_tracking_demo.mp4")
    gif_quant = Path("figures/cine_all_models_quantization_comparison.gif")
    mp4_quant = Path("figures/cine_all_models_quantization_comparison.mp4")

    assert gif_5x3.exists() and gif_5x3.stat().st_size > 2_000_000, "Missing 5x3 tracking GIF"
    assert mp4_5x3.exists() and mp4_5x3.stat().st_size > 500_000, "Missing 5x3 tracking MP4"
    assert gif_quant.exists() and gif_quant.stat().st_size > 2_000_000, "Missing All-Models Quantization GIF"
    assert mp4_quant.exists() and mp4_quant.stat().st_size > 500_000, "Missing All-Models Quantization MP4"

    df_v = pd.read_csv(csv_p)
    assert len(df_v) == 5, f"Expected 5 videos, got {len(df_v)}"
    for col in ["tci_raw", "tci_ema", "edge_fps", "latency_ms"]:
        assert not df_v[col].isna().any(), f"NaN detected in {col}"
