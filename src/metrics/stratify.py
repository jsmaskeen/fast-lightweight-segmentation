import json
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion

from src.metrics.bootstrap import bootstrap_ci


def compute_boundary_sharpness(img_gray: np.ndarray, mask: np.ndarray) -> float:
    """Mean Sobel gradient magnitude along the ground-truth contour."""
    b = (mask > 0) ^ binary_erosion(mask > 0)
    if not np.any(b):
        return 0.0
    gx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return float(np.mean(mag[b]))


def compute_training_terciles(
    train_ids: List[str],
    data_dir: str = "data/processed"
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Compute cut points for lesion area and boundary sharpness on the TRAINING set only.
    Returns: ((area_t1, area_t2), (sharpness_t1, sharpness_t2))
    """
    areas = []
    sharpnesses = []
    img_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"

    for img_id in train_ids:
        if "normal" in img_id:
            continue
        mask = cv2.imread(str(mask_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(str(img_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        if mask is None or img is None:
            continue

        area = int(np.count_nonzero(mask > 127))
        sharp = compute_boundary_sharpness(img, mask)
        areas.append(area)
        sharpnesses.append(sharp)

    area_cut1, area_cut2 = float(np.percentile(areas, 33.33)), float(np.percentile(areas, 66.67))
    sharp_cut1, sharp_cut2 = float(np.percentile(sharpnesses, 33.33)), float(np.percentile(sharpnesses, 66.67))

    print(f"Training terciles computed from {len(areas)} lesion scans:")
    print(f"  Area Terciles (px): Small < {area_cut1:.1f} <= Medium < {area_cut2:.1f} <= Large")
    print(f"  Sharpness Terciles: Soft < {sharp_cut1:.1f} <= Medium < {sharp_cut2:.1f} <= Sharp")

    return (area_cut1, area_cut2), (sharp_cut1, sharp_cut2)


def assign_strata(
    df: pd.DataFrame,
    area_cuts: Tuple[float, float],
    sharp_cuts: Tuple[float, float],
    data_dir: str = "data/processed"
) -> pd.DataFrame:
    """Assign class, area stratum, and sharpness stratum to each test evaluation row."""
    img_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"

    area_t1, area_t2 = area_cuts
    sharp_t1, sharp_t2 = sharp_cuts

    area_strata = []
    sharp_strata = []

    for _, row in df.iterrows():
        img_id = row["image_id"]
        if row["class"] == "normal":
            area_strata.append("normal")
            sharp_strata.append("normal")
            continue

        area = row["gt_area_px"]
        if area < area_t1:
            a_label = "Small"
        elif area < area_t2:
            a_label = "Medium"
        else:
            a_label = "Large"
        area_strata.append(a_label)

        # Compute sharpness
        img = cv2.imread(str(img_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_dir / f"{img_id}.png"), cv2.IMREAD_GRAYSCALE)
        sharp = compute_boundary_sharpness(img, mask)

        if sharp < sharp_t1:
            s_label = "Soft"
        elif sharp < sharp_t2:
            s_label = "Medium"
        else:
            s_label = "Sharp"
        sharp_strata.append(s_label)

    df["area_stratum"] = area_strata
    df["sharpness_stratum"] = sharp_strata
    return df


def analyze_stratified_damage(
    fp32_csv: str,
    int8_csv: str,
    mixed_csv: Optional[str] = None,
    splits_file: str = "data/splits.json",
    data_dir: str = "data/processed"
) -> pd.DataFrame:
    """
    Mini-Study S1: Computes paired ΔDice and ΔHD95 (INT8 and Mixed minus FP32) across strata:
    - Pathology class (benign, malignant)
    - Lesion size (Small, Medium, Large)
    - Boundary sharpness (Soft, Medium, Sharp)
    """
    with open(splits_file, "r") as f:
        splits = json.load(f)

    area_cuts, sharp_cuts = compute_training_terciles(splits["train"], data_dir=data_dir)

    df_fp32 = pd.read_csv(fp32_csv)
    df_int8 = pd.read_csv(int8_csv)

    # Filter to lesion test images only
    df_fp32 = df_fp32[df_fp32["class"].isin(["benign", "malignant"])].copy()
    df_int8 = df_int8[df_int8["class"].isin(["benign", "malignant"])].copy()

    df_fp32 = assign_strata(df_fp32, area_cuts, sharp_cuts, data_dir=data_dir)
    df_int8 = assign_strata(df_int8, area_cuts, sharp_cuts, data_dir=data_dir)

    merged = pd.merge(df_fp32, df_int8, on="image_id", suffixes=("_fp32", "_int8"))
    merged["delta_dsc_int8"] = merged["dsc_int8"] - merged["dsc_fp32"]
    merged["delta_hd95_int8"] = merged["hd95_int8"] - merged["hd95_fp32"]

    has_mixed = False
    if mixed_csv is not None and Path(mixed_csv).exists():
        df_mixed = pd.read_csv(mixed_csv)
        df_mixed = df_mixed[df_mixed["class"].isin(["benign", "malignant"])].copy()
        df_mixed = assign_strata(df_mixed, area_cuts, sharp_cuts, data_dir=data_dir)
        df_mixed = df_mixed[["image_id", "dsc", "hd95", "is_empty_pred"]].rename(
            columns={"dsc": "dsc_mixed", "hd95": "hd95_mixed", "is_empty_pred": "is_empty_pred_mixed"}
        )
        merged = pd.merge(merged, df_mixed, on="image_id")
        merged["delta_dsc_mixed"] = merged["dsc_mixed"] - merged["dsc_fp32"]
        merged["delta_hd95_mixed"] = merged["hd95_mixed"] - merged["hd95_fp32"]
        has_mixed = True

    results = []

    def summarize_group(grp_df, group_type, group_name):
        n = len(grp_df)
        d_dice_int8 = bootstrap_ci(grp_df["delta_dsc_int8"].values, n_resamples=1000, seed=42)
        d_hd95_int8 = bootstrap_ci(grp_df["delta_hd95_int8"].values, n_resamples=1000, seed=42)
        empty_fp32 = int(grp_df["is_empty_pred_fp32"].sum())
        empty_int8 = int(grp_df["is_empty_pred_int8"].sum())

        row = {
            "Stratum Type": group_type,
            "Stratum": group_name,
            "N": n,
            "Mean ΔDice (INT8)": f"{d_dice_int8['mean']:.4f}",
            "ΔDice 95% CI (INT8)": f"[{d_dice_int8['ci_lower']:.4f}, {d_dice_int8['ci_upper']:.4f}]",
            "Mean ΔHD95 (INT8)": f"{d_hd95_int8['mean']:.2f}",
            "Empty Pred (FP32/INT8)": f"{empty_fp32} / {empty_int8}",
        }
        if has_mixed:
            d_dice_mix = bootstrap_ci(grp_df["delta_dsc_mixed"].values, n_resamples=1000, seed=42)
            d_hd95_mix = bootstrap_ci(grp_df["delta_hd95_mixed"].values, n_resamples=1000, seed=42)
            row["Mean ΔDice (Mixed)"] = f"{d_dice_mix['mean']:.4f}"
            row["Mean ΔHD95 (Mixed)"] = f"{d_hd95_mix['mean']:.2f}"
            row["Empty Pred (Mixed)"] = int(grp_df["is_empty_pred_mixed"].sum())

        results.append(row)

    # Overall
    summarize_group(merged, "Overall", "All Lesions")

    # Pathology Class
    for cls in ["benign", "malignant"]:
        summarize_group(merged[merged["class_fp32"] == cls], "Class", cls)

    # Size terciles
    for size in ["Small", "Medium", "Large"]:
        summarize_group(merged[merged["area_stratum_fp32"] == size], "Area Tercile", size)

    # Sharpness terciles
    for sharp in ["Soft", "Medium", "Sharp"]:
        summarize_group(merged[merged["sharpness_stratum_fp32"] == sharp], "Sharpness Tercile", sharp)

    report_df = pd.DataFrame(results)
    out_path = Path("results/study_s1_stratified_damage.csv")
    report_df.to_csv(out_path, index=False)
    print(f"\n--- Mini-Study S1 Results saved to {out_path} ---")
    print(report_df.to_string(index=False))
    return report_df


if __name__ == "__main__":
    fp32_path = "results/eval_test_mbv3s_fp32.csv"
    int8_path = "results/eval_test_mbv3s_int8_minmax.csv"
    mixed_path = "results/eval_test_mbv3s_mixed_k13.csv"
    if not Path(mixed_path).exists():
        mixed_cands = list(Path("results").glob("eval_test_mbv3s_mixed_*.csv"))
        mixed_path = str(mixed_cands[0]) if mixed_cands else None
    if Path(fp32_path).exists() and Path(int8_path).exists():
        analyze_stratified_damage(fp32_path, int8_path, mixed_csv=mixed_path)
    else:
        print("Waiting for FP32 and INT8 evaluations to run S1.")
