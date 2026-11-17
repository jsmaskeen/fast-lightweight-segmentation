import json
from pathlib import Path
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

from src.bench.literature_compare import load_shifaa_pretrained
from src.metrics.dsc_iou_hd95 import compute_dsc_iou_hd95


def draw_contour_overlay(
    img_gray: np.ndarray,
    mask_bin: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.35,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Creates a transparent overlay of a binary mask over a grayscale image."""
    base_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = base_bgr.copy()

    # Fill mask
    mask_bool = mask_bin > 127
    overlay[mask_bool] = color
    blended = cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)

    # Draw sharp boundary contour
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, contour_thickness)
    return blended


def generate_benchmark_overlay_plot():
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_overlay_comparison.png"

    with open("data/splits.json") as f:
        test_ids = json.load(f)["test"]

    # Select 4 representative test cases with diverse morphology:
    # 1. Benign oval regular fibroadenoma
    # 2. Benign large complex lesion
    # 3. Malignant irregular lobulated tumor
    # 4. Malignant small aggressive microlobulated lesion
    benign_ids = [tid for tid in test_ids if tid.startswith("benign")]
    malignant_ids = [tid for tid in test_ids if tid.startswith("malignant")]

    selected_ids = [
        benign_ids[0],
        benign_ids[min(len(benign_ids) - 1, 12)],
        malignant_ids[0],
        malignant_ids[min(len(malignant_ids) - 1, 8)],
    ]
    print(f"Generating visual benchmark overlay for selected cases: {selected_ids}")

    # Load Models
    print("Loading models for visual comparison...")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    shifaa_model = load_shifaa_pretrained()

    # Sessions
    sess_r34_fp32 = ort.InferenceSession("models/fp32/r34_prep.onnx", providers=["CPUExecutionProvider"])
    sess_r34_int8 = ort.InferenceSession("models/int8/r34_int8_minmax.onnx", providers=["CPUExecutionProvider"])
    sess_mbv3s_fp32 = ort.InferenceSession("models/fp32/mbv3s_prep.onnx", providers=["CPUExecutionProvider"])
    sess_mbv3s_int8 = ort.InferenceSession("models/int8/mbv3s_int8_minmax.onnx", providers=["CPUExecutionProvider"])
    sess_mbv3s_mix = ort.InferenceSession("models/mixed/mbv3s_mixed_best.onnx", providers=["CPUExecutionProvider"])

    # Colors (BGR for OpenCV, RGB for Matplotlib)
    # 1: Raw scan (no overlay)
    # 2: Ground Truth (Green)
    # 3: Shifaa U-Net (Cyan)
    # 4: Attention U-Net (Orange)
    # 5: Our r34 FP32 (Royal Blue)
    # 6: Our r34 INT8 (Red)
    # 7: Our mbv3s FP32 (Gold)
    # 8: Our mbv3s Naive INT8 (Crimson/Pink)
    # 9: Our mbv3s Rescued Mixed k=13 (Bright Magenta)

    cols = [
        "Raw B-Mode Scan",
        "Ground Truth",
        "Shifaa U-Net\n(Online HF)",
        "Our ResNet-34\n(FP32)",
        "Our ResNet-34\n(INT8 QDQ)",
        "Our MobileNetV3\n(FP32)",
        "MobileNetV3 v1\n(Naive INT8)",
        "MobileNetV3 v2\n(Rescued Mixed)",
    ]

    n_rows = len(selected_ids)
    n_cols = len(cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(26, 14), gridspec_kw={"wspace": 0.08, "hspace": 0.22})

    for row_idx, case_id in enumerate(selected_ids):
        img_p = Path(f"data/processed/images/{case_id}.png")
        mask_p = Path(f"data/processed/masks/{case_id}.png")

        img_gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_mask > 127).astype(np.uint8) * 255

        # Standardized inputs
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        norm = (img_rgb.astype(np.float32) / 255.0 - mean) / std
        inp_ort = norm.transpose(2, 0, 1)[np.newaxis, ...]
        inp_shifaa = torch.from_numpy(img_gray.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

        # 1. Shifaa U-Net
        with torch.no_grad():
            shifaa_out = shifaa_model(inp_shifaa)
            shifaa_bin = (torch.sigmoid(shifaa_out)[0, 0].cpu().numpy() > 0.5).astype(np.uint8) * 255

        # 2. Our r34 FP32
        r34_fp32_logits = sess_r34_fp32.run(None, {"input": inp_ort})[0][0, 0]
        r34_fp32_bin = (r34_fp32_logits > 0.0).astype(np.uint8) * 255

        # 3. Our r34 INT8
        r34_int8_logits = sess_r34_int8.run(None, {"input": inp_ort})[0][0, 0]
        r34_int8_bin = (r34_int8_logits > 0.0).astype(np.uint8) * 255

        # 4. Our mbv3s FP32
        mbv3s_fp32_logits = sess_mbv3s_fp32.run(None, {"input": inp_ort})[0][0, 0]
        mbv3s_fp32_bin = (mbv3s_fp32_logits > 0.0).astype(np.uint8) * 255

        # 5. Our mbv3s Naive INT8
        mbv3s_int8_logits = sess_mbv3s_int8.run(None, {"input": inp_ort})[0][0, 0]
        mbv3s_int8_bin = (mbv3s_int8_logits > 0.0).astype(np.uint8) * 255

        # 6. Our mbv3s Rescued Mixed
        mbv3s_mix_logits = sess_mbv3s_mix.run(None, {"input": inp_ort})[0][0, 0]
        mbv3s_mix_bin = (mbv3s_mix_logits > 0.0).astype(np.uint8) * 255

        # Calculate DSCs for subtitle display
        d_gt = 1.000
        d_shifaa = compute_dsc_iou_hd95(shifaa_bin, gt_bin)["dsc"]
        d_r34_fp32 = compute_dsc_iou_hd95(r34_fp32_bin, gt_bin)["dsc"]
        d_r34_int8 = compute_dsc_iou_hd95(r34_int8_bin, gt_bin)["dsc"]
        d_mbv3s_fp32 = compute_dsc_iou_hd95(mbv3s_fp32_bin, gt_bin)["dsc"]
        d_mbv3s_int8 = compute_dsc_iou_hd95(mbv3s_int8_bin, gt_bin)["dsc"]
        d_mbv3s_mix = compute_dsc_iou_hd95(mbv3s_mix_bin, gt_bin)["dsc"]

        # Render Overlays (in RGB for matplotlib)
        # Raw scan
        panel_0 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        # Ground Truth: Green (0, 255, 0)
        panel_1 = cv2.cvtColor(draw_contour_overlay(img_gray, gt_bin, (0, 255, 0)), cv2.COLOR_BGR2RGB)
        # Shifaa: Cyan (0, 220, 255)
        panel_2 = cv2.cvtColor(draw_contour_overlay(img_gray, shifaa_bin, (255, 220, 0)), cv2.COLOR_BGR2RGB)
        # Our r34 FP32: Royal Blue (255, 120, 0)
        panel_3 = cv2.cvtColor(draw_contour_overlay(img_gray, r34_fp32_bin, (255, 120, 0)), cv2.COLOR_BGR2RGB)
        # Our r34 INT8: Bright Red (0, 0, 255)
        panel_4 = cv2.cvtColor(draw_contour_overlay(img_gray, r34_int8_bin, (0, 0, 255)), cv2.COLOR_BGR2RGB)
        # Our mbv3s FP32: Yellow/Amber (0, 215, 255)
        panel_5 = cv2.cvtColor(draw_contour_overlay(img_gray, mbv3s_fp32_bin, (0, 215, 255)), cv2.COLOR_BGR2RGB)
        # Our mbv3s Naive INT8: Gray/Red (0, 50, 200)
        panel_6 = cv2.cvtColor(draw_contour_overlay(img_gray, mbv3s_int8_bin, (0, 50, 200)), cv2.COLOR_BGR2RGB)
        # Our mbv3s Rescued Mixed: Magenta (255, 0, 255)
        panel_7 = cv2.cvtColor(draw_contour_overlay(img_gray, mbv3s_mix_bin, (255, 0, 220)), cv2.COLOR_BGR2RGB)

        panels = [
            (panel_0, None),
            (panel_1, "GT (1.000)"),
            (panel_2, f"DSC: {d_shifaa:.3f}"),
            (panel_3, f"DSC: {d_r34_fp32:.3f}"),
            (panel_4, f"DSC: {d_r34_int8:.3f}"),
            (panel_5, f"DSC: {d_mbv3s_fp32:.3f}"),
            (panel_6, f"DSC: {d_mbv3s_int8:.3f}"),
            (panel_7, f"DSC: {d_mbv3s_mix:.3f}"),
        ]

        # Case label
        case_title = f"Case {row_idx+1}: {case_id.replace('_', ' ').capitalize()}"

        for col_idx, (p_img, dsc_lbl) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(p_img)
            ax.set_xticks([])
            ax.set_yticks([])

            # Header on top row
            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=14.5, fontweight="bold", pad=10)

            # Subtitle / DSC label on bottom
            if dsc_lbl is not None:
                color = "green" if "✓" in dsc_lbl or "GT" in dsc_lbl else ("red" if "⚠" in dsc_lbl else "navy")
                ax.set_xlabel(dsc_lbl, fontsize=14.5, fontweight="semibold", color=color, labelpad=6)

            # Y-axis label on leftmost column
            if col_idx == 0:
                ax.set_ylabel(case_title, fontsize=15, fontweight="bold", labelpad=8)

    plt.suptitle(
        "Breast Ultrasound Tumor Segmentation: Comprehensive Visual Benchmark Comparison\n"
        "(Held-out BUSI Test Set Scans: Community Open Pretrained vs. Literature Baselines vs. Our FP32, INT8 & Rescued Mixed Models)",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Multi-panel visual overlay comparison successfully saved to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


if __name__ == "__main__":
    generate_benchmark_overlay_plot()
