import json
from pathlib import Path
from typing import Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch

from src.bench.literature_compare import load_shifaa_pretrained


def draw_contour_overlay(
    img_gray: np.ndarray,
    mask_bin: np.ndarray,
    color: Tuple[int, int, int] = (0, 0, 255),
    alpha: float = 0.40,
    contour_thickness: int = 2,
) -> np.ndarray:
    """Creates a transparent overlay of a binary mask over a grayscale image."""
    base_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    overlay = base_bgr.copy()

    mask_bool = mask_bin > 127
    if not np.any(mask_bool):
        return base_bgr

    overlay[mask_bool] = color
    blended = cv2.addWeighted(overlay, alpha, base_bgr, 1 - alpha, 0)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, color, contour_thickness)
    return blended


def generate_normal_cases_overlay_plot():
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benchmark_overlay_normal_cases.png"

    with open("data/splits.json") as f:
        test_ids = json.load(f)["test"]

    normal_ids = [tid for tid in test_ids if tid.startswith("normal")]
    selected_ids = [
        normal_ids[0],
        normal_ids[min(len(normal_ids) - 1, 10)],
        normal_ids[min(len(normal_ids) - 1, 25)],
        normal_ids[min(len(normal_ids) - 1, 45)],
    ]
    print(f"Generating normal cases visual benchmark overlay for: {selected_ids}")

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    shifaa_model = load_shifaa_pretrained()

    sess_r34_fp32 = ort.InferenceSession("models/fp32/r34_prep.onnx", providers=["CPUExecutionProvider"])
    sess_r34_int8 = ort.InferenceSession("models/int8/r34_int8_minmax.onnx", providers=["CPUExecutionProvider"])
    sess_mbv3s_fp32 = ort.InferenceSession("models/fp32/mbv3s_prep.onnx", providers=["CPUExecutionProvider"])
    sess_mbv3s_v1_int8 = ort.InferenceSession("models/int8/mbv3s_int8_minmax.onnx", providers=["CPUExecutionProvider"])
    sess_mbv3s_v2_mix = ort.InferenceSession("models/mixed/mbv3s_mixed_best.onnx", providers=["CPUExecutionProvider"])

    cols = [
        "Raw B-Mode Scan\n(Healthy Tissue)",
        "Ground Truth\n(Normal / Empty)",
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
        gt_bin = (gt_mask > 127).astype(np.uint8) * 255  # Always 0 for normal

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

        # 5. Our mbv3s v1 (Naive INT8)
        mbv3s_v1_logits = sess_mbv3s_v1_int8.run(None, {"input": inp_ort})[0][0, 0]
        mbv3s_v1_bin = (mbv3s_v1_logits > 0.0).astype(np.uint8) * 255

        # 6. Our mbv3s v2 (Rescued Mixed)
        mbv3s_v2_logits = sess_mbv3s_v2_mix.run(None, {"input": inp_ort})[0][0, 0]
        mbv3s_v2_bin = (mbv3s_v2_logits > 0.0).astype(np.uint8) * 255

        # FP pixel counts
        fp_gt = 0
        fp_shifaa = int(np.sum(shifaa_bin > 127))
        fp_r34_fp32 = int(np.sum(r34_fp32_bin > 127))
        fp_r34_int8 = int(np.sum(r34_int8_bin > 127))
        fp_mbv3s_fp32 = int(np.sum(mbv3s_fp32_bin > 127))
        fp_mbv3s_v1 = int(np.sum(mbv3s_v1_bin > 127))
        fp_mbv3s_v2 = int(np.sum(mbv3s_v2_bin > 127))

        # Render Overlays
        panel_0 = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        panel_1 = cv2.cvtColor(draw_contour_overlay(img_gray, gt_bin, (0, 255, 0)), cv2.COLOR_BGR2RGB)
        panel_2 = cv2.cvtColor(draw_contour_overlay(img_gray, shifaa_bin, (255, 0, 0)), cv2.COLOR_BGR2RGB)
        panel_3 = cv2.cvtColor(draw_contour_overlay(img_gray, r34_fp32_bin, (255, 120, 0)), cv2.COLOR_BGR2RGB)
        panel_4 = cv2.cvtColor(draw_contour_overlay(img_gray, r34_int8_bin, (0, 0, 255)), cv2.COLOR_BGR2RGB)
        panel_5 = cv2.cvtColor(draw_contour_overlay(img_gray, mbv3s_fp32_bin, (0, 215, 255)), cv2.COLOR_BGR2RGB)
        panel_6 = cv2.cvtColor(draw_contour_overlay(img_gray, mbv3s_v1_bin, (0, 0, 255)), cv2.COLOR_BGR2RGB)
        panel_7 = cv2.cvtColor(draw_contour_overlay(img_gray, mbv3s_v2_bin, (0, 255, 50)), cv2.COLOR_BGR2RGB)

        def make_lbl(fp_count):
            if fp_count <= 50:
                return f"FP: {fp_count} px (Clean ✓)", "green"
            else:
                return f"FP: {fp_count:,} px (Alarm ⚠)", "red"

        panels = [
            (panel_0, None, "black"),
            (panel_1, "GT: 0 px (Clean ✓)", "green"),
            (panel_2, *make_lbl(fp_shifaa)),
            (panel_3, *make_lbl(fp_r34_fp32)),
            (panel_4, *make_lbl(fp_r34_int8)),
            (panel_5, *make_lbl(fp_mbv3s_fp32)),
            (panel_6, *make_lbl(fp_mbv3s_v1)),
            (panel_7, *make_lbl(fp_mbv3s_v2)),
        ]

        case_title = f"Case {row_idx+1}: {case_id.replace('_', ' ').capitalize()}"

        for col_idx, (p_img, fp_lbl, lbl_color) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            ax.imshow(p_img)
            ax.set_xticks([])
            ax.set_yticks([])

            if row_idx == 0:
                ax.set_title(cols[col_idx], fontsize=14, fontweight="bold", pad=10)

            if fp_lbl is not None:
                ax.set_xlabel(fp_lbl, fontsize=12.5, fontweight="semibold", color=lbl_color, labelpad=6)

            if col_idx == 0:
                ax.set_ylabel(case_title, fontsize=15, fontweight="bold", labelpad=8)

    plt.suptitle(
        "Normal Tissue (Healthy Breast) Evaluation: False Positive & Specificity Audit\n"
        "(Healthy tissue without lesions: Specificity and False Alarm Audit across FP32, INT8 & Rescued Models)",
        fontsize=20,
        fontweight="bold",
        y=0.99,
    )
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Normal cases visual benchmark overlay successfully saved to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


if __name__ == "__main__":
    generate_normal_cases_overlay_plot()
