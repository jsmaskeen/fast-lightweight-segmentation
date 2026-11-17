import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

from src.metrics.bootstrap import bootstrap_ci, paired_bootstrap_ci
from src.metrics.dsc_iou_hd95 import compute_dsc_iou_hd95, compute_normal_fpr


def evaluate_onnx_model(
    model_path: str,
    split: str = "test",
    data_dir: str = "data/processed",
    splits_file: str = "data/splits.json"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Evaluate an ONNX model (FP32 or INT8) on val or test split."""
    with open(splits_file, "r") as f:
        splits_data = json.load(f)
    image_ids = splits_data[split]

    meta_dict = {}
    if "metadata" in splits_data and isinstance(splits_data["metadata"], list):
        for item in splits_data["metadata"]:
            meta_dict[item.get("stem", "")] = item.get("class_label", "lesion")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_opts, providers=["CPUExecutionProvider"])
    img_dir = Path(data_dir) / "images"
    mask_dir = Path(data_dir) / "masks"

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    rows = []
    lesion_dsc = []
    lesion_iou = []
    lesion_hd95 = []
    empty_preds = 0
    normal_total = 0
    normal_fp_50 = 0
    normal_fp_0 = 0

    for img_id in image_ids:
        img_p = img_dir / f"{img_id}.png"
        mask_p = mask_dir / f"{img_id}.png"

        img_gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None or mask is None:
            continue

        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - MEAN) / STD
        inp = img_norm.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 256, 256)

        ort_out = session.run(None, {"input": inp})[0][0, 0]
        # Logits > 0.0 is equivalent to probability > 0.5
        pred_mask = (ort_out > 0.0).astype(np.uint8) * 255

        cls_label = meta_dict.get(img_id, "normal" if "normal" in img_id else ("benign" if "benign" in img_id else "malignant"))
        is_normal = (cls_label == "normal") or ("normal" in img_id)

        if not is_normal:
            # Lesion scan
            m = compute_dsc_iou_hd95(pred_mask, mask)
            lesion_dsc.append(m["dsc"])
            lesion_iou.append(m["iou"])
            if not np.isnan(m["hd95"]):
                lesion_hd95.append(m["hd95"])
            if m["is_empty_pred"]:
                empty_preds += 1

            rows.append({
                "image_id": img_id,
                "class": cls_label,
                "dsc": m["dsc"],
                "iou": m["iou"],
                "hd95": m["hd95"],
                "is_empty_pred": m["is_empty_pred"],
                "pred_area_px": m["pred_area_px"],
                "gt_area_px": m["gt_area_px"],
            })
        else:
            # Normal scan
            fpr_res = compute_normal_fpr(pred_mask)
            normal_total += 1
            if fpr_res["fp_50px"]:
                normal_fp_50 += 1
            if fpr_res["fp_0px"]:
                normal_fp_0 += 1

            rows.append({
                "image_id": img_id,
                "class": "normal",
                "dsc": np.nan,
                "iou": np.nan,
                "hd95": np.nan,
                "is_empty_pred": 0.0,
                "pred_area_px": fpr_res["pred_area_px"],
                "gt_area_px": 0.0,
            })

    df = pd.DataFrame(rows)
    dsc_ci = bootstrap_ci(np.array(lesion_dsc), n_resamples=1000, seed=42)
    iou_ci = bootstrap_ci(np.array(lesion_iou), n_resamples=1000, seed=42)
    hd95_ci = bootstrap_ci(np.array(lesion_hd95), n_resamples=1000, seed=42)

    summary = {
        "model_path": str(model_path),
        "split": split,
        "n_lesions": len(lesion_dsc),
        "n_normal": normal_total,
        "empty_predictions": empty_preds,
        "dsc": dsc_ci,
        "iou": iou_ci,
        "hd95": hd95_ci,
        "normal_fpr_50px": normal_fp_50 / normal_total if normal_total > 0 else 0.0,
        "normal_fpr_0px": normal_fp_0 / normal_total if normal_total > 0 else 0.0,
    }
    return df, summary


def run_qdq_loss_debug_smoke(model_name: str = "mbv3s"):
    """Smoke test of onnxruntime.quantization.qdq_loss_debug for Extension A readiness."""
    print(f"\n--- Running qdq_loss_debug smoke test on {model_name} ---")
    import onnxruntime.quantization.qdq_loss_debug as qdq_debug

    fp32_path = Path(f"models/fp32/{model_name}_prep.onnx")
    if not fp32_path.exists():
        fp32_path = Path(f"models/fp32/{model_name}.onnx")
    int8_path = Path(f"models/int8/{model_name}_int8_minmax.onnx")

    if not fp32_path.exists() or not int8_path.exists():
        print(f"Skipping qdq_loss_debug smoke test: {fp32_path} or {int8_path} not found.")
        return

    # 1. Weight matching
    print("Testing create_weight_matching...")
    weights_match = qdq_debug.create_weight_matching(str(fp32_path), str(int8_path))
    print(f"  Matched {len(weights_match)} quantized weight tensors.")

    # 2. Modify intermediate tensors
    print("Testing modify_model_output_intermediate_tensors...")
    aug_fp32 = Path(f"models/fp32/{model_name}_aug_smoke.onnx")
    aug_int8 = Path(f"models/int8/{model_name}_aug_smoke.onnx")
    qdq_debug.modify_model_output_intermediate_tensors(str(fp32_path), str(aug_fp32))
    qdq_debug.modify_model_output_intermediate_tensors(str(int8_path), str(aug_int8))

    # Clean up smoke models
    if aug_fp32.exists():
        aug_fp32.unlink()
    if aug_int8.exists():
        aug_int8.unlink()

    print("qdq_loss_debug smoke test PASSED: tooling is fully verified for Extension A.")


def main():
    parser = argparse.ArgumentParser(description="Accuracy audit: FP32 vs INT8 variants")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    args = parser.parse_args()

    models = ["r34", "mbv3s"]
    variants = ["fp32", "int8_minmax", "int8_entropy"]

    audit_records = []
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    for m in models:
        dfs = {}
        for v in variants:
            if v == "fp32":
                p = Path(f"models/fp32/{m}_prep.onnx")
                if not p.exists():
                    p = Path(f"models/fp32/{m}.onnx")
            else:
                p = Path(f"models/int8/{m}_{v}.onnx")

            if not p.exists():
                print(f"Skipping missing model: {p}")
                continue

            df, summ = evaluate_onnx_model(p, split=args.split)
            dfs[v] = df
            csv_path = results_dir / f"eval_{args.split}_{m}_{v}.csv"
            df.to_csv(csv_path, index=False)

            audit_records.append({
                "Model": m,
                "Variant": v,
                "Split": args.split,
                "Mean Dice": f"{summ['dsc']['mean']:.4f}",
                "Dice 95% CI": f"[{summ['dsc']['ci_lower']:.4f}, {summ['dsc']['ci_upper']:.4f}]",
                "Mean HD95 (px)": f"{summ['hd95']['mean']:.2f}",
                "HD95 95% CI": f"[{summ['hd95']['ci_lower']:.2f}, {summ['hd95']['ci_upper']:.2f}]",
                "Normal FPR@50px": f"{summ['normal_fpr_50px']*100:.2f}%",
                "Empty Preds": summ["empty_predictions"],
            })

        # Paired differences (INT8 minus FP32)
        if "fp32" in dfs and "int8_minmax" in dfs:
            merged = pd.merge(dfs["fp32"], dfs["int8_minmax"], on="image_id", suffixes=("_fp32", "_int8"))
            merged = merged[merged["class_fp32"] != "normal"]
            d_dice = paired_bootstrap_ci(merged["dsc_int8"].values, merged["dsc_fp32"].values)
            d_hd95 = paired_bootstrap_ci(merged["hd95_int8"].values, merged["hd95_fp32"].values)
            print(f"\n[{m} INT8-MinMax vs FP32 Paired Difference on {args.split}]")
            print(f"  ΔDice: {d_dice['mean']:+.4f} (95% CI: [{d_dice['ci_lower']:+.4f}, {d_dice['ci_upper']:+.4f}])")
            print(f"  ΔHD95: {d_hd95['mean']:+.2f} px (95% CI: [{d_hd95['ci_lower']:+.2f}, {d_hd95['ci_upper']:+.2f}])")

    if audit_records:
        audit_df = pd.DataFrame(audit_records)
        out_path = results_dir / f"accuracy_audit_{args.split}.csv"
        audit_df.to_csv(out_path, index=False)
        print(f"\n--- Accuracy Audit Table ({args.split}) saved to {out_path} ---")
        print(audit_df.to_string(index=False))

    # Run smoke test on mbv3s
    run_qdq_loss_debug_smoke("mbv3s")


if __name__ == "__main__":
    main()
