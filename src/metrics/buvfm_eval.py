import glob
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import cv2
import numpy as np
import pandas as pd
import onnxruntime as ort

from src.utils.seed import seed_everything
from src.metrics.bootstrap import bootstrap_ci


# Standard ImageNet normalization used during training
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_boundary_entropy(prob_map: np.ndarray, margin: float = 0.20) -> float:
    """Mean Shannon entropy in ambiguous boundary regions |p - 0.5| < margin."""
    ambiguous = np.abs(prob_map - 0.5) < margin
    if not np.any(ambiguous):
        return 0.0
    p = np.clip(prob_map[ambiguous], 1e-6, 1.0 - 1e-6)
    entropy = - (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(np.mean(entropy))


def load_buvfm_val_data(cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads all 500 BUVFM validation image paths and YOLO bounding-box annotations."""
    if cache_dir is None:
        cache_dir = os.environ.get("BUVFM_CACHE_DIR")
    if cache_dir is None:
        candidate_paths = [
            os.path.expanduser("~/.cache/huggingface/hub/datasets--xenosscu--BUVFM/snapshots/12a2a1e563269563bc0a3404128b5f01360415bf"),
            "data/buvfm",
            "data/buvfm_val",
        ]
        for p in candidate_paths:
            if os.path.exists(p):
                cache_dir = p
                break
        if cache_dir is None:
            cache_dir = candidate_paths[0]

    label_files = sorted(glob.glob(os.path.join(cache_dir, "detection", "labels", "val2017", "*.txt")))
    if not label_files:
        raise FileNotFoundError(f"No BUVFM label files found in {cache_dir}")

    samples = []
    for lp in label_files:
        img_p = lp.replace("labels", "images").replace(".txt", ".jpg")
        if not os.path.exists(img_p):
            continue

        boxes = []
        is_normal = False
        with open(lp, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls_id = int(parts[0])
                if cls_id == 1:
                    is_normal = True
                elif cls_id == 0:
                    xc, yc, bw, bh = [float(x) for x in parts[1:5]]
                    boxes.append((xc, yc, bw, bh))

        samples.append({
            "img_path": img_p,
            "img_id": os.path.basename(img_p).replace(".jpg", ""),
            "is_normal": is_normal or len(boxes) == 0,
            "boxes": boxes,  # list of (xc, yc, bw, bh) in normalized coords
        })

    return samples


def preprocess_image(img_bgr: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Standardizes image to (1, 3, 256, 256) float32 tensor."""
    resized = cv2.resize(img_bgr, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
    tensor = norm.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 256, 256)
    return tensor.astype(np.float32)


def evaluate_models_on_buvfm(
    samples: List[Dict[str, Any]],
    num_threads: int = 4,
    cascade_threshold: float = 0.75,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluates 6 model variants across all 500 BUVFM validation frames:
    1. r34_fp32
    2. r34_int8
    3. mbv3s_fp32
    4. mbv3s_int8
    5. mbv3s_mixed_k13
    6. Cascade (MBV3S Mixed -> R34 INT8)
    """
    model_configs = [
        {"name": "r34_fp32", "path": "models/fp32/r34_prep.onnx", "type": "single"},
        {"name": "r34_int8", "path": "models/int8/r34_int8_minmax.onnx", "type": "single"},
        {"name": "mbv3s_fp32", "path": "models/fp32/mbv3s_prep.onnx", "type": "single"},
        {"name": "mbv3s_int8", "path": "models/int8/mbv3s_int8_minmax.onnx", "type": "single"},
        {"name": "mbv3s_mixed_k13", "path": "models/mixed/mbv3s_mixed_best.onnx", "type": "single"},
        {"name": "cascade_mixed_r34", "path": "models/mixed/mbv3s_mixed_best.onnx", "type": "cascade"},
    ]

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = 1

    # Load sessions
    sessions = {}
    for m in model_configs:
        if m["path"] not in sessions:
            sessions[m["path"]] = ort.InferenceSession(m["path"], sess_opts, providers=["CPUExecutionProvider"])
    sess_r34_int8 = sessions["models/int8/r34_int8_minmax.onnx"]

    # Pre-load all images to memory for exact fair latency timing
    print(f"Pre-loading {len(samples)} BUVFM validation images...")
    loaded_data = []
    for s in samples:
        img_bgr = cv2.imread(s["img_path"])
        tensor = preprocess_image(img_bgr)
        h, w = img_bgr.shape[:2]
        loaded_data.append({
            "img_id": s["img_id"],
            "is_normal": s["is_normal"],
            "boxes": s["boxes"],
            "h": h,
            "w": w,
            "tensor": tensor,
        })

    all_per_image_rows = []
    summary_rows = []

    for cfg in model_configs:
        mname = cfg["name"]
        mtype = cfg["type"]
        sess = sessions[cfg["path"]]
        inp_name = sess.get_inputs()[0].name

        print(f"\nEvaluating '{mname}' on {len(loaded_data)} BUVFM frames...")

        # Warmup
        dummy = np.zeros((1, 3, 256, 256), dtype=np.float32)
        for _ in range(10):
            _ = sess.run(None, {inp_name: dummy})

        latencies_ms = []
        lesion_bbox_ious = []
        lesion_bbox_dices = []
        lesion_mask_precs = []
        lesion_centroid_errs = []
        empty_lesion_count = 0
        total_lesion_count = 0

        normal_total_count = 0
        normal_fp_50_count = 0
        normal_fp_areas = []

        cascade_trigger_count = 0

        for item in loaded_data:
            tensor = item["tensor"]
            H, W = item["h"], item["w"]
            is_normal = item["is_normal"]
            gt_boxes = item["boxes"]

            t0 = time.perf_counter_ns()
            if mtype == "cascade":
                logits_fast = sess.run(None, {inp_name: tensor})[0][0, 0]
                prob_fast = 1.0 / (1.0 + np.exp(-logits_fast))
                uncertainty = compute_boundary_entropy(prob_fast)
                if uncertainty > cascade_threshold:
                    cascade_trigger_count += 1
                    logits = sess_r34_int8.run(None, {sess_r34_int8.get_inputs()[0].name: tensor})[0][0, 0]
                else:
                    logits = logits_fast
            else:
                logits = sess.run(None, {inp_name: tensor})[0][0, 0]
            dt_ms = (time.perf_counter_ns() - t0) / 1e6
            latencies_ms.append(dt_ms)

            # Binary mask at 256x256, then upsampled to original image dimensions (W, H)
            pred_mask_256 = (logits > 0.0).astype(np.uint8)
            pred_mask = cv2.resize(pred_mask_256, (W, H), interpolation=cv2.INTER_NEAREST)
            pred_area = int(np.sum(pred_mask))

            # Bounding box of prediction
            if pred_area > 0:
                ys, xs = np.where(pred_mask > 0)
                p_bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
                pred_cx = float(np.mean(xs))
                pred_cy = float(np.mean(ys))
            else:
                p_bbox = None
                pred_cx, pred_cy = None, None

            if is_normal:
                normal_total_count += 1
                is_fp = pred_area >= 50
                if is_fp:
                    normal_fp_50_count += 1
                normal_fp_areas.append(pred_area)

                all_per_image_rows.append({
                    "model": mname,
                    "img_id": item["img_id"],
                    "category": "normal",
                    "pred_area_px": pred_area,
                    "bbox_iou": np.nan,
                    "bbox_dice": np.nan,
                    "mask_in_bbox_prec": np.nan,
                    "centroid_err_px": np.nan,
                    "latency_ms": dt_ms,
                })
            else:
                total_lesion_count += 1
                if pred_area == 0:
                    empty_lesion_count += 1
                    lesion_bbox_ious.append(0.0)
                    lesion_bbox_dices.append(0.0)
                    lesion_mask_precs.append(0.0)
                    # For empty predictions, set centroid error to image diagonal (hypothetical max)
                    diag = np.sqrt(H**2 + W**2)
                    lesion_centroid_errs.append(diag)

                    all_per_image_rows.append({
                        "model": mname,
                        "img_id": item["img_id"],
                        "category": "lesion",
                        "pred_area_px": 0,
                        "bbox_iou": 0.0,
                        "bbox_dice": 0.0,
                        "mask_in_bbox_prec": 0.0,
                        "centroid_err_px": diag,
                        "latency_ms": dt_ms,
                    })
                else:
                    # Match against ground-truth lesion boxes
                    best_iou = 0.0
                    best_dice = 0.0
                    best_c_err = float("inf")

                    # Combined GT mask for multi-box precision
                    gt_mask_union = np.zeros((H, W), dtype=np.uint8)

                    for (xc, yc, bw, bh) in gt_boxes:
                        gx1 = int(round((xc - bw / 2.0) * W))
                        gy1 = int(round((yc - bh / 2.0) * H))
                        gx2 = int(round((xc + bw / 2.0) * W))
                        gy2 = int(round((yc + bh / 2.0) * H))
                        gx1, gy1 = max(0, gx1), max(0, gy1)
                        gx2, gy2 = min(W, gx2), min(H, gy2)
                        gt_mask_union[gy1:gy2, gx1:gx2] = 1

                        gt_cx = (gx1 + gx2) / 2.0
                        gt_cy = (gy1 + gy2) / 2.0
                        c_err = np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2)
                        if c_err < best_c_err:
                            best_c_err = c_err

                        # Box intersection & union
                        ix1 = max(p_bbox[0], gx1)
                        iy1 = max(p_bbox[1], gy1)
                        ix2 = min(p_bbox[2], gx2)
                        iy2 = min(p_bbox[3], gy2)
                        iw = max(0, ix2 - ix1)
                        ih = max(0, iy2 - iy1)
                        i_area = iw * ih
                        p_area = (p_bbox[2] - p_bbox[0]) * (p_bbox[3] - p_bbox[1])
                        g_area = (gx2 - gx1) * (gy2 - gy1)
                        u_area = p_area + g_area - i_area
                        iou = i_area / max(u_area, 1e-6)
                        dice = (2.0 * i_area) / max(p_area + g_area, 1e-6)

                        if iou > best_iou:
                            best_iou = iou
                            best_dice = dice

                    # Mask-in-BBox precision: % of predicted pixels inside union of GT boxes
                    in_box_pixels = int(np.sum((pred_mask > 0) & (gt_mask_union > 0)))
                    mask_prec = in_box_pixels / max(pred_area, 1)

                    lesion_bbox_ious.append(best_iou)
                    lesion_bbox_dices.append(best_dice)
                    lesion_mask_precs.append(mask_prec)
                    lesion_centroid_errs.append(best_c_err)

                    all_per_image_rows.append({
                        "model": mname,
                        "img_id": item["img_id"],
                        "category": "lesion",
                        "pred_area_px": pred_area,
                        "bbox_iou": best_iou,
                        "bbox_dice": best_dice,
                        "mask_in_bbox_prec": mask_prec,
                        "centroid_err_px": best_c_err,
                        "latency_ms": dt_ms,
                    })

        # Compute summary metrics with 1,000 bootstrap CIs
        iou_ci = bootstrap_ci(lesion_bbox_ious, n_resamples=1000)
        dice_ci = bootstrap_ci(lesion_bbox_dices, n_resamples=1000)
        prec_ci = bootstrap_ci(lesion_mask_precs, n_resamples=1000)
        cent_ci = bootstrap_ci(lesion_centroid_errs, n_resamples=1000)

        mean_lat = float(np.mean(latencies_ms))
        fps = 1000.0 / mean_lat
        empty_pct = (empty_lesion_count / max(total_lesion_count, 1)) * 100.0
        normal_fpr_50 = (normal_fp_50_count / max(normal_total_count, 1)) * 100.0
        normal_spec = 100.0 - normal_fpr_50
        coverage_pct = (sum(1 for iou in lesion_bbox_ious if iou > 0.0) / max(total_lesion_count, 1)) * 100.0

        summary_rows.append({
            "model": mname,
            "total_frames": len(loaded_data),
            "lesion_frames": total_lesion_count,
            "normal_frames": normal_total_count,
            "bbox_iou_mean": iou_ci["mean"],
            "bbox_iou_ci_low": iou_ci["ci_lower"],
            "bbox_iou_ci_high": iou_ci["ci_upper"],
            "bbox_iou_median": iou_ci["median"],
            "bbox_dice_mean": dice_ci["mean"],
            "bbox_dice_ci_low": dice_ci["ci_lower"],
            "bbox_dice_ci_high": dice_ci["ci_upper"],
            "bbox_dice_median": dice_ci["median"],
            "mask_in_bbox_prec_mean": prec_ci["mean"],
            "mask_in_bbox_prec_ci_low": prec_ci["ci_lower"],
            "mask_in_bbox_prec_ci_high": prec_ci["ci_upper"],
            "centroid_err_px_mean": cent_ci["mean"],
            "centroid_err_px_ci_low": cent_ci["ci_lower"],
            "centroid_err_px_ci_high": cent_ci["ci_upper"],
            "lesion_coverage_pct": coverage_pct,
            "empty_pred_pct": empty_pct,
            "normal_specificity_pct": normal_spec,
            "normal_fpr50_pct": normal_fpr_50,
            "cascade_trigger_pct": (cascade_trigger_count / len(loaded_data)) * 100.0 if mtype == "cascade" else 0.0,
            "latency_ms_mean": mean_lat,
            "edge_fps": fps,
        })

        print(f"  -> Mean BBox IoU: {iou_ci['mean']:.4f} [{iou_ci['ci_lower']:.4f}, {iou_ci['ci_upper']:.4f}]")
        print(f"  -> Mean BBox Dice: {dice_ci['mean']:.4f} [{dice_ci['ci_lower']:.4f}, {dice_ci['ci_upper']:.4f}]")
        print(f"  -> Mask-in-BBox Prec: {prec_ci['mean']:.4f} [{prec_ci['ci_lower']:.4f}, {prec_ci['ci_upper']:.4f}]")
        print(f"  -> Centroid Error: {cent_ci['mean']:.1f} px | Coverage: {coverage_pct:.1f}%")
        print(f"  -> Normal Spec: {normal_spec:.1f}% (FPR@50px: {normal_fpr_50:.1f}%)")
        print(f"  -> CPU Throughput: {fps:.1f} FPS ({mean_lat:.2f} ms/frame)")

    df_per_image = pd.DataFrame(all_per_image_rows)
    df_summary = pd.DataFrame(summary_rows)

    return df_per_image, df_summary


def main():
    seed_everything(42)
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("BUVFM Clinical Breast Ultrasound Ground-Truth Benchmark (Val2017)")
    print("=================================================================")

    samples = load_buvfm_val_data()
    print(f"Loaded {len(samples)} BUVFM validation cases ({sum(1 for s in samples if not s['is_normal'])} lesion, {sum(1 for s in samples if s['is_normal'])} normal).")

    df_per_image, df_summary = evaluate_models_on_buvfm(samples)

    per_image_path = out_dir / "buvfm_per_image_benchmark.csv"
    summary_path = out_dir / "buvfm_groundtruth_benchmark.csv"

    df_per_image.to_csv(per_image_path, index=False)
    df_summary.to_csv(summary_path, index=False)

    print(f"\nSaved detailed evaluation to: {per_image_path}")
    print(f"Saved benchmark summary table to: {summary_path}")


if __name__ == "__main__":
    main()
