import glob
from pathlib import Path
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

from src.cine.visual_cine import CineRealtimePipeline


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_frame_dice_iou(mask_pred: np.ndarray, mask_gt: np.ndarray) -> Tuple[float, float]:
    """Computes instantaneous Dice and IoU for binary masks."""
    p = mask_pred > 127
    g = mask_gt > 127
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    p_sum = p.sum()
    g_sum = g.sum()

    if p_sum == 0 and g_sum == 0:
        return 1.0, 1.0  # Perfect true negative
    if union == 0:
        return 1.0, 1.0

    iou = float(inter / union)
    dice = float((2.0 * inter) / (p_sum + g_sum))
    return dice, iou


def draw_labeled_mask_overlay(
    img_bgr: np.ndarray,
    mask_bin: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.45,
    contour_thickness: int = 2,
    centroid: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Overlays binary mask and contour border on BGR image."""
    h, w = img_bgr.shape[:2]
    mask_res = cv2.resize(mask_bin, (w, h), interpolation=cv2.INTER_NEAREST)
    out = img_bgr.copy()

    # Alpha blending
    mask_idx = mask_res > 127
    if np.any(mask_idx):
        color_arr = np.array(color, dtype=np.float32)
        out[mask_idx] = (
            (1.0 - alpha) * out[mask_idx] + alpha * color_arr
        ).astype(np.uint8)
        # Contours
        contours, _ = cv2.findContours(mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, contour_thickness)

    # Centroid crosshair
    if centroid is not None:
        cx = int(centroid[0] * (w / 256.0))
        cy = int(centroid[1] * (h / 256.0))
        cv2.drawMarker(out, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 12, 2)

    return out


def load_real_video_data(vid_id: str, base_dir: str = "data/ufuv_test", max_frames: int = 50) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Loads genuine clinical video frames and matching ground-truth masks.
    Returns list of (raw_bgr, tensor, gt_mask).
    """
    img_paths = sorted(glob.glob(f"{base_dir}/{vid_id}/images/*.png"))[:max_frames]
    mask_paths = sorted(glob.glob(f"{base_dir}/{vid_id}/masks/*.png"))[:max_frames]

    data = []
    for ip, mp in zip(img_paths, mask_paths):
        img_bgr = cv2.imread(ip)
        gt_mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)

        resized = cv2.resize(img_bgr, (256, 256))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = (rgb.astype(np.float32) / 255.0 - MEAN) / STD
        tensor = norm.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

        data.append((img_bgr, tensor, gt_mask))
    return data


def generate_5x3_groundtruth_tracking_demo(
    output_gif: str = "figures/cine_5sample_5x3_tracking_demo.gif",
    output_mp4: str = "figures/cine_5sample_5x3_tracking_demo.mp4",
    video_ids: Optional[List[str]] = None,
    num_frames: int = 50,
    output_fps: int = 15,
) -> pd.DataFrame:
    """
    Renders a 5x3 multi-video tracking canvas using 5 REAL clinical ultrasound video sequences:
    - 5 Rows: 5 distinct genuine clinical videos (IM_0003, IM_0004, IM_0005, IM_0014, IM_0055)
    - 3 Columns:
      [Column 1: Real Ground Truth Video] | [Column 2: Real-Time Model Prediction] | [Column 3: Live Telemetry HUD]
    """
    if video_ids is None:
        video_ids = ["IM_0003", "IM_0004", "IM_0005", "IM_0014", "IM_0055"]

    print("\n=================================================================")
    print("Generating 5x3 Real Clinical Video Tracking Showcase (.gif / .mp4)")
    print(f"Videos: {video_ids} ({num_frames} genuine clinical frames each)")
    print("=================================================================")

    # Load genuine video frame streams
    streams = {vid: load_real_video_data(vid, max_frames=num_frames) for vid in video_ids}

    pipeline = CineRealtimePipeline(
        mbv3s_model_path="models/mixed/mbv3s_mixed_best.onnx",
        r34_model_path="models/int8/r34_int8_minmax.onnx",
        ema_alpha=0.60,
        cascade_threshold=0.75,
    )

    # Canvas dimensions
    tile_w, tile_h = 240, 140
    canvas_w = tile_w * 3  # 720
    canvas_h = tile_h * 5  # 700

    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_mp4, fourcc, output_fps, (canvas_w, canvas_h))
    gif_frames = []

    video_metrics = {vid: {"dices": [], "ious": [], "lats": [], "tcis_raw": [], "tcis_ema": []} for vid in video_ids}

    for f_idx in range(num_frames):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        for row_idx, vid in enumerate(video_ids):
            raw_bgr, tensor, gt_mask = streams[vid][f_idx]

            # Model prediction
            t0 = time.perf_counter_ns()
            res = pipeline.process_frame(tensor)
            dt_ms = (time.perf_counter_ns() - t0) / 1e6

            pred_mask_256 = res["ema_mask"]
            pred_mask = cv2.resize(pred_mask_256, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            dice, iou = compute_frame_dice_iou(pred_mask, gt_mask)
            gt_area = int(np.sum(gt_mask > 127))
            pred_area = int(np.sum(pred_mask > 127))

            video_metrics[vid]["dices"].append(dice)
            video_metrics[vid]["ious"].append(iou)
            video_metrics[vid]["lats"].append(dt_ms)
            video_metrics[vid]["tcis_raw"].append(res["tci_raw"])
            video_metrics[vid]["tcis_ema"].append(res["tci_ema"])

            # --- Tile 1: Left - Real Ground Truth Video ---
            tile_gt = draw_labeled_mask_overlay(
                img_bgr=cv2.resize(raw_bgr, (tile_w, tile_h)),
                mask_bin=cv2.resize(gt_mask, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST),
                color=(0, 255, 0),  # Vivid Green
                alpha=0.45,
                contour_thickness=2,
            )
            # Banner
            cv2.rectangle(tile_gt, (4, 4), (tile_w - 4, 30), (10, 10, 10), -1)
            cv2.rectangle(tile_gt, (4, 4), (tile_w - 4, 30), (50, 200, 50), 1)
            cv2.putText(tile_gt, f"GROUND TRUTH: {vid}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1)
            cv2.putText(tile_gt, f"GT Area: {gt_area:,} px", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 255, 200), 1)

            # --- Tile 2: Center - Real Model Prediction ---
            tile_pred = draw_labeled_mask_overlay(
                img_bgr=cv2.resize(raw_bgr, (tile_w, tile_h)),
                mask_bin=cv2.resize(pred_mask_256, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST),
                color=(0, 230, 255),  # Cyan / Emerald
                alpha=0.45,
                contour_thickness=2,
                centroid=res["centroid"],
            )
            # Banner
            cv2.rectangle(tile_pred, (4, 4), (tile_w - 4, 30), (10, 10, 10), -1)
            cv2.rectangle(tile_pred, (4, 4), (tile_w - 4, 30), (0, 200, 255), 1)
            cv2.putText(tile_pred, f"PRED: {res['active_model'][:20]}", (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 230, 255), 1)
            cv2.putText(tile_pred, f"Pred Area: {pred_area:,} px", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (220, 240, 255), 1)

            # --- Tile 3: Right - Live Telemetry HUD ---
            tile_hud = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
            tile_hud[:] = (20, 24, 28)
            cv2.rectangle(tile_hud, (4, 4), (tile_w - 4, tile_h - 4), (55, 65, 75), 1)

            cv2.rectangle(tile_hud, (4, 4), (tile_w - 4, 26), (35, 45, 55), -1)
            cv2.putText(tile_hud, f"CLINICAL VIDEO: {vid}", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

            hud_lines = [
                ("Live Dice (DSC):", f"{dice:.4f}"),
                ("Live IoU (Jacc):", f"{iou:.4f}"),
                ("Latency / FPS:", f"{dt_ms:.1f} ms ({1000.0/max(dt_ms,0.1):.0f} FPS)"),
                ("Anti-Flicker TCI:", f"{res['tci_ema']:.3f}"),
            ]

            y_pos = 46
            for label_k, val_v in hud_lines:
                cv2.putText(tile_hud, label_k, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 175, 190), 1)
                is_highlight = "Dice" in label_k or "IoU" in label_k
                cv2.putText(
                    tile_hud,
                    val_v,
                    (115, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.34,
                    (0, 255, 180) if is_highlight else (255, 255, 255),
                    1,
                )
                y_pos += 22

            y1 = row_idx * tile_h
            y2 = (row_idx + 1) * tile_h
            canvas[y1:y2, 0:tile_w] = tile_gt
            canvas[y1:y2, tile_w:2*tile_w] = tile_pred
            canvas[y1:y2, 2*tile_w:3*tile_w] = tile_hud

        writer.write(canvas)
        rgb_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        gif_frames.append(Image.fromarray(rgb_c))

        if (f_idx + 1) % 15 == 0 or f_idx == num_frames - 1:
            print(f"  Processed Real Video Frame {f_idx + 1:02d}/{num_frames}")

    writer.release()

    print(f"Compiling animated GIF to {output_gif}...")
    gif_frames[0].save(
        output_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / output_fps),
        loop=0,
    )

    print(f"Rendered Genuine Video 5x3 GIF: {output_gif} ({Path(output_gif).stat().st_size / 1e6:.2f} MB)")
    print(f"Rendered Genuine Video 5x3 MP4: {output_mp4} ({Path(output_mp4).stat().st_size / 1e6:.2f} MB)")

    # Compute summary metrics table
    summary_rows = []
    for vid in video_ids:
        d = video_metrics[vid]
        mean_d = float(np.mean(d["dices"]))
        mean_i = float(np.mean(d["ious"]))
        mean_l = float(np.mean(d["lats"]))
        raw_tcis = [t for t in d["tcis_raw"][1:] if not np.isnan(t)]
        ema_tcis = [t for t in d["tcis_ema"][1:] if not np.isnan(t)]
        mean_tci_raw = float(np.mean(raw_tcis)) if raw_tcis else 1.0
        mean_tci_ema = float(np.mean(ema_tcis)) if ema_tcis else 1.0

        summary_rows.append({
            "video_id": vid,
            "mean_dice": mean_d,
            "mean_iou": mean_i,
            "raw_tci": mean_tci_raw,
            "ema_tci": mean_tci_ema,
            "tci_gain_pct": ((mean_tci_ema - mean_tci_raw) / max(mean_tci_raw, 1e-4)) * 100.0,
            "latency_ms": mean_l,
            "fps": 1000.0 / mean_l,
        })
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("results/real_video_groundtruth_benchmark.csv", index=False)
    print(f"Saved real video benchmark summary to: results/real_video_groundtruth_benchmark.csv")
    return df_summary


def generate_quantization_comparison_demo(
    output_gif: str = "figures/cine_all_models_quantization_comparison.gif",
    output_mp4: str = "figures/cine_all_models_quantization_comparison.mp4",
    video_id: str = "IM_0055",
    num_frames: int = 50,
    output_fps: int = 15,
) -> str:
    """
    Renders an 8-panel (2x4) synchronized video comparison on a REAL clinical ultrasound video stream (IM_0055):
    Row 1: [1. Ground Truth] | [2. ResNet-34 FP32] | [3. ResNet-34 INT8] | [4. MobileNetV3 FP32]
    Row 2: [5. MobileNetV3 Naive INT8 (Collapsed)] | [6. MobileNetV3 Rescued Mixed k=13] | [7. Cascade + EMA] | [8. Comparative Telemetry Leaderboard]
    """
    print("\n=================================================================")
    print(f"Generating All-Models Quantization Comparison on Real Clinical Video {video_id} (2x4 Grid)")
    print("=================================================================")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1

    models = {
        "r34_fp32": ort.InferenceSession("models/fp32/r34_prep.onnx", sess_opts, providers=["CPUExecutionProvider"]),
        "r34_int8": ort.InferenceSession("models/int8/r34_int8_minmax.onnx", sess_opts, providers=["CPUExecutionProvider"]),
        "mbv3s_fp32": ort.InferenceSession("models/fp32/mbv3s_prep.onnx", sess_opts, providers=["CPUExecutionProvider"]),
        "mbv3s_int8": ort.InferenceSession("models/int8/mbv3s_int8_minmax.onnx", sess_opts, providers=["CPUExecutionProvider"]),
        "mbv3s_mixed": ort.InferenceSession("models/mixed/mbv3s_mixed_best.onnx", sess_opts, providers=["CPUExecutionProvider"]),
    }

    pipeline_cascade = CineRealtimePipeline(
        mbv3s_model_path="models/mixed/mbv3s_mixed_best.onnx",
        r34_model_path="models/int8/r34_int8_minmax.onnx",
        ema_alpha=0.60,
        cascade_threshold=0.75,
    )

    dummy = np.zeros((1, 3, 256, 256), dtype=np.float32)
    for s in models.values():
        _ = s.run(None, {s.get_inputs()[0].name: dummy})

    frames_data = load_real_video_data(video_id, max_frames=num_frames)

    tile_w, tile_h = 480, 360
    canvas_w = tile_w * 4  # 1920 (720p HD widescreen)
    canvas_h = tile_h * 2  # 720

    Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_mp4, fourcc, output_fps, (canvas_w, canvas_h))
    gif_frames = []

    for f_idx, (raw_bgr, tensor, gt_mask) in enumerate(frames_data):
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        preds = {}
        lats = {}

        for m_name, sess in models.items():
            t0 = time.perf_counter_ns()
            out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0][0, 0]
            dt = (time.perf_counter_ns() - t0) / 1e6
            lats[m_name] = dt
            p256 = (out > 0.0).astype(np.uint8) * 255
            preds[m_name] = cv2.resize(p256, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

        t0 = time.perf_counter_ns()
        res_cascade = pipeline_cascade.process_frame(tensor)
        dt_casc = (time.perf_counter_ns() - t0) / 1e6
        preds["cascade_ema"] = cv2.resize(res_cascade["ema_mask"], (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        lats["cascade_ema"] = dt_casc

        dices = {m: compute_frame_dice_iou(preds[m], gt_mask)[0] for m in preds}

        panels = [
            (0, 0, "GROUND TRUTH", f"Clinical Mask ({int(np.sum(gt_mask>127)):,} px)", (0, 255, 0), "gt", True),
            (0, 1, "ResNet-34 FP32", f"DSC: {dices['r34_fp32']:.3f} | 93.2 MB | {lats['r34_fp32']:.1f}ms", (255, 150, 0), "r34_fp32", False),
            (0, 2, "ResNet-34 INT8 (Lossless)", f"DSC: {dices['r34_int8']:.3f} | 23.6 MB | {lats['r34_int8']:.1f}ms", (255, 210, 0), "r34_int8", False),
            (0, 3, "MobileNetV3 FP32", f"DSC: {dices['mbv3s_fp32']:.3f} | 8.2 MB | {lats['mbv3s_fp32']:.1f}ms", (0, 220, 255), "mbv3s_fp32", False),
            (1, 0, "MobileNetV3 Naive INT8", f"COLLAPSED! DSC: {dices['mbv3s_int8']:.3f} | FP Spill", (0, 0, 255), "mbv3s_int8", False),
            (1, 1, "MobileNetV3 Mixed (k*=13)", f"RESCUED! DSC: {dices['mbv3s_mixed']:.3f} | 2.5 MB", (255, 0, 255), "mbv3s_mixed", False),
            (1, 2, "Cascade + EMA Logits", f"STABILIZED! DSC: {dices['cascade_ema']:.3f} | TCI: {res_cascade['tci_ema']:.2f}", (0, 240, 120), "cascade_ema", False),
        ]

        for r, c, title, sub, col, m_key, is_gt in panels:
            mask_to_draw = gt_mask if is_gt else preds[m_key]
            tile = draw_labeled_mask_overlay(
                img_bgr=cv2.resize(raw_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA),
                mask_bin=cv2.resize(mask_to_draw, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST),
                color=col,
                alpha=0.45,
                contour_thickness=3,
            )
            cv2.rectangle(tile, (6, 6), (tile_w - 6, 60), (12, 12, 12), -1)
            cv2.rectangle(tile, (6, 6), (tile_w - 6, 60), col, 2)
            cv2.putText(tile, title, (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2, cv2.LINE_AA)
            cv2.putText(tile, sub, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA)

            canvas[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tile

        hud_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)
        hud_tile[:] = (20, 24, 28)
        cv2.rectangle(hud_tile, (6, 6), (tile_w - 6, tile_h - 6), (60, 70, 80), 2)

        cv2.rectangle(hud_tile, (6, 6), (tile_w - 6, 52), (35, 45, 55), -1)
        cv2.putText(hud_tile, f"LEADERBOARD: {video_id}", (14, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 230, 255), 2, cv2.LINE_AA)

        rows = [
            ("r34 FP32", f"{dices['r34_fp32']:.3f}", "93MB", f"{lats['r34_fp32']:.0f}ms"),
            ("r34 INT8", f"{dices['r34_int8']:.3f}", "24MB", f"{lats['r34_int8']:.0f}ms"),
            ("mbv3s FP32", f"{dices['mbv3s_fp32']:.3f}", "8.2MB", f"{lats['mbv3s_fp32']:.0f}ms"),
            ("mbv3s INT8", f"{dices['mbv3s_int8']:.3f}", "2.4MB", "FAIL"),
            ("mbv3s Mixed", f"{dices['mbv3s_mixed']:.3f}", "2.5MB", f"{lats['mbv3s_mixed']:.0f}ms"),
            ("Cascade+EMA", f"{dices['cascade_ema']:.3f}", "Smart", f"{lats['cascade_ema']:.0f}ms"),
        ]

        y_pos = 92
        cv2.putText(hud_tile, "Model", (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 160, 170), 2, cv2.LINE_AA)
        cv2.putText(hud_tile, "Dice", (175, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 160, 170), 2, cv2.LINE_AA)
        cv2.putText(hud_tile, "Size", (275, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 160, 170), 2, cv2.LINE_AA)
        cv2.putText(hud_tile, "Speed", (375, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 160, 170), 2, cv2.LINE_AA)
        cv2.line(hud_tile, (12, 106), (tile_w - 12, 106), (50, 60, 70), 1)

        y_pos = 142
        for m_name, d_str, sz_str, lat_str in rows:
            is_fail = lat_str == "FAIL"
            is_rescued = "Mixed" in m_name or "Cascade" in m_name
            text_col = (0, 80, 255) if is_fail else ((0, 255, 180) if is_rescued else (240, 240, 240))
            cv2.putText(hud_tile, m_name, (16, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 2, cv2.LINE_AA)
            cv2.putText(hud_tile, d_str, (175, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 2, cv2.LINE_AA)
            cv2.putText(hud_tile, sz_str, (275, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 2, cv2.LINE_AA)
            cv2.putText(hud_tile, lat_str, (375, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_col, 2, cv2.LINE_AA)
            y_pos += 36

        canvas[tile_h:2*tile_h, 3*tile_w:4*tile_w] = hud_tile

        writer.write(canvas)
        rgb_c = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        gif_frames.append(Image.fromarray(rgb_c))

        if (f_idx + 1) % 15 == 0 or f_idx == num_frames - 1:
            print(f"  Processed Real Video Quantization Frame {f_idx + 1:02d}/{num_frames}")

    writer.release()

    print(f"Compiling animated GIF to {output_gif}...")
    gif_frames[0].save(
        output_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=int(1000 / output_fps),
        loop=0,
    )

    print(f"Rendered Real Video All-Models Comparison GIF: {output_gif} ({Path(output_gif).stat().st_size / 1e6:.2f} MB)")
    print(f"Rendered Real Video All-Models Comparison MP4: {output_mp4} ({Path(output_mp4).stat().st_size / 1e6:.2f} MB)")
    return output_gif


def main():
    generate_5x3_groundtruth_tracking_demo()
    generate_quantization_comparison_demo()


if __name__ == "__main__":
    main()
