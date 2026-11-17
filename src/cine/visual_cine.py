from pathlib import Path
import time
from typing import Any, Dict, Optional
import cv2
import numpy as np
import onnxruntime as ort

from src.data.buv_video import POCUSContinuousSweepStream


def compute_boundary_entropy(prob_map: np.ndarray, margin: float = 0.20) -> float:
    """
    Computes mean Shannon entropy strictly within the ambiguous boundary region
    where probability |p - 0.5| < margin.
    """
    ambiguous = np.abs(prob_map - 0.5) < margin
    if not np.any(ambiguous):
        return 0.0
    p = np.clip(prob_map[ambiguous], 1e-6, 1.0 - 1e-6)
    entropy = - (p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(np.mean(entropy))


def compute_tci(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Temporal Consistency Index (Jaccard similarity between consecutive frame masks)."""
    a = mask_a > 127
    b = mask_b > 127
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


class CineRealtimePipeline:
    def __init__(
        self,
        mbv3s_model_path: str = "models/mixed/mbv3s_mixed_best.onnx",
        r34_model_path: str = "models/int8/r34_int8_minmax.onnx",
        ema_alpha: float = 0.60,
        cascade_threshold: float = 0.75,
        num_threads: int = 4,
    ):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.inter_op_num_threads = 1
        self.sess_fast = ort.InferenceSession(mbv3s_model_path, sess_opts, providers=["CPUExecutionProvider"])
        self.sess_high = ort.InferenceSession(r34_model_path, sess_opts, providers=["CPUExecutionProvider"])
        self.ema_alpha = ema_alpha
        self.cascade_threshold = cascade_threshold

        self.prev_logits_ema = None
        self.prev_mask_raw = None
        self.prev_mask_ema = None

    def process_frame(self, inp_tensor: np.ndarray) -> Dict[str, Any]:
        t0 = time.perf_counter_ns()

        # Step 1: Run fast edge model: MobileNetV3 v2 (Rescued Mixed k=13)
        logits_fast = self.sess_fast.run(None, {"input": inp_tensor})[0][0, 0]
        prob_fast = 1.0 / (1.0 + np.exp(-logits_fast))
        uncertainty = compute_boundary_entropy(prob_fast)

        # Step 2: Uncertainty-Gated Cascade check
        active_model = "MobileNetV3 v2 (Rescued Mixed)"
        if uncertainty > self.cascade_threshold:
            # Trigger high-capacity ResNet-34 INT8 model
            active_model = "Cascade -> ResNet-34 INT8"
            logits = self.sess_high.run(None, {"input": inp_tensor})[0][0, 0]
        else:
            logits = logits_fast

        dt_ms = (time.perf_counter_ns() - t0) / 1e6

        # Step 3: Raw frame binary mask (threshold 0.0 on logits)
        raw_mask = (logits > 0.0).astype(np.uint8) * 255

        # Step 4: Temporal EMA logit smoothing (Novelty 1)
        if self.prev_logits_ema is None:
            logits_ema = logits.copy()
        else:
            logits_ema = self.ema_alpha * logits + (1.0 - self.ema_alpha) * self.prev_logits_ema
        self.prev_logits_ema = logits_ema

        ema_mask = (logits_ema > 0.0).astype(np.uint8) * 255

        # Step 5: Temporal Consistency Index
        tci_raw = compute_tci(raw_mask, self.prev_mask_raw) if self.prev_mask_raw is not None else 1.0
        tci_ema = compute_tci(ema_mask, self.prev_mask_ema) if self.prev_mask_ema is not None else 1.0
        self.prev_mask_raw = raw_mask.copy()
        self.prev_mask_ema = ema_mask.copy()

        # Centroid & Lesion Area
        area_px = int(np.sum(ema_mask > 127))
        centroid = None
        if area_px > 50:
            M = cv2.moments(ema_mask)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)

        return {
            "logits": logits,
            "raw_mask": raw_mask,
            "ema_mask": ema_mask,
            "latency_ms": dt_ms,
            "fps": 1000.0 / max(dt_ms, 0.1),
            "uncertainty": uncertainty,
            "active_model": active_model,
            "tci_raw": tci_raw,
            "tci_ema": tci_ema,
            "area_px": area_px,
            "centroid": centroid,
        }


def render_hud_panel(
    raw_bgr: np.ndarray,
    gt_mask: Optional[np.ndarray],
    raw_mask: np.ndarray,
    ema_mask: np.ndarray,
    telemetry: Dict[str, Any],
    frame_idx: int,
) -> np.ndarray:
    """
    Renders a unified 3-panel widescreen frame (1100x380) with telemetry HUD:
    [Panel 1: Continuous Ultrasound Stream] | [Panel 2: Raw Detection vs GT] | [Panel 3: EMA Stabilized Overlay + HUD]
    """
    h_target, w_target = 360, 360

    # Panel 1: Original B-Mode Ultrasound Cine Stream
    p1 = cv2.resize(raw_bgr, (w_target, h_target))
    # Probe orientation banner
    cv2.rectangle(p1, (8, 8), (w_target - 8, 48), (20, 20, 20), -1)
    cv2.rectangle(p1, (8, 8), (w_target - 8, 48), (100, 100, 100), 1)
    cv2.putText(p1, "Panel 1: B-Mode Transducer Stream", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
    cv2.putText(p1, f"Frame #{frame_idx:04d} (30 FPS POCUS Sweep)", (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)

    # Panel 2: Raw Detection vs Radiologist Ground Truth
    p2 = cv2.resize(raw_bgr, (w_target, h_target))
    raw_mask_res = cv2.resize(raw_mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

    # If Ground Truth mask available: draw dashed / distinct Cyan boundary
    if gt_mask is not None:
        gt_res = cv2.resize(gt_mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)
        contours_gt, _ = cv2.findContours(gt_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(p2, contours_gt, -1, (255, 255, 0), 2)  # Cyan GT

    # Draw Raw Detection (Orange outline + light orange fill)
    overlay_raw = p2.copy()
    overlay_raw[raw_mask_res > 127] = (0, 140, 255)
    p2 = cv2.addWeighted(overlay_raw, 0.35, p2, 0.65, 0)
    contours_raw, _ = cv2.findContours(raw_mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p2, contours_raw, -1, (0, 165, 255), 2)

    cv2.rectangle(p2, (8, 8), (w_target - 8, 48), (20, 20, 20), -1)
    cv2.rectangle(p2, (8, 8), (w_target - 8, 48), (0, 140, 255), 1)
    cv2.putText(p2, "Panel 2: Raw Frame-by-Frame Detection", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 165, 255), 1)
    cv2.putText(p2, f"Cyan=GT | Orange=Raw (TCI: {telemetry['tci_raw']:.2f})", (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)

    # Panel 3: Temporal EMA-Smoothed Overlay (Vibrant Neon Green) + Centroid Tracker + HUD
    p3 = cv2.resize(raw_bgr, (w_target, h_target))
    ema_mask_res = cv2.resize(ema_mask, (w_target, h_target), interpolation=cv2.INTER_NEAREST)

    # High-Visibility Neon Green Overlay: (0, 255, 30) with 45% alpha
    overlay_ema = p3.copy()
    overlay_ema[ema_mask_res > 127] = (0, 255, 40)
    p3 = cv2.addWeighted(overlay_ema, 0.45, p3, 0.55, 0)

    # Outer glowing thick boundary (thickness = 3)
    contours_ema, _ = cv2.findContours(ema_mask_res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(p3, contours_ema, -1, (0, 255, 0), 3)

    # Centroid target tracker (Crosshair & Bounding Box)
    if telemetry["centroid"] is not None and telemetry["area_px"] > 200:
        # Scale centroid from 256x256 to 360x360
        scale_c = w_target / 256.0
        cx = int(telemetry["centroid"][0] * scale_c)
        cy = int(telemetry["centroid"][1] * scale_c)
        # Red tracking crosshair
        cv2.drawMarker(p3, (cx, cy), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)

    # Telemetry HUD on Panel 3
    # Semi-transparent HUD background box
    cv2.rectangle(p3, (8, 8), (w_target - 8, 126), (15, 15, 15), -1)
    cv2.rectangle(p3, (8, 8), (w_target - 8, 126), (0, 255, 0), 1)

    fps_val = telemetry["fps"]
    lat_val = telemetry["latency_ms"]
    u_val = telemetry["uncertainty"]
    model_lbl = telemetry["active_model"]
    tci_ema = telemetry["tci_ema"]
    area_px = telemetry["area_px"]

    cv2.putText(p3, "LIVE TELEMETRY HUD (Edge INT8)", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 0), 2)
    cv2.putText(p3, f"Throughput: {fps_val:4.1f} FPS  |  Latency: {lat_val:4.1f} ms", (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    cv2.putText(p3, f"Model: {model_lbl}", (14, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 255) if "Cascade" in model_lbl else (120, 255, 120), 1)

    status_str = f"Lesion: DETECTED ({area_px:,} px)" if area_px > 200 else "Lesion: NONE (Normal Tissue)"
    cv2.putText(p3, f"{status_str}  |  EMA TCI: {tci_ema:.2f}", (14, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0) if area_px > 200 else (200, 200, 200), 1)

    # Uncertainty Meter bar
    bar_x = 14
    bar_y = 100
    bar_w = 180
    bar_h = 10
    cv2.rectangle(p3, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), 1)
    filled_w = int(min(u_val, 1.0) * bar_w)
    bar_color = (0, 0, 255) if u_val > 0.70 else ((0, 200, 255) if u_val > 0.40 else (0, 255, 0))
    cv2.rectangle(p3, (bar_x + 1, bar_y + 1), (bar_x + filled_w, bar_y + bar_h - 1), bar_color, -1)
    cv2.putText(p3, f"Uncertainty: {u_val:.2f}", (bar_x + bar_w + 10, bar_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

    # Combine 3 panels horizontally with divider lines
    divider = np.zeros((h_target, 4, 3), dtype=np.uint8) + 80
    combined = np.hstack([p1, divider, p2, divider, p3])
    return combined


def generate_cine_video(num_frames: int = 120, output_fps: int = 30):
    """
    Renders real-time moving mask cine-loop video (.mp4 and .gif)
    streaming continuous POCUS transducer motion sweep over verified tumor scans.
    Shows prominent, brightly visible moving segmentation mask, EMA anti-flicker smoothing,
    centroid tracking, and uncertainty-gated cascade HUD.
    """
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4_path = out_dir / "cine_realtime_demo.mp4"
    gif_path = out_dir / "cine_realtime_demo.gif"

    print("\n=======================================================")
    print("Generating Real-Time Cine Moving Mask Video (POCUS Sweep)")
    print("=======================================================")

    # Use continuous POCUS transducer sweep over confirmed lesion
    stream = POCUSContinuousSweepStream(case_id="benign (10)", num_frames=num_frames, fps=output_fps)
    pipeline = CineRealtimePipeline()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = None
    frames_for_gif = []

    tcis_raw = []
    tcis_ema = []
    latencies = []

    for f_idx, raw_bgr, tensor, gt_mask in stream:
        res = pipeline.process_frame(tensor)
        tcis_raw.append(res["tci_raw"])
        tcis_ema.append(res["tci_ema"])
        latencies.append(res["latency_ms"])

        frame_rendered = render_hud_panel(raw_bgr, gt_mask, res["raw_mask"], res["ema_mask"], res, f_idx)

        if writer is None:
            h, w = frame_rendered.shape[:2]
            writer = cv2.VideoWriter(str(mp4_path), fourcc, output_fps, (w, h))

        writer.write(frame_rendered)

        # Downsample frames for GIF export (every 2nd frame)
        if f_idx % 2 == 0:
            small_rgb = cv2.cvtColor(cv2.resize(frame_rendered, (w // 2, h // 2)), cv2.COLOR_BGR2RGB)
            frames_for_gif.append(small_rgb)

        if f_idx % 30 == 0 or f_idx == num_frames - 1:
            print(f"  Frame {f_idx:3d}/{num_frames} -> Latency: {res['latency_ms']:.1f} ms ({res['fps']:.1f} FPS), Mask Area: {res['area_px']:,} px, Model: {res['active_model']}")

    if writer:
        writer.release()
    stream.release()

    # Save animated GIF using PIL
    print("Rendering high-quality GIF preview...")
    from PIL import Image
    pil_images = [Image.fromarray(f) for f in frames_for_gif]
    pil_images[0].save(
        str(gif_path),
        save_all=True,
        append_images=pil_images[1:],
        duration=int(1000 / (output_fps / 2)),
        loop=0,
    )

    avg_tci_raw = np.mean(tcis_raw[1:])
    avg_tci_ema = np.mean(tcis_ema[1:])
    avg_lat = np.mean(latencies)
    avg_fps = 1000.0 / avg_lat

    print(f"\nCine Video Rendered:")
    print(f"  MP4 Video: {mp4_path} ({mp4_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  GIF Preview: {gif_path} ({gif_path.stat().st_size / 1e6:.2f} MB)")
    print(f"  Average Throughput: {avg_fps:.1f} FPS ({avg_lat:.2f} ms latency)")
    print(f"  Temporal Consistency Index (TCI): Raw = {avg_tci_raw:.3f} -> EMA = {avg_tci_ema:.3f} (+{(avg_tci_ema - avg_tci_raw)*100:.1f}% anti-flicker stability!)")

    return mp4_path, gif_path


if __name__ == "__main__":
    generate_cine_video(num_frames=90)
