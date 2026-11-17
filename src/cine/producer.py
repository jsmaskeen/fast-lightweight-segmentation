import os
import time
import cv2
from pathlib import Path
from typing import List

from src.cine.ring_buffer import CineRingBuffer, FramePacket


def set_core_affinity(core_id: int = 0):
    """Set CPU core affinity for producer thread (Linux/RPi4)."""
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, {core_id})
        except Exception as e:
            pass


def run_producer(
    ring_buffer: CineRingBuffer,
    test_image_paths: List[Path],
    rate_hz: float = 30.0,
    duration_sec: float = 60.0,
    core_id: int = 0
) -> List[dict]:
    """
    Producer thread emitting frames at strict 1/f absolute deadlines.
    Avoids cumulative sleep drift by sleeping until near deadline then spin-waiting.
    """
    set_core_affinity(core_id)

    # Pre-load all test images into RAM at native resolution
    preloaded = []
    for p in test_image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            preloaded.append((p.name, img))

    if not preloaded:
        raise ValueError("No test images available for cine-loop.")

    num_images = len(preloaded)
    period = 1.0 / rate_hz
    total_frames = int(round(duration_sec * rate_hz))

    jitter_logs = []
    t_start = time.perf_counter()

    for idx in range(total_frames):
        target_t = t_start + idx * period

        # Hybrid sleep + spin-wait for high-precision absolute deadline
        now = time.perf_counter()
        sleep_time = target_t - now - 0.0015
        if sleep_time > 0:
            time.sleep(sleep_time)

        # Spin-wait last 1.5 milliseconds
        while time.perf_counter() < target_t:
            pass

        t_capture = time.perf_counter()
        jitter_ms = (t_capture - target_t) * 1000.0

        img_name, frame_img = preloaded[idx % num_images]
        packet = FramePacket(frame_id=idx, frame_img=frame_img, t_capture=t_capture)

        was_dropped = ring_buffer.push(packet)

        jitter_logs.append({
            "frame_id": idx,
            "target_t": target_t,
            "t_capture": t_capture,
            "jitter_ms": jitter_ms,
            "dropped_on_push": was_dropped,
        })

    ring_buffer.close()
    return jitter_logs
