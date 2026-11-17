import time
import threading
from pathlib import Path
import numpy as np
import pytest

from src.cine.ring_buffer import CineRingBuffer, FramePacket
from src.cine.producer import run_producer


def test_ring_buffer_concurrency_invariant():
    """Verify invariant: produced == processed + dropped under heavy saturation."""
    buf = CineRingBuffer(capacity=4)

    # Fast producer pushing 50 packets
    def producer_worker():
        for i in range(50):
            dummy_img = np.zeros((256, 256), dtype=np.uint8)
            pkt = FramePacket(frame_id=i, frame_img=dummy_img, t_capture=time.perf_counter())
            buf.push(pkt)
            time.sleep(0.001)  # 1 ms period
        buf.close()

    # Slow consumer popping packets with 5 ms delay
    consumer_count = [0]
    def consumer_worker():
        while True:
            item = buf.pop_latest(timeout=0.1)
            if item is None:
                if buf.is_closed:
                    break
                continue
            consumer_count[0] += 1
            time.sleep(0.005)

    t1 = threading.Thread(target=producer_worker)
    t2 = threading.Thread(target=consumer_worker)

    t1.start()
    t2.start()

    t1.join()
    t2.join()

    stats = buf.get_stats()
    assert stats["produced"] == 50, f"Expected 50 produced, got {stats['produced']}"
    assert stats["processed"] + stats["dropped"] == 50, f"Invariant violated: processed({stats['processed']}) + dropped({stats['dropped']}) != 50"
    assert stats["dropped"] > 0, "Slow consumer should have triggered frame drops in capacity 4 buffer"


def test_producer_jitter_noop():
    """Verify producer emission jitter remains low relative to frame period."""
    buf = CineRingBuffer(capacity=10)
    dummy_imgs = [Path("data/processed/images/benign (1).png")]
    if not dummy_imgs[0].exists():
        pytest.skip("Processed images not found yet.")

    # Run for 0.5s at 30 Hz (15 frames)
    jitters = run_producer(buf, dummy_imgs, rate_hz=30.0, duration_sec=0.5)
    jitter_vals = [abs(j["jitter_ms"]) for j in jitters]

    # Mean jitter should be well under 5 ms
    mean_jitter = np.mean(jitter_vals)
    assert mean_jitter < 5.0, f"Mean producer jitter {mean_jitter:.2f} ms exceeds 5 ms threshold"


def test_pop_latest_returns_freshest_frame():
    """Verify pop_latest() strictly returns the newest frame and marks older ones dropped."""
    buf = CineRingBuffer(capacity=4)
    # Push 3 frames
    buf.push(FramePacket(0, np.zeros((10, 10)), time.perf_counter()))
    buf.push(FramePacket(1, np.zeros((10, 10)), time.perf_counter()))
    buf.push(FramePacket(2, np.zeros((10, 10)), time.perf_counter()))

    # Consumer pops once: should get frame 2 (freshest)
    item = buf.pop_latest(timeout=0.1)
    assert item is not None
    pkt, _ = item
    assert pkt.frame_id == 2, f"Expected newest frame (id=2), got frame id={pkt.frame_id}"

    stats = buf.get_stats()
    assert stats["produced"] == 3
    assert stats["processed"] == 1
    assert stats["dropped"] == 2  # Frames 0 and 1 dropped
    assert stats["in_flight"] == 0
    assert stats["produced"] == stats["processed"] + stats["dropped"]

