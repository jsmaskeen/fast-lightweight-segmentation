import gc
import time
import cv2
import numpy as np
import onnxruntime as ort
from typing import List

from src.cine.ring_buffer import CineRingBuffer


class CineConsumer:
    """
    Consumer thread:
    - Always pops newest available frame from ring buffer.
    - Preprocessing: resize to 256x256, RGB conversion, ImageNet normalization.
    - Inference: ONNX Runtime session.run (releases GIL).
    - Postprocessing: thresholding at logit > 0.0 + cv2.findContours.
    - Disables garbage collection during active loop to avoid timing spikes.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str, num_threads: int = 4, num_warmup: int = 20):
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = num_threads
        sess_opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        
        # Warmup session to initialize threadpools and memory allocations
        dummy = np.zeros((1, 3, 256, 256), dtype=np.float32)
        for _ in range(num_warmup):
            _ = self.session.run(None, {self.input_name: dummy})

    def run(self, ring_buffer: CineRingBuffer) -> List[dict]:
        gc.disable()
        frame_logs = []

        try:
            while True:
                item = ring_buffer.pop_latest(timeout=0.2)
                if item is None:
                    if ring_buffer.is_closed:
                        break
                    continue

                packet, t_dequeue = item

                # 1. Preprocessing
                t_pre_start = time.perf_counter()
                frame_gray = packet.image
                resized = cv2.resize(frame_gray, (256, 256), interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
                norm = (rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
                inp = norm.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 256, 256)
                t_pre_done = time.perf_counter()

                # 2. Inference (releases GIL)
                ort_out = self.session.run(None, {self.input_name: inp})[0][0, 0]
                t_infer_done = time.perf_counter()

                # 3. Postprocessing
                bin_mask = (ort_out > 0.0).astype(np.uint8) * 255
                contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                t_post_done = time.perf_counter()

                e2e_latency_ms = (t_post_done - packet.t_capture) * 1000.0
                queue_latency_ms = (t_dequeue - packet.t_capture) * 1000.0
                pre_ms = (t_pre_done - t_pre_start) * 1000.0
                infer_ms = (t_infer_done - t_pre_done) * 1000.0
                post_ms = (t_post_done - t_infer_done) * 1000.0

                frame_logs.append({
                    "frame_id": packet.frame_id,
                    "t_capture": packet.t_capture,
                    "t_dequeue": t_dequeue,
                    "t_pre_done": t_pre_done,
                    "t_infer_done": t_infer_done,
                    "t_post_done": t_post_done,
                    "queue_ms": queue_latency_ms,
                    "pre_ms": pre_ms,
                    "infer_ms": infer_ms,
                    "post_ms": post_ms,
                    "e2e_ms": e2e_latency_ms,
                    "num_contours": len(contours),
                })
        finally:
            gc.enable()

        return frame_logs
