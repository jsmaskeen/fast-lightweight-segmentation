import argparse
import json
import threading
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd

from src.cine.ring_buffer import CineRingBuffer
from src.cine.producer import run_producer
from src.cine.consumer import CineConsumer


def parse_args():
    parser = argparse.ArgumentParser(description="Extension B: Simulated Real-Time POCUS Cine-Loop")
    parser.add_argument("--duration", type=float, default=60.0, help="Duration in seconds per run (default: 60.0)")
    parser.add_argument("--rates", type=int, nargs="+", default=[10, 15, 20, 30], help="Cine frame rates to test (Hz)")
    parser.add_argument("--repeats", type=int, default=3, help="Repeats per rate (default: 3)")
    parser.add_argument("--models", type=str, nargs="+", default=None, help="Model variants to evaluate")
    parser.add_argument("--num-threads", type=int, default=4, help="Inference intra-op threads")
    return parser.parse_args()


def run_single_cine_session(
    model_path: str,
    rate_hz: float,
    duration_sec: float,
    test_image_paths: List[Path],
    num_threads: int = 4,
    capacity: int = 4,
) -> Dict[str, Any]:
    """Execute a single Producer -> RingBuffer -> Consumer cine-loop streaming session."""
    ring_buf = CineRingBuffer(capacity=capacity)
    consumer = CineConsumer(model_path, num_threads=num_threads)

    producer_logs = []
    consumer_logs = []

    def prod_worker():
        nonlocal producer_logs
        producer_logs = run_producer(
            ring_buf,
            test_image_paths,
            rate_hz=rate_hz,
            duration_sec=duration_sec,
            core_id=0
        )

    def cons_worker():
        nonlocal consumer_logs
        consumer_logs = consumer.run(ring_buf)

    t_prod = threading.Thread(target=prod_worker)
    t_cons = threading.Thread(target=cons_worker)

    t_cons.start()
    t_prod.start()

    t_prod.join()
    t_cons.join()

    stats = ring_buf.get_stats()
    produced = stats["produced"]
    processed = stats["processed"]
    dropped = stats["dropped"]
    drop_rate = (dropped / produced) if produced > 0 else 0.0

    # Compute consumer e2e latencies
    e2e_latencies = [l["e2e_ms"] for l in consumer_logs] if consumer_logs else [np.nan]
    p50_e2e = float(np.percentile(e2e_latencies, 50))
    p95_e2e = float(np.percentile(e2e_latencies, 95))
    p99_e2e = float(np.percentile(e2e_latencies, 99))

    # Deadline miss rate against 100 ms budget
    deadline_misses = sum(1 for lat in e2e_latencies if lat > 100.0)
    deadline_miss_rate = (deadline_misses / len(e2e_latencies)) if e2e_latencies else 1.0

    # Producer jitter
    jitters = [abs(l["jitter_ms"]) for l in producer_logs] if producer_logs else [0.0]
    p50_jitter = float(np.median(jitters))

    return {
        "rate_hz": rate_hz,
        "produced": produced,
        "processed": processed,
        "dropped": dropped,
        "drop_rate": drop_rate,
        "p50_e2e_ms": p50_e2e,
        "p95_e2e_ms": p95_e2e,
        "p99_e2e_ms": p99_e2e,
        "deadline_miss_rate": deadline_miss_rate,
        "p50_jitter_ms": p50_jitter,
        "consumer_logs": consumer_logs,
    }


def main():
    args = parse_args()
    results_dir = Path("results/cine")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open("data/splits.json", "r") as f:
        test_ids = json.load(f)["test"]
    test_image_paths = [Path("data/processed/images") / f"{img_id}.png" for img_id in test_ids]

    candidate_models = {
        "mbv3s_fp32": Path("models/fp32/mbv3s_prep.onnx"),
        "mbv3s_int8": Path("models/int8/mbv3s_int8_minmax.onnx"),
        "r34_fp32": Path("models/fp32/r34_prep.onnx"),
        "r34_int8": Path("models/int8/r34_int8_minmax.onnx"),
    }
    # Check for mixed precision model dynamically
    best_mixed_path = Path("models/mixed/mbv3s_mixed_best.onnx")
    sweep_path = Path("results/extA_sweep.json")
    if best_mixed_path.exists():
        candidate_models["mbv3s_mixed"] = best_mixed_path
    elif sweep_path.exists():
        try:
            with open(sweep_path, "r") as f:
                sweep_data = json.load(f)
            k_star = sweep_data.get("k_star")
            if k_star is not None and Path(f"models/mixed/mbv3s_mix_k{k_star}.onnx").exists():
                candidate_models["mbv3s_mixed"] = Path(f"models/mixed/mbv3s_mix_k{k_star}.onnx")
        except Exception:
            pass
    if "mbv3s_mixed" not in candidate_models:
        mixed_models = list(Path("models/mixed").glob("mbv3s_mix_k*.onnx"))
        if mixed_models:
            candidate_models["mbv3s_mixed"] = mixed_models[0]

    matrix_records = []
    print("\n=======================================================")
    print("Extension B: Simulated Real-Time POCUS Cine-Loop Matrix")
    print("=======================================================")

    for v_name, m_path in candidate_models.items():
        if not m_path.exists():
            print(f"Skipping {v_name}: {m_path} does not exist.")
            continue

        print(f"\n--- Testing Variant: {v_name} ({m_path}) ---")
        for rate in args.rates:
            for rep in range(1, args.repeats + 1):
                res = run_single_cine_session(
                    model_path=str(m_path),
                    rate_hz=rate,
                    duration_sec=args.duration,
                    test_image_paths=test_image_paths,
                    num_threads=args.num_threads,
                )
                print(f"  Rate: {rate:2d} Hz | Rep: {rep}/{args.repeats} -> Drops: {res['dropped']}/{res['produced']} ({res['drop_rate']*100:.2f}%) | e2e p50: {res['p50_e2e_ms']:.1f}ms, p95: {res['p95_e2e_ms']:.1f}ms")

                matrix_records.append({
                    "variant": v_name,
                    "model_path": str(m_path),
                    "rate_hz": rate,
                    "repeat": rep,
                    "duration_sec": args.duration,
                    "produced": res["produced"],
                    "processed": res["processed"],
                    "dropped": res["dropped"],
                    "drop_rate": res["drop_rate"],
                    "p50_e2e_ms": res["p50_e2e_ms"],
                    "p95_e2e_ms": res["p95_e2e_ms"],
                    "p99_e2e_ms": res["p99_e2e_ms"],
                    "deadline_miss_rate": res["deadline_miss_rate"],
                    "p50_jitter_ms": res["p50_jitter_ms"],
                })

    df = pd.DataFrame(matrix_records)
    summary_csv = Path("results/cine_matrix_summary.csv")
    df.to_csv(summary_csv, index=False)
    print(f"\nCine matrix results saved to {summary_csv}")

    # Maximum Sustainable Rate per variant (drop rate < 1%)
    print("\n--- Maximum Sustainable Frame Rate (Drop Rate < 1%) ---")
    for v_name, grp in df.groupby("variant"):
        mean_drops_per_rate = grp.groupby("rate_hz")["drop_rate"].mean()
        sustainable_rates = [r for r, dr in mean_drops_per_rate.items() if dr < 0.01]
        max_rate = max(sustainable_rates) if sustainable_rates else 0
        print(f"  {v_name:15s}: {max_rate:2d} Hz sustainable (Mean drop at 30Hz: {mean_drops_per_rate.get(30, 0.0)*100:.1f}%)")


if __name__ == "__main__":
    main()
