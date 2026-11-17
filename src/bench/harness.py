import os
import sys
import argparse
import subprocess
import time
import json
import psutil
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import onnxruntime as ort

from src.utils.meta import make_metadata


def check_throttling() -> Dict[str, Any]:
    """Check temperature and CPU throttling flags (RPi4 or local)."""
    info = {"temp_c": np.nan, "throttled_flag": "N/A"}
    try:
        # Raspberry Pi specific
        temp_out = subprocess.check_output(["vcgencmd", "measure_temp"], text=True, stderr=subprocess.DEVNULL)
        info["temp_c"] = float(temp_out.replace("temp=", "").replace("'C", "").strip())
        throt_out = subprocess.check_output(["vcgencmd", "get_throttled"], text=True, stderr=subprocess.DEVNULL)
        info["throttled_flag"] = throt_out.replace("throttled=", "").strip()
    except Exception:
        pass
    return info


def benchmark_single_process(
    model_path: str,
    num_warmup: int = 50,
    num_timed: int = 500,
    num_threads: int = 4,
) -> Dict[str, Any]:
    """
    Run benchmark in a fresh process:
    - Exactly 50 warm-up runs.
    - Exactly 500 timed runs, timing ONLY session.run with perf_counter_ns().
    - Records peak RSS and latency percentiles (p50, p95, p99).
    """
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model file {m_path} does not exist.")

    disk_size_mb = m_path.stat().st_size / (1024 * 1024)

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(m_path), sess_opts, providers=["CPUExecutionProvider"])

    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, 256, 256).astype(np.float32)
    feed = {input_name: dummy_input}

    throt_before = check_throttling()

    # 50 Warm-up runs
    for _ in range(num_warmup):
        session.run(None, feed)

    # 500 Timed runs (ONLY session.run timed with perf_counter_ns)
    latencies_ns = np.empty(num_timed, dtype=np.int64)
    for i in range(num_timed):
        t0 = time.perf_counter_ns()
        session.run(None, feed)
        latencies_ns[i] = time.perf_counter_ns() - t0

    throt_after = check_throttling()

    # Memory RSS (Peak working set on Windows, RSS on POSIX)
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    peak_rss_mb = getattr(mem_info, "peak_wset", mem_info.rss) / (1024 * 1024)

    latencies_ms = latencies_ns / 1e6
    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))
    mean_lat = float(np.mean(latencies_ms))
    fps = (1000.0 / mean_lat) if mean_lat > 0 else 0.0

    return {
        "model_path": str(m_path),
        "disk_size_mb": disk_size_mb,
        "peak_rss_mb": peak_rss_mb,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "mean_ms": mean_lat,
        "fps": fps,
        "temp_before_c": throt_before["temp_c"],
        "temp_after_c": throt_after["temp_c"],
        "throttled_flag": throt_after["throttled_flag"],
    }


def main():
    parser = argparse.ArgumentParser(description="Edge CPU Benchmark Harness")
    parser.add_argument("--model-path", type=str, default=None, help="Path to single model for child subprocess")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed", type=int, default=500)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--single-run-json", action="store_true", help="Child subprocess mode: print JSON and exit")
    args = parser.parse_args()

    # Child subprocess mode: execute single run and output JSON
    if args.single_run_json:
        res = benchmark_single_process(
            model_path=args.model_path,
            num_warmup=args.warmup,
            num_timed=args.timed,
            num_threads=args.threads
        )
        print("__JSON_START__" + json.dumps(res) + "__JSON_END__")
        return

    # Master orchestrator: runs fresh subprocess per model and repeat
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Discover candidate models
    models_to_test = []
    for p in Path("models/fp32").glob("*_prep.onnx"):
        m_name = p.stem.replace("_prep", "")
        models_to_test.append((m_name, "fp32", p))
    for p in Path("models/int8").glob("*.onnx"):
        if "tmp" in p.name:
            continue
        parts = p.stem.split("_int8_")
        m_name = parts[0]
        v_name = f"int8-{parts[1]}" if len(parts) > 1 else "int8"
        models_to_test.append((m_name, v_name, p))
    k_star = 2
    if Path("results/extA_sweep.json").exists():
        try:
            with open("results/extA_sweep.json") as f:
                k_star = json.load(f).get("k_star", 2)
        except Exception:
            pass
    target_mixed = [f"mbv3s_mix_k{k_star}", "mbv3s_mixed_best", "mbv3s_mix_heuristic_first_last"]
    for p in Path("models/mixed").glob("*.onnx"):
        if getattr(args, "include_all_mixed", False) or p.stem in target_mixed:
            models_to_test.append(("mbv3s", p.stem.replace("mbv3s_", ""), p))

    if not models_to_test:
        print("No exported ONNX models found in models/fp32 or models/int8.")
        return

    all_rows = []
    print(f"\n=======================================================")
    print(f"Edge CPU Benchmark: {len(models_to_test)} models, {args.repeats} repeats each")
    print(f"Config: 50 warm-up + 500 timed runs, intra_op_threads={args.threads}")
    print(f"=======================================================")

    for m_name, v_name, m_path in models_to_test:
        print(f"\n--- Benchmarking {m_name} ({v_name}) ---")
        for rep in range(1, args.repeats + 1):
            cmd = [
                sys.executable, "-m", "src.bench.harness",
                "--model-path", str(m_path),
                "--warmup", str(args.warmup),
                "--timed", str(args.timed),
                "--threads", str(args.threads),
                "--single-run-json"
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"Error running benchmark for {m_path}: {proc.stderr}")
                continue

            # Extract JSON output
            out = proc.stdout
            start = out.find("__JSON_START__")
            end = out.find("__JSON_END__")
            if start == -1 or end == -1:
                print(f"Failed to parse JSON output: {out}")
                continue

            res = json.loads(out[start + len("__JSON_START__") : end])
            meta = make_metadata(m_name, v_name)

            row = {
                **meta,
                "repeat": rep,
                "disk_size_mb": res["disk_size_mb"],
                "peak_rss_mb": res["peak_rss_mb"],
                "p50_ms": res["p50_ms"],
                "p95_ms": res["p95_ms"],
                "p99_ms": res["p99_ms"],
                "fps": res["fps"],
                "temp_c": res["temp_after_c"],
                "throttled": res["throttled_flag"],
            }
            all_rows.append(row)
            print(f"  Rep {rep}/{args.repeats}: p50 = {res['p50_ms']:.2f} ms | p95 = {res['p95_ms']:.2f} ms | FPS = {res['fps']:.1f} | Peak RSS = {res['peak_rss_mb']:.1f} MB")

    df = pd.DataFrame(all_rows)
    out_csv = results_dir / "edge_benchmark_table.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nBenchmark table saved to {out_csv}")
    print(df.groupby(["model", "variant"])[["p50_ms", "p95_ms", "fps", "disk_size_mb", "peak_rss_mb"]].mean())


if __name__ == "__main__":
    main()
