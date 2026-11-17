import argparse
import json
from pathlib import Path
import subprocess
import sys
import pandas as pd

from src.utils.meta import make_metadata
from src.bench.harness import benchmark_single_process


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Distilled Models (FP32 vs INT8)")
    parser.add_argument("--model-path", type=str, default=None, help="Path to single model for child subprocess")
    parser.add_argument("--distil-dir", type=str, default="models/distil", help="Directory containing distilled models")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup runs")
    parser.add_argument("--timed", type=int, default=500, help="Timed benchmark iterations")
    parser.add_argument("--repeats", type=int, default=3, help="Subprocess repeats per model")
    parser.add_argument("--threads", type=int, default=4, help="intra_op_num_threads")
    parser.add_argument("--single-run-json", action="store_true", help="Child subprocess mode: print JSON and exit")
    parser.add_argument("--out-csv", type=str, default="results/distil_benchmark_table.csv", help="Output CSV path")
    return parser.parse_args()


def benchmark_distilled_models(
    distil_dir: str = "models/distil",
    warmup: int = 50,
    timed: int = 500,
    repeats: int = 3,
    threads: int = 4,
    out_csv: str = "results/distil_benchmark_table.csv",
):
    results_dir = Path(out_csv).parent
    results_dir.mkdir(parents=True, exist_ok=True)

    models_dir = Path(distil_dir)
    models_to_test = []

    # 1. FP32 prep ONNX
    for p in models_dir.glob("*_prep.onnx"):
        variant = "fp32-distilled"
        models_to_test.append((p.stem.replace("_distilled_prep", "").replace("_prep", ""), variant, p))

    # 2. INT8 ONNX
    for p in models_dir.glob("*_int8_*.onnx"):
        parts = p.stem.split("_int8_")
        m_name = parts[0].replace("_distilled", "")
        method = parts[1] if len(parts) > 1 else "unknown"
        variant = f"int8-{method}-distilled"
        models_to_test.append((m_name, variant, p))

    # Fallback to any .onnx in distil_dir if not matched above
    if not models_to_test:
        for p in models_dir.glob("*.onnx"):
            models_to_test.append(("mbv3s", p.stem, p))

    if not models_to_test:
        print(f"No ONNX models found in {distil_dir}. Run export/quantize first.")
        return

    print(f"\n=======================================================")
    print(f"Distilled Model Benchmark: {len(models_to_test)} model variants, {repeats} repeats each")
    print(f"Config: {warmup} warm-up + {timed} timed runs, intra_op_threads={threads}")
    print(f"=======================================================")

    all_rows = []
    for m_name, v_name, m_path in models_to_test:
        print(f"\n--- Benchmarking {m_name} ({v_name}) -> {m_path.name} ---")
        for rep in range(1, repeats + 1):
            cmd = [
                sys.executable, "-m", "src.distil.bench",
                "--model-path", str(m_path),
                "--warmup", str(warmup),
                "--timed", str(timed),
                "--threads", str(threads),
                "--single-run-json",
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"Error benchmarking {m_path}: {proc.stderr}")
                continue

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
                "mean_ms": res["mean_ms"],
                "fps": res["fps"],
                "temp_c": res["temp_after_c"],
                "throttled": res["throttled_flag"],
            }
            all_rows.append(row)
            print(
                f"  Rep {rep}/{repeats}: p50 = {res['p50_ms']:.2f} ms | "
                f"p95 = {res['p95_ms']:.2f} ms | FPS = {res['fps']:.1f} | "
                f"Peak RSS = {res['peak_rss_mb']:.1f} MB | Size = {res['disk_size_mb']:.2f} MB"
            )

    if all_rows:
        df = pd.DataFrame(all_rows)
        csv_file = Path(out_csv)
        df.to_csv(csv_file, index=False)
        print(f"\nBenchmark table saved to {csv_file}")
        summary = df.groupby(["model", "variant"])[["p50_ms", "p95_ms", "fps", "disk_size_mb", "peak_rss_mb"]].mean()
        print("\n--- Summary Averages ---")
        print(summary)


def main():
    args = parse_args()
    if args.single_run_json:
        res = benchmark_single_process(
            model_path=args.model_path,
            num_warmup=args.warmup,
            num_timed=args.timed,
            num_threads=args.threads,
        )
        print("__JSON_START__" + json.dumps(res) + "__JSON_END__")
        return

    benchmark_distilled_models(
        distil_dir=args.distil_dir,
        warmup=args.warmup,
        timed=args.timed,
        repeats=args.repeats,
        threads=args.threads,
        out_csv=args.out_csv,
    )


if __name__ == "__main__":
    main()
