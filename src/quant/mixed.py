import json
from pathlib import Path
import time
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort

from src.quant.audit import evaluate_onnx_model
from src.quant.quantize import quantize_model


def measure_cpu_latency(model_path: str, num_runs: int = 50, num_threads: int = 4) -> float:
    """CPU latency benchmark on validation samples for k-sweep comparison (4-thread edge cpu)."""
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_path), sess_opts, providers=["CPUExecutionProvider"])
    inp = np.random.randn(1, 3, 256, 256).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {"input": inp})

    times = []
    for _ in range(num_runs):
        t0 = time.perf_counter_ns()
        session.run(None, {"input": inp})
        times.append((time.perf_counter_ns() - t0) / 1e6)  # ms

    return float(np.median(times))


def run_mixed_precision_sweep(
    model_name: str = "mbv3s",
    k_grid: Optional[List[int]] = None,
    recovery_threshold: float = 0.90,
) -> Dict[str, Any]:
    """
    Extension A: Run k-sweep, baselines, and select optimal k* on validation data.
    """
    if k_grid is None:
        try:
            import yaml
            with open("configs/quant.yaml", "r") as f:
                qcfg = yaml.safe_load(f)
            k_grid = qcfg.get("extA", {}).get("k_grid", [0, 1, 2, 4, 6, 8, 10, 13])
            if 13 not in k_grid:
                k_grid.append(13)
        except Exception:
            k_grid = [0, 1, 2, 4, 6, 8, 10, 13]
    ranking_csv = Path(f"results/sensitivity_ranking_{model_name}.csv")
    if not ranking_csv.exists():
        from src.quant.sensitivity import analyze_sensitivity
        ranking_df = analyze_sensitivity(model_name)
    else:
        ranking_df = pd.read_csv(ranking_csv)

    ranked_nodes = list(ranking_df["node_name"].values)
    fp32_path = Path(f"models/fp32/{model_name}_prep.onnx")
    if not fp32_path.exists():
        fp32_path = Path(f"models/fp32/{model_name}.onnx")

    mixed_dir = Path("models/mixed")
    mixed_dir.mkdir(parents=True, exist_ok=True)

    # For mbv3s, use u8s8 and restrict quantization to Conv ops
    scheme = "u8s8" if model_name == "mbv3s" else "s8s8"
    op_types = ["Conv", "Gemm", "MatMul"] if model_name == "mbv3s" else None

    print(f"\n==================================================")
    print(f"Extension A: Hybrid Mixed-Precision k-Sweep ({model_name})")
    print(f"Scheme: {scheme}, Op Types: {op_types}")
    print(f"==================================================")

    # 1. Evaluate FP32 baseline on val
    _, fp32_val_summ = evaluate_onnx_model(fp32_path, split="val")
    fp32_val_dice = fp32_val_summ["dsc"]["mean"]
    fp32_size = fp32_path.stat().st_size / (1024 * 1024)
    fp32_lat = measure_cpu_latency(str(fp32_path))

    sweep_results = []

    # 2. Sweep across k
    for k in k_grid:
        excluded = ranked_nodes[:k]
        out_model_path = mixed_dir / f"{model_name}_mix_k{k}.onnx"

        quantize_model(
            model_name=model_name,
            method="minmax",
            num_calib=100,
            scheme=scheme,
            nodes_to_exclude=excluded,
            op_types_to_quantize=op_types,
            output_path=out_model_path,
        )

        _, val_summ = evaluate_onnx_model(str(out_model_path), split="val")
        size_mb = out_model_path.stat().st_size / (1024 * 1024)
        lat_ms = measure_cpu_latency(str(out_model_path))

        sweep_results.append({
            "k": k,
            "excluded_nodes": excluded,
            "val_dsc": val_summ["dsc"]["mean"],
            "val_hd95": val_summ["hd95"]["mean"],
            "val_fpr_50px": val_summ["normal_fpr_50px"],
            "size_mb": size_mb,
            "p50_latency_ms": lat_ms,
            "model_path": str(out_model_path),
        })
        print(f"  k={k:2d} -> Val Dice: {val_summ['dsc']['mean']:.4f}, HD95: {val_summ['hd95']['mean']:.2f} px, Size: {size_mb:.2f} MB, Latency: {lat_ms:.2f} ms")

    int8_val_dice = sweep_results[0]["val_dsc"]
    total_drop = fp32_val_dice - int8_val_dice
    print(f"\nFP32 Val Dice: {fp32_val_dice:.4f} | Full INT8 (k=0) Val Dice: {int8_val_dice:.4f} (Drop: {total_drop:.4f})")

    # 3. Calculate recovery ratio and pick k*
    k_star = 0
    k_star_entry = sweep_results[0]
    contingency_invoked = False

    if total_drop < 0.005:
        # Contingency: full INT8 drop < 0.5 points
        contingency_invoked = True
        print("Note: INT8 accuracy loss is < 0.5 Dice points (near-lossless). Selecting k=1 as primary mixed model.")
        k_star = 1
        k_star_entry = next((e for e in sweep_results if e["k"] == 1), sweep_results[0])
    else:
        for entry in sweep_results:
            rec = (entry["val_dsc"] - int8_val_dice) / total_drop
            entry["recovery_ratio"] = float(rec)
            if rec >= recovery_threshold and k_star == 0 and entry["k"] > 0:
                k_star = entry["k"]
                k_star_entry = entry

        if k_star == 0:
            # Fallback: pick k with highest recovery
            best_entry = max(sweep_results, key=lambda e: e.get("recovery_ratio", 0))
            k_star = best_entry["k"]
            k_star_entry = best_entry

    print(f"\nOptimal setting selected on validation data: k* = {k_star}")
    print(f"  Model Path: {k_star_entry['model_path']}")
    print(f"  Val Dice: {k_star_entry['val_dsc']:.4f}, Val HD95: {k_star_entry['val_hd95']:.2f} px, Size: {k_star_entry['size_mb']:.2f} MB")

    # 4. Baselines:
    # Baseline (a): Exclude first and last conv only
    onnx_fp32 = onnx.load(str(fp32_path))
    conv_nodes = [n.name for n in onnx_fp32.graph.node if n.op_type == "Conv"]
    heuristic_excluded = [conv_nodes[0], conv_nodes[-1]] if len(conv_nodes) >= 2 else []
    heur_path = mixed_dir / f"{model_name}_mix_heuristic_first_last.onnx"
    quantize_model(
        model_name,
        method="minmax",
        scheme=scheme,
        nodes_to_exclude=heuristic_excluded,
        op_types_to_quantize=op_types,
        output_path=heur_path,
    )
    _, heur_summ = evaluate_onnx_model(str(heur_path), split="val")

    # Baseline (b): Random k* nodes excluded (5 seeds)
    rng = np.random.RandomState(42)
    random_dice_scores = []
    quantizable_nodes = ranked_nodes
    k_rand = max(1, k_star)
    for s in [42, 43, 44, 45, 46]:
        rand_nodes = list(np.random.RandomState(s).choice(quantizable_nodes, size=k_rand, replace=False))
        rand_path = mixed_dir / f"{model_name}_mix_random_s{s}.onnx"
        quantize_model(
            model_name,
            method="minmax",
            scheme=scheme,
            nodes_to_exclude=rand_nodes,
            op_types_to_quantize=op_types,
            output_path=rand_path,
        )
        _, rand_summ = evaluate_onnx_model(str(rand_path), split="val")
        random_dice_scores.append(rand_summ["dsc"]["mean"])
        if rand_path.exists():
            rand_path.unlink()

    # Save a stable reference to the best mixed model
    import shutil
    best_stable_path = mixed_dir / f"{model_name}_mixed_best.onnx"
    shutil.copyfile(k_star_entry["model_path"], best_stable_path)
    print(f"Copied best mixed model to stable path: {best_stable_path}")

    # 5. Evaluate chosen k* strictly ONCE on the test split
    print(f"\nEvaluating optimal model k*={k_star} on held-out test split...")
    test_df, test_summ = evaluate_onnx_model(k_star_entry["model_path"], split="test")
    test_csv = Path(f"results/eval_test_{model_name}_mixed_k{k_star}.csv")
    test_df.to_csv(test_csv, index=False)

    payload = {
        "model": model_name,
        "k_star": k_star,
        "total_quantizable_nodes": len(quantizable_nodes),
        "contingency_invoked": contingency_invoked,
        "fp32_val": {"dsc": fp32_val_dice, "size_mb": fp32_size, "latency_ms": fp32_lat},
        "k_sweep": sweep_results,
        "baselines": {
            "first_last_heuristic": {
                "excluded": heuristic_excluded,
                "val_dsc": heur_summ["dsc"]["mean"],
                "val_hd95": heur_summ["hd95"]["mean"],
            },
            "random_k": {
                "k": k_rand,
                "seeds": [42, 43, 44, 45, 46],
                "val_dsc_mean": float(np.mean(random_dice_scores)),
                "val_dsc_min": float(np.min(random_dice_scores)),
                "val_dsc_max": float(np.max(random_dice_scores)),
            }
        },
        "k_star_test_results": {
            "model_path": k_star_entry["model_path"],
            "dsc": test_summ["dsc"],
            "hd95": test_summ["hd95"],
            "normal_fpr_50px": test_summ["normal_fpr_50px"],
            "empty_predictions": test_summ["empty_predictions"],
        }
    }

    out_json = Path("results/extA_sweep.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Extension A sweep results saved to {out_json}")
    return payload


if __name__ == "__main__":
    run_mixed_precision_sweep("mbv3s")
