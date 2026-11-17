import argparse
import json
import math
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extension A: User-adjustable Mixed-Precision Sensitivity Knob (Zero-Recomputation)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--keep-fp32", type=int, help="Fixed count: keep N most sensitive nodes in FP32")
    group.add_argument("--keep-fp32-pct", type=float, help="Fraction: keep top P% of nodes (rounded up) in FP32")
    group.add_argument("--max-dice-drop", type=float, help="Accuracy budget: max allowed val Dice drop from FP32 (e.g. 0.01 or 0.5)")
    parser.add_argument("--sweep-file", type=str, default="results/extA_sweep.json", help="Path to precomputed sweep JSON")
    return parser.parse_args()


def snap_k(target_k: int, available_k: list) -> tuple:
    """Snap target k upward to the next available swept k."""
    sorted_k = sorted(available_k)
    for k in sorted_k:
        if k >= target_k:
            snapped = (k != target_k)
            return k, snapped
    # If higher than max, clamp to max
    return sorted_k[-1], True


def main():
    args = parse_args()
    sweep_path = Path(args.sweep_file)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep file {sweep_path} not found. Run 'make extA' first.")

    with open(sweep_path, "r") as f:
        data = json.load(f)

    sweep_entries = data["k_sweep"]
    available_k = [e["k"] for e in sweep_entries]
    total_quantizable_nodes = data.get("total_quantizable_nodes", 63)
    fp32_val_dice = data["fp32_val"]["dsc"]

    selected_k = None
    was_snapped = False
    requested_desc = ""

    if args.keep_fp32 is not None:
        target = max(0, args.keep_fp32)
        selected_k, was_snapped = snap_k(target, available_k)
        requested_desc = f"--keep-fp32 {args.keep_fp32}"

    elif args.keep_fp32_pct is not None:
        target = int(math.ceil((args.keep_fp32_pct / 100.0) * total_quantizable_nodes))
        selected_k, was_snapped = snap_k(target, available_k)
        requested_desc = f"--keep-fp32-pct {args.keep_fp32_pct}% (computed k={target})"

    elif args.max_dice_drop is not None:
        # Interpret drop: if user gives 0.5 (meaning 0.5% or 0.005) or 0.005
        max_drop = args.max_dice_drop if args.max_dice_drop < 0.1 else (args.max_dice_drop / 100.0)
        requested_desc = f"--max-dice-drop {max_drop:.4f}"

        # Find smallest k whose val_dsc >= fp32_val_dice - max_drop
        feasible = [e for e in sweep_entries if (fp32_val_dice - e["val_dsc"]) <= max_drop]
        if feasible:
            best = min(feasible, key=lambda e: e["k"])
            selected_k = best["k"]
        else:
            print(f"Warning: No setting meets accuracy budget (max drop {max_drop:.4f}).")
            closest = max(sweep_entries, key=lambda e: e["val_dsc"])
            selected_k = closest["k"]
            print(f"Falling back to closest available setting: k={selected_k} (drop: {fp32_val_dice - closest['val_dsc']:.4f})")

    entry = next(e for e in sweep_entries if e["k"] == selected_k)

    print("\n=======================================================")
    print("BUSI Mixed-Precision Sensitivity Knob Selection")
    print("=======================================================")
    print(f"Request:          {requested_desc}")
    if was_snapped:
        print(f"Snapping Notice:  Value snapped upward to swept grid k={selected_k}")
    print(f"Selected Model:   {entry['model_path']}")
    print(f"Nodes in FP32:    k={selected_k} ({entry['excluded_nodes']})")
    print(f"Val DSC:          {entry['val_dsc']:.4f} (Drop vs FP32: {fp32_val_dice - entry['val_dsc']:+.4f})")
    print(f"Val HD95:         {entry['val_hd95']:.2f} px")
    print(f"Model Disk Size:  {entry['size_mb']:.2f} MB (FP32: {data['fp32_val']['size_mb']:.2f} MB)")
    print(f"p50 CPU Latency:  {entry['p50_latency_ms']:.2f} ms (FP32: {data['fp32_val']['latency_ms']:.2f} ms)")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
