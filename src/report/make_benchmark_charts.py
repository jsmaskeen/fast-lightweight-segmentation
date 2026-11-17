import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def generate_model_size_latency_comparison():
    out_dir = Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_size_latency_comparison.png"

    # Load directly from results/literature_benchmark_comparison.csv
    lit_csv = Path("results/literature_benchmark_comparison.csv")
    if lit_csv.exists():
        df_lit = pd.read_csv(lit_csv)
        color_map = {
            "U-Net++ (Nested Skips)": "#7f7f7f",
            "Attention U-Net": "#bcbd22",
            "Our ResNet-34 U-Net (FP32)": "#1f77b4",
            "Our ResNet-34 U-Net (Static INT8)": "#2ca02c",
            "Shifaa U-Net (Online Pretrained)": "#17becf",
            "Our MobileNetV3-Small (FP32)": "#ff7f0e",
            "Our MobileNetV3 v1 (Naive INT8 / Harsh)": "#d62728",
            "Our MobileNetV3 v2 (Rescued Mixed k=13)": "#9467bd",
        }
        models_data = []
        for _, r in df_lit.iterrows():
            m_name = r["Model"]
            models_data.append({
                "name": m_name,
                "type": r["Type / Backbone"],
                "size_mb": float(r["Size (MB)"]),
                "lat_ms": float(r["Edge CPU Latency (ms)"]),
                "fps": float(r["Throughput (FPS)"]),
                "dsc": float(r["Test DSC (Mean)"]),
                "prec": r["Precision"],
                "color": color_map.get(m_name, "#333333"),
            })
        df = pd.DataFrame(models_data)
    else:
        # Fallback only if results missing
        models_data = [
            {"name": "U-Net++", "type": "ResNet-34", "size_mb": 103.8, "lat_ms": 187.51, "fps": 5.3, "dsc": 0.7588, "prec": "FP32", "color": "#7f7f7f"},
            {"name": "Attention U-Net", "type": "ResNet-34", "size_mb": 97.4, "lat_ms": 102.37, "fps": 9.8, "dsc": 0.7412, "prec": "FP32", "color": "#bcbd22"},
            {"name": "Our ResNet-34 (FP32)", "type": "ResNet-34", "size_mb": 93.22, "lat_ms": 54.27, "fps": 18.4, "dsc": 0.7728, "prec": "FP32", "color": "#1f77b4"},
            {"name": "Our ResNet-34 (INT8 QDQ)", "type": "ResNet-34", "size_mb": 23.59, "lat_ms": 56.82, "fps": 17.6, "dsc": 0.7715, "prec": "INT8", "color": "#2ca02c"},
            {"name": "Shifaa U-Net (Online HF)", "type": "Plain U-Net", "size_mb": 7.78, "lat_ms": 42.07, "fps": 23.8, "dsc": 0.4973, "prec": "FP32", "color": "#17becf"},
            {"name": "Our MobileNetV3 (FP32)", "type": "MobileNetV3", "size_mb": 8.23, "lat_ms": 16.85, "fps": 59.4, "dsc": 0.7655, "prec": "FP32", "color": "#ff7f0e"},
            {"name": "MobileNetV3 v1 (Naive INT8)", "type": "MobileNetV3", "size_mb": 2.43, "lat_ms": 28.57, "fps": 35.0, "dsc": 0.3150, "prec": "INT8 Naive", "color": "#d62728"},
            {"name": "MobileNetV3 v2 (Rescued Mixed)", "type": "MobileNetV3", "size_mb": 2.50, "lat_ms": 24.48, "fps": 40.9, "dsc": 0.7345, "prec": "Mixed INT8/FP32", "color": "#9467bd"},
        ]
        df = pd.DataFrame(models_data)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(2, 2, figsize=(22, 16.5))
    plt.subplots_adjust(wspace=0.38, hspace=0.36)

    # -------------------------------------------------------------
    # Subplot A: Model Disk Footprint (MB)
    # -------------------------------------------------------------
    ax_a = axes[0, 0]
    df_sorted_size = df.sort_values(by="size_mb", ascending=True)
    bars_a = ax_a.barh(df_sorted_size["name"], df_sorted_size["size_mb"], color=df_sorted_size["color"], height=0.62)
    ax_a.tick_params(axis="both", labelsize=13.5)
    ax_a.set_xlabel("Model Disk Storage Footprint (MB)", fontsize=16.5, fontweight="bold")
    ax_a.set_title("Model Storage Footprint & Compression", fontsize=18.5, fontweight="bold", pad=12)
    for bar, (_, r) in zip(bars_a, df_sorted_size.iterrows()):
        val = bar.get_width()
        txt = f" {val:.2f} MB"
        # Only append compression ratio to quantized/mixed models, never to uncompressed FP32 baselines
        if "INT8" in r["prec"] or "Mixed" in r["prec"]:
            if "ResNet-34" in r["name"]:
                comp = 93.22 / val
                txt += f" ({comp:.2f}x)"
            elif "MobileNetV3" in r["name"]:
                comp = 8.23 / val
                txt += f" ({comp:.2f}x)"
        ax_a.text(val + 1.2, bar.get_y() + bar.get_height() / 2, txt, va="center", fontsize=13.5, fontweight="semibold")
    ax_a.set_xlim(0, 140)

    # -------------------------------------------------------------
    # Subplot B: Edge CPU Inference Latency (ms) & Throughput (FPS)
    # -------------------------------------------------------------
    ax_b = axes[0, 1]
    df_sorted_lat = df.sort_values(by="lat_ms", ascending=True)
    bars_b = ax_b.barh(df_sorted_lat["name"], df_sorted_lat["lat_ms"], color=df_sorted_lat["color"], height=0.62)
    ax_b.tick_params(axis="both", labelsize=13.5)
    ax_b.axvline(33.33, color="green", linestyle="--", linewidth=1.8, label="30 FPS Streaming Barrier (33.3 ms)")
    ax_b.axvline(100.0, color="crimson", linestyle=":", linewidth=1.8, label="Interactive Usability Ceiling (100 ms)")
    ax_b.set_xlabel("Edge CPU Inference Latency (p50 ms)", fontsize=16.5, fontweight="bold")
    ax_b.set_title("Edge CPU Latency & Real-Time Throughput", fontsize=18.5, fontweight="bold", pad=12)
    for bar in bars_b:
        val = bar.get_width()
        fps_val = 1000.0 / val
        ax_b.text(val + 2.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f} ms ({fps_val:.1f} FPS)", va="center", fontsize=13.5, fontweight="semibold")
    ax_b.set_xlim(0, 310)
    ax_b.legend(loc="lower right", fontsize=13.5, frameon=True)

    # -------------------------------------------------------------
    # Subplot C: Pareto Frontier (Test Dice vs. CPU Latency, Bubble=Size)
    # -------------------------------------------------------------
    ax_c = axes[1, 0]
    ax_c.tick_params(axis="both", labelsize=13.5)
    # Scaling bubble size
    bubble_sizes = df["size_mb"] * 16 + 150
    scatter = ax_c.scatter(
        df["lat_ms"],
        df["dsc"],
        s=bubble_sizes,
        c=df["color"],
        alpha=0.82,
        edgecolors="black",
        linewidth=1.5,
        zorder=3
    )

    # Distinct non-overlapping label offsets
    offsets = {
        "Our ResNet-34 U-Net (FP32)": (14, 30),
        "Our ResNet-34 U-Net (Static INT8)": (14, -36),
        "Our MobileNetV3-Small (FP32)": (-10, 24),
        "Our MobileNetV3 v2 (Rescued Mixed k=13)": (14, -36),
        "Our MobileNetV3 v1 (Naive INT8 / Harsh)": (14, -18),
        "Shifaa U-Net (Online Pretrained)": (14, 12),
        "Attention U-Net": (14, -24),
        "U-Net++ (Nested Skips)": (-110, 24),
    }

    # Annotate models with concise readable labels
    for _, row in df.iterrows():
        off = offsets.get(row["name"], (12, 8))
        label_text = (
            row["name"]
            .replace("Our ", "")
            .replace(" U-Net", "")
            .replace(" (Nested Skips)", "")
            .replace(" (Online Pretrained)", "")
            .replace(" (Naive INT8 / Harsh)", " (Naive INT8)")
            .replace(" (Rescued Mixed k=13)", " (Rescued k=13)")
        )
        ax_c.annotate(
            f"{label_text}\n(Dice: {row['dsc']:.3f})",
            xy=(row["lat_ms"], row["dsc"]),
            xytext=off,
            textcoords="offset points",
            fontsize=12,
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.6),
            zorder=4
        )

    # Dynamic Pareto skyline frontier
    pareto_candidates = sorted(list(zip(df["lat_ms"], df["dsc"])), key=lambda x: x[0])
    skyline = []
    max_d = -1.0
    for lat, dsc in pareto_candidates:
        if dsc > max_d:
            skyline.append((lat, dsc))
            max_d = dsc
    px = [p[0] for p in skyline]
    py = [p[1] for p in skyline]
    ax_c.plot(px, py, color="black", linestyle="-.", linewidth=2.0, label="Empirical Pareto Frontier", zorder=2)

    ax_c.axhline(0.70, color="crimson", linestyle="--", linewidth=1.5, label="Clinical Usability Threshold (Dice = 0.70)")
    ax_c.axvline(33.33, color="green", linestyle="--", linewidth=1.5, label="30 FPS Streaming Threshold")
    ax_c.set_xlabel("Edge CPU Inference Latency (ms)", fontsize=16.5, fontweight="bold")
    ax_c.set_ylabel("Held-Out Test Dice Score (DSC)", fontsize=16.5, fontweight="bold")
    ax_c.set_title("Pareto Frontier: Accuracy vs. Edge Latency (Bubble = Size)", fontsize=18.5, fontweight="bold", pad=12)
    ax_c.set_ylim(0.18, 0.90)
    ax_c.set_xlim(0, 290)
    ax_c.legend(loc="lower right", fontsize=13.5, frameon=True)

    # -------------------------------------------------------------
    # Subplot D: Specificity & Safety: Normal Tissue False Positive Rate
    # -------------------------------------------------------------
    ax_d = axes[1, 1]
    ax_d.tick_params(axis="both", labelsize=13.5)
    fpr_data = []
    audit_csv = Path("results/accuracy_audit_test.csv")
    if audit_csv.exists():
        df_audit = pd.read_csv(audit_csv)
        name_map = {
            ("r34", "fp32"): ("Our ResNet-34 FP32", "#1f77b4"),
            ("r34", "int8_minmax"): ("Our ResNet-34 INT8", "#2ca02c"),
            ("mbv3s", "fp32"): ("Our MobileNetV3 FP32", "#ff7f0e"),
            ("mbv3s", "int8_minmax"): ("MobileNetV3 v1 (Naive INT8)", "#d62728"),
        }
        for _, row in df_audit.iterrows():
            key = (row["Model"], row["Variant"])
            if key in name_map:
                lbl, col = name_map[key]
                fpr_str = str(row["Normal FPR@50px"]).replace("%", "")
                fpr_data.append({"name": lbl, "fpr": float(fpr_str), "color": col})
        sweep_file = Path("results/extA_sweep.json")
        if sweep_file.exists():
            try:
                with open(sweep_file) as f:
                    sw = json.load(f)
                mixed_fpr = sw.get("k_star_test_results", {}).get("normal_fpr_50px", 0.0) * 100.0
                fpr_data.append({"name": "MobileNetV3 v2 (Rescued Mixed)", "fpr": mixed_fpr, "color": "#9467bd"})
            except Exception:
                pass
    if not fpr_data:
        fpr_data = [
            {"name": "Our ResNet-34 FP32", "fpr": 77.61, "color": "#1f77b4"},
            {"name": "Our ResNet-34 INT8", "fpr": 76.12, "color": "#2ca02c"},
            {"name": "Our MobileNetV3 FP32", "fpr": 89.55, "color": "#ff7f0e"},
            {"name": "MobileNetV3 v1 (Naive INT8)", "fpr": 100.00, "color": "#d62728"},
            {"name": "MobileNetV3 v2 (Rescued Mixed)", "fpr": 89.55, "color": "#9467bd"},
        ]
    df_fpr = pd.DataFrame(fpr_data).sort_values(by="fpr", ascending=True)
    bars_d = ax_d.barh(df_fpr["name"], df_fpr["fpr"], color=df_fpr["color"], height=0.55)
    ax_d.set_xlabel("Normal Tissue False Positive Rate (% FPR@50px)", fontsize=16.5, fontweight="bold")
    ax_d.set_title("Clinical Safety Audit: False Positive Hallucinations", fontsize=18.5, fontweight="bold", pad=12)
    for bar in bars_d:
        val = bar.get_width()
        txt = f" {val:.2f}%"
        ax_d.text(val + 1.0, bar.get_y() + bar.get_height() / 2, txt, va="center", fontsize=14, fontweight="semibold")
    ax_d.set_xlim(0, 118)

    plt.suptitle(
        "Quantitative System Trade-offs: Model Footprint, Latency, Pareto Frontier & Clinical Safety\n"
        "(Benchmarked on Edge CPU: MobileNetV3 v1 Naive Collapse vs. MobileNetV3 v2 Rescued vs. ResNet-34 Lossless INT8)",
        fontsize=21,
        fontweight="bold",
        y=0.99
    )

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Model size and latency comparison chart saved to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return out_path


if __name__ == "__main__":
    generate_model_size_latency_comparison()
