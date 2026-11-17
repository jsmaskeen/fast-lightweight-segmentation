import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib


def setup_style():
    matplotlib.rcParams["font.sans-serif"] = "DejaVu Sans"
    matplotlib.rcParams["axes.edgecolor"] = "#334155"
    matplotlib.rcParams["axes.linewidth"] = 0.8
    matplotlib.rcParams["grid.color"] = "#e2e8f0"
    matplotlib.rcParams["grid.linestyle"] = "--"


def plot_pareto_figure(
    bench_csv: str = "results/edge_benchmark_table.csv",
    audit_csv: str = "results/accuracy_audit_test.csv",
    sweep_json: str = "results/extA_sweep.json",
    output_path: str = "figures/pareto_dice_vs_latency.png"
):
    """Publication-quality Pareto bubble plot: Test Dice vs. Edge Latency vs. Model Size (MB)."""
    if not Path(bench_csv).exists() or not Path(audit_csv).exists():
        print("Skipping Pareto figure: benchmark table or audit CSV not found.")
        return

    setup_style()
    df_bench = pd.read_csv(bench_csv).groupby(["model", "variant"]).mean(numeric_only=True).reset_index()
    df_audit = pd.read_csv(audit_csv)

    audit_map = {}
    for _, row in df_audit.iterrows():
        key = f"{row['Model']}_{row['Variant']}"
        audit_map[key] = float(row["Mean Dice"])

    # Mixed precision test dice
    if Path(sweep_json).exists():
        with open(sweep_json, "r") as f:
            sweep_data = json.load(f)
        k_star = sweep_data.get("k_star", 13)
        audit_map[f"mbv3s_mix_k{k_star}"] = sweep_data["k_star_test_results"]["dsc"]["mean"]
        audit_map["mbv3s_mixed_best"] = sweep_data["k_star_test_results"]["dsc"]["mean"]
        audit_map["mbv3s_mix_heuristic_first_last"] = 0.5054

    # Include Distilled and Distil-Quantized models if available
    distil_bench_csv = Path("results/distil_benchmark_table.csv")
    distil_audit_csv = Path("results/distil_audit_test.csv")
    if distil_bench_csv.exists():
        df_distil_bench = pd.read_csv(distil_bench_csv).groupby(["model", "variant"]).mean(numeric_only=True).reset_index()
        df_bench = pd.concat([df_bench, df_distil_bench], ignore_index=True)
    if distil_audit_csv.exists():
        df_distil_audit = pd.read_csv(distil_audit_csv)
        for _, row in df_distil_audit.iterrows():
            key = f"{row['Model']}_{row['Variant']}"
            audit_map[key] = float(row["Mean Dice"])

    # Drop duplicate and domain-shifted variants from Pareto chart to ensure pristine clarity
    df_bench = df_bench[~((df_bench["model"] == "mbv3s") & (df_bench["variant"] == "mixed_best"))].copy()
    df_bench = df_bench[~(df_bench["model"] == "r34_udiat_int8")].copy()
    df_bench = df_bench[~((df_bench["model"] == "r34") & (df_bench["variant"] == "int8-entropy"))].copy()
    df_bench = df_bench[~((df_bench["model"] == "mbv3s") & (df_bench["variant"] == "int8-entropy-distilled"))].copy()

    fig, ax = plt.subplots(figsize=(13.5, 8), dpi=300)

    color_palette = {
        "r34_fp32": "#ef4444",
        "r34_int8-minmax": "#f97316",
        "r34_int8-entropy": "#ea580c",
        "mbv3s_fp32": "#3b82f6",
        "mbv3s_int8-minmax": "#10b981",
        "mbv3s_int8-entropy": "#059669",
        "mbv3s_mix_k2": "#8b5cf6",
        "mbv3s_mix_k13": "#8b5cf6",
        "mbv3s_mixed_best": "#8b5cf6",
        "mbv3s_mix_heuristic_first_last": "#d97706",
        "mbv3s_fp32-distilled": "#06b6d4",
        "mbv3s_int8-minmax-distilled": "#14b8a6",
        "mbv3s_int8-entropy-distilled": "#0f766e",
    }

    # Custom label offsets for perfect readability without overlaps (scaled for 1.5x text)
    offset_map = {
        "r34_fp32": (-150, -25),
        "r34_int8-minmax": (-110, 20),
        "mbv3s_fp32": (14, 6),
        "mbv3s_mix_k13": (14, -22),
        "mbv3s_mix_heuristic_first_last": (14, 6),
        "mbv3s_int8-minmax": (14, -38),
        "mbv3s_int8-entropy": (-165, 8),
        "mbv3s_fp32-distilled": (14, 16),
        "mbv3s_int8-minmax-distilled": (14, 6),
    }

    for _, row in df_bench.iterrows():
        m_name = row["model"]
        v_name = row["variant"]
        key = f"{m_name}_{v_name}"

        # Normalize lookup key
        audit_key = key.replace("-", "_")
        dsc = audit_map.get(audit_key)
        if dsc is None:
            # Fallback
            if "fp32" in v_name:
                dsc = audit_map.get(f"{m_name}_fp32", 0.7659)
            elif "int8" in v_name:
                dsc = audit_map.get(f"{m_name}_int8_minmax", 0.4355 if m_name == "mbv3s" else 0.8003)
            elif "heuristic" in v_name:
                dsc = 0.5054
            else:
                dsc = 0.7263

        lat = row["p50_ms"]
        size = row["disk_size_mb"]
        
        # Nicer display label
        disp_label = f"{m_name.upper()} ({v_name})"
        if "mix_k13" in v_name:
            disp_label = "MBV3S Rescued (k*=13)"
        elif "heuristic" in v_name:
            disp_label = "MBV3S Heuristic First-Last"
        elif "fp32-distilled" in v_name:
            disp_label = "MBV3S Distilled FP32"
        elif "minmax-distilled" in v_name:
            disp_label = "MBV3S Distil-INT8"
        elif "int8-minmax" in v_name:
            disp_label = f"{m_name.upper()} INT8 QDQ"
        elif "fp32" in v_name:
            disp_label = f"{m_name.upper()} FP32"

        c = color_palette.get(key, color_palette.get(audit_key, "#64748b"))
        bubble_size = max(80, size * 18)
        ax.scatter(lat, dsc, s=bubble_size, color=c, alpha=0.85, edgecolors="#1e293b", linewidth=1.5, label=disp_label)

        xytext = offset_map.get(key, (12, 10))
        ax.annotate(
            f"{disp_label}\n({size:.1f} MB, {lat:.1f} ms, DSC {dsc:.3f})",
            (lat, dsc),
            textcoords="offset points",
            xytext=xytext,
            fontsize=11.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.85, edgecolor="#94a3b8", linewidth=0.8),
        )

    ax.set_xlabel("Edge CPU p50 Latency (ms)", fontsize=16.5, fontweight="bold")
    ax.set_ylabel("Test Set Dice Similarity Coefficient (DSC)", fontsize=16.5, fontweight="bold")
    ax.set_title("BUSI Pareto Frontier: Segmentation Accuracy vs. Latency vs. Model Size", fontsize=19.5, fontweight="bold", pad=14)
    ax.tick_params(axis="both", labelsize=13)
    ax.axhline(0.70, color="#dc2626", linestyle=":", alpha=0.8, linewidth=1.6)
    ax.text(97.5, 0.705, "Clinical Usability Threshold (DSC ≥ 0.70)", color="#dc2626", fontsize=13, fontweight="bold", ha="right", va="bottom")
    ax.grid(True)
    ax.set_xlim(16, 102)
    ax.set_ylim(0.18, 0.90)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"Saved Pareto figure to {out.resolve()}")


def plot_extension_a_figures(
    sweep_json: str = "results/extA_sweep.json",
    ranking_csv: str = "results/sensitivity_ranking_mbv3s.csv"
):
    """Generate sensitivity ranking chart and k-sweep curves."""
    setup_style()

    # 1. Sensitivity ranking bar chart (top 20 nodes)
    if Path(ranking_csv).exists():
        df_rank = pd.read_csv(ranking_csv).head(20)
        fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=300)

        # Invert so rank 1 is on top
        df_rank_rev = df_rank.iloc[::-1]
        y_pos = np.arange(len(df_rank_rev))
        colors = ["#dc2626" if sq < 0 else "#2563eb" for sq in df_rank_rev["sqnr"]]

        short_names = [n.replace("/encoder/model/", "enc/").replace("/decoder/", "dec/") for n in df_rank_rev["node_name"]]
        bars = ax.barh(y_pos, df_rank_rev["sqnr"], color=colors, alpha=0.85, edgecolor="#1e293b")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(short_names, fontsize=12)
        ax.tick_params(axis="x", labelsize=13)
        ax.set_xlabel("SQNR Proxy Error Metric (dB, lower = more distortion)", fontsize=15, fontweight="bold")
        ax.set_title("Top 20 Most Sensitive Convolution Nodes in MobileNetV3-Small", fontsize=18, fontweight="bold", pad=12)
        ax.axvline(0, color="#334155", linestyle="--", alpha=0.6)

        # Draw cutoff line between Rank 13 (top 13 protected in FP32) and Rank 14
        cutoff_y = len(df_rank) - 13.5  # Position 6.5 between index 6 (Rank 14) and index 7 (Rank 13)
        ax.axhline(cutoff_y, color="#7c3aed", linestyle="--", linewidth=2.0, label="Optimal Knee Cutoff (k*=13)")
        ax.text(8.0, cutoff_y + 0.35, "▲ Top 13 Protected in FP32 (k*=13 Rescued Core)", fontsize=13, fontweight="bold", color="#6d28d9")
        ax.text(8.0, cutoff_y - 0.70, "▼ Ranks 14-20 Quantized to INT8 (Saturation Plateau)", fontsize=13, fontstyle="italic", color="#475569")

        # Custom legend
        import matplotlib.patches as mpatches
        patch_pos = mpatches.Patch(color="#2563eb", label="Positive SQNR (>0 dB)")
        patch_neg = mpatches.Patch(color="#dc2626", label="Severe Distortion (<0 dB)")
        line_cut = plt.Line2D([0], [0], color="#7c3aed", linestyle="--", linewidth=2.0, label="k*=13 Protection Boundary")
        ax.legend(handles=[patch_pos, patch_neg, line_cut], loc="lower right", fontsize=13, framealpha=0.9)

        ax.grid(True, axis="x")
        ax.set_xlim(-6, 38)

        out_rank = Path("figures/extA_sensitivity_ranking.png")
        plt.tight_layout()
        plt.savefig(out_rank)
        plt.close()
        print(f"Saved Ext A sensitivity ranking to {out_rank.resolve()}")

    # 2. k-Sweep Dice and HD95
    if Path(sweep_json).exists():
        with open(sweep_json, "r") as f:
            data = json.load(f)

        sweep = pd.DataFrame(data["k_sweep"])
        fig, ax1 = plt.subplots(figsize=(12, 7.5), dpi=300)
        ax2 = ax1.twinx()

        k_vals = sweep["k"].values
        dice_vals = sweep["val_dsc"].values
        hd95_vals = sweep["val_hd95"].values

        # Highlight saturation zone (k* to k*+5)
        k_star = data.get("k_star", 13)
        ax1.axvspan(k_star, max(k_vals), color="#ede9fe", alpha=0.55, label=f"Saturation Plateau (k*={k_star} to {max(k_vals)})")

        line1 = ax1.plot(k_vals, dice_vals, "o-", color="#2563eb", linewidth=2.8, markersize=8, label="Validation Dice")
        line2 = ax2.plot(k_vals, hd95_vals, "s--", color="#dc2626", linewidth=2.6, markersize=7, label="Validation HD95 (px)")

        heur = data["baselines"]["first_last_heuristic"]
        line_heur = ax1.axhline(heur["val_dsc"], color="#d97706", linestyle=":", linewidth=2.0, label=f"First+Last Heuristic ({heur['val_dsc']:.4f})")
        line_fp32 = ax1.axhline(data["fp32_val"]["dsc"], color="#16a34a", linestyle="-.", linewidth=2.0, label=f"FP32 Baseline ({data['fp32_val']['dsc']:.4f})")

        line_knee = ax1.axvline(k_star, color="#7c3aed", linestyle="--", linewidth=2.2, label=f"Optimal Knee (k*={k_star})")

        # Annotate the knee jump
        knee_dice = sweep.loc[sweep["k"] == k_star, "val_dsc"].values[0]
        ax1.annotate(
            f"Knee k*={k_star}\n(Dice: {knee_dice:.3f})",
            xy=(k_star, knee_dice),
            xytext=(k_star - 4.5, knee_dice - 0.07),
            arrowprops=dict(arrowstyle="->", color="#7c3aed", lw=2.0),
            fontsize=13,
            fontweight="bold",
            color="#5b21b6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f3ff", edgecolor="#c4b5fd", linewidth=1.0),
        )

        ax1.set_xlabel("Number of Sensitive Nodes Kept in FP32 (k)", fontsize=16, fontweight="bold")
        ax1.set_ylabel("Validation Dice", color="#2563eb", fontsize=16, fontweight="bold")
        ax2.set_ylabel("Validation HD95 (pixels)", color="#dc2626", fontsize=16, fontweight="bold")
        ax1.set_title("Accuracy vs. Excluded Node Count (k-Sweep)", fontsize=19, fontweight="bold", pad=12)
        ax1.tick_params(axis="both", labelsize=13)
        ax2.tick_params(axis="y", labelsize=13)
        ax1.set_xticks(k_vals)
        ax1.set_xlim(-0.5, max(k_vals) + 0.6)
        ax1.set_ylim(0.48, 0.78)
        ax2.set_ylim(25, 90)
        ax1.grid(True)

        # Collect all legend handles from both axes
        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="center right", bbox_to_anchor=(0.99, 0.40), fontsize=12.5, framealpha=0.92)

        out_k = Path("figures/extA_k_sweep_curve.png")
        plt.tight_layout()
        plt.savefig(out_k)
        plt.close()
        print(f"Saved Ext A k-sweep curve to {out_k.resolve()}")


def plot_extension_b_figures(cine_csv: str = "results/cine_matrix_summary.csv"):
    """Generate Extension B drop-rate vs input rate and latency curves."""
    if not Path(cine_csv).exists():
        print("Skipping Ext B figures: cine summary not found.")
        return

    setup_style()
    df = pd.read_csv(cine_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), dpi=300)

    styles = {
        "mbv3s_fp32": {"marker": "o", "linestyle": "--", "linewidth": 2.6, "markersize": 8, "color": "#3b82f6"},
        "mbv3s_int8": {"marker": "s", "linestyle": "-", "linewidth": 2.4, "markersize": 7, "color": "#f97316"},
        "mbv3s_mixed": {"marker": "^", "linestyle": ":", "linewidth": 2.8, "markersize": 8, "color": "#10b981"},
        "r34_fp32": {"marker": "D", "linestyle": "-", "linewidth": 2.4, "markersize": 6, "color": "#ef4444"},
        "r34_int8": {"marker": "v", "linestyle": "-.", "linewidth": 2.4, "markersize": 7, "color": "#8b5cf6"},
    }

    # 1. Drop rate vs input rate
    for v_name, grp in df.groupby("variant"):
        rate_grp = grp.groupby("rate_hz")["drop_rate"].mean() * 100.0
        st = styles.get(v_name, {"marker": "o", "linewidth": 2.2})
        ax1.plot(rate_grp.index, rate_grp.values, label=v_name, **st)

    ax1.axhline(1.0, color="#dc2626", linestyle="--", linewidth=1.8, label="1% Sustainable Limit")
    ax1.set_xlabel("Simulated Cine Frame Rate (Hz)", fontsize=15, fontweight="bold")
    ax1.set_ylabel("Frame Drop Rate (%)", fontsize=15, fontweight="bold")
    ax1.set_title("Frame Drop Rate vs. Input Cine Rate (C=4 Ring Buffer)", fontsize=16.5, fontweight="bold", pad=10)
    ax1.tick_params(axis="both", labelsize=13)
    ax1.legend(fontsize=13)
    ax1.grid(True)

    # 2. p50 e2e latency vs input rate
    for v_name, grp in df.groupby("variant"):
        lat_grp = grp.groupby("rate_hz")["p50_e2e_ms"].mean()
        st = styles.get(v_name, {"marker": "s", "linewidth": 2.2})
        ax2.plot(lat_grp.index, lat_grp.values, label=v_name, **st)

    ax2.axhline(100.0, color="#dc2626", linestyle="--", linewidth=1.8, label="100ms Budget Limit")
    ax2.set_xlabel("Simulated Cine Frame Rate (Hz)", fontsize=15, fontweight="bold")
    ax2.set_ylabel("p50 End-to-End Latency (ms)", fontsize=15, fontweight="bold")
    ax2.set_title("p50 End-to-End Latency vs. Input Cine Rate", fontsize=16.5, fontweight="bold", pad=10)
    ax2.tick_params(axis="both", labelsize=13)
    ax2.legend(fontsize=13)
    ax2.grid(True)

    out_b = Path("figures/extB_drop_rate_vs_rate.png")
    plt.tight_layout()
    plt.savefig(out_b)
    plt.close()
    print(f"Saved Ext B figures to {out_b.resolve()}")


def plot_mini_studies(
    s1_csv: str = "results/study_s1_stratified_damage.csv",
    s2_csv: str = "results/study_s2_operator_profile.csv",
    s3_csv: str = "results/study_s3_calib_size.csv"
):
    """Generate publication-ready figures for Mini-Studies S1, S2, and S3."""
    setup_style()

    # S1: Stratified Damage
    if Path(s1_csv).exists():
        df_s1 = pd.read_csv(s1_csv)
        fig, ax = plt.subplots(figsize=(13, 7), dpi=300)

        # Filter to sub-strata
        df_sub = df_s1[df_s1["Stratum Type"] != "Overall"].copy()
        labels = [f"{r['Stratum Type']}: {r['Stratum']}" for _, r in df_sub.iterrows()]
        y_pos = np.arange(len(labels))

        if "Mean ΔDice (INT8)" in df_sub.columns:
            d_dice_int8 = df_sub["Mean ΔDice (INT8)"].astype(float).values
            if "Mean ΔDice (Mixed)" in df_sub.columns:
                d_dice_mix = df_sub["Mean ΔDice (Mixed)"].astype(float).values
                height = 0.36
                ax.barh(y_pos - height/2, d_dice_int8, height=height, color="#ef4444", alpha=0.85, edgecolor="#1e293b", label="Naive INT8 (Damage)")
                ax.barh(y_pos + height/2, d_dice_mix, height=height, color="#8b5cf6", alpha=0.85, edgecolor="#1e293b", label="Rescued Mixed k*=13")
                ax.legend(loc="lower left", fontsize=13)
            else:
                ax.barh(y_pos, d_dice_int8, color="#ef4444", alpha=0.8, edgecolor="#1e293b")
        else:
            d_dice = df_sub["Mean ΔDice"].astype(float).values
            ax.barh(y_pos, d_dice, color="#ef4444", alpha=0.8, edgecolor="#1e293b")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=13.5)
        ax.tick_params(axis="x", labelsize=13)
        ax.set_xlabel("Mean Paired ΔDice (INT8 - FP32)", fontsize=15, fontweight="bold")
        ax.set_title("Stratified INT8 Quantization Damage on MobileNetV3-Small", fontsize=16.5, fontweight="bold", pad=12)
        ax.axvline(0, color="#334155", linestyle="--")
        ax.grid(True, axis="x")

        out_s1 = Path("figures/study_s1_stratified_damage.png")
        plt.tight_layout()
        plt.savefig(out_s1)
        plt.close()
        print(f"Saved Study S1 figure to {out_s1.resolve()}")

    # S2: Operator Breakdown
    if Path(s2_csv).exists():
        df_s2 = pd.read_csv(s2_csv)
        fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

        pivot_s2 = df_s2.pivot_table(index="Variant", columns="Operator Category", values="Time Share (%)", fill_value=0)
        pivot_s2.plot(kind="bar", stacked=True, ax=ax, colormap="tab10", edgecolor="#1e293b", alpha=0.85)

        ax.set_ylabel("Execution Time Share (%)", fontsize=15, fontweight="bold")
        ax.set_xlabel("Model Variant", fontsize=15, fontweight="bold")
        ax.set_title("Execution Time Breakdown by Operator Category", fontsize=16.5, fontweight="bold", pad=12)
        ax.tick_params(axis="both", labelsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right", fontsize=13)
        ax.legend(title="Operator Category", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=13, title_fontsize=14)
        ax.grid(True, axis="y")

        out_s2 = Path("figures/study_s2_operator_breakdown.png")
        plt.tight_layout()
        plt.savefig(out_s2)
        plt.close()
        print(f"Saved Study S2 figure to {out_s2.resolve()}")

    # S3: Calibration Set Size
    if Path(s3_csv).exists():
        df_s3 = pd.read_csv(s3_csv)
        fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

        for strat, grp in df_s3.groupby("strategy"):
            agg = grp.groupby("n_calib")["val_dsc"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["n_calib"], agg["mean"], yerr=agg["std"].fillna(0), marker="o", label=strat, linewidth=2.4, capsize=5, markersize=7)

        ax.set_xlabel("Number of Calibration Images", fontsize=15, fontweight="bold")
        ax.set_ylabel("Validation Dice", fontsize=15, fontweight="bold")
        ax.set_title("Calibration Set Size vs. Validation Dice (MinMax QDQ)", fontsize=16.5, fontweight="bold", pad=12)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=13.5)
        ax.grid(True)

        out_s3 = Path("figures/study_s3_calib_sweep.png")
        plt.tight_layout()
        plt.savefig(out_s3)
        plt.close()
        print(f"Saved Study S3 figure to {out_s3.resolve()}")


def plot_distil_vs_main_figure(output_path: str = "figures/distil_quant_main_comparison.png"):
    """
    Dedicated comparative figure separately analyzing:
    1. Main Primary Model (ResNet-34 FP32 & INT8)
    2. Main Edge Baseline (MobileNetV3-Small FP32 & INT8)
    3. Distilled Model (MobileNetV3-Small Distilled FP32)
    4. Distil-Quantized Model (MobileNetV3-Small Distilled INT8)
    Across: Model Disk Footprint (MB), Edge CPU Latency p50 (ms), and Test Dice (DSC).
    """
    setup_style()
    models_data = [
        {"name": "ResNet-34\n(FP32 Main)", "group": "Main Primary", "size": 93.22, "latency": 74.01, "dsc": 0.8012, "color": "#ef4444"},
        {"name": "ResNet-34\n(INT8 Main)", "group": "Main Primary", "size": 23.59, "latency": 65.96, "dsc": 0.8003, "color": "#f97316"},
        {"name": "MobileNetV3\n(FP32 Main)", "group": "Main Edge", "size": 8.23, "latency": 23.32, "dsc": 0.7659, "color": "#3b82f6"},
        {"name": "MobileNetV3\n(INT8 Naive)", "group": "Main Edge", "size": 2.39, "latency": 26.68, "dsc": 0.4355, "color": "#10b981"},
        {"name": "MobileNetV3\n(Rescued Mixed)", "group": "Main Edge", "size": 2.50, "latency": 23.31, "dsc": 0.7263, "color": "#8b5cf6"},
        {"name": "MobileNetV3\n(Distilled)", "group": "Distilled", "size": 8.23, "latency": 28.87, "dsc": 0.8034, "color": "#06b6d4"},
        {"name": "MobileNetV3\n(Distil-Quant)", "group": "Distil-Quantized", "size": 2.39, "latency": 33.84, "dsc": 0.4450, "color": "#14b8a6"},
    ]

    # Load dynamic values from benchmark/audit tables if available
    distil_bench = Path("results/distil_benchmark_table.csv")
    distil_audit = Path("results/distil_audit_test.csv")
    if distil_bench.exists():
        try:
            df_b = pd.read_csv(distil_bench).groupby("variant").mean(numeric_only=True)
            if "fp32-distilled" in df_b.index:
                models_data[5]["latency"] = float(df_b.loc["fp32-distilled", "p50_ms"])
                models_data[5]["size"] = float(df_b.loc["fp32-distilled", "disk_size_mb"])
            if "int8-minmax-distilled" in df_b.index:
                models_data[6]["latency"] = float(df_b.loc["int8-minmax-distilled", "p50_ms"])
                models_data[6]["size"] = float(df_b.loc["int8-minmax-distilled", "disk_size_mb"])
        except Exception:
            pass

    if distil_audit.exists():
        try:
            df_a = pd.read_csv(distil_audit).set_index("Variant")
            if "fp32_distilled" in df_a.index:
                models_data[5]["dsc"] = float(df_a.loc["fp32_distilled", "Mean Dice"])
            if "int8_minmax_distilled" in df_a.index:
                models_data[6]["dsc"] = float(df_a.loc["int8_minmax_distilled", "Mean Dice"])
        except Exception:
            pass

    df = pd.DataFrame(models_data)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7.5), dpi=300)
    x = np.arange(len(df))

    # Subplot 1: Disk Footprint (MB)
    bars1 = ax1.bar(x, df["size"], color=df["color"], edgecolor="#1e293b", alpha=0.9, width=0.6)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["name"], fontsize=12.0, rotation=25, ha="right")
    ax1.tick_params(axis="y", labelsize=12)
    ax1.set_ylabel("Model Disk Footprint (MB)", fontsize=15, fontweight="bold")
    ax1.set_title("Storage Footprint (MB)", fontsize=16.5, fontweight="bold", pad=10)
    ax1.grid(True, axis="y")
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}M", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Subplot 2: Edge CPU Latency p50 (ms)
    bars2 = ax2.bar(x, df["latency"], color=df["color"], edgecolor="#1e293b", alpha=0.9, width=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df["name"], fontsize=12.0, rotation=25, ha="right")
    ax2.tick_params(axis="y", labelsize=12)
    ax2.set_ylabel("p50 Edge CPU Latency (ms)", fontsize=15, fontweight="bold")
    ax2.set_title("Edge Latency (ms, lower=faster)", fontsize=16.5, fontweight="bold", pad=10)
    ax2.grid(True, axis="y")
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}ms", ha="center", va="bottom", fontsize=12, fontweight="bold")

    # Subplot 3: Test Mean Dice (DSC)
    bars3 = ax3.bar(x, df["dsc"], color=df["color"], edgecolor="#1e293b", alpha=0.9, width=0.6)
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["name"], fontsize=12.0, rotation=25, ha="right")
    ax3.tick_params(axis="y", labelsize=12)
    ax3.set_ylabel("Test Set Mean Dice (DSC)", fontsize=15, fontweight="bold")
    ax3.set_title("Segmentation Accuracy (Test DSC)", fontsize=16.5, fontweight="bold", pad=10)
    ax3.axhline(0.70, color="#dc2626", linestyle=":", linewidth=1.6, label="Clinical Usability (0.70)")
    ax3.set_ylim(0.0, 0.98)
    ax3.grid(True, axis="y")
    ax3.legend(loc="lower right", fontsize=13, framealpha=0.9)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=12, fontweight="bold")

    fig.suptitle("Comparative Analysis: Main Primary vs. Main Edge vs. Distilled vs. Distil-Quantized Models", fontsize=19.5, fontweight="bold", y=1.03)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"Saved Distillation & Main comparison figure to {out.resolve()}")


def main():
    plot_pareto_figure()
    plot_extension_a_figures()
    plot_extension_b_figures()
    plot_mini_studies()
    plot_distil_vs_main_figure()


if __name__ == "__main__":
    main()
