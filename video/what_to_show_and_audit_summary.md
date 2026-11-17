# Audit Summary & Video Presentation Master Plan

This document serves as the **single source of truth** for all experimental metrics, data integrity audits, architectural comparisons, and the 7-minute video presentation strategy. Every value below is cross-verified directly against the CSV/JSON artifacts in `results/`.

---

## 1. Experimental Integrity Audit & Verification

Every metric presented in our final report, slides, and script has been rigorously audited against held-out test splits and execution benchmarks. **No fallback, interpolated, or simulated values exist.**

### Resolved Inconsistencies & Corrected Artifacts
1. **Pareto Plot Fallback Removal:** Corrected `src/report/make_figures.py` which previously defaulted to `0.31` for mixed-precision models. The exact held-out test score for `mbv3s_mixed_best` (`k*=13`) is **0.7263 [0.6690, 0.7784]** (Median **0.8448**, HD95 **35.20 px**).
2. **Distilled Model Audit Verification:** Re-evaluated the latest distilled model checkpoints on the 163-image BUSI held-out test split (96 lesion, 67 normal) in `results/distil_audit_test.csv`. Distilled MobileNetV3-Small (`T=2.0, alpha=0.50`) achieves **0.8034 [0.7617, 0.8430]** Mean DSC (Median **0.8916**, HD95 **26.44 px**), outperforming the ResNet-34 teacher (0.8012 DSC) while utilizing **11.4× fewer parameters**.
3. **Distil-Quantized INT8 Verification:** Verified naive static INT8 quantization of the distilled checkpoint achieves **0.4450 [0.3810, 0.5080]** Mean DSC (Median **0.4850**, HD95 **82.64 px**), corroborating that distillation alone does not eliminate the depthwise separable quantization bottleneck without mixed-precision preservation.
4. **Literature & Cross-Center Sync:** Cross-checked all tables in `video/` against `results/literature_benchmark_comparison.csv`, `results/udiat_cross_center_results.csv`, `results/study_s1_stratified_damage.csv`, `results/study_s2_operator_profile.csv`, and `results/cine_matrix_summary.csv`.
5. **In-Browser WebViewer Model Sync:** Deployed `mbv3s_mixed_best.onnx` (2.50 MB, 0.726 DSC) and `mbv3s_distilled.onnx` (8.23 MB, 0.803 DSC) into `server/static/models/` and updated `server/static/index.html` model selection.

---

## 2. Master Metrics Reference Table (Held-Out BUSI Test Split & UDIAT Zero-Shot)

| Architecture | Precision / Variant | Parameters | Storage Footprint | Edge CPU Latency (p50) | Throughput (FPS) | Sustained Stream Rate (30 Hz Drops) | BUSI Test Mean Dice [95% CI] | BUSI Test Median Dice | BUSI Test Mean HD95 | UDIAT Zero-Shot Mean / Median Dice |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-34** | FP32 Baseline | 24.44M | 93.22 MB | 74.01 ms | 13.5 FPS | 10 Hz (Drops 61.9% @ 30Hz) | 0.8012 [0.7535, 0.8457] | 0.9033 | 27.71 px | 0.7902 / 0.8748 |
| **ResNet-34** | Static INT8 QDQ | 24.44M | **23.59 MB (3.95×)** | 65.96 ms | 15.2 FPS | 10 Hz (Drops 56.3% @ 30Hz) | **0.8003** [0.7515, 0.8462] | **0.9048** | 28.56 px | **0.7928 / 0.8746** |
| **MobileNetV3-Small** | FP32 Baseline | 2.15M | 8.23 MB | 23.32 ms | 42.9 FPS | 30 Hz (0.02% drops) | 0.7659 [0.7123, 0.8163] | 0.8716 | 35.13 px | 0.7244 / 0.8278 |
| **MobileNetV3-Small** | Naive Static INT8 | 2.15M | 2.39 MB (3.44×) | 26.68 ms | 37.5 FPS | 30 Hz (4.41% drops, Collapsed) | 0.4355 [0.3612, 0.5002] | 0.4680 | 86.97 px | 0.1519 / 0.1248 |
| **MobileNetV3-Small** | **Rescued Mixed ($k^*=13$)** | 2.15M | **2.50 MB (3.30×)** | **23.31 ms** | **42.9 FPS** | **30 Hz (0.06% drops)** | **0.7263** [0.6690, 0.7784] | **0.8448** | **35.20 px** | **0.7002 / 0.8562** |
| **MobileNetV3-Small** | **Distilled FP32** | 2.15M | **8.23 MB** | **28.87 ms** | **34.9 FPS** | **30 Hz (Beats Teacher!)** | **0.8034** [0.7617, 0.8430] | **0.8916** | **26.44 px** | — *(BUSI Primary)* |
| **MobileNetV3-Small** | Distilled INT8 (Naive) | 2.15M | 2.39 MB (3.44×) | 33.84 ms | 29.7 FPS | 30 Hz (Degraded) | 0.4450 [0.3810, 0.5080] | 0.4850 | 82.64 px | — *(BUSI Primary)* |

*(Note: Fine-tuning ResNet-34 on UDIAT achieves **0.8249 DSC** in FP32 and **0.8283 DSC** in Static INT8 QDQ, per `results/udiat_cross_center_results.csv`.)*

---

## 3. Master Literature Benchmark Comparison (`results/literature_benchmark_comparison.csv`)

| Model / Architecture | Type / Backbone | Params | Size (MB) | Test Mean DSC | Test Median DSC | Test Mean IoU | Test HD95 (px) | Edge CPU Latency | Throughput | Source / Deployment Note |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Shifaa U-Net** | 4-Stage Plain U-Net | 1.94M | 7.78 MB | 0.4973 | 0.5279 | 0.3673 | 88.72 px | 57.68 ms | 17.3 FPS | Hugging Face (`Ahmed-Selem`) |
| **Attention U-Net** | ResNet-34 + SCSE | 24.55M | 97.40 MB | 0.7412 | 0.8124 | 0.6385 | 38.42 px | 139.99 ms | 7.1 FPS | Yap et al. (Overloads Edge) |
| **U-Net++ (Nested)** | ResNet-34 Nested Dense | 26.08M | 103.80 MB | 0.7588 | 0.8340 | 0.6542 | 35.12 px | 248.03 ms | 4.0 FPS | Al-Dhabyani et al. (4.0 FPS) |
| **Our ResNet-34 FP32** | ResNet-34 U-Net | 24.44M | 93.22 MB | 0.8012 | 0.9033 | 0.7188 | 27.71 px | 74.01 ms | 13.5 FPS | Ours (Heavyweight Teacher) |
| **Our ResNet-34 INT8** | ResNet-34 QDQ INT8 | 24.44M | **23.59 MB** | **0.8003** | **0.9048** | **0.7187** | **28.56 px** | 65.96 ms | 15.2 FPS | **Ours (Lossless 4× Compression)** |
| **Our MobileNetV3 FP32** | MobileNetV3-Small | **2.15M** | 8.23 MB | **0.7659** | **0.8716** | **0.6770** | 35.13 px | 23.32 ms | 42.9 FPS | **Ours (Beats U-Net++ at 10× FPS)** |
| **Our MobileNetV3 Distilled** | MobileNetV3-Small KD | **2.15M** | 8.23 MB | **0.8034** | **0.8916** | **0.7042** | **26.44 px** | 28.87 ms | 34.9 FPS | **Ours (Highest Overall DSC)** |
| **Our MobileNetV3 Rescued** | Hybrid INT8/FP32 ($k^*=13$) | **2.15M** | **2.50 MB** | **0.7263** | **0.8448** | **0.6314** | **35.20 px** | **23.31 ms** | **42.9 FPS** | **Ours (Sub-3 MB, 42.9 FPS Edge)** |

---

## 4. What to Show vs. What to Omit (7-Minute Time Allocation)

Because the presentation is strictly limited to 7 minutes (420 seconds), we allocate time strategically to maximize evaluation score: **Over 50% of the time is dedicated to Methodology and Engineering Rigor**.

### High-Priority Elements Featured in Sequential Slide Order (Slides 1–12)
1. **Slides 1–2: Clinical Problem & The POCUS Edge Dilemma (~65s):** Point-of-Care Ultrasound systems run on constrained mobile APUs/CPUs with battery limits, requiring $\ge 30$ Hz real-time frame rates ($\le 33.3$ ms latency) and $<10$ MB storage without sacrificing diagnostic accuracy.
2. **Slide 3: Architectural Progression & Compound Loss (~40s):** Heavyweight ResNet-34 U-Net (24.44M params, 93.2 MB) vs. edge-native MobileNetV3-Small U-Net (2.15M params, 8.23 MB) trained with $0.5\,\mathcal{L}_{\text{BCE}} + 0.5\,\mathcal{L}_{\text{Dice}}$.
3. **Slide 4: Quantization Failure & Stratified Damage Analysis (~45s):** ResNet-34 quantizes losslessly to INT8 (0.8012 $\to$ 0.8003 DSC), whereas MobileNetV3 suffers severe collapse (0.7659 $\to$ 0.4355 DSC). Mini-Study S1 demonstrates that soft/diffuse margins ($\Delta\text{DSC} = -0.40$) and malignant lesions ($\Delta\text{DSC} = -0.40$) suffer up to $2\times$ worse degradation than sharp margins ($-0.20$).
4. **Slide 5: Extension A: Graph Surgery & Sensitivity Knob ($k^*=13$) (~45s):** Empirical activation SQNR + topological graph surgery (`src/quant/knob.py`) isolates the 13 fragile depthwise and boundary convolutions. Keeping 13/63 nodes in FP32 recovers test DSC to **0.7263** (Median **0.8448**) at **2.50 MB** (3.30× compression).
5. **Slide 6: Knowledge Distillation Breakthrough (~30s):** Temperature-scaled distillation ($T=2.0, \alpha=0.50$) propels MobileNetV3 FP32 to **0.8034 DSC**, beating the ResNet-34 teacher while proving naive INT8 distil-quant (0.4450 DSC) still requires mixed precision.
6. **Slide 7: Systems Profiling Deep-Dive (Mini-Study S2) (~25s):** ONNX Runtime operator profiling reveals QDQ nodes consume **26.6%** of runtime in naive INT8 MobileNetV3; pruning QDQ around the 13 rescued nodes drops QDQ overhead to **14.6%**, making Rescued Mixed (23.31 ms) faster than naive INT8 (26.68 ms).
7. **Slide 8: Master Quantitative Evaluation & Pareto Frontier (~40s):** Three optimal Pareto operating points (ResNet-34 INT8, MobileNetV3 Distilled FP32, MobileNetV3 Rescued Mixed).
8. **Slide 9: Literature Benchmark & UDIAT Cross-Center Generalization (~35s):** Outperforming Shifaa U-Net (0.4973), Attention U-Net (0.7412), and U-Net++ (0.7588), plus zero-shot validation on Spanish UDIAT Dataset B (ResNet-34 INT8: 0.7928 DSC; Rescued Mixed: 0.7002 Mean / 0.8562 Median DSC).
9. **Slide 10: Extension B: 30 Hz Cine-Loop Streaming & Temporal EMA (~40s):** Drop-oldest ring buffer ($C=4$), $<0.1\%$ frame drops on MobileNetV3 vs. 56.3–61.9% drops on ResNet-34, and EMA logit smoothing ($\alpha=0.60$) achieving +18.5% to +65.2% (mean +37.7%) TCI stability gain on UFUV MICCAI clinical videos.
10. **Slides 11–12: Live WebViewer Demo, Clinical Limitations & Conclusion (~55s):** Zero-install client-side WebGPU/WASM browser inference, followed by honest clinical limitations (posterior acoustic shadowing, out-of-plane probe motion, gain outliers).

---

## 5. Speaker Role & Time Breakdown (Equal Contribution, Sequential Slides 1–12)

| Speaker | Assigned Role & Section | Sequential Slides | Allocated Time | Spoken Word Count (~135 wpm) |
| :--- | :--- | :---: | :---: | :---: |
| **Speaker A** | **Clinical Problem, Architectural Progression, Compound Loss & WebGPU WebViewer** | Slides 1–3, 11 | **2 min 15 sec** | ~305 words |
| **Speaker B** | **Quantization Collapse, Extension A Graph Surgery ($k^*=13$), Distillation & Profiling (S2)** | Slides 4–7 | **2 min 25 sec** | ~330 words |
| **Speaker C** | **Master Pareto Frontier, Literature/UDIAT Benchmarks, Extension B Streaming & Limitations** | Slides 8–10, 12 | **2 min 20 sec** | ~315 words |
| **Total** | **Complete Project Defense** | **Slides 1–12** | **7 min 00 sec** | **~950 words** |
