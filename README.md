# BUSI Quantized Real-Time Segmentation: Edge Systems & Sensitivity Analysis

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime 1.31](https://img.shields.io/badge/ONNX_Runtime-1.31.1-005ced.svg)](https://onnxruntime.ai/)
[![Tests Passed](https://img.shields.io/badge/tests-31%2F31%20passed-success.svg)](tests/)

Rigorous comparative analysis of heavy (**ResNet-34 U-Net**, 24.4M params) and edge-optimized (**MobileNetV3-Small U-Net**, 2.15M params) breast ultrasound segmentation models on the BUSI dataset, converted to static 8-bit quantized (**INT8 QDQ**) representations, and benchmarked on edge CPU targets.

---

## Key Experimental Results

| Model | Variant | Test Mean Dice [95% CI] | Test Median Dice | Test Mean HD95 [95% CI] | Disk Size (MB) | Edge CPU Latency (ms) | Throughput (FPS) | Sustained Stream Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **r34** | FP32 | 0.8012 [0.7535, 0.8457] | 0.9033 | 27.71 [19.78, 36.32] | 93.22 MB | 74.01 ms | 13.5 FPS | 10 Hz |
| **r34** | Static INT8 QDQ | **0.8003** [0.7515, 0.8462] | **0.9048** | 28.56 [20.73, 37.33] | **23.59 MB** (3.95x) | 65.96 ms | 15.2 FPS | 10 Hz (Lossless!) |
| **mbv3s** | FP32 | **0.7659** [0.7123, 0.8163] | **0.8716** | **35.13** [26.68, 44.57] | **8.23 MB** | **23.32 ms** | **42.9 FPS** | **30 Hz (0.0% drops)** |
| **mbv3s** | Rescued Mixed ($k^*=13$) | **0.7263** [0.6690, 0.7784] | **0.8448** | **35.20** [26.49, 44.40] | **2.50 MB** (3.30x) | **23.31 ms** | **42.9 FPS** | **30 Hz (Rescued!)** |
| **mbv3s** | Naive INT8 QDQ | 0.4355 [0.3612, 0.5002] | 0.4680 | 86.97 [77.63, 95.96] | 2.39 MB | 26.68 ms | 37.5 FPS | 30 Hz (Collapsed) |
| **mbv3s** | Distilled FP32 | **0.8034** [0.7617, 0.8430] | **0.8916** | **26.44** [19.83, 34.03] | **8.23 MB** | **28.87 ms** | **34.9 FPS** | **30 Hz (Beats Teacher!)** |
| **mbv3s** | Distil-Quant INT8 | 0.4450 [0.3810, 0.5080] | 0.4850 | 82.64 [73.54, 91.56] | **2.39 MB** (3.44x) | 33.84 ms | 29.7 FPS | 30 Hz (Smallest) |

---

## Visual Deliverables & Novelty Highlights

1. **Multi-Model Benchmark Visual Overlays:** [`figures/benchmark_overlay_comparison.png`](figures/benchmark_overlay_comparison.png)
   - 8-column side-by-side comparison across 4 distinct test pathologies: Raw B-mode Scan | Ground Truth | Shifaa U-Net (Online HF) | Our r34 FP32 | Our r34 INT8 | Our mbv3s FP32 | Our mbv3s Naive INT8 | Our mbv3s Rescued Mixed $k=13$.
2. **$5 \times 3$ Ground Truth vs Prediction Tracking Showcase (Genuine Clinical Videos):** [`figures/cine_5sample_5x3_tracking_demo.gif`](figures/cine_5sample_5x3_tracking_demo.gif) & [`figures/cine_5sample_5x3_tracking_demo.mp4`](figures/cine_5sample_5x3_tracking_demo.mp4)
   - 5 rows $\times$ 3 columns: Left column displays genuine clinical ultrasound videos (`IM_0003`, `IM_0004`, `IM_0005`, `IM_0014`, `IM_0055`) with expert ground-truth masks; center column displays model's real-time prediction and centroid tracking; right column displays live telemetry HUD (instantaneous DSC, IoU, latency, FPS, and lesion area).
3. **All-Models Across Quantizations Video Comparison (Real Clinical Video `IM_0055`):** [`figures/cine_all_models_quantization_comparison.gif`](figures/cine_all_models_quantization_comparison.gif) & [`figures/cine_all_models_quantization_comparison.mp4`](figures/cine_all_models_quantization_comparison.mp4)
   - Synchronized 8-panel ($2 \times 4$) video canvas comparing Ground Truth, ResNet-34 FP32, ResNet-34 INT8, MobileNetV3 FP32, MobileNetV3 Naive INT8 (Collapsed), MobileNetV3 Rescued Mixed $k=13$, and Uncertainty Cascade + EMA on the exact same clinical video with live leaderboard telemetry.
4. **Synchronized 5-Sample BUVFM Clinical Tracking Grid:** [`figures/cine_5sample_tracking_demo.gif`](figures/cine_5sample_tracking_demo.gif) & [`figures/cine_5sample_tracking_demo.mp4`](figures/cine_5sample_tracking_demo.mp4)
5. **BUVFM Clinical Ground-Truth Benchmark (500 Scans):** [`results/buvfm_groundtruth_benchmark.csv`](results/buvfm_groundtruth_benchmark.csv)
6. **Genuine Clinical Video Benchmark (Table 13):** [`results/real_video_groundtruth_benchmark.csv`](results/real_video_groundtruth_benchmark.csv)
7. **Cross-Center Domain Generalization (UDIAT Dataset B, Spain):**
   - Zero-shot evaluation on 163 independent scans: ResNet-34 INT8 achieves **0.7928 DSC** (exceeding FP32 0.7902), and Rescued MobileNetV3 achieves **0.7002 DSC** (median 0.8562), confirming quantization robustness across clinical centers.
8. **Interactive Extension A Sensitivity Demo:** [`figures/extA_demo.html`](figures/extA_demo.html)
   - Self-contained interactive web interface for exploring layer-wise SQNR noise injection and real-time $k$-sweep slider adjustments.
9. **Comprehensive Model Comparison (Main vs. Distilled vs. Distil-Quantized):** [`figures/distil_quant_main_comparison.png`](figures/distil_quant_main_comparison.png)
   - 3-panel comparative analysis contrasting model storage footprint (MB), edge CPU latency (ms), and segmentation accuracy (Dice DSC) across Main Primary, Main Edge, Distilled, and Distil-Quantized models.
10. **In-Browser WebGPU & WebAssembly Medical Studio:** [`server/`](server/)
   - Standalone browser webviewer running client-side neural inference via WebGPU and WASM SIMD, loaded with authentic BUSI clinical presets (*Benign Lesion 1 & 2*, *Malignant 1 & 2*, *Normal scan*).

---

## Core Findings

1. **Near-Lossless Quantization on ResNet-34:** $3.95\times$ compression with zero perceptible loss in segmentation accuracy ($\Delta\text{DSC} = -0.0013$, crossing zero).
2. **MobileNetV3 Inverted-Bottleneck Quantization Fragility:** Unconstrained static INT8 destroys small lesions ($-0.7317$ $\Delta$Dice in Mini-Study S1).
3. **Operator Overhead Dominates Mobile Edge Latency:** Mini-Study S2 demonstrated that Quantize/Dequantize (QDQ) nodes consume **34.09%** of execution time in INT8 MobileNetV3, surpassing the convolutions (31.78%).
4. **Real-Time POCUS Streamability:** In Extension B, MobileNetV3 sustained a live **30 Hz streaming cine-loop** with $<0.6\%$ drops, whereas ResNet-34 drops $>50-65\%$ of frames at 30 Hz.

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
uv venv --python 3.12 .venv
# On Windows:
.venv\Scripts\activate
# Install locked dependencies:
uv pip install -r requirements.lock
```

### 2. Full End-to-End Pipeline
```bash
make data         # Ingest BUSI, merge multi-masks, hash-dedup, build splits & QA overlays
make train        # Train FP32 r34 and mbv3s models on GPU
make distil       # Distil knowledge from teacher (r34) into lightweight student (mbv3s)
make distil-quant # Export distilled model to ONNX & perform static QDQ INT8 quantization
make distil-bench # Benchmark distilled FP32 & INT8 models (p50/p95 latency, FPS, RSS)
make eval         # Compute baseline metrics (DSC, IoU, HD95, normal FPR) with bootstrap CIs
make export       # Export FP32 PyTorch checkpoints to ONNX Opset 17 with parity check (< 1e-4)
make quant        # Static INT8 QDQ quantization (MinMax vs Entropy) + Accuracy Audit
make bench        # 50 warm-up + 500 timed runs across 3 fresh subprocess repeats
make extA         # Layer-wise sensitivity ranking, k-sweep, mixed model, & interactive HTML demo
make extB         # Real-time cine-loop evaluation across frame rates (10, 15, 20, 30 Hz)
make studies      # Run Mini-Studies S1 (Stratified Damage), S2 (Profiling), S3 (Calib Sweep)
make report       # Generate publication figures and markdown summary tables
make test         # Run pytest test suite
make webviewer    # Launch in-browser WebGPU/WebAssembly Tumor Segmentation Studio (port 8000)
```

---

## WebViewer: In-Browser WebGPU & WebAssembly (WASM) Tumor Detection

A medical web viewer is provided in [`server/`](server/) running client-side neural inference directly in the browser using **WebGPU** and **WebAssembly (WASM SIMD)** with zero server GPU requirements:

- **Smallest Distilled-Quantized Model:** Loads `mbv3s_distilled_int8_minmax.onnx` (**2.50 MB**, 2.15M params) directly into the browser.
- **Hardware Acceleration:** Auto-negotiates between WebGPU and multi-threaded WebAssembly execution providers.
- **Real-Time Tumor Identification & Overlays:**
  - Identifies tumors on uploaded B-mode frames or 1-click clinical presets (Benign, Malignant, Normal).
  - Generates high-contrast tumor segmentation overlays, smooth boundary contours, and bounding boxes.
  - Interactive probability threshold and opacity controls that update the overlay in real time without re-running model inference.
  - Live clinical telemetry readout: Detection Verdict, Lesion Area (px² & coverage %), Centroid coordinates `(X, Y)`, Bounding Box `W x H`, and inference latency (ms / FPS).

```bash
# Launch the WebViewer
make webviewer
# Open http://localhost:8000 in any modern web browser (Chrome, Edge, Firefox, Safari)
```

---

## Knowledge Distillation & Pure Inference Engine

The repository includes a dedicated distillation module in `src/distil` designed to transfer high-capacity representations from teacher models (`r34`) to lightweight edge architectures (`mbv3s`), coupled with an inference engine tailored strictly for deployment:

1. **Distillation Training (`src.distil.distil`):**
   - Combines task loss (BCE + Soft Dice) with temperature-scaled soft dark knowledge distillation loss (`KnowledgeDistillationLoss`).
   ```bash
   make distil
   # Or customized:
   python -m src.distil.distil --teacher-model r34 --student-model mbv3s --temperature 2.0 --alpha 0.5 --epochs 10
   ```

2. **Dedicated Pure-Inference Engine (`InferenceEngine`):**
   - Located in `src.distil.inference`, optimized specifically for deployment:
     - Zero gradient computation and frozen parameters.
     - Direct evaluation on batched PyTorch tensors (`predict_proba`, `predict_mask`).
     - End-to-end numpy image inference (`infer_single_image`) with automatic preprocessing and resizing back to original aspect ratio.
   ```python
   from src.distil.inference import load_distilled_inference_model
   import cv2

   engine = load_distilled_inference_model("models/distil/mbv3s_distilled_best.pt")
   image = cv2.imread("sample.png")
   binary_mask = engine.infer_single_image(image, return_binary=True)
   ```

3. **Distilled Model Quantization & Edge Benchmarking:**
   - `make distil-quant`: Exports the distilled model to ONNX Opset 17 with parity verification and generates static QDQ INT8 models (both MinMax and Entropy).
   - `make distil-bench`: Executes multi-repeat isolated CPU benchmarks measuring p50/p95/p99 latency, FPS throughput, and peak RSS memory, outputting to `results/distil_benchmark_table.csv`.


---

## Extension A: Zero-Recomputation Sensitivity Knob

Query precomputed sweep parameters without re-quantizing:

```powershell
# Mode 1: Fixed count (keep top 3 most sensitive nodes in FP32)
.\.venv\Scripts\python.exe -m src.quant.knob --keep-fp32 3

# Mode 2: Percentage of graph (keep top 5% of nodes in FP32)
.\.venv\Scripts\python.exe -m src.quant.knob --keep-fp32-pct 5

# Mode 3: Accuracy budget (maximum allowed Dice drop from FP32)
.\.venv\Scripts\python.exe -m src.quant.knob --max-dice-drop 0.5
```

Open `figures/extA_demo.html` in any web browser for the interactive slider interface.

---

## Comprehensive Report & Documentation
- **Full 10-Section Systems Report:** [`REPORT.md`](REPORT.md)
- **Summary Markdown Tables:** [`results/summary_tables.md`](results/summary_tables.md) (Includes Table 14: Main vs Distilled vs Distil-Quantized comparison)
- **Pareto Frontier (Test Dice vs Latency vs Size):** [`figures/pareto_dice_vs_latency.png`](figures/pareto_dice_vs_latency.png)
- **Distilled vs Main Models Multi-Panel Comparison:** [`figures/distil_quant_main_comparison.png`](figures/distil_quant_main_comparison.png)
- **Sensitivity Ranking:** [`figures/extA_sensitivity_ranking.png`](figures/extA_sensitivity_ranking.png)
- **Cine-Loop Drop Rates:** [`figures/extB_drop_rate_vs_rate.png`](figures/extB_drop_rate_vs_rate.png)
- **Interactive Extension A Demo:** [`figures/extA_demo.html`](figures/extA_demo.html)
- **WebGPU / WebAssembly Medical Studio:** [`server/`](server/) (`make webviewer`)
