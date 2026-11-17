# Fast and Lightweight Segmentation for Breast Ultrasound

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime 1.31](https://img.shields.io/badge/ONNX_Runtime-1.31.1-005ced.svg)](https://onnxruntime.ai/)
[![Tests Passed](https://img.shields.io/badge/tests-31%2F31%20passed-success.svg)](tests/)


## Experimental Results

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

## Findings

1. **Near-Lossless Quantization on ResNet-34:** $3.95\times$ compression with zero perceptible loss in segmentation accuracy ($\Delta\text{DSC} = -0.0013$, crossing zero).
2. **MobileNetV3 Inverted-Bottleneck Quantization Fragility:** Unconstrained static INT8 destroys small lesions ($-0.7317$ $\Delta$Dice in Mini-Study S1).
3. **Operator Overhead Dominates Mobile Edge Latency:** Mini-Study S2 demonstrated that Quantize/Dequantize (QDQ) nodes consume **34.09%** of execution time in INT8 MobileNetV3, surpassing the convolutions (31.78%).
4. **Real-Time POCUS Streamability:** In Extension B, MobileNetV3 sustained a live **30 Hz streaming cine-loop** with $<0.6\%$ drops, whereas ResNet-34 drops $>50-65\%$ of frames at 30 Hz.

---

## Quickstart

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