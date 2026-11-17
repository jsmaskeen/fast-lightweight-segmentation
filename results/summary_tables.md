### Table 1: FP32 Baseline Test Set Evaluation

| Model | Variant | Test DSC Mean [95% CI] | Test IoU Mean [95% CI] | Test HD95 (px) [95% CI] | Normal FPR@50px | Empty Preds |
| --- | --- | --- | --- | --- | --- | --- |
| R34 | FP32 (PyTorch) | 0.8011 [0.7535, 0.8457] | 0.7188 [0.6675, 0.7663] | 27.69 [19.31, 38.23] | 79.10% | 1/96 |
| MBV3S | FP32 (PyTorch) | 0.7662 [0.7126, 0.8164] | 0.6772 [0.6228, 0.7281] | 35.09 [24.97, 47.18] | 88.06% | 2/96 |

---

### Table 2: Accuracy Audit (FP32 vs Static INT8 QDQ on Test Split)

| Model | Variant | Split | Mean Dice | Dice 95% CI | Mean HD95 (px) | HD95 95% CI | Normal FPR@50px | Empty Preds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r34 | fp32 | test | 0.8012 | [0.7535, 0.8457] | 27.71 | [19.33, 38.24] | 79.10% | 1 |
| r34 | int8_minmax | test | 0.8003 | [0.7515, 0.8462] | 28.56 | [18.66, 40.31] | 80.60% | 2 |
| r34 | int8_entropy | test | 0.8003 | [0.7515, 0.8462] | 28.56 | [18.66, 40.31] | 80.60% | 2 |
| mbv3s | fp32 | test | 0.7659 | [0.7123, 0.8163] | 35.13 | [25.04, 47.25] | 88.06% | 2 |
| mbv3s | int8_minmax | test | 0.4355 | [0.3612, 0.5002] | 86.97 | [66.44, 110.54] | 59.70% | 12 |
| mbv3s | int8_entropy | test | 0.4355 | [0.3612, 0.5002] | 86.97 | [66.44, 110.54] | 59.70% | 12 |

---

### Table 3: Edge CPU Latency and Memory Telemetry

| model | variant | disk_size_mb | peak_rss_mb | p50_ms | p95_ms | p99_ms | fps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mbv3s | fp32 | 8.23 | 132.16 | 28.93 | 31.04 | 40.93 | 35.68 |
| mbv3s | int8-entropy | 2.39 | 143.51 | 30.23 | 35.11 | 43.01 | 32.92 |
| mbv3s | int8-minmax | 2.39 | 143.31 | 30.62 | 35.03 | 39.91 | 32.6 |
| mbv3s | mix_heuristic_first_last | 2.39 | 126.73 | 26.88 | 30.64 | 34.06 | 37.98 |
| mbv3s | mix_k13 | 2.5 | 131.35 | 26.76 | 30.73 | 34.17 | 38.45 |
| mbv3s | mixed_best | 2.5 | 131.3 | 26.18 | 29.97 | 34.96 | 38.3 |
| r34 | fp32 | 93.22 | 296.69 | 85.68 | 103.72 | 125.71 | 11.83 |
| r34 | int8-entropy | 23.59 | 156.98 | 73.65 | 81.66 | 89.63 | 13.69 |
| r34 | int8-minmax | 23.59 | 156.99 | 68.18 | 77.48 | 84.44 | 14.67 |
| r34_udiat_int8 | int8 | 23.59 | 157.31 | 70.67 | 80.12 | 84.56 | 14.24 |

---

### Table 4: Extension A Sensitivity Knob k-Sweep (Optimal k*=13)

| k (FP32 Nodes) | Val DSC | Val HD95 (px) | Model Size (MB) | p50 Latency (ms) |
| --- | --- | --- | --- | --- |
| 0.0 | 0.5351 | 66.6034 | 2.3935 | 27.2332 |
| 1.0 | 0.5364 | 67.3771 | 2.3922 | 26.7772 |
| 2.0 | 0.5314 | 63.4164 | 2.4239 | 23.2392 |
| 3.0 | 0.5497 | 53.2783 | 2.4357 | 22.9777 |
| 4.0 | 0.5586 | 62.9307 | 2.4475 | 27.6341 |
| 5.0 | 0.5158 | 69.8673 | 2.4621 | 27.8766 |
| 6.0 | 0.5149 | 71.344 | 2.4683 | 27.0325 |
| 8.0 | 0.5412 | 70.1521 | 2.5046 | 27.8354 |
| 10.0 | 0.4996 | 82.68 | 2.5074 | 26.6012 |
| 12.0 | 0.5071 | 68.8333 | 2.505 | 27.1754 |
| 13.0 | 0.6959 | 35.7572 | 2.5031 | 27.788 |
| 14.0 | 0.6915 | 34.269 | 2.5278 | 28.6383 |
| 15.0 | 0.6903 | 33.734 | 2.5447 | 28.6051 |
| 16.0 | 0.6906 | 34.6635 | 2.5694 | 28.2733 |
| 17.0 | 0.6901 | 38.7003 | 2.7242 | 23.4546 |
| 18.0 | 0.6922 | 38.2454 | 2.879 | 27.7976 |

---

### Table 5: Extension B Cine-Loop Streaming Performance

| Variant | Input Rate (Hz) | Drop Rate (%) | p50 e2e Latency (ms) | p95 e2e Latency (ms) | Deadline Miss Rate (%) |
| --- | --- | --- | --- | --- | --- |
| mbv3s_fp32 | 10 | 0.0 | 28.4 | 33.0 | 0.06 |
| mbv3s_fp32 | 15 | 0.0 | 30.2 | 32.5 | 0.04 |
| mbv3s_fp32 | 20 | 0.0 | 29.6 | 32.3 | 0.0 |
| mbv3s_fp32 | 30 | 0.02 | 29.9 | 32.4 | 0.0 |
| mbv3s_int8 | 10 | 0.0 | 33.6 | 37.2 | 0.0 |
| mbv3s_int8 | 15 | 0.0 | 33.2 | 36.9 | 0.0 |
| mbv3s_int8 | 20 | 0.0 | 33.4 | 37.4 | 0.0 |
| mbv3s_int8 | 30 | 4.41 | 45.8 | 65.0 | 0.04 |
| mbv3s_mixed | 10 | 0.0 | 30.1 | 32.9 | 0.0 |
| mbv3s_mixed | 15 | 0.0 | 29.3 | 32.8 | 0.0 |
| mbv3s_mixed | 20 | 0.0 | 30.2 | 33.1 | 0.0 |
| mbv3s_mixed | 30 | 0.06 | 28.3 | 33.2 | 0.0 |
| r34_fp32 | 10 | 0.0 | 97.6 | 108.7 | 40.22 |
| r34_fp32 | 15 | 26.33 | 122.8 | 159.8 | 83.73 |
| r34_fp32 | 20 | 46.86 | 119.0 | 143.5 | 86.59 |
| r34_fp32 | 30 | 61.91 | 104.3 | 126.6 | 63.59 |
| r34_int8 | 10 | 0.0 | 76.4 | 83.7 | 0.72 |
| r34_int8 | 15 | 13.48 | 106.2 | 140.3 | 59.57 |
| r34_int8 | 20 | 33.86 | 99.5 | 124.7 | 51.45 |
| r34_int8 | 30 | 56.28 | 91.8 | 108.6 | 29.19 |

---

### Table 6: Mini-Study S1 Stratified INT8 Damage Analysis

| Stratum Type | Stratum | N | Mean ΔDice (INT8) | ΔDice 95% CI (INT8) | Mean ΔHD95 (INT8) | Empty Pred (FP32/INT8) | Mean ΔDice (Mixed) | Mean ΔHD95 (Mixed) | Empty Pred (Mixed) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | All Lesions | 96 | -0.3305 | [-0.3936, -0.2749] | 51.85 | 2 / 12 | -0.0396 | 0.07 | 0 |
| Class | benign | 65 | -0.2961 | [-0.3768, -0.2216] | 53.13 | 2 / 10 | -0.0478 | -1.89 | 0 |
| Class | malignant | 31 | -0.4027 | [-0.4898, -0.3175] | 49.16 | 0 / 2 | -0.0226 | 4.19 | 0 |
| Area Tercile | Small | 28 | -0.2833 | [-0.4103, -0.1542] | 59.46 | 0 / 4 | -0.0611 | 9.25 | 0 |
| Area Tercile | Medium | 30 | -0.2575 | [-0.3692, -0.1552] | 26.53 | 1 / 3 | -0.0157 | -10.17 | 0 |
| Area Tercile | Large | 38 | -0.4229 | [-0.5075, -0.3386] | 66.22 | 1 / 5 | -0.0426 | 1.4 | 0 |
| Sharpness Tercile | Soft | 35 | -0.3977 | [-0.4850, -0.3171] | 65.14 | 1 / 6 | -0.0399 | -2.1 | 0 |
| Sharpness Tercile | Medium | 28 | -0.4059 | [-0.5370, -0.2750] | 57.45 | 0 / 3 | -0.0174 | 1.4 | 0 |
| Sharpness Tercile | Sharp | 33 | -0.1952 | [-0.3010, -0.1038] | 32.99 | 1 / 3 | -0.0582 | 1.25 | 0 |

---

### Table 7: Mini-Study S2 Execution Time Shares by Operator Category (%)

| Variant | Activations | Concat | Convolution (FP32/INT8) | Other | Quantize / Dequantize (QDQ) | Resize |
| --- | --- | --- | --- | --- | --- | --- |
| mbv3s_fp32 | 0.0 | 1.89 | 87.27 | 10.84 | 0.0 | 0.0 |
| mbv3s_int8 | 4.78 | 1.48 | 32.69 | 11.76 | 26.59 | 22.7 |
| mbv3s_mixed | 3.04 | 1.84 | 30.96 | 19.66 | 14.64 | 29.86 |
| r34_fp32 | 0.0 | 1.01 | 97.38 | 1.61 | 0.0 | 0.0 |
| r34_int8 | 2.54 | 1.08 | 57.23 | 4.29 | 24.45 | 10.42 |

---

### Table 8: Mini-Study S3 Calibration Set Size Sweep (Val DSC & HD95)

| n_calib | strategy | val_dsc | val_hd95 |
| --- | --- | --- | --- |
| 10 | class-stratified | 0.2635 | 134.2727 |
| 10 | hard-case | 0.3195 | 125.4229 |
| 10 | random | 0.1896 | 189.3485 |
| 25 | class-stratified | 0.1861 | 159.2206 |
| 25 | hard-case | 0.0624 | 215.9411 |
| 25 | random | 0.1674 | 211.1607 |
| 50 | class-stratified | 0.2605 | 133.7725 |
| 50 | hard-case | 0.043 | 198.5661 |
| 50 | random | 0.1611 | 207.983 |
| 100 | class-stratified | 0.2608 | 133.7785 |
| 100 | hard-case | 0.2317 | 125.5265 |
| 100 | random | 0.2461 | 114.2151 |
| 200 | class-stratified | 0.2431 | 139.8416 |
| 200 | hard-case | 0.2586 | 136.8359 |
| 200 | random | 0.2474 | 138.7999 |

---

### Table 9: Master Literature Benchmark Comparison (Held-out BUSI Test Set)

| Model | Type / Backbone | Parameters (M) | Size (MB) | Precision | Test DSC (Mean) | Test DSC (Median) | Test IoU (Mean) | Test HD95 (px) | Edge CPU Latency (ms) | Throughput (FPS) | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Shifaa U-Net (Online Pretrained) | 4-Stage Plain U-Net | 1.94 | 7.78 | FP32 | 0.4973 | 0.5279 | 0.3673 | 88.72 | 57.68 | 17.3 | Hugging Face (Ahmed-Selem) |
| Attention U-Net | ResNet-34 + SCSE Attention | 24.55 | 97.4 | FP32 | 0.7412 | 0.8124 | 0.6385 | 38.42 | 139.99 | 7.1 | Reported in Literature (Yap et al.); Latency benchmarked |
| U-Net++ (Nested Skips) | ResNet-34 Nested Dense | 26.08 | 103.8 | FP32 | 0.7588 | 0.834 | 0.6542 | 35.12 | 248.03 | 4.0 | Reported in Literature (Al-Dhabyani et al.); Latency benchmarked |
| Our ResNet-34 U-Net (FP32) | ResNet-34 | 24.44 | 93.22 | FP32 | 0.8012 | 0.9033 | 0.7188 | 27.71 | 74.01 | 13.5 | Ours (Trained) |
| Our ResNet-34 U-Net (Static INT8) | ResNet-34 | 24.44 | 23.59 | INT8 QDQ | 0.8003 | 0.9048 | 0.7187 | 28.56 | 65.96 | 15.2 | Ours (Quantized) |
| Our MobileNetV3-Small (FP32) | MobileNetV3-Small | 2.15 | 8.23 | FP32 | 0.7659 | 0.8716 | 0.677 | 35.13 | 23.32 | 42.9 | Ours (Trained) |
| Our MobileNetV3 v1 (Naive INT8 / Harsh) | MobileNetV3-Small | 2.15 | 2.39 | INT8 QDQ (Naive) | 0.4355 | 0.468 | 0.3523 | 86.97 | 26.68 | 37.5 | Ours (Collapsed) |
| Our MobileNetV3 v2 (Rescued Mixed k=13) | MobileNetV3-Small | 2.15 | 2.5 | Hybrid INT8/FP32 | 0.7263 | 0.8448 | 0.6314 | 35.2 | 23.31 | 42.9 | Ours (Rescued) |

---

### Table 10: Cross-Center Dataset Generalization (BUSI -> UDIAT Dataset B, Sabadell, Spain)

| model_name | regime | target_dataset | model_size_mb | test_dsc_mean | test_dsc_median | test_hd95_mean_px | test_hd95_median_px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ResNet-34 FP32 (BUSI) | Zero-Shot | UDIAT Dataset B | 93.22 | 0.7902 | 0.8748 | 17.03 | 6.61 |
| ResNet-34 INT8 (BUSI) | Zero-Shot | UDIAT Dataset B | 23.59 | 0.7928 | 0.8746 | 16.66 | 6.52 |
| MobileNetV3-Small FP32 (BUSI) | Zero-Shot | UDIAT Dataset B | 8.23 | 0.7244 | 0.8278 | 33.73 | 18.81 |
| MobileNetV3 v1 (Naive INT8 / Harsh) [BUSI] | Zero-Shot | UDIAT Dataset B | 2.43 | 0.1519 | 0.1248 | 148.71 | 145.13 |
| MobileNetV3 v2 (Rescued Mixed k=13) [BUSI] | Zero-Shot | UDIAT Dataset B | 2.5 | 0.7002 | 0.8562 | 31.36 | 16.04 |
| ResNet-34 FP32 (UDIAT Fine-Tuned) | Fine-Tuned | UDIAT Dataset B | 93.21 | 0.8249 | 0.9081 | 22.09 | 5.92 |
| ResNet-34 INT8 (UDIAT Fine-Tuned) | Fine-Tuned | UDIAT Dataset B | 23.59 | 0.8283 | 0.9067 | 21.51 | 5.92 |

---

### Table 11: BUVFM Clinical Breast Ultrasound Ground-Truth Benchmark (500 Scans: 399 Lesion, 101 Normal)

| Model Variant | BBox IoU Mean [95% CI] | BBox Dice Mean [95% CI] | Mask-in-BBox Precision | Centroid Error (px) | Lesion Coverage | Normal Spec (%) | Latency (ms) | Edge FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r34_fp32 | 0.4918 [0.4603, 0.5261] | 0.5797 [0.5458, 0.6160] | 0.6759 [0.6388, 0.7111] | 72.4 | 85.0% | 39.6% | 100.46 | 10.0 |
| r34_int8 | 0.4916 [0.4593, 0.5253] | 0.5807 [0.5462, 0.6161] | 0.6803 [0.6437, 0.7158] | 81.1 | 84.5% | 39.6% | 80.22 | 12.5 |
| mbv3s_fp32 | 0.4271 [0.3967, 0.4623] | 0.5102 [0.4770, 0.5464] | 0.6062 [0.5679, 0.6456] | 107.1 | 81.0% | 26.7% | 28.80 | 34.7 |
| mbv3s_int8 | 0.0429 [0.0366, 0.0494] | 0.0764 [0.0669, 0.0862] | 0.0626 [0.0535, 0.0723] | 96.9 | 100.0% | 0.0% | 33.18 | 30.1 |
| mbv3s_mixed_k13 | 0.3504 [0.3200, 0.3850] | 0.4270 [0.3925, 0.4633] | 0.5391 [0.4988, 0.5809] | 120.3 | 73.7% | 31.7% | 28.38 | 35.2 |
| cascade_mixed_r34 | 0.4724 [0.4404, 0.5062] | 0.5547 [0.5197, 0.5913] | 0.6455 [0.6074, 0.6836] | 126.3 | 79.4% | 51.5% | 92.77 | 10.8 |

---

### Table 12: Real Clinical Ultrasound Continuous Tracking Performance (BUVFM 5-Sequence Cohort)

| Video ID | Clinical Case | Class | Frames | Mean Area (px) | Raw TCI | EMA TCI (α=0.60) | Stability Gain | Cascade Triggers | Edge FPS |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V1 | Sample 1: Class 0 (Full Exam) | lesion | 50 | 520.8 | 0.004 | 0.041 | +1023.6% | 100.0% | 10.8 |
| V2 | Sample 2: Class 0 (Val 18 FPS) | lesion | 50 | 6576.6 | 0.179 | 0.069 | -61.6% | 76.0% | 13.3 |
| V3 | Sample 3: Class 1 (Val 18 FPS) | class_1 | 50 | 40.6 | 0.163 | 0.184 | +12.6% | 100.0% | 11.0 |
| V4 | Sample 4: Class 2 (Val 18 FPS) | class_2 | 50 | 657.8 | 0.105 | 0.270 | +157.2% | 100.0% | 11.1 |
| V5 | Sample 5: Class 2 (Val 18 FPS) | class_2 | 50 | 476.9 | 0.064 | 0.238 | +275.3% | 100.0% | 11.0 |

---

### Table 13: Genuine Clinical Ultrasound Video Benchmark (Dense Per-Frame Ground-Truth Masks)

| Video Sequence | Mean Dice (DSC) | Mean IoU (Jaccard) | Raw TCI | EMA TCI (α=0.60) | TCI Stability Gain | Frame Latency (ms) | Edge FPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IM_0003 | 0.2806 | 0.1774 | 0.202 | 0.333 | +65.2% | 94.7 | 10.6 |
| IM_0004 | 0.6252 | 0.4987 | 0.220 | 0.302 | +37.0% | 96.3 | 10.4 |
| IM_0005 | 0.6645 | 0.5298 | 0.430 | 0.510 | +18.5% | 95.4 | 10.5 |
| IM_0014 | 0.7140 | 0.5749 | 0.412 | 0.538 | +30.6% | 96.4 | 10.4 |
| IM_0055 | 0.4917 | 0.3560 | 0.357 | 0.490 | +37.0% | 97.2 | 10.3 |

---

### Table 14: Comprehensive Model Comparison (Main Primary vs. Main Edge vs. Distilled vs. Distil-Quantized)

| Category | Model | Variant / Precision | Test DSC [95% CI] | Test HD95 (px) | Disk Size (MB) | p50 Latency (ms) | FPS | Compression |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Main Primary (Teacher) | ResNet-34 | FP32 (Baseline) | 0.8012 [0.7535, 0.8457] | 27.71 | 93.22 | 74.01 | 13.5 | 1.00x |
| Main Primary (Teacher) | ResNet-34 | Static INT8 QDQ (MinMax) | 0.8003 [0.7515, 0.8462] | 28.56 | 23.59 | 65.96 | 15.2 | 3.95x |
| Main Edge Baseline | MobileNetV3-Small | FP32 (Baseline) | 0.7659 [0.7123, 0.8163] | 35.13 | 8.23 | 23.32 | 42.9 | 1.00x |
| Main Edge Baseline | MobileNetV3-Small | Static INT8 QDQ (Naive) | 0.4355 [0.3612, 0.5002] | 86.97 | 2.39 | 26.68 | 37.5 | 3.44x |
| Main Edge Baseline | MobileNetV3-Small | Rescued Mixed (k*=13) | 0.7263 [0.6690, 0.7784] | 35.20 | 2.50 | 23.31 | 42.9 | 3.30x |
| Distilled Student | MobileNetV3-Small | FP32 Distilled | 0.8034 [0.7617, 0.8430] | 26.44 | 8.23 | 28.87 | 34.9 | 1.00x |
| Distil-Quantized Student | MobileNetV3-Small | Static INT8 QDQ (MinMax) | 0.4450 [0.3810, 0.5080] | 82.64 | 2.39 | 33.84 | 29.7 | 3.44x |
| Distil-Quantized Student | MobileNetV3-Small | Static INT8 QDQ (Entropy) | 0.4450 [0.3810, 0.5080] | 82.64 | 2.39 | 33.76 | 29.7 | 3.44x |