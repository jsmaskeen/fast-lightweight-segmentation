# 10-Minute Professor Defense & Q&A Preparation Guide

**Project:** Edge-Native Real-Time Breast Ultrasound Tumor Segmentation  
**Format:** 10-Minute Oral Examination / Viva with Professor  
**Team Roster:** Speaker A, Speaker B, Speaker C  
**Primary Defense Objective:** Demonstrate deep technical mastery, mathematical rigor, architectural understanding, systems profiling acumen, and honest awareness of clinical limitations. Every citation and metric below is verified directly against the repository's `src/` code and `results/` CSV/JSON tables.

---

## Quick Navigation: Question Categories

- **[Category 1: Clinical Problem, Dataset Integrity & Metrics (Q1 – Q5)](#category-1-clinical-problem-dataset-integrity--metrics)** — *Speaker A Primary*
- **[Category 2: Quantization Mechanics & Graph Surgery Knob (Q6 – Q10)](#category-2-quantization-mechanics--graph-surgery-knob)** — *Speaker B Primary*
- **[Category 3: Knowledge Distillation & Distil-Quant Dynamics (Q11 – Q13)](#category-3-knowledge-distillation--distil-quant-dynamics)** — *Speaker B / C Primary*
- **[Category 4: Cine-Loop Streaming Systems & Temporal Filtering (Q14 – Q17)](#category-4-cine-loop-streaming-systems--temporal-filtering)** — *Speaker C Primary*
- **[Category 5: Cross-Center Generalization & Browser WebGPU (Q18 – Q20)](#category-5-cross-center-generalization--browser-webgpu)** — *Speaker A / C Primary*

---

## Category 1: Clinical Problem, Dataset Integrity & Metrics

### Q1: Why did you evaluate on BUSI with both Dice and HD95 instead of just Dice? What does HD95 capture that Dice misses?
- **Target Speaker:** Speaker A
- **The Core Concept:** Spatial boundary errors vs. volumetric overlap.
- **Master Answer:**  
  "Dice measures volumetric overlap: $\frac{2|X \cap Y|}{|X| + |Y|}$. However, in clinical breast oncology, Dice is insensitive to distant outlier false-positive clusters or boundary spiculation. A model can achieve a seemingly acceptable Dice score of 0.70 while predicting a rogue pixel island centimeters away from the true tumor margin, which would misguide biopsy needle placement.  
  The 95th-percentile Hausdorff Distance (HD95) computes the maximum distance between predicted and ground-truth boundary contours, clipped at the 95th percentile to suppress single-pixel acoustic noise. It directly penalizes geometric distortion. For example, when MobileNetV3 naive INT8 collapsed, its Dice score dropped by 43% (from 0.7659 to 0.4355), but its HD95 surged by **148%** (from 35.13 px to **86.97 px**), immediately alerting us to phantom margin errors that overlap metrics under-reported."

### Q2: How did you handle normal (tumor-free) ultrasound scans during evaluation? Isn't Dice undefined when both prediction and ground truth are empty?
- **Target Speaker:** Speaker A
- **The Core Concept:** Zero-division convention and clinical False Positive Rate (FPR).
- **Master Answer:**  
  "When evaluating the 67 normal scans in the held-out BUSI test split, both the ground truth $Y$ and ideal prediction $X$ are empty sets, making standard $\frac{0}{0}$ Dice undefined.  
  We adhered to the medical imaging convention implemented in `src/metrics/dsc_iou_hd95.py` and verified in `tests/test_metrics.py`: if both mask and prediction are empty, DSC is assigned 1.0 (true negative); if one is empty and the other is non-empty, DSC is assigned 0.0.  
  To prevent normal scans from artificially inflating or deflating tumor overlap metrics, we compute Test DSC, IoU, and HD95 strictly over the **96 held-out lesion scans**, and separately report **Normal FPR@50px** on the 67 normal scans (counting a false positive whenever a predicted mask exceeds 50 foreground pixels on healthy tissue)."

### Q3: Why is class imbalance so severe in ultrasound segmentation, and how does your compound loss address it?
- **Target Speaker:** Speaker A
- **The Core Concept:** Vanishing gradients in cross-entropy for small foreground targets.
- **Master Answer:**  
  "In a standard $256 \times 256$ B-mode ultrasound frame (65,536 pixels), a typical early-stage malignant nodule may occupy fewer than 1,000 pixels—under 1.5% of the frame. Standard Binary Cross-Entropy (BCE) averages across all 65,536 pixels; consequently, the background class overwhelmingly dominates the gradient, biasing the network toward all-zero predictions.  
  In `src/train/losses.py` (`BCEDiceLoss`), we formulated an equal-weighted compound objective: $\mathcal{L}_{\text{task}} = 0.5\,\mathcal{L}_{\text{BCE}} + 0.5\,\mathcal{L}_{\text{Dice}}$. While BCE provides smooth, convex pixel-wise gradients in early training, soft Dice ($\epsilon=10^{-6}$) normalizes gradients by the foreground sum $\frac{2 \sum p_i y_i + \epsilon}{\sum p_i + \sum y_i + \epsilon}$. This forces the gradient magnitude to depend on relative tumor overlap rather than background frame resolution."

### Q4: How were the train, validation, and test splits generated? Did you prevent near-duplicate patient scan leakage?
- **Target Speaker:** Speaker A
- **The Core Concept:** Perceptual deduplication, cluster-atomic splitting, and data leakage prevention.
- **Master Answer:**  
  "We implemented a rigorous deduplication and cluster-atomic splitting pipeline in `src/data/merge_masks.py` (audited in `data/dedup_report.json` and `tests/test_data.py`):  
  1. **Deduplication:** Starting from the 780 raw BUSI scans (437 benign, 210 malignant, 133 normal), we computed SHA-256 byte hashes and 64-bit perceptual difference hashes (`dhash`). We detected and removed **4 exact/cross-class conflicting duplicates** (`malignant (51)`, `malignant (52)`, `malignant (94)`, `malignant (145)`), leaving **776 clean scans** (643 lesion, 133 normal).  
  2. **Union-Find Clustering:** We grouped near-duplicate cross-sectional views within the same class (`dhash` Hamming distance $\le 2$) into **686 atomic clusters** using connected components, ensuring multi-view scans of the same patient never split across partitions.  
  3. **Stratified Partitioning (`data/splits.json`, seed 42):** The 643 lesion scans were stratified **70% / 15% / 15%** into Train (**450**), Val (**97**), and Test (**96**). All 133 normal scans were strictly excluded from training (**0 in Train**) and split **50% / 50%** into Val (**66**) and Test (**67**), giving 450 Train, 163 Val, and 163 Held-Out Test scans."

### Q5: Did you observe any multi-mask cases where an ultrasound scan contained multiple distinct tumor lesions?
- **Target Speaker:** Speaker A
- **The Core Concept:** Mask combination at native resolution via logical bitwise-OR.
- **Master Answer:**  
  "Yes. In BUSI, multifocal cases contain multiple ground-truth mask files per scan (e.g., `_mask.png`, `_mask_1.png`, `_mask_2.png`).  
  In `src/data/merge_masks.py`, all ground-truth masks belonging to the same scan ID are combined via `np.bitwise_or.reduce` **at native resolution prior to resizing**, resized to $256 \times 256$ using nearest-neighbor interpolation (`cv2.INTER_NEAREST`) to prevent boundary anti-aliasing, and strictly re-binarized at $>127$. This guarantees our models learn multifocal lesions without gray-level interpolation artifacts."

---

## Category 2: Quantization Mechanics & Graph Surgery Knob

### Q6: Walk me through the exact mathematics of your static INT8 QDQ quantization. What is the difference between MinMax and KL-Entropy calibration?
- **Target Speaker:** Speaker B
- **The Core Concept:** Affine mapping, clipping thresholds, and activation histogram divergence.
- **Master Answer:**  
  "Our Quantize-Dequantize (QDQ) pipeline (`src/quant/quantize.py`) maps floating-point values into 8-bit signed integers $[-128, 127]$ via:
  $$q = \text{clip}\left(\left\lfloor \frac{x}{S} \right\rceil, -128, 127\right), \quad \hat{x} = S \cdot q$$
  Weights are quantized using symmetric per-channel scales $S_c = \frac{\max |W_c|}{127}$, which avoids inter-channel weight skew.  
  For activations, dynamic ranges are calibrated on 100 training scans (`BUSICalibrationDataReader`, `seed=42`):  
  - **MinMax Calibration:** Sets $S = \frac{\max(|x_{\min}|, |x_{\max}|)}{127}$. While preserving extreme peaks, any rare activation spike inflates $S$, collapsing resolution for the bulk of values near zero.  
  - **KL-Entropy Calibration:** Collects activation histograms and minimizes the Kullback-Leibler divergence $D_{\text{KL}}(P \| Q)$ between the FP32 distribution $P$ and quantized distribution $Q$, clipping extreme outliers to preserve fine bit-depth for typical acoustic intensities."

### Q7: Why did ResNet-34 quantize losslessly to INT8 while MobileNetV3 suffered catastrophic collapse?
- **Target Speaker:** Speaker B
- **The Core Concept:** Channel capacity, group convolution dynamic range, and representational redundancy.
- **Master Answer:**  
  "This was our primary architectural finding (`results/accuracy_audit_test.csv`):  
  ResNet-34 uses dense $3 \times 3$ convolutions with high channel redundancy ($C \in [64, 512]$). Each output feature averages across up to 512 input channels. By the Central Limit Theorem, this cross-channel summation smooths activation distributions into well-behaved bell curves that quantize cleanly with $<0.1\%$ loss (0.8012 $\to$ **0.8003 DSC**, 3.95× compression to **23.59 MB**).  
  In contrast, MobileNetV3-Small relies on Depthwise Separable Convolutions where $group = C_{\text{in}}$ (only 9 weights per channel with zero cross-channel summation). Consequently, different channels exhibit wildly varying dynamic ranges. Applying per-tensor activation quantization clips low-amplitude channels to zero, obliterating faint ultrasound boundaries and causing test Dice to collapse from 0.7659 to **0.4355** ($\Delta\text{DSC} = -0.3304$)."

### Q8: How does your Extension A sensitivity metric work? How did you define SQNR and decide which nodes to keep in FP32?
- **Target Speaker:** Speaker B
- **The Core Concept:** Empirical layer-wise SQNR profiling and topological graph surgery.
- **Master Answer:**  
  "In `src/quant/sensitivity.py`, we augmented the FP32 and INT8 ONNX graphs to dump intermediate activation tensors across calibration scans and computed the empirical Signal-to-Quantization-Noise Ratio (SQNR):
  $$\text{SQNR}_l = 10 \log_{10} \left( \frac{\sum_i x_{l,i}^2}{\sum_i (x_{l,i} - \hat{x}_{l,i})^2} \right) \quad [\text{dB}]$$
  We combined negative SQNR with structural graph topology attributes:
  1. The output segmentation head (`/segmentation_head/segmentation_head.0/Conv`) and input stem (`/encoder/model/conv_stem/Conv`).  
  2. The inverted bottleneck depthwise convolutions (`conv_dw/Conv`, where $group > 1$).  
  Our CLI tool `src/quant/knob.py` ranks all 63 `Conv` nodes and excludes the top-$k$ most sensitive nodes from INT8 quantization. In `results/extA_sweep.json`, sweeping $k$ identified the sharp recovery inflection point at **$k^*=13$**: keeping just 13 of 63 convolutions in FP32 recovers test Dice to **0.7263** (Median **0.8448**, HD95 **35.20 px**) while outperforming both the first/last heuristic ($k=2$, Val DSC 0.5016) and random $k=13$ selection (mean Val DSC 0.5614), at **2.50 MB** (3.30× compression)."

### Q9: Why is your Rescued Mixed model (23.31 ms) faster than the Naive INT8 model (26.68 ms)? Shouldn't an INT8 model always be faster than a mixed model?
- **Target Speaker:** Speaker B
- **The Core Concept:** Memory bandwidth bounds, QDQ operator overhead, and cache thrashing.
- **Master Answer:**  
  "This is a classic systems insight (`results/study_s2_operator_profile.csv`). While integer MACs are faster for heavy dense convolutions (speeding up ResNet-34 from 74.01 ms to 65.96 ms), depthwise separable convolutions have very low arithmetic intensity (FLOPs-per-byte) and are strictly **memory-bandwidth bound**.  
  In Mini-Study S2, ONNX Runtime operator-level profiling revealed that in Naive INT8 MobileNetV3, **QuantizeLinear and DequantizeLinear (QDQ) nodes consumed 26.59% of total CPU execution time** (nearly matching the 32.69% spent in convolutions!). The constant memory traffic of casting activation tensors between FP32 and INT8 overwhelmed the integer arithmetic savings.  
  In our Rescued Mixed model ($k^*=13$), pruning QDQ nodes around the 13 rescued layers slashed QDQ runtime share from **26.59% down to 14.64%**, allowing Rescued Mixed (**23.31 ms**) to execute faster than Naive INT8 (**26.68 ms**)."

### Q10: What did Mini-Study S1 reveal about which clinical pathologies suffer most from INT8 quantization?
- **Target Speaker:** Speaker B
- **The Core Concept:** Stratified error analysis across pathology class, boundary sharpness, and lesion area.
- **Master Answer:**  
  "In Mini-Study S1 (`results/study_s1_stratified_damage.csv`, `src/metrics/stratify.py`), we stratified paired test damage ($\text{DSC}_{\text{INT8}} - \text{DSC}_{\text{FP32}}$) across all 96 held-out lesion scans:
  1. **Boundary Sharpness Terciles:** Tumors with **Soft / diffuse boundaries** ($\Delta\text{DSC} = -0.3977$) and **Medium boundaries** ($\Delta\text{DSC} = -0.4059$) suffered **2.0× greater INT8 damage** than **Sharp boundaries** ($\Delta\text{DSC} = -0.1952$). Coarse 8-bit activation quantization erases subtle acoustic transition gradients at poorly defined margins.  
  2. **Pathology Class:** **Malignant lesions** ($N=31$, $\Delta\text{DSC} = -0.4027$) suffered **1.36× worse degradation** than **Benign lesions** ($N=65$, $\Delta\text{DSC} = -0.2961$), because malignant tumors exhibit irregular, infiltrated borders.  
  3. **Rescue Verification:** Across all strata, our Rescued Mixed model ($k^*=13$) reduced overall degradation from $-0.3305$ to just **$-0.0396$** ($\Delta\text{HD95} = +0.07\text{ px}$), recovering malignant tumors to within **$-0.0226$ DSC** of FP32 and eliminating all 12 empty predictions!"

---

## Category 3: Knowledge Distillation & Distil-Quant Dynamics

### Q11: In Knowledge Distillation, what temperature $T$ and $\alpha$ did you use and why? What is the function of dark knowledge in tumor segmentation?
- **Target Speaker:** Speaker B / C
- **The Core Concept:** Soft label logit temperature scaling and spatial dark knowledge transfer.
- **Master Answer:**  
  "In `src/distil/loss.py` and `src/distil/distil.py`, we trained the MobileNetV3-Small student from the frozen ResNet-34 FP32 teacher using temperature **$T = 2.0$** and balance weight **$\alpha = 0.50$**:
  $$\mathcal{L}_{\text{KD}} = 0.5\,\mathcal{L}_{\text{task}}(y, z_s) + 0.5 \cdot (2.0)^2\,\mathcal{L}_{\text{BCE}}\!\left(\sigma\!\left(\frac{z_s}{2.0}\right),\; \sigma\!\left(\frac{z_t}{2.0}\right)\right)$$
  In hard binary masks, every pixel is strictly 0 or 1, providing zero supervision on boundary uncertainty. However, ultrasound tumor margins are zones of gradual acoustic impedance transition.  
  Scaling logits by $T=2.0$ softens the teacher's spatial probability map across tumor perimeters, transferring continuous 'dark knowledge' about boundary ambiguity and speckle suppression. This regularizer propelled our 2.15M-parameter MobileNetV3 student from 0.7659 DSC to **0.8034 DSC** (Median **0.8916**, HD95 **26.44 px**), surpassing the 24.44M-parameter ResNet-34 teacher (0.8012 DSC)."

### Q12: Why did naive INT8 quantization of the distilled student model still fail (0.4450 DSC)?
- **Target Speaker:** Speaker B
- **The Core Concept:** Representation learning vs. hardware numeric precision.
- **Master Answer:**  
  "This was a critical control experiment (`results/distil_audit_test.csv`). One might hypothesize that knowledge distillation smooths weight distributions enough to survive uniform INT8 quantization.  
  Our audit disproved this for depthwise separable networks: unconstrained static INT8 quantization (both MinMax and Entropy) of the distilled checkpoint dropped test Dice from 0.8034 to **0.4450** (HD95 **82.64 px**).  
  While distillation trains superior feature representations in 32-bit floating point, it does not alter the $group = C_{\text{in}}$ depthwise convolution topology. The isolated single-channel activations still experience high inter-channel dynamic range variance under per-tensor 8-bit clipping. Thus, distillation solves the **capacity gap**, while Extension A mixed-precision graph surgery solves the **quantization bottleneck**."

### Q13: Did you try Quantization-Aware Training (QAT) instead of post-training quantization?
- **Target Speaker:** Speaker B
- **The Core Concept:** Post-Training Quantization (PTQ) vs QAT complexity and zero-recomputation benefits.
- **Master Answer:**  
  "We intentionally focused on Post-Training Quantization (PTQ) with mixed-precision graph surgery (`src/quant/knob.py`) because of its practical deployment advantages: it requires **zero retraining**, executes in seconds on a standard CPU, and allows dynamic adjustment of the precision knob $k \in [0, 63]$ without GPU clusters.  
  Furthermore, between our Distilled FP32 student (**0.8034 DSC** at 8.23 MB / 28.87 ms) and our Rescued Mixed PTQ model (**0.7263 DSC** at 2.50 MB / 23.31 ms), we already span the entire clinical accuracy-efficiency frontier without the training instability of fake-quantization QAT."

---

## Category 4: Cine-Loop Streaming Systems & Temporal Filtering

### Q14: How does your Extension B ring buffer handle producer-consumer concurrency? Why capacity 4?
- **Target Speaker:** Speaker C
- **The Core Concept:** Bounded ring buffer, thread synchronization, and drop-oldest freshness invariant.
- **Master Answer:**  
  "In live ultrasound scanning, the transducer pushes frames at 30 Hz (every 33.3 ms). If the inference consumer experiences a momentary lag spike, an unbounded FIFO queue accumulates stale frames, causing the screen to lag behind the sonographer's hand movement.  
  In `src/cine/ring_buffer.py` (`CineRingBuffer`), we built a thread-safe bounded circular buffer of capacity $C = 4$ synchronized via `threading.Lock` and `threading.Condition`. It enforces a **double drop-oldest freshness invariant**:  
  1. On `push()`: if the buffer is full ($count = 4$), the write pointer overwrites the oldest frame at `tail`.  
  2. On `pop_latest()`: the consumer immediately pops the newest frame at `(head - 1) % capacity` and discards any older queued frames, resetting `count = 0`.  
  This guarantees the sonographer **always views the single freshest frame**, bounding queue staleness to a single frame interval, verified by concurrent stress tests in `tests/test_cine.py`."

### Q15: What were the exact streaming drop rates at 30 Hz for ResNet-34 vs MobileNetV3?
- **Target Speaker:** Speaker C
- **The Core Concept:** Throughput bounds and deadline miss telemetry.
- **Master Answer:**  
  "Across our 60-second (1,800-frame) multi-threaded streaming runs at 30 Hz (`results/cine_matrix_summary.csv`, Table 5):
  - **ResNet-34 FP32:** **61.91% mean drop rate** (up to 65.06% peak), 104.3 ms p50 e2e latency, 63.59% deadline miss rate.  
  - **ResNet-34 INT8:** **56.28% mean drop rate**, 91.8 ms p50 e2e latency, 29.19% deadline miss rate.  
  - **MobileNetV3 Naive INT8:** **4.41% mean drop rate**, 45.8 ms p50 e2e latency (degraded by QDQ overhead).  
  - **MobileNetV3 FP32:** **0.02% mean drop rate** (0 to 1 frame dropped out of 1,800), **29.9 ms** p50 e2e latency, **0.0% deadline misses**.  
  - **MobileNetV3 Rescued Mixed ($k^*=13$):** **0.06% mean drop rate** (1 to 2 frames out of 1,800), **28.3 ms** p50 e2e latency, **0.0% deadline misses**.  
  Thus, ResNet-34 can only sustain 10 Hz without dropping frames, whereas MobileNetV3 FP32 and Rescued Mixed sustain full 30 Hz real-time streaming."

### Q16: How does your Temporal EMA smoothing work, and how did you measure Temporal Consistency (TCI)?
- **Target Speaker:** Speaker C
- **The Core Concept:** Recursive logit filtering and pairwise Jaccard consistency across video frames.
- **Master Answer:**  
  "Frame-by-frame 2D segmentation suffers from high-frequency perimeter flicker due to shifting acoustic speckle.  
  In `src/cine/visual_cine.py` (`CineRealtimePipeline`), we apply an Exponential Moving Average (EMA) filter directly to the output spatial logits before thresholding:
  $$Z_t^{\text{EMA}} = \alpha Z_t + (1-\alpha) Z_{t-1}^{\text{EMA}}, \quad \alpha = 0.60$$
  We quantify stability via the **Temporal Consistency Index (TCI)** (`compute_tci`), defined as the consecutive-frame Jaccard overlap:
  $$\text{TCI}_t = \frac{|M_t \cap M_{t-1}|}{|M_t \cup M_{t-1}|}$$
  Evaluated across 5 genuine clinical ultrasound video sequences with dense per-frame ground-truth masks from the UFUV MICCAI 2024 cohort (`results/real_video_groundtruth_benchmark.csv`, Table 13), EMA smoothing improved TCI across every sequence by **+18.5% to +65.2%** (mean **+37.7% stability gain**)."

### Q17: How do you handle ambiguous frames or sudden probe movements during live video tracking?
- **Target Speaker:** Speaker C
- **The Core Concept:** Boundary Shannon entropy and Uncertainty-Gated Cascade.
- **Master Answer:**  
  "First, with $\alpha=0.60$, the current frame carries 60% weight and the effective exponential memory horizon is $\frac{1}{\alpha} \approx 1.67$ frames (~55 ms), preventing noticeable visual lag during normal sweeps.  
  Second, in `src/cine/visual_cine.py` (`compute_boundary_entropy`), we implemented an **Uncertainty-Gated Cascade**: on every frame, we compute the mean Shannon entropy $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$ strictly within the ambiguous boundary band where $|p - 0.5| < 0.20$. Whenever boundary entropy exceeds $\tau_{\text{unc}} = 0.75$ bits—indicating an acoustically ambiguous margin or rapid probe shift—the pipeline can dynamically cascade to the high-capacity ResNet-34 INT8 model."

---

## Category 5: Cross-Center Generalization & Browser WebGPU

### Q18: How did you test cross-center domain shift? What is the UDIAT dataset and what were the exact numbers?
- **Target Speaker:** Speaker C / A
- **The Core Concept:** Out-of-distribution zero-shot evaluation across scanner manufacturers.
- **Master Answer:**  
  "To test whether our quantized models overfit to BUSI's acquisition hardware (LOGIQ E9 / LOGIQ E9 Agile in Cairo, Egypt), we evaluated them **zero-shot** on **UDIAT Dataset B from Hospital Parc Taulí in Sabadell, Spain** (`src/metrics/cross_dataset.py`, `results/udiat_cross_center_results.csv`, 163 lesion scans: 110 benign, 53 malignant, acquired on a Siemens Sonoline Elegra):  
  - **ResNet-34 FP32 (Zero-Shot):** **0.7902** Mean DSC (Median **0.8748**), HD95 **17.03 px**.  
  - **ResNet-34 INT8 (Zero-Shot):** **0.7928** Mean DSC (Median **0.8746**), HD95 **16.66 px** *(INT8 slightly exceeds FP32 zero-shot!)*.  
  - **MobileNetV3 FP32 (Zero-Shot):** **0.7244** Mean DSC (Median **0.8278**), HD95 **33.73 px**.  
  - **MobileNetV3 Rescued Mixed $k^*=13$ (Zero-Shot):** **0.7002** Mean DSC (Median **0.8562**), HD95 **31.36 px**.  
  - **MobileNetV3 Naive INT8 (Zero-Shot):** Collapsed to **0.1519** Mean DSC (HD95 **148.71 px**).  
  Furthermore, fine-tuning ResNet-34 on UDIAT pushed DSC to **0.8249 (FP32)** and **0.8283 (INT8)**."

### Q19: Why build an In-Browser WebGPU/WASM WebViewer? How does it work under the hood?
- **Target Speaker:** Speaker A
- **The Core Concept:** Client-side ONNX Runtime Web, Cross-Origin Isolation, WebGPU shaders, and HIPAA privacy.
- **Master Answer:**  
  "Hospital IT environments restrict installing custom native executables or drivers. Our WebViewer (`server/app.py`, `server/static/js/app.js`) runs entirely inside standard web browsers using **ONNX Runtime Web**:  
  1. **Cross-Origin Isolation:** `server/app.py` injects `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` headers, enabling `SharedArrayBuffer` for multi-threaded WebAssembly SIMD and WebGPU.  
  2. **Progressive Hardware Fallback:** `createSessionWithFallback()` attempts **WebGPU** first (achieving ~22.4 ms inference via GPU compute shaders) and automatically falls back to **4-threaded WASM SIMD** (~31.8 ms) on devices without WebGPU.  
  3. **Reactive Client Telemetry & 100% Privacy:** Sigmoid probability maps are cached in a `Float32Array(256 * 256)` with bilinear interpolation (`sampleProb`), allowing clinicians to drag the threshold slider ($\tau \in [0.05, 0.95]$) and update contours, bounding boxes, and centroids at 60 FPS with zero network uploads—keeping patient scans 100% on-device."

### Q20: What are the primary clinical failure modes and limitations of this system in practice?
- **Target Speaker:** Speaker C / All
- **The Core Concept:** Realistic appraisal of clinical edge cases, artifacts, and translational hurdles.
- **Master Answer:**  
  "We identified three concrete clinical limitations:  
  1. **Posterior Acoustic Shadowing:** Dense calcifications or fibrotic masses attenuate the ultrasound beam, casting dark posterior shadows. Because deep boundary echoes are absent, 2D models can under-segment the posterior wall.  
  2. **Out-of-Plane Probe Translation:** When a sonographer tilts or translates the transducer perpendicular to the 2D imaging plane, the anatomical cross-section shifts abruptly, which 2D temporal EMA smoothing cannot track across 3D volume space.  
  3. **Normal Scan False-Positive Sensitivity:** Because training (`data/splits.json`) used exclusively lesion scans to maximize boundary recall, raw uncalibrated normal specificity requires post-processing area/confidence gating (or uncertainty cascade filtering) when scanning healthy breast tissue.  
  Accordingly, our system is designed as an interactive real-time sonographer decision-support overlay rather than an autonomous diagnostic replacement."

---

## Defense Quick-Reference Card (Keep Open During Viva)

| Architecture | Precision / Variant | Size (MB) | Latency (p50) | Edge FPS | 30 Hz Stream Drops | BUSI Test Mean / Median DSC | BUSI HD95 | UDIAT Zero-Shot Mean / Median DSC |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-34** | FP32 Baseline | 93.22 MB | 74.01 ms | 13.5 | 61.91% | 0.8012 / 0.9033 | 27.71 px | 0.7902 / 0.8748 |
| **ResNet-34** | **Static INT8 QDQ** | **23.59 MB** | 65.96 ms | 15.2 | 56.28% | **0.8003 / 0.9048** | 28.56 px | **0.7928 / 0.8746** |
| **MobileNetV3** | FP32 Baseline | 8.23 MB | 23.32 ms | 42.9 | **0.02%** | 0.7659 / 0.8716 | 35.13 px | 0.7244 / 0.8278 |
| **MobileNetV3** | Naive Static INT8 | 2.39 MB | 26.68 ms | 37.5 | 4.41% | 0.4355 / 0.4680 | 86.97 px | 0.1519 / 0.1248 |
| **MobileNetV3** | **Rescued Mixed ($k^*=13$)** | **2.50 MB** | **23.31 ms** | **42.9** | **0.06%** | **0.7263 / 0.8448** | **35.20 px** | **0.7002 / 0.8562** |
| **MobileNetV3** | **Distilled FP32 ($T=2.0$)** | **8.23 MB** | **28.87 ms** | **34.9** | **<0.1%** | **0.8034 / 0.8916** | **26.44 px** | — *(BUSI Primary)* |
| **MobileNetV3** | Distilled INT8 (Naive) | 2.39 MB | 33.84 ms | 29.7 | — | 0.4450 / 0.4850 | 82.64 px | — *(BUSI Primary)* |
