# 7-Minute Video Presentation Script (Word-for-Word)

**Project Title:** Edge-Native Real-Time Breast Ultrasound Tumor Segmentation  
**Subtitle:** Extreme Quantization, Mixed-Precision Graph Surgery, and Low-Latency POCUS Streaming  
**Target Duration:** Exactly 7 minutes (420 seconds)  
**Total Spoken Word Count:** ~945 words (~135 words per minute pacing)  
**Speaker Roster:** Speaker A, Speaker B, Speaker C (Equal contribution throughout)  
**Slide Progression:** Strictly sequential **Slides 1 through 12** (no backward or out-of-order jumps)

---

## Timing & Segment Allocation Overview

```
[0:00 - 1:45] Speaker A : Slides 1–3   | Clinical Motivation, Edge Constraints & Architectural Progression (105s)
[1:45 - 4:10] Speaker B : Slides 4–7   | Quantization Collapse, Graph Surgery Knob, Distillation & Profiling (145s)
[4:10 - 6:05] Speaker C : Slides 8–10  | Master Evaluation, Literature Benchmarks & 30 Hz Streaming (115s)
[6:05 - 6:35] Speaker A : Slide 11     | In-Browser WebGPU/WASM Diagnostic WebViewer Demo (30s)
[6:35 - 7:00] Speaker C : Slide 12     | Clinical Limitations, Trade-offs & Conclusion (25s)
```

---

## Detailed Spoken Script

### [0:00 - 0:25] | Slide 1: Title & Introduction
**Speaker A:**  
"Hello everyone. Welcome to our project presentation on **Edge-Native Real-Time Breast Ultrasound Tumor Segmentation**.  
My teammates Speaker B, Speaker C, and I investigated how to deploy deep segmentation networks directly onto battery-powered, handheld ultrasound probes without sacrificing diagnostic accuracy or real-time clinical responsiveness.  
Every member of our team contributed equally across the architecture design, quantization mathematics, streaming systems engineering, and clinical validation that you will see today."

---

### [0:25 - 1:05] | Slide 2: Clinical Problem & The POCUS Dilemma
**Speaker A:**  
*[Speaker A gestures toward the two comparison columns on Slide 2]*  
"Breast cancer remains the most prevalent malignancy among women worldwide. Point-of-Care Ultrasound, or POCUS, offers a non-ionizing, cost-effective tool for rapid bedside triage.  
However, clinical ultrasound is plagued by acoustic speckle, low signal-to-noise ratios, and faint, irregular tumor boundaries.  
Compounding this challenge is the edge computing bottleneck: handheld probes run on constrained mobile APUs and CPUs with strict thermal limits. A clinical ultrasound feed produces 30 frames per second, imposing a non-negotiable **33.3-millisecond latency deadline** per frame, while storage must remain under 10 megabytes.  
Transmitting scans to cloud servers introduces privacy violations and unacceptable network jitter. Our goal was clear: engineer a sub-3-megabyte neural model capable of zero-drop 30-hertz streaming on local edge hardware."

---

### [1:05 - 1:45] | Slide 3: Architectural Progression & Loss Formulation
**Speaker A:**  
*[Speaker A highlights the backbone diagram and compound loss formulas]*  
"We began by contrasting two distinct architectural philosophies.  
First, a heavyweight **ResNet-34 U-Net** representing the clinical standard. With standard 3-by-3 convolutions and 24.44 million parameters, it achieves a high Dice score of 0.8012 on the BUSI test split. But at 93.2 megabytes and 74 milliseconds latency, it only manages 13.5 frames per second—dropping over 60 percent of frames in a live 30-hertz feed.  
Second, we engineered an edge-native **MobileNetV3-Small U-Net**. By leveraging inverted residual bottlenecks, depthwise separable convolutions, and Squeeze-and-Excitation attention, we compressed parameter count by **11.4-fold** down to 2.15 million weights. It runs natively at **42.9 frames per second** on CPU with an 8.23-megabyte footprint, achieving 0.7659 Dice.  
To supervise both models against extreme foreground-background class imbalance, we optimized an equal-weighted compound loss combining Binary Cross-Entropy with soft Dice loss.  
Now, I hand over to Speaker B to reveal what happened when we quantized these networks."

---

### [1:45 - 2:30] | Slide 4: Quantization Breakdown & Failure Discovery
**Speaker B:**  
*[Speaker B steps in and points to the INT8 mathematical formulation on Slide 4]*  
"Thank you, Speaker A.  
To fit embedded memory budgets, we applied static 8-bit integer Quantize-Dequantize—or QDQ—quantization. We mapped FP32 tensors into signed 8-bit integers using symmetric per-channel scaling for weights and symmetric calibration on 100 training scans for activations.  
*[Speaker B points to the diverging red and blue callout boxes]*  
Here, we uncovered a striking architectural divergence.  
On ResNet-34, static INT8 quantization was completely **lossless**. It shrank disk footprint by 4-fold to **23.59 megabytes** while maintaining a **0.8003 Dice score**—a negligible change of under 0.1 percent.  
However, when we applied the exact same pipeline to MobileNetV3, the model suffered a **catastrophic accuracy collapse**: Dice plummeted from 0.7659 to **0.4355**, and 95th-percentile Hausdorff distance exploded from 35.1 to **86.97 pixels**.  
Why? In depthwise separable layers, each channel is convolved in isolation. Without cross-channel averaging, activation dynamic ranges fluctuate wildly, and coarse INT8 clipping obliterates weak acoustic tumor boundaries. As shown in Mini-Study S1 on the bottom right, tumors with soft, diffuse margins and malignant spiculations suffered up to 2 times worse degradation than sharp margins."

---

### [2:30 - 3:15] | Slide 5: Extension A: Sensitivity Knob & Graph Surgery
**Speaker B:**  
*[Speaker B gestures to the SQNR formula and the validation k-sweep curve on Slide 5]*  
"To solve this collapse without inflating model size, we developed **Extension A: Mixed-Precision Graph Surgery**.  
We profiled the empirical Signal-to-Quantization-Noise Ratio, or SQNR, across every layer. Structural graph analysis confirmed that sensitivity concentrated heavily in the first and last boundary layers, as well as intermediate depthwise bottlenecks.  
We engineered an automated zero-recomputation graph surgery knob, implemented in `src/quant/knob.py`.  
By sweeping the parameter $k$, we selectively kept the top-$k$ most sensitive operators in 32-bit floating point while quantizing all remaining layers.  
At the optimal inflection point, **$k^* = 13$**, preserving just 13 of 63 convolutions in FP32 completely rescued the model: test Dice rebounded to **0.7263**—with a median of **0.8448**—while preserving a **3.30-fold compression** to just **2.50 megabytes** at **42.9 frames per second**."

---

### [3:15 - 3:45] | Slide 6: Knowledge Distillation & Distil-Quantization
**Speaker B:**  
*[Speaker B highlights the KD loss formula and the 3-panel comparison figure on Slide 6]*  
"Next, to determine whether student capacity could be elevated further, we implemented Knowledge Distillation.  
We distilled spatial dark knowledge from the 24.44-million parameter ResNet-34 teacher into our 2.15-million parameter MobileNetV3 student using temperature-scaled soft targets at temperature $T = 2.0$ and balance $\alpha = 0.50$.  
The result was remarkable: distilled MobileNetV3 reached **0.8034 Dice**, **outperforming its own teacher** while utilizing over eleven times fewer parameters.  
Critically, as shown on the right, naively quantizing this distilled model still resulted in collapse at 0.4450 Dice, proving that knowledge distillation elevates feature representations but cannot overcome bit-depth clipping without mixed-precision preservation."

---

### [3:45 - 4:10] | Slide 7: Systems Profiling Deep-Dive (Mini-Study S2)
**Speaker B:**  
*[Speaker B directs attention to the operator breakdown bar chart on Slide 7]*  
"Finally, we conducted deep systems profiling with ONNX Runtime to inspect real edge CPU execution.  
Counterintuitively, we discovered that in naive INT8 MobileNetV3, **Quantize and Dequantize nodes consumed 26.6 percent of total runtime**, negating integer math speedups due to constant memory casting.  
By pruning QDQ nodes around the 13 rescued layers, our Rescued Mixed model reduced QDQ overhead to 14.6 percent. Consequently, our Rescued Mixed model at **23.31 milliseconds** actually runs faster than naive INT8 at 26.68 milliseconds.  
I now hand over to Speaker C to present our master results and streaming benchmarks."

---

### [4:10 - 4:50] | Slide 8: Master Quantitative Results & Pareto Frontier
**Speaker C:**  
*[Speaker C points to the master evaluation table and Pareto curve on Slide 8]*  
"Thank you, Speaker B.  
Here we synthesize our complete experimental benchmark on the held-out BUSI test split.  
Rather than a single winner, our research establishes a clear three-point Pareto frontier:  
First, for hospital ultrasound carts with moderate compute, **ResNet-34 INT8** offers lossless diagnostic fidelity: 0.8003 Dice, 23.59 megabytes, and 15.2 frames per second.  
Second, where raw accuracy on edge devices is paramount, **MobileNetV3 Distilled FP32** achieves peak performance: 0.8034 Dice at 34.9 frames per second and 8.23 megabytes.  
Third, for ultra-low-power handheld probes, our **Rescued Mixed model** occupies the optimal sweet spot: 0.7263 Dice, a tiny **2.50-megabyte footprint**, and **42.9 frames per second**."

---

### [4:50 - 5:25] | Slide 9: Literature Benchmark & Cross-Center Generalization
**Speaker C:**  
*[Speaker C points to the literature table and qualitative visual overlays on Slide 9]*  
"To contextualize our contributions against the state-of-the-art, we benchmarked our models directly against published literature on the BUSI test set.  
Our models significantly outperform HuggingFace's pretrained Shifaa U-Net at 0.4973 Dice, Attention U-Net at 0.7412, and U-Net++ at 0.7588. Notably, our 2.15-million parameter Distilled MobileNetV3 achieves **0.8034 Dice** at 34.9 FPS—surpassing U-Net++ by 4.5 Dice points while using 12 times fewer parameters and running nearly 9 times faster.  
To verify out-of-distribution robustness, we tested our models zero-shot on **UDIAT Dataset B from Sabadell, Spain**, comprising 163 unseen scans.  
ResNet-34 INT8 achieved **0.7928 Dice**, demonstrating zero domain degradation, while our Rescued Mixed model sustained **0.7002 Dice** with a median of **0.8562**, completely avoiding the 0.1519 collapse of naive INT8."

---

### [5:25 - 6:05] | Slide 10: Extension B: Cine-Loop Streaming & Temporal Smoothing
**Speaker C:**  
*[Speaker C highlights the drop-rate graph and temporal EMA formulation on Slide 10]*  
"Moving to **Extension B**, we evaluated real-time cine-loop streaming under simulated probe scanning.  
We built a multi-threaded producer-consumer engine with a lock-free drop-oldest ring buffer of capacity 4.  
At a full 30-hertz input rate, ResNet-34 dropped 56.3 to 61.9 percent of frames, causing severe visual lag. In contrast, our MobileNetV3 FP32 and Rescued Mixed variants sustained 30 hertz with **only 0.02 to 0.06 percent frame drops** at a 28-to-30-millisecond end-to-end pipeline latency.  
Furthermore, to resolve inter-frame mask flicker during probe motion, we applied temporal Exponential Moving Average logit smoothing with $\alpha = 0.60$. Evaluated across clinical video sequences from the UFUV MICCAI 2024 cohort, this boosted Temporal Consistency by **18.5 to 65.2 percent**—a mean gain of **37.7 percent**—producing smooth, anatomically stable contours.  
Now, Speaker A will showcase our live browser diagnostic viewer."

---

### [6:05 - 6:35] | Slide 11: In-Browser WebGPU/WASM WebViewer Demo
**Speaker A:**  
*[Speaker A gestures to the browser architecture diagram on Slide 11]*  
"To prove immediate deployability, we developed a zero-install diagnostic web viewer powered by ONNX Runtime Web.  
The application executes our 2.50-megabyte Rescued Mixed model entirely inside client browsers like Chrome and Edge.  
It features a dual execution pipeline: leveraging client **WebGPU hardware acceleration** for 22.4-millisecond inference, with seamless fallback to **4-threaded WebAssembly SIMD** at 31.8 milliseconds for legacy hospital tablets.  
Clinicians can interactively adjust probability thresholds, view real-time tumor contours, bounding boxes, and centroids with zero server roundtrips.  
Crucially, because all inference runs locally, **no patient data ever leaves the clinic**, satisfying medical privacy regulations."

---

### [6:35 - 7:00] | Slide 12: Clinical Limitations, Trade-offs & Conclusion
**Speaker C:**  
*[Speaker C steps in for the closing remarks on Slide 12]*  
"In conclusion, we have demonstrated that extreme quantization on edge medical devices is achievable through principled graph surgery and knowledge distillation.  
We must also highlight clinical limitations: acoustic shadows beneath calcifications remain a source of false negatives, and abrupt out-of-plane probe rotations can momentarily disrupt 2D temporal filters.  
Nevertheless, our 2.50-megabyte, 42.9-FPS pipeline bridges the gap between deep learning research and practical bedside POCUS triage.  
Thank you for your time. Speaker A, Speaker B, and I look forward to your questions."

---

## Presentation Checklist & Delivery Notes

- [ ] **Pacing:** Maintain a steady, measured delivery of ~135 words per minute. Do not rush.
- [ ] **Sequential Slides:** Press the **Right Arrow / Next Page** key sequentially from **Slide 1 to Slide 12**—the slides in `presentation_slides.pdf` match this script in exact 1-to-1 chronological order.
- [ ] **Handoffs:** Make physical eye contact or nod during verbal transitions (`"I now hand over to..."`).
- [ ] **Pointer Guidance:** Actively reference slide quadrants, formulas, and metric callout boxes as indicated in the script stage directions.
