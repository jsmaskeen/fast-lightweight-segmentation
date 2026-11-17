# BUSI Quantized Real-Time Segmentation: Project Plan (Source of Truth)

*Version 1.1 · 7 Nov 2026 · Status: Active. Any change to scope, metrics or protocol is logged in the Decision Log (Section 13) before it is made.*

## 1. Project in one paragraph

Train a heavy (ResNet-34 U-Net) and a lightweight (MobileNetV3-Small U-Net) breast-ultrasound segmentation model on BUSI, convert both to static INT8 (QDQ) ONNX, and measure the accuracy, size, latency and memory trade-offs on a resource-constrained CPU target. Two extensions push the project beyond a standard quantization script:

- **Extension A: Layer-wise sensitivity analysis and hybrid mixed precision** with a user-adjustable sensitivity knob (the accuracy side of the Pareto front).
- **Extension B: Simulated real-time POCUS cine-loop with a ring buffer** (the systems side: does the model actually keep up with a live 30 Hz stream?).

## 2. Why these two extensions

Selection criteria: low implementation risk, no new training, reuse of artifacts we already produce, and a headline result that is easy to explain and defend.

| Extension | Effort | Marks upside | Main risk | Decision |
| --- | --- | --- | --- | --- |
| 1. C++ engine + cache analysis | High (C++ build or cross-compile, OpenCV linking) | High but fragile | `perf` cache events are often unavailable on a Pi kernel or inside VMs/containers | Dropped |
| 2. Mixed precision via sensitivity debugging | Low to medium (ONNX Runtime ships the tooling; the work is re-quantizing and re-evaluating) | High: a concrete engineering finding that extends the core work | INT8 may already be near-lossless, leaving little to recover | **Selected: Extension A** |
| 3. Cine-loop ring buffer | Low to medium (Python threads; ONNX Runtime releases the GIL during `run`) | High: the only item that demonstrates the real-time claim end to end | Timing jitter on a shared host | **Selected: Extension B** |
| 4. Speckle / gain robustness | Lowest | Medium | Less systems content | Reserve: only if A and B are finished early |
| 5. Knowledge distillation | Medium (new training, tuning, and a retrain cascade) | Medium to high | Improves accuracy, not systems; delays everything downstream | Dropped |

Together, A covers the accuracy axis and B covers the latency/throughput axis of the final Pareto report. Both run on the ONNX models and harnesses built in Phases 0 to 9.

## 3. Goals, hypotheses and acceptance criteria

Hypotheses are things to **measure**, not assumed outcomes. The report states what was found even when it contradicts them.

| ID | Hypothesis | How it is tested |
| --- | --- | --- |
| H1 | INT8 models are about 3.5x to 4x smaller on disk | File size in MB, FP32 vs INT8 |
| H2 | INT8 reduces CPU latency by 2x to 3x | p50 latency on the edge target. Caveat: a Raspberry Pi 4 (Cortex-A72) has no int8 dot-product instructions, so the gain there may be smaller than on newer cores. Report the measured number. |
| H3 | Full INT8 costs a measurable Dice/HD95 drop on weak lesion boundaries | FP32 vs INT8 on the held-out test set with paired bootstrap CIs |
| H4 (Ext A) | Keeping a small fraction of sensitive nodes in FP32 recovers most of the lost Dice | Sweep of k excluded nodes, recovery ratio defined in Section 8 |
| H5 (Ext B) | At 30 Hz the heavy FP32 model drops frames and the INT8 edge model does not | Frame drop rate and end-to-end latency per variant and input rate |

**Definition of done**

- All 2 architectures x (FP32, INT8-MinMax, INT8-Entropy) evaluated on the test set (DSC, IoU, HD95, normal-image false-positive rate).
- Latency, throughput, disk size and peak RSS recorded for every variant on the edge target, with 3 repeated runs.
- Extension A: sensitivity ranking, k-sweep, and at least one mixed-precision model selected on validation data and reported on test data, plus the adjustable sensitivity knob (CLI and demo page).
- Extension B: full variant x rate matrix run, drop-rate and latency plots produced, maximum sustainable frame rate reported per variant.
- A Pareto figure (Dice vs latency, bubble size = model size), a written report, and three mini-studies (Section 10).
- Everything reproducible from a single `make all` (or documented script sequence) with fixed seeds and pinned library versions.

**Non-goals:** C++ inference engine, knowledge distillation, other datasets, and any clinical-performance claim. This is a research prototype, not a medical device.

## 4. Locked technical decisions

| Item | Decision |
| --- | --- |
| Dataset | BUSI: 437 benign, 210 malignant, 133 normal |
| Mask handling | Merge multi-mask scans (`_mask`, `_mask_1`, ...) with bitwise OR **before** resizing; resize masks with nearest-neighbour, then re-binarize at >127 |
| Input size | 256 x 256, grayscale replicated to 3 channels, ImageNet mean/std normalization (pretrained encoders) |
| Duplicate check | Hash all images and drop or group exact and near duplicates before splitting (BUSI has reported duplicates, and BUSI has no patient IDs, so leakage control relies on hashing) |
| Splits | Lesion images (647): stratified benign/malignant 70/15/15 (train/val/test), seed 42. Normal images (133): **none in training**; split roughly 50/50 into val and test to measure false-positive rate. Splits saved to `data/splits.json` and never regenerated. |
| Test-set rule | The test set is untouched until final evaluation. All model selection, calibration choice and node exclusion use train/val only. |
| Heavy model | `segmentation_models_pytorch` U-Net, ResNet-34 encoder (about 24M parameters) |
| Edge model | U-Net with a MobileNetV3-Small encoder (`tu-mobilenetv3_small_100`, or `timm-mobilenetv3_small_100` on older smp versions). The default decoder can have more parameters than this encoder, so **shrink `decoder_channels`** and verify the total is about 2M to 3M in Phase 2. |
| Loss | 0.5 x BCE + 0.5 x Dice |
| Training | AdamW, cosine schedule, early stopping on val Dice (patience 15), flips, small affine, brightness/contrast (gain drift), light noise |
| Output head | Export **logits** (no Sigmoid in the ONNX graph). Post-processing thresholds at 0.5 probability, which equals logit > 0. This keeps the head out of the quantization problem by design. |
| ONNX | opset 17, fixed input shape 1x3x256x256, parity check vs PyTorch (max abs logit diff < 1e-4) |
| Quantization | Static PTQ, QDQ format, per-channel weights, 100 calibration images from the training set using the exact inference preprocessing. S8S8 on ARM; U8S8 with reduce\_range or S8S8 on x86 (AVX2 without VNNI can saturate with U8S8). Calibration methods compared: MinMax vs Entropy. |
| Edge target | Primary: Raspberry Pi 4 (or similar ARM64 SBC) over SSH. Fallback: local Linux, `taskset -c 0-3`, cgroup CPU/memory limits. Whichever is used is recorded in the report; do not mix targets inside one comparison. |

## 5. Repository layout

```
busi-int8/
  README.md            # how to reproduce
  Makefile             # make data | train | export | quant | eval | bench | extA | extB | studies | report
  configs/             # yaml: model, train, quant, bench
  data/                # raw (read-only), processed, splits.json
  src/
    data/              # merge_masks.py, dataset.py
    train/             # train.py, losses.py
    metrics/           # dsc_iou_hd95.py, bootstrap.py, stratify.py
    export/            # to_onnx.py, preprocess.py
    quant/             # calib_reader.py, calib_study.py, quantize.py, sensitivity.py, mixed.py, knob.py, demo_page.py
    bench/             # harness.py, telemetry.py, perf_stat.sh, op_profile.py
    cine/              # ring_buffer.py, producer.py, consumer.py, run_cine.py
  models/              # fp32/, int8/, mixed/ (+ checksums.txt)
  results/             # raw CSV logs, one file per run, never edited by hand
  figures/             # generated only by scripts in src/report
  tests/               # metric tests, ring-buffer tests, parity tests
  PLAN.md              # this document (kept in sync)
```

**Conventions:** model IDs are `r34` and `mbv3s`. Variant IDs are `fp32`, `int8-minmax`, `int8-entropy`, `mix-k{N}` (N = number of nodes kept in FP32). Every result row carries model, variant, git commit, ORT version, hardware ID and timestamp.

## 6. Core pipeline (Phases 0 to 9)

Phases are gated, not timed: a phase starts only when the previous gate passes.

| Phase | Work | Output | Gate to proceed |
| --- | --- | --- | --- |
| 0 | Foundations: create the repo from the Section 5 layout, pin library versions in a lockfile, set up the Makefile, configs and a global seed helper, confirm access to the edge target (or fall back to the container) | requirements.lock, Makefile skeleton, docs/env.md stub | make targets exist and a CPU smoke test runs |
| 1 | Merge masks (OR), resize, hash-dedup, create splits, overlay QA | `data/processed`, `splits.json` | Counts match (647 lesion + 133 normal before dedup); 20 random overlays visually checked |
| 2 | Train `r34` and `mbv3s` with fixed seeds | FP32 checkpoints, training logs | Val Dice sane (about 0.70 or above; otherwise debug before continuing) |
| 3 | Implement and unit-test DSC, IoU, HD95; evaluate both on test; bootstrap CIs | Baseline metrics table | Metric code passes tests on synthetic masks |
| 4 | Export to ONNX opset 17, parity check, run `quantization.preprocess` | FP32 ONNX models | Parity test passes |
| 5 | Calibration data reader (100 train images), static QDQ quantization, MinMax vs Entropy, both models. Mini-study S3 (calibration-set size) runs in this phase | 4 INT8 models | Models load and run in ONNX Runtime |
| 6 | Accuracy audit FP32 vs INT8 on val, then test, plus mini-study S1 (stratified damage). **Also run `qdq_loss_debug` once on `mbv3s` as a smoke test for Extension A.** | Audit table | Drop in Dice and HD95 quantified with CIs |
| 7 | Set up edge target; fix thread count, governor to `performance`, record hardware/OS/kernel/ORT versions | `docs/env.md` | Reproducible run of one model on the target |
| 8 | Benchmark harness: 50 warm-up + 500 timed runs, only `session.run` timed with `perf_counter_ns`, one fresh process per variant. **Start the ring-buffer skeleton (Extension B) at the end of this phase.** | `harness.py` | Run-to-run variation understood |
| 9 | Full telemetry: p50/p95/p99, FPS, disk MB, peak RSS, `perf stat` where the kernel allows it (mark N/A otherwise), temperature and throttle flags; 3 repeats per variant; mini-study S2 (per-operator profiling) | Benchmark table | No throttling flagged, or flagged runs repeated |

HD95 convention (fixed): if both masks are empty the image is excluded from HD95; if the prediction is empty but ground truth is not, HD95 is set to the image diagonal (256 x sqrt(2), about 362 px) and the number of such cases is reported.

## 7. Evaluation protocol (applies to everything)

- **Segmentation:** DSC and IoU per image on lesion images; HD95 as above. Report mean, median and 95% bootstrap CI (1000 resamples). FP32 vs INT8 comparisons use **paired** per-image differences.
- **Normal images:** false-positive rate = fraction of normal test images with predicted area above 50 px (and the same at 0 px for reference).
- **Latency:** p50, p95, p99 in ms over 500 timed runs after 50 warm-ups; FPS = 1000 / mean latency.
- **Memory:** disk MB and peak RSS (`/usr/bin/time -v` or `resource.getrusage`).
- **Environment hygiene:** performance governor, no background jobs, temperature and `vcgencmd get_throttled` (on Pi) logged with every run, thread counts fixed and stated.

## 8. Extension A: Layer-wise sensitivity and hybrid mixed precision

**Question:** Which few nodes cause most of the INT8 accuracy loss, and how much Dice and HD95 do we recover by leaving them in FP32?

**Inputs:** FP32 ONNX and best full-INT8 QDQ model of `mbv3s` (repeat for `r34` only if time remains), the 100 calibration images, the val split.

**Procedure**

1. **Weight matching.** Use `onnxruntime.quantization.qdq_loss_debug.create_weight_matching` between the FP32 and INT8 models to get per-weight quantization error.
2. **Activation matching.** Augment both models to expose intermediate tensors (`modify_model_output_intermediate_tensors`), collect activations on a **subset of 16 to 32 calibration images** (all tensors on all 100 images can exhaust RAM at decoder resolution), then `create_activation_matching` and `compute_activation_error`.
3. **Rank nodes** by activation error (lowest SQNR = most sensitive). Keep the per-node table as a result.
4. **k-sweep.** Re-quantize with `nodes_to_exclude` = top-k ranked nodes for k in {0, 1, 2, 3, 4, 5, 6, 8, 10, 12}. Every other setting is unchanged.
5. **Baselines for comparison:** (a) exclude the first and last convolution only (the common heuristic), (b) exclude k random nodes (5 seeds, mean and range). This shows whether the ranking adds value.
6. **Evaluate every variant on val:** DSC, HD95, normal FP rate, model size, and p50 latency on the edge target (boundary Q/DQ nodes can add overhead, so latency must be measured, not assumed).
7. **Choose k\*** on val: the smallest k whose recovery ratio is at least 0.9, where

   recovery = (Dice\_mixed - Dice\_int8) / (Dice\_fp32 - Dice\_int8).

   Then report that single configuration on test.
8. **Stretch (only if ahead of schedule):** leave-one-out validation. Exclude each node alone, measure true delta-Dice, and report the Spearman correlation between the proxy error ranking and true impact.

**Adjustable sensitivity knob.** The sweep in steps 3 to 7 runs once and is saved to `results/extA_sweep.json`: the node ranking and, for every k, the val Dice, HD95, normal-image FP rate, model size, p50 latency and model path. A small CLI, `src/quant/knob.py`, then lets the user choose how much of the model stays in FP32 without re-quantizing or re-evaluating anything.

| Mode | Flag | Behaviour |
| --- | --- | --- |
| Fixed count | `--keep-fp32 3` | Keeps the 3 most sensitive nodes in FP32 |
| Fraction of graph | `--keep-fp32-pct 5` | Keeps the top 5% of nodes (rounded up) in FP32 |
| Accuracy budget | `--max-dice-drop 0.5` | Picks the smallest k whose val Dice is within 0.5 point of FP32; reports that no setting qualifies if none does |

- **Output:** the path of the chosen model plus a one-line summary: k, names of the FP32 nodes, val Dice and HD95, size, p50 latency.
- **Snapping:** a requested value between swept k values is rounded up to the next swept k, and the CLI says so. The grid is dense for k up to 6, so this rarely matters.
- **Leakage rule:** the knob only reads the val-based ranking and sweep. The test set is evaluated once, for the chosen setting, and reported next to the full curve.
- **Demo page:** a static HTML page generated by `src/quant/demo_page.py` from the saved JSON, with no backend. A slider for k (and an accuracy-budget field) updates Dice, HD95, size and p50 latency and highlights which nodes are in FP32. Latency may not fall smoothly with k because Q/DQ boundaries add overhead, and the page should show that rather than hide it.
- **Done when:** the CLI reproduces the k\* choice from the same inputs, and `make extA` regenerates the sweep file, the CLI output and the demo page.

**Outputs:** sensitivity bar chart (top 20 nodes), Dice and HD95 vs k curve (with random and heuristic baselines), latency vs k, final mixed model `models/mixed/mbv3s_mix-kN.onnx`, a 1-paragraph finding.

**Contingency:** if the full-INT8 Dice drop is under about 0.5 point, there is little to recover. The finding then becomes: INT8 is near-lossless for this model, and the sensitivity ranking still identifies where the (small) loss concentrates. Report HD95 as the more sensitive boundary metric.

**Pitfalls**

- Choosing nodes on the test set is leakage. Use calibration/val data only.
- Node names come from the exporter (for example `/decoder/blocks.4/conv2/conv2.0/Conv`). Read them from the graph; do not guess.
- Concat nodes in skip connections force shared quantization parameters across inputs with different ranges. Note any Concat-adjacent nodes in the top-ranked list, since this is the mechanism the project expects.
- The Sigmoid is not in the graph by design (Section 4), so "exclude the head" means excluding the final convolution.

## 9. Extension B: Simulated real-time POCUS cine-loop

**Question:** At a live frame rate of 20 to 30 Hz, which model variants keep up, and what happens to latency and dropped frames when they do not?

**Architecture**

```
[Producer thread]  --push(frame, t_capture)-->  [Ring buffer, capacity C]  --pop latest-->  [Consumer thread]
  strict 1/f deadline                          overwrite-oldest when full                  resize + normalize
  pinned to core 0                             counts every overwritten frame             ORT session.run (INT8/FP32)
                                                                                          threshold + contour (OpenCV)
                                                                                          log timestamps
```

**Design rules**

- **Producer:** pre-loads test frames into RAM at native resolution and emits them using absolute deadlines (`next_t += 1/f`, sleep until near the deadline, then spin for the last millisecond). Plain `sleep(1/f)` drifts and is not allowed. Log the actual emit time to quantify producer jitter.
- **Ring buffer:** bounded (default C = 4), thread-safe, **drop-oldest** policy (the display should show the newest frame). Each overwritten frame increments the drop counter. A drop-newest policy may be compared as a stretch.
- **Consumer:** always takes the newest available frame. Preprocessing, inference and post-processing (threshold + `findContours`) are all inside the timed path, with per-stage timestamps.
- **Threading:** Python threads are acceptable because ONNX Runtime releases the GIL in `run`. Pin the producer with `os.sched_setaffinity(0, {0})` called inside the thread, and give ONNX Runtime the remaining cores via `intra_op_num_threads`. Disable garbage collection during a run (`gc.disable()` after warm-up) to avoid pauses.

**Logged per frame:** frame id, `t_capture`, `t_dequeue`, `t_pre_done`, `t_infer_done`, `t_post_done`, dropped flag.

**Metrics**

- Frame drop rate = frames never processed / frames produced.
- End-to-end latency = `t_post_done` - `t_capture` (p50, p95, p99).
- Deadline-miss rate against a latency budget of 100 ms. This budget is an engineering assumption stated in the report, not a clinical standard.
- Effective output FPS, producer jitter, and queue occupancy over time.
- **Maximum sustainable rate** per variant: the highest input rate with drop rate under 1%.

**Experiment matrix**

| Factor | Levels |
| --- | --- |
| Variants | `r34` fp32, `mbv3s` fp32, `mbv3s` best INT8, `mbv3s` mixed (k\*), `r34` INT8 |
| Input rate | 10, 15, 20, 30 Hz |
| Repeats | 3 runs of 60 s each |

That is 5 x 4 x 3 = 60 runs, roughly one hour of wall-clock time plus overhead.

**Sanity check:** when the consumer is saturated, the expected drop rate is about 1 - (service rate / input rate). Compare measured drops against this prediction using the service times from the Phase 9 benchmark. A mismatch signals a harness bug.

**Tests before any real run**

- Producer-only test with a no-op consumer: jitter stays small relative to the frame period.
- Ring buffer concurrency test: produced = processed + dropped, always.
- Parity test: the cine consumer produces the same masks as the offline evaluation for the same frames.

**Outputs:** drop rate vs input rate (per variant), latency CDF at 30 Hz, a 5-second timeline of capture/inference/output events, a max-sustainable-FPS table. Optional: a side-by-side overlay video of FP32 and INT8 at 30 Hz.

**Pitfalls:** thermal throttling mid-run (log temperature and flag runs), background processes, comparing variants on different threads settings, and counting warm-up frames.

## 10. Mini-studies

Three small analyses explain results the core phases already produce. Each needs no new training or data, runs on existing outputs, and ends in one figure or table plus a short paragraph in the report. Either outcome is reportable, so they are findings, not extensions.

| Study | Question | Runs in | Output |
| --- | --- | --- | --- |
| S1: Stratified INT8 damage | Does INT8 hurt some lesions more than the average suggests? | Phase 6 | Table of paired ΔDice and ΔHD95 per stratum, with CIs |
| S2: Per-operator profiling | Where does inference time go, and why is the speedup what it is? | Phase 9 | Stacked bar of time per operator type, FP32 vs INT8 |
| S3: Calibration-set size | How many calibration images does INT8 need? | Phase 5 | Dice and HD95 vs number of calibration images |

**S1: Stratified INT8 damage**

- Strata: class (benign, malignant); lesion size (terciles of ground-truth area); boundary sharpness (terciles of mean gradient magnitude along the ground-truth contour). Tercile cut points are computed on the training split and applied to test, so the strata do not depend on test data.
- Metric: paired per-image difference INT8 minus FP32 in Dice and HD95, with the mean and 95% bootstrap CI per stratum, plus the count of empty predictions.
- Models: the best full-INT8 model. Once Phase 10 is done, add the mixed-precision model k\* as an extra column to show whether it protects the hard strata.
- Caveat: with about 97 test lesion images, each tercile holds roughly 32. Intervals will be wide, so present the results as exploratory and say so.

**S2: Per-operator profiling**

- Enable the ONNX Runtime profiler for 100 iterations after warm-up, per variant (FP32, best INT8, mixed k\*), on the edge target with the same thread count as the benchmark.
- Aggregate durations by operator type as the runtime reports them: convolutions (fused quantized convolutions in INT8), QuantizeLinear/DequantizeLinear, Concat, Resize and activations. Also list the top 10 individual nodes.
- Report time shares, not absolute latency. The profiler adds overhead, so absolute latency comes from the Phase 8 and 9 harness.
- Link to Extension A: the Q/DQ share helps explain why latency does not fall smoothly as more nodes stay in FP32.

**S3: Calibration-set size**

- Sizes n in {10, 25, 50, 100, 200}, drawn from the training split only. Selection strategies: random (3 seeds), class-stratified (3 seeds) and hard-case (the n training images with the highest FP32 loss, deterministic). That is about 35 quantize-and-evaluate runs on `mbv3s`.
- Use only the better calibration algorithm from Phase 5 (MinMax or Entropy) and evaluate on val. The test set is not used.
- Output: Dice and HD95 vs n, one line per strategy, with the seed range as a band.
- The main pipeline keeps n = 100 whatever this study finds (Decision D-11), so it cannot change the headline numbers.

## 11. Phases 10 to 12 and cut order

| Phase | Plan | Exit criterion |
| --- | --- | --- |
| 10 | Extension A: sensitivity ranking, k-sweep, baselines, select k\* on val, test evaluation, latency of mixed models, knob CLI and demo page (small extra effort) | Mixed model, knob CLI, demo page and figures exist |
| 11 | Extension B: finish harness, run the 60-run matrix, produce plots (and the demo video if time permits) | Max-sustainable-FPS table exists |
| 12 | Final Pareto figure (Dice vs latency, bubble = size), mini-study write-ups, report writing, README, freeze: tag the git commit, store model checksums | Report complete, `make all` reproduces headline numbers and the knob CLI runs from the saved sweep |

**Cut order if time runs short** (cut from the top first): demo video, demo page (keep the CLI), mini-study S3 (calibration-set size), leave-one-out validation (A step 8), mixed precision for `r34`, U8S8 vs S8S8 comparison, 10 Hz and 15 Hz cine rates. Never cut: val-based selection discipline, the 3 repeats, or the environment logging.

## 12. Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| INT8 speedup is small on the edge CPU (no int8 dot-product on Cortex-A72) | Medium | Medium | Report honestly; show size and cache-friendliness benefits; if a newer board is available, add it as a second target |
| INT8 Dice drop is tiny, so Extension A has little to recover | Medium | Medium | Contingency in Section 8; HD95 as the main sensitive metric |
| `perf stat` counters unavailable | High on VMs and some Pi kernels | Low | Mark N/A, rely on latency, RSS and disk size; counters are not required by the acceptance criteria |
| Timing noise (throttling, background load) | Medium | High | Governor to `performance`, temperature logging, repeated runs, report p95/p99 and flagged runs |
| MobileNetV3 U-Net is much worse than R34 | Low to medium | Medium | Tune decoder width; report the gap, because it is part of the trade-off story |
| Mask-merging or resize bug silently corrupts labels | Low | High | Overlay QA in Phase 1, count checks, unit test for OR merge |
| Split leakage from duplicates | Medium | High | Hash-based dedup before splitting |
| Schedule slip | Medium | Medium | Gates in Section 6 and the cut order in Section 11 |
| Accuracy-budget mode finds no k that meets the budget | Medium | Low | CLI reports no feasible setting and the closest k; default budgets are set from the observed full-INT8 drop |

## 13. Decision log

| ID | Decision | Reason |
| --- | --- | --- |
| D-01 | Extensions A (mixed precision) and B (cine-loop) selected; others dropped or reserved | Best marks-per-effort, no new training, covers accuracy and systems axes |
| D-02 | Export logits, apply sigmoid/threshold outside the graph | Removes the head from quantization and simplifies Extension A |
| D-03 | Normal images excluded from training, split between val and test | Avoids all-zero collapse while still measuring false positives |
| D-04 | Node selection and calibration choice use val data only | Prevents test-set leakage |
| D-05 | Per-channel QDQ quantization as default | Depthwise convolutions in MobileNet are fragile under per-tensor weights |
| D-06 | Python threads for the cine harness | ONNX Runtime releases the GIL; avoids the C++ build cost |
| D-07 | 100 ms end-to-end latency budget in Extension B | Stated assumption, adjustable, not a clinical standard |
| D-08 | Extension A exposes a user-adjustable knob (fixed count, fraction, accuracy budget) over a precomputed sweep | Lets the user trade accuracy for size and latency without re-running quantization; adds a small amount of work to Phase 10 |
| D-09 | The knob reads only the val-based ranking and sweep; test is evaluated once for the chosen setting, shown beside the full curve | Extends D-04: keeps test data out of the selection |
| D-10 | Phases replace days as the planning unit; each phase ends at a gate, not a date | Work is sequenced by dependency and acceptance criteria, so a slip in one phase does not invalidate the plan |
| D-11 | Mini-studies S1 to S3 are analysis only; the main pipeline keeps 100 calibration images and its fixed settings | S3 varies the calibration set, so letting it change the main pipeline would be circular |

New entries are appended with a date and the reason. Existing entries are never deleted, only superseded.

## 14. Final deliverables checklist

- [ ] Cleaned dataset script, `splits.json`, overlay QA images
- [ ] FP32 checkpoints and ONNX models with checksums
- [ ] INT8 models: MinMax, Entropy, and the mixed-precision model
- [ ] Baseline and post-quantization metrics table with bootstrap CIs
- [ ] Edge benchmark table: p50/p95/p99, FPS, disk MB, peak RSS (and `perf stat` if available)
- [ ] Extension A: sensitivity chart, k-sweep curves, baselines, final finding, knob CLI and demo page
- [ ] Extension B: drop-rate and latency plots, max-sustainable-FPS table
- [ ] Pareto figure and written report
- [ ] README with exact reproduction commands and library versions

**Report outline:** 1) Problem and clinical motivation, 2) Data and preprocessing, 3) Models and training, 4) Quantization method and accuracy audit, 5) Edge benchmark methodology and results, 6) Extension A, 7) Extension B, 8) Pareto analysis, 9) Limitations (small dataset, single scanner source, no patient-level split, prototype only), 10) Conclusions.
