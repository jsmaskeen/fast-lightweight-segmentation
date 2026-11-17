/**
 * BUSI POCUS Edge Tumor Segmentation WebViewer
 * Real-time In-Browser Inference via WebGPU and WebAssembly (WASM SIMD)
 * Model: MobileNetV3-Small Distilled INT8 QDQ (Smallest Model: 2.50 MB)
 */

// Configure ONNX Runtime Web environment with absolute URLs to avoid bare specifier errors
function setupOrtWasmEnvironment() {
  if (typeof ort === "undefined") return;

  try {
    // Resolve absolute URL for the wasm and jsep modules
    const baseUrl = new URL("js/", window.location.href).href;
    console.log("[WebViewer] Configuring ONNX Runtime wasmPaths with base:", baseUrl);
    
    // Explicitly provide absolute paths for WASM and JSEP WebGPU binaries
    ort.env.wasm.wasmPaths = {
      wasm: baseUrl + "ort-wasm-simd-threaded.wasm",
      mjs: baseUrl + "ort-wasm-simd-threaded.mjs",
      "jsep.wasm": baseUrl + "ort-wasm-simd-threaded.jsep.wasm",
      "jsep.mjs": baseUrl + "ort-wasm-simd-threaded.jsep.mjs",
    };
    
    ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
    ort.env.wasm.simd = true;
  } catch (e) {
    console.warn("[WebViewer] URL resolution note:", e);
    ort.env.wasm.wasmPaths = "/js/";
  }
}

setupOrtWasmEnvironment();

// Global Application State
const state = {
  session: null,
  activeBackend: "unknown",
  currentModelPath: "models/mbv3s_distilled_int8_minmax.onnx",
  loadedImage: null,
  rawProbMap256: null, // Float32Array(256 * 256) probabilities
  viewMode: "composite", // 'composite' | 'sidebyside' | 'heatmap' | 'binary'
  threshold: 0.50,
  opacity: 0.65,
  colorRgb: [0, 240, 255], // Electric cyan
  drawContour: true,
  drawBBox: true,
  drawCentroid: true,
};

// UI Element References
const elements = {
  backendBadge: document.getElementById("backendBadge"),
  backendText: document.getElementById("backendText"),
  modelBadge: document.getElementById("modelBadge"),
  modelText: document.getElementById("modelText"),
  modelSelect: document.getElementById("modelSelect"),
  epSelect: document.getElementById("epSelect"),
  reloadModelBtn: document.getElementById("reloadModelBtn"),
  dropZone: document.getElementById("dropZone"),
  fileInput: document.getElementById("fileInput"),
  threshSlider: document.getElementById("threshSlider"),
  threshVal: document.getElementById("threshVal"),
  opacitySlider: document.getElementById("opacitySlider"),
  opacityVal: document.getElementById("opacityVal"),
  colorDots: document.querySelectorAll(".color-dot"),
  contourCheck: document.getElementById("contourCheck"),
  bboxCheck: document.getElementById("bboxCheck"),
  centroidCheck: document.getElementById("centroidCheck"),
  tabBtns: document.querySelectorAll(".tab-btn"),
  outputCanvas: document.getElementById("outputCanvas"),
  secondaryCanvas: document.getElementById("secondaryCanvas"),
  viewportMeta: document.getElementById("imageDim"),
  spinnerOverlay: document.getElementById("spinnerOverlay"),
  spinnerText: document.getElementById("spinnerText"),
  verdictValue: document.getElementById("verdictValue"),
  verdictSub: document.getElementById("verdictSub"),
  areaValue: document.getElementById("areaValue"),
  areaPct: document.getElementById("areaPct"),
  coordsValue: document.getElementById("coordsValue"),
  boxValue: document.getElementById("boxValue"),
  latencyValue: document.getElementById("latencyValue"),
  fpsValue: document.getElementById("fpsValue"),
};

/**
 * Creates an ONNX Runtime InferenceSession with progressive fallback:
 * 1. Try specified execution provider (WebGPU or WASM) with local binaries
 * 2. If WebGPU fails, try WebAssembly (WASM SIMD) with local binaries
 * 3. If local binaries fail, switch wasmPaths to CDN and retry WebAssembly
 */
async function createSessionWithFallback(modelUrl, epMode) {
  const hasWebGPU = typeof navigator !== "undefined" && "gpu" in navigator;

  // Try 1: WebGPU if requested and available
  if (epMode === "webgpu" || (epMode === "auto" && hasWebGPU)) {
    try {
      console.log("[WebViewer] Attempting WebGPU session creation...");
      const session = await ort.InferenceSession.create(modelUrl, {
        executionProviders: ["webgpu"],
        graphOptimizationLevel: "all",
      });
      return { session, backend: "WebGPU (GPU Accelerated)" };
    } catch (gpuErr) {
      console.warn("[WebViewer] WebGPU initialization failed, falling back to WebAssembly:", gpuErr);
      if (epMode === "webgpu") {
        throw gpuErr; // If user explicitly forced WebGPU, bubble up error
      }
    }
  }

  // Try 2: WebAssembly with local wasm binaries
  try {
    console.log("[WebViewer] Attempting local WebAssembly session creation...");
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return { session, backend: "WebAssembly (WASM SIMD)" };
  } catch (wasmErr) {
    console.warn("[WebViewer] Local WebAssembly initialization failed:", wasmErr);

    // Try 3: Switch wasmPaths to CDN and retry
    console.log("[WebViewer] Attempting CDN fallback for WebAssembly...");
    ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/";
    const session = await ort.InferenceSession.create(modelUrl, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    });
    return { session, backend: "WebAssembly (WASM CDN)" };
  }
}

/**
 * Initializes or reloads the ONNX Runtime session.
 */
async function initEngine() {
  showSpinner("Initializing WebGPU / WebAssembly Inference Engine...");
  const modelUrl = elements.modelSelect.value;
  const epMode = elements.epSelect.value;
  state.currentModelPath = modelUrl;

  try {
    console.log(`[WebViewer] Loading model: ${modelUrl} | Mode: ${epMode}`);
    const { session, backend } = await createSessionWithFallback(modelUrl, epMode);

    state.session = session;
    state.activeBackend = backend;

    elements.backendText.textContent = backend;
    elements.backendBadge.className = backend.includes("WebGPU")
      ? "status-badge"
      : "status-badge warning";

    const isInt8 = modelUrl.includes("int8");
    elements.modelText.textContent = isInt8
      ? "mbv3s Distilled INT8 (2.50 MB)"
      : "mbv3s Distilled FP32 (8.63 MB)";

    console.log(`[WebViewer] Engine initialized successfully: ${backend}`);
    hideSpinner();

    // If an image is already loaded, re-run inference
    if (state.loadedImage) {
      await runInference();
    }
  } catch (err) {
    console.error("[WebViewer] Failed to initialize engine:", err);
    elements.backendText.textContent = "Engine Load Error";
    elements.backendBadge.className = "status-badge warning";
    hideSpinner();
    alert("Failed to load neural model.\nDetails: " + err.message);
  }
}

/**
 * Standard ImageNet Mean and Standard Deviation for 256x256 normalization
 */
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];

/**
 * Preprocesses the loaded image into a 1x3x256x256 planar float32 tensor
 */
function preprocessImage(img) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d");

  // Draw image scaled to 256x256
  ctx.drawImage(img, 0, 0, 256, 256);
  const imgData = ctx.getImageData(0, 0, 256, 256).data;

  // Allocate planar tensor buffer [1, 3, 256, 256]
  const floatData = new Float32Array(3 * 256 * 256);
  const planeSize = 256 * 256;

  for (let i = 0; i < planeSize; i++) {
    const r = imgData[i * 4 + 0] / 255.0;
    const g = imgData[i * 4 + 1] / 255.0;
    const b = imgData[i * 4 + 2] / 255.0;

    // Normalize per ImageNet
    floatData[i] = (r - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
    floatData[planeSize + i] = (g - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
    floatData[2 * planeSize + i] = (b - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
  }

  return new ort.Tensor("float32", floatData, [1, 3, 256, 256]);
}

/**
 * Executes model inference on the uploaded B-Mode image.
 */
async function runInference() {
  if (!state.session || !state.loadedImage) return;

  showSpinner("Analyzing B-Mode ultrasound frame...");
  const tTotalStart = performance.now();

  try {
    // 1. Preprocessing
    const tPre0 = performance.now();
    const inputTensor = preprocessImage(state.loadedImage);
    const tPre = performance.now() - tPre0;

    // 2. Model Inference
    const tInf0 = performance.now();
    const inputName = state.session.inputNames[0];
    const feeds = {};
    feeds[inputName] = inputTensor;

    const results = await state.session.run(feeds);
    const tInf = performance.now() - tInf0;

    // 3. Extract output logits & compute sigmoid probabilities
    const tPost0 = performance.now();
    const outputName = state.session.outputNames[0];
    const logits = results[outputName].data; // Float32Array of 1x1x256x256

    const probMap = new Float32Array(256 * 256);
    for (let i = 0; i < logits.length; i++) {
      // Sigmoid activation: 1 / (1 + exp(-x))
      probMap[i] = 1.0 / (1.0 + Math.exp(-logits[i]));
    }
    state.rawProbMap256 = probMap;
    const tPost = performance.now() - tPost0;

    const tTotal = performance.now() - tTotalStart;
    const fps = (1000.0 / tTotal).toFixed(1);

    // Update Benchmark Telemetry
    elements.latencyValue.textContent = `${tTotal.toFixed(1)} ms`;
    elements.fpsValue.textContent = `${fps} FPS (${tInf.toFixed(1)} ms net infer)`;

    // 4. Render Overlay on Canvas
    renderOverlay();
  } catch (err) {
    console.error("[WebViewer] Inference failed:", err);
    alert("Inference execution failed: " + err.message);
  } finally {
    hideSpinner();
  }
}

/**
 * Bilinear sampling from the 256x256 probability map
 */
function sampleProb(xNorm, yNorm) {
  const x = Math.max(0, Math.min(255, xNorm * 255));
  const y = Math.max(0, Math.min(255, yNorm * 255));

  const x0 = Math.floor(x);
  const x1 = Math.min(255, x0 + 1);
  const y0 = Math.floor(y);
  const y1 = Math.min(255, y0 + 1);

  const fx = x - x0;
  const fy = y - y0;

  const p00 = state.rawProbMap256[y0 * 256 + x0];
  const p10 = state.rawProbMap256[y0 * 256 + x1];
  const p01 = state.rawProbMap256[y1 * 256 + x0];
  const p11 = state.rawProbMap256[y1 * 256 + x1];

  const top = p00 * (1 - fx) + p10 * fx;
  const bottom = p01 * (1 - fx) + p11 * fx;
  return top * (1 - fy) + bottom * fy;
}

/**
 * Renders the tumor overlay, bounding box, and telemetry metrics.
 * Runs reactively whenever sliders (threshold, opacity, color) or view mode change.
 */
function renderOverlay() {
  if (!state.loadedImage || !state.rawProbMap256) return;

  const origW = state.loadedImage.naturalWidth || state.loadedImage.width;
  const origH = state.loadedImage.naturalHeight || state.loadedImage.height;

  elements.viewportMeta.textContent = `${origW} × ${origH} px B-Mode`;

  const canvas = elements.outputCanvas;
  const secCanvas = elements.secondaryCanvas;
  const ctx = canvas.getContext("2d");

  // Adjust canvas buffer resolution
  canvas.width = origW;
  canvas.height = origH;

  // Handle Side-by-Side view mode
  if (state.viewMode === "sidebyside") {
    secCanvas.style.display = "block";
    secCanvas.width = origW;
    secCanvas.height = origH;
    const secCtx = secCanvas.getContext("2d");
    secCtx.drawImage(state.loadedImage, 0, 0, origW, origH);
  } else {
    secCanvas.style.display = "none";
  }

  // Draw base B-Mode image
  ctx.drawImage(state.loadedImage, 0, 0, origW, origH);

  // Extract pixel buffer to construct overlay
  const frameImageData = ctx.getImageData(0, 0, origW, origH);
  const pixels = frameImageData.data;

  // Binary tumor mask grid for connected component & contour extraction
  const binaryMask = new Uint8Array(origW * origH);
  let tumorPixelCount = 0;
  let sumX = 0;
  let sumY = 0;
  let minX = origW, maxX = 0, minY = origH, maxY = 0;

  const [colR, colG, colB] = state.colorRgb;
  const thresh = state.threshold;
  const alphaWeight = state.opacity;

  for (let y = 0; y < origH; y++) {
    const yNorm = y / origH;
    const rowOffset = y * origW;

    for (let x = 0; x < origW; x++) {
      const xNorm = x / origW;
      const prob = sampleProb(xNorm, yNorm);
      const idx = (rowOffset + x) * 4;

      if (state.viewMode === "binary") {
        const val = prob >= thresh ? 255 : 0;
        pixels[idx + 0] = val;
        pixels[idx + 1] = val;
        pixels[idx + 2] = val;
        pixels[idx + 3] = 255;
      } else if (state.viewMode === "heatmap") {
        // Jet / Turbo heatmap representation
        const heat = turboColormap(prob);
        pixels[idx + 0] = heat[0];
        pixels[idx + 1] = heat[1];
        pixels[idx + 2] = heat[2];
        pixels[idx + 3] = 255;
      } else {
        // Composite or Side-by-side mode: Alpha blend tumor overlay
        if (prob >= thresh) {
          const bgR = pixels[idx + 0];
          const bgG = pixels[idx + 1];
          const bgB = pixels[idx + 2];

          pixels[idx + 0] = Math.round(bgR * (1 - alphaWeight) + colR * alphaWeight);
          pixels[idx + 1] = Math.round(bgG * (1 - alphaWeight) + colG * alphaWeight);
          pixels[idx + 2] = Math.round(bgB * (1 - alphaWeight) + colB * alphaWeight);
        }
      }

      if (prob >= thresh) {
        binaryMask[rowOffset + x] = 1;
        tumorPixelCount++;
        sumX += x;
        sumY += y;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // Put image data back onto canvas
  ctx.putImageData(frameImageData, 0, 0);

  // Compute Clinical Findings
  const totalPixels = origW * origH;
  const areaCoveragePct = ((tumorPixelCount / totalPixels) * 100).toFixed(2);
  const tumorDetected = tumorPixelCount > 50;

  if (tumorDetected) {
    const cx = Math.round(sumX / tumorPixelCount);
    const cy = Math.round(sumY / tumorPixelCount);
    const boxW = Math.max(0, maxX - minX);
    const boxH = Math.max(0, maxY - minY);

    // Update Telemetry Cards
    elements.verdictValue.textContent = "TUMOR IDENTIFIED";
    elements.verdictValue.className = "t-value verdict-positive";
    elements.verdictSub.textContent = `High-confidence lesion detection`;

    elements.areaValue.textContent = `${tumorPixelCount.toLocaleString()} px²`;
    elements.areaPct.textContent = `${areaCoveragePct}% of scan field`;

    elements.coordsValue.textContent = `X: ${cx} | Y: ${cy}`;
    elements.boxValue.textContent = `W: ${boxW} px | H: ${boxH} px`;

    // Draw Annotations on Composite Views
    if (state.viewMode === "composite" || state.viewMode === "sidebyside") {
      // 1. Draw Boundary Contour
      if (state.drawContour) {
        drawBoundaryContours(ctx, binaryMask, origW, origH, colR, colG, colB);
      }

      // 2. Draw Bounding Box
      if (state.drawBBox && boxW > 0 && boxH > 0) {
        ctx.strokeStyle = `rgba(${colR}, ${colG}, ${colB}, 0.9)`;
        ctx.lineWidth = Math.max(2, Math.round(origW / 250));
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(minX, minY, boxW, boxH);
        ctx.setLineDash([]);

        // Label above bbox
        ctx.fillStyle = `rgba(${colR}, ${colG}, ${colB}, 0.95)`;
        ctx.font = `bold ${Math.max(12, Math.round(origW / 35))}px var(--font-sans)`;
        ctx.fillText(`Tumor: ${boxW}×${boxH}px`, minX, Math.max(16, minY - 6));
      }

      // 3. Draw Centroid (+)
      if (state.drawCentroid) {
        const markerSize = Math.max(8, Math.round(origW / 40));
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(cx - markerSize, cy);
        ctx.lineTo(cx + markerSize, cy);
        ctx.moveTo(cx, cy - markerSize);
        ctx.lineTo(cx, cy + markerSize);
        ctx.stroke();

        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(cx, cy, 3, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
  } else {
    // Normal Scan (No Tumor Detected)
    elements.verdictValue.textContent = "NORMAL SCAN";
    elements.verdictValue.className = "t-value verdict-negative";
    elements.verdictSub.textContent = "No significant lesion detected";

    elements.areaValue.textContent = "0 px²";
    elements.areaPct.textContent = "0.0% coverage";
    elements.coordsValue.textContent = "None";
    elements.boxValue.textContent = "W: 0 px | H: 0 px";
  }
}

/**
 * Draws smooth boundary contour around detected tumor mask.
 */
function drawBoundaryContours(ctx, mask, w, h, r, g, b) {
  ctx.strokeStyle = "#ffffff";
  ctx.lineWidth = Math.max(2, Math.round(w / 300));
  ctx.shadowColor = `rgb(${r}, ${g}, ${b})`;
  ctx.shadowBlur = 6;

  ctx.beginPath();
  for (let y = 1; y < h - 1; y++) {
    const row = y * w;
    for (let x = 1; x < w - 1; x++) {
      if (mask[row + x]) {
        if (!mask[row + x - 1] || !mask[row + x + 1] || !mask[row - w + x] || !mask[row + w + x]) {
          ctx.rect(x, y, 1.2, 1.2);
        }
      }
    }
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

/**
 * Turbo Colormap for scientific probability heatmaps
 */
function turboColormap(t) {
  const kRedVec4 = [0.13572138, 4.61539260, -42.66032258, 132.13108234];
  const kGreenVec4 = [0.09140261, 2.19418839, 4.84296658, -14.18503333];
  const kBlueVec4 = [0.10667330, 12.64194608, -60.58204836, 110.36276771];
  const kRedVec2 = [-152.94239396, 59.28637943];
  const kGreenVec2 = [4.27729857, 2.82956604];
  const kBlueVec2 = [-89.90310912, 27.34824973];

  const v4 = [1.0, t, t * t, t * t * t];
  const v2 = [v4[2] * v4[2], v4[3] * v4[1]];

  let r = (kRedVec4[0]*v4[0] + kRedVec4[1]*v4[1] + kRedVec4[2]*v4[2] + kRedVec4[3]*v4[3]) + (kRedVec2[0]*v2[0] + kRedVec2[1]*v2[1]);
  let g = (kGreenVec4[0]*v4[0] + kGreenVec4[1]*v4[1] + kGreenVec4[2]*v4[2] + kGreenVec4[3]*v4[3]) + (kGreenVec2[0]*v2[0] + kGreenVec2[1]*v2[1]);
  let b = (kBlueVec4[0]*v4[0] + kBlueVec4[1]*v4[1] + kBlueVec4[2]*v4[2] + kBlueVec4[3]*v4[3]) + (kBlueVec2[0]*v2[0] + kBlueVec2[1]*v2[1]);

  return [
    Math.min(255, Math.max(0, Math.round(r * 255))),
    Math.min(255, Math.max(0, Math.round(g * 255))),
    Math.min(255, Math.max(0, Math.round(b * 255))),
  ];
}

/**
 * Loads image from URL or File object into the workspace
 */
function loadImageSource(source) {
  const img = new Image();
  img.onload = () => {
    state.loadedImage = img;
    runInference();
  };
  if (typeof source === "string") {
    img.src = source;
  } else if (source instanceof File) {
    const reader = new FileReader();
    reader.onload = (e) => { img.src = e.target.result; };
    reader.readAsDataURL(source);
  }
}

/**
 * Event Listeners & Interactive UI Wiring
 */
function setupEventListeners() {
  // Model & Execution Provider Selection
  elements.reloadModelBtn.addEventListener("click", () => initEngine());
  elements.modelSelect.addEventListener("change", () => initEngine());
  elements.epSelect.addEventListener("change", () => initEngine());

  // File Upload
  elements.fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      loadImageSource(e.target.files[0]);
    }
  });

  // Drag and Drop
  elements.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    elements.dropZone.classList.add("dragover");
  });
  elements.dropZone.addEventListener("dragleave", () => {
    elements.dropZone.classList.remove("dragover");
  });
  elements.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    elements.dropZone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      loadImageSource(e.dataTransfer.files[0]);
    }
  });

  // Sample Presets from BUSI Dataset
  document.querySelectorAll(".sample-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const url = btn.getAttribute("data-url");
      loadImageSource(url);
    });
  });

  // Threshold Slider (instant reactive update)
  elements.threshSlider.addEventListener("input", (e) => {
    state.threshold = parseFloat(e.target.value);
    elements.threshVal.textContent = state.threshold.toFixed(2);
    renderOverlay();
  });

  // Opacity Slider (instant reactive update)
  elements.opacitySlider.addEventListener("input", (e) => {
    state.opacity = parseFloat(e.target.value);
    elements.opacityVal.textContent = `${Math.round(state.opacity * 100)}%`;
    renderOverlay();
  });

  // Color Palette Dots
  elements.colorDots.forEach((dot) => {
    dot.addEventListener("click", () => {
      elements.colorDots.forEach((d) => d.classList.remove("active"));
      dot.classList.add("active");
      const rgbStr = dot.style.background;
      const rgbMatch = rgbStr.match(/\d+/g);
      if (rgbMatch && rgbMatch.length >= 3) {
        state.colorRgb = [parseInt(rgbMatch[0]), parseInt(rgbMatch[1]), parseInt(rgbMatch[2])];
        renderOverlay();
      }
    });
  });

  // Checkbox Annotations
  elements.contourCheck.addEventListener("change", (e) => {
    state.drawContour = e.target.checked;
    renderOverlay();
  });
  elements.bboxCheck.addEventListener("change", (e) => {
    state.drawBBox = e.target.checked;
    renderOverlay();
  });
  elements.centroidCheck.addEventListener("change", (e) => {
    state.drawCentroid = e.target.checked;
    renderOverlay();
  });

  // Viewport Tabs
  elements.tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      elements.tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.viewMode = btn.getAttribute("data-view");
      renderOverlay();
    });
  });
}

function showSpinner(msg = "Processing...") {
  elements.spinnerText.textContent = msg;
  elements.spinnerOverlay.style.display = "flex";
}

function hideSpinner() {
  elements.spinnerOverlay.style.display = "none";
}

// Bootstrapping
window.addEventListener("DOMContentLoaded", () => {
  setupEventListeners();
  initEngine().then(() => {
    // Automatically load Benign Lesion 1 on startup
    loadImageSource("samples/benign_sample1.png");
  });
});
