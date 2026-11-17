import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
import pandas as pd

from src.quant.calib_reader import BUSICalibrationDataReader
from src.metrics.dsc_iou_hd95 import compute_dsc_iou_hd95


def compute_fp32_losses(
    model_path: str,
    train_ids: List[str],
    data_dir: str = "data/processed"
) -> Dict[str, float]:
    """Compute FP32 training losses for hard-case calibration selection."""
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    losses = {}
    for img_id in train_ids:
        img_p = Path(data_dir) / "images" / f"{img_id}.png"
        mask_p = Path(data_dir) / "masks" / f"{img_id}.png"
        img_gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
        if img_gray is None or mask is None:
            continue
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - MEAN) / STD
        inp = img_norm.transpose(2, 0, 1)[np.newaxis, ...]
        logits = session.run(None, {"input": inp})[0][0, 0]
        prob = 1.0 / (1.0 + np.exp(-logits))
        gt = (mask > 127).astype(np.float32)
        # Binary cross entropy + dice loss
        bce = - (gt * np.log(prob + 1e-7) + (1.0 - gt) * np.log(1.0 - prob + 1e-7)).mean()
        dice = (2.0 * np.sum(prob * gt) + 1e-6) / (np.sum(prob) + np.sum(gt) + 1e-6)
        losses[img_id] = float(0.5 * bce + 0.5 * (1.0 - dice))
    return losses


def select_calib_images(
    train_ids: List[str],
    n: int,
    strategy: str,
    seed: int,
    fp32_losses: Dict[str, float] = None,
    data_dir: str = "data/processed",
) -> List[str]:
    """Select n calibration images using random, class-stratified, or hard-case strategy."""
    rng = np.random.RandomState(seed)

    if strategy == "random":
        return list(rng.choice(train_ids, size=n, replace=False))

    elif strategy == "class-stratified":
        benign_ids = [i for i in train_ids if "benign" in i]
        malignant_ids = [i for i in train_ids if "malignant" in i]
        n_b = int(round(n * (len(benign_ids) / len(train_ids))))
        n_m = n - n_b
        sel_b = list(rng.choice(benign_ids, size=n_b, replace=False))
        sel_m = list(rng.choice(malignant_ids, size=n_m, replace=False))
        return sel_b + sel_m

    elif strategy == "hard-case":
        # Top-n highest FP32 training loss (deterministic)
        if fp32_losses is None:
            # Fallback if losses not precomputed: sort by area
            return list(train_ids[:n])
        sorted_by_loss = sorted(train_ids, key=lambda i: fp32_losses.get(i, 0.0), reverse=True)
        return list(sorted_by_loss[:n])
    else:
        raise ValueError(f"Unknown strategy {strategy}")


def evaluate_quant_on_val(model_path: str, data_dir: str = "data/processed", splits_file: str = "data/splits.json") -> Tuple[float, float]:
    """Evaluate quantized ONNX model strictly on validation set."""
    with open(splits_file, "r") as f:
        val_ids = json.load(f)["val"]

    # Filter to lesion images for calibration accuracy
    val_lesions = [i for i in val_ids if "normal" not in i]
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    dsc_list = []
    hd95_list = []

    for img_id in val_lesions:
        img_p = Path(data_dir) / "images" / f"{img_id}.png"
        mask_p = Path(data_dir) / "masks" / f"{img_id}.png"

        img_gray = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)

        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - MEAN) / STD
        inp = img_norm.transpose(2, 0, 1)[np.newaxis, ...]

        ort_out = session.run(None, {"input": inp})[0][0, 0]
        pred_mask = (ort_out > 0.0).astype(np.uint8) * 255

        m = compute_dsc_iou_hd95(pred_mask, mask)
        dsc_list.append(m["dsc"])
        if not np.isnan(m["hd95"]):
            hd95_list.append(m["hd95"])

    mean_dsc = float(np.mean(dsc_list))
    mean_hd95 = float(np.mean(hd95_list)) if hd95_list else 0.0
    return mean_dsc, mean_hd95


def run_calib_size_study(
    model_name: str = "mbv3s",
    method: str = "minmax",
    splits_file: str = "data/splits.json",
    data_dir: str = "data/processed",
):
    """Run Mini-Study S3: Calibration-Set Size sensitivity sweep."""
    with open(splits_file, "r") as f:
        train_ids = json.load(f)["train"]

    prep_model = Path("models/fp32") / f"{model_name}_prep.onnx"
    if not prep_model.exists():
        prep_model = Path("models/fp32") / f"{model_name}.onnx"

    sizes = [10, 25, 50, 100, 200]
    strategies = [
        ("random", [42, 43, 44]),
        ("class-stratified", [42, 43, 44]),
        ("hard-case", [42]),
    ]

    calib_method = CalibrationMethod.MinMax if method == "minmax" else CalibrationMethod.Entropy
    results = []
    tmp_out = Path("models/int8/tmp_s3_eval.onnx")

    print(f"\nComputing FP32 training losses for hard-case calibration strategy...")
    fp32_losses = compute_fp32_losses(str(prep_model), train_ids, data_dir=data_dir)

    print(f"\n--- Running Mini-Study S3 on {model_name} ({method.upper()}) ---")
    for n in sizes:
        for strat, seeds in strategies:
            for s in seeds:
                selected_ids = select_calib_images(train_ids, n=n, strategy=strat, seed=s, fp32_losses=fp32_losses)
                reader = BUSICalibrationDataReader(image_ids=selected_ids)

                quantize_static(
                    model_input=str(prep_model),
                    model_output=str(tmp_out),
                    calibration_data_reader=reader,
                    quant_format=QuantFormat.QDQ,
                    per_channel=True,
                    activation_type=QuantType.QInt8,
                    weight_type=QuantType.QInt8,
                    calibrate_method=calib_method,
                    extra_options={"ActivationSymmetric": True, "WeightSymmetric": True},
                )

                val_dsc, val_hd95 = evaluate_quant_on_val(str(tmp_out))
                print(f"  n={n:3d} | strategy={strat:16s} | seed={s} -> Val DSC: {val_dsc:.4f}, Val HD95: {val_hd95:.2f} px")
                results.append({
                    "n_calib": n,
                    "strategy": strat,
                    "seed": s,
                    "val_dsc": val_dsc,
                    "val_hd95": val_hd95,
                })

    if tmp_out.exists():
        tmp_out.unlink()

    df = pd.DataFrame(results)
    out_csv = Path("results/study_s3_calib_size.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n--- Mini-Study S3 completed and saved to {out_csv} ---")
    return df


if __name__ == "__main__":
    run_calib_size_study()
