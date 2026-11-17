import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.utils.meta import make_metadata
from src.data.dataset import BUSIDataset
from src.train.models import get_model
from src.metrics.dsc_iou_hd95 import compute_dsc_iou_hd95, compute_normal_fpr
from src.metrics.bootstrap import bootstrap_ci


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate FP32 PyTorch baseline on test set")
    parser.add_argument("--model", type=str, default="all", choices=["r34", "mbv3s", "all"], help="Model to evaluate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def evaluate_model_on_test(model_name: str, device_str: str = "cpu") -> dict:
    device = torch.device(device_str)
    ckpt_path = Path("models/fp32") / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Load model
    model = get_model(model_name, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_ds = BUSIDataset(split="test", transform=None)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    per_image_rows = []
    lesion_dsc = []
    lesion_iou = []
    lesion_hd95 = []
    empty_pred_count = 0
    normal_total = 0
    normal_fp_50_count = 0
    normal_fp_0_count = 0

    print(f"\nEvaluating {model_name} (FP32) on held-out test set ({len(test_ds)} images)...")

    for batch in test_loader:
        img = batch["image"].to(device)
        mask_gt = batch["mask"].numpy()[0, 0]  # (256, 256)
        img_id = batch["image_id"][0]
        cls_label = batch["class_label"][0]

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logit = model(img)

        # Raw logit > 0.0 equals probability > 0.5
        pred_mask = (logit > 0.0).cpu().numpy()[0, 0].astype(np.uint8) * 255

        if cls_label in ["benign", "malignant"]:
            metrics = compute_dsc_iou_hd95(pred_mask, mask_gt)
            lesion_dsc.append(metrics["dsc"])
            lesion_iou.append(metrics["iou"])
            if not np.isnan(metrics["hd95"]):
                lesion_hd95.append(metrics["hd95"])
            if metrics["is_empty_pred"]:
                empty_pred_count += 1

            per_image_rows.append({
                "image_id": img_id,
                "class": cls_label,
                "dsc": metrics["dsc"],
                "iou": metrics["iou"],
                "hd95": metrics["hd95"],
                "is_empty_pred": metrics["is_empty_pred"],
                "pred_area_px": metrics["pred_area_px"],
                "gt_area_px": metrics["gt_area_px"],
            })
        else:
            fpr_res = compute_normal_fpr(pred_mask)
            normal_total += 1
            if fpr_res["fp_50px"]:
                normal_fp_50_count += 1
            if fpr_res["fp_0px"]:
                normal_fp_0_count += 1

            per_image_rows.append({
                "image_id": img_id,
                "class": cls_label,
                "dsc": np.nan,
                "iou": np.nan,
                "hd95": np.nan,
                "is_empty_pred": 0.0,
                "pred_area_px": fpr_res["pred_area_px"],
                "gt_area_px": 0.0,
            })

    # Save per-image CSV
    out_csv = Path("results") / f"eval_test_{model_name}_fp32.csv"
    pd.DataFrame(per_image_rows).to_csv(out_csv, index=False)
    print(f"Per-image test results saved to {out_csv}")

    # Bootstrap CIs (1,000 resamples)
    dsc_ci = bootstrap_ci(np.array(lesion_dsc), n_resamples=1000, seed=42)
    iou_ci = bootstrap_ci(np.array(lesion_iou), n_resamples=1000, seed=42)
    hd95_ci = bootstrap_ci(np.array(lesion_hd95), n_resamples=1000, seed=42)

    fpr_50 = (normal_fp_50_count / normal_total) if normal_total > 0 else 0.0
    fpr_0 = (normal_fp_0_count / normal_total) if normal_total > 0 else 0.0

    meta = make_metadata(model_name, "fp32")
    summary = {
        "metadata": meta,
        "n_lesion_test": len(lesion_dsc),
        "n_normal_test": normal_total,
        "empty_predictions_count": empty_pred_count,
        "dsc": dsc_ci,
        "iou": iou_ci,
        "hd95": hd95_ci,
        "normal_fpr_50px": fpr_50,
        "normal_fpr_0px": fpr_0,
    }

    print(f"[{model_name} Test Results]")
    print(f"  DSC:  {dsc_ci['mean']:.4f} (95% CI: [{dsc_ci['ci_lower']:.4f}, {dsc_ci['ci_upper']:.4f}], median: {dsc_ci['median']:.4f})")
    print(f"  IoU:  {iou_ci['mean']:.4f} (95% CI: [{iou_ci['ci_lower']:.4f}, {iou_ci['ci_upper']:.4f}], median: {iou_ci['median']:.4f})")
    print(f"  HD95: {hd95_ci['mean']:.2f} px (95% CI: [{hd95_ci['ci_lower']:.2f}, {hd95_ci['ci_upper']:.2f}], median: {hd95_ci['median']:.2f} px)")
    print(f"  Normal FPR@50px: {fpr_50*100:.2f}% ({normal_fp_50_count}/{normal_total})")
    print(f"  Empty lesion predictions: {empty_pred_count}/{len(lesion_dsc)}")

    return summary


def main():
    args = parse_args()
    models = ["r34", "mbv3s"] if args.model == "all" else [args.model]

    summaries = {}
    summary_path = Path("results/baseline_metrics_summary.json")
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summaries = json.load(f)

    for m in models:
        try:
            res = evaluate_model_on_test(m, args.device)
            summaries[m] = res
        except FileNotFoundError as e:
            print(f"Skipping {m}: {e}")

    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"Updated baseline summary: {summary_path}")


if __name__ == "__main__":
    main()
