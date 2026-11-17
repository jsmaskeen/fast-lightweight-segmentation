import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.quant.audit import evaluate_onnx_model
from src.train.models import get_model
from src.train.losses import BCEDiceLoss
from src.utils.seed import seed_everything
from src.quant.quantize import quantize_model


class UdiatTorchDataset(Dataset):
    def __init__(self, split: str = "train"):
        with open("data/udiat/splits.json") as f:
            stems = json.load(f)[split]
        self.img_paths = [Path("data/udiat/images") / f"{s}.png" for s in stems]
        self.mask_paths = [Path("data/udiat/masks") / f"{s}.png" for s in stems]
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_gray = cv2.imread(str(self.img_paths[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(self.mask_paths[idx]), cv2.IMREAD_GRAYSCALE)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        img_norm = (img_rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        tensor_img = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()
        tensor_mask = torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)
        return tensor_img, tensor_mask


def run_cross_center_benchmarks():
    """
    Evaluates zero-shot cross-center generalization (BUSI -> UDIAT) across:
    1. ResNet-34 FP32 baseline
    2. ResNet-34 INT8 QDQ (MinMax)
    3. MobileNetV3-Small FP32 baseline
    4. MobileNetV3-Small Naive INT8 QDQ (showing collapse)
    5. MobileNetV3-Small Rescued Mixed Precision (k*=13)
    """
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    models_to_test = [
        ("ResNet-34 FP32 (BUSI)", "models/fp32/r34_prep.onnx", "Zero-Shot"),
        ("ResNet-34 INT8 (BUSI)", "models/int8/r34_int8_minmax.onnx", "Zero-Shot"),
        ("MobileNetV3-Small FP32 (BUSI)", "models/fp32/mbv3s_prep.onnx", "Zero-Shot"),
        ("MobileNetV3 v1 (Naive INT8 / Harsh) [BUSI]", "models/int8/mbv3s_int8_minmax.onnx", "Zero-Shot"),
        ("MobileNetV3 v2 (Rescued Mixed k=13) [BUSI]", "models/mixed/mbv3s_mixed_best.onnx", "Zero-Shot"),
    ]

    print("\n=======================================================")
    print("Cross-Center Evaluation: BUSI Models -> UDIAT Dataset B")
    print("=======================================================")

    results = []
    for display_name, model_path, regime in models_to_test:
        p = Path(model_path)
        if not p.exists():
            print(f"Skipping {display_name}: {model_path} not found.")
            continue

        size_mb = p.stat().st_size / (1024 * 1024)
        _, summ = evaluate_onnx_model(
            str(p),
            split="test",
            data_dir="data/udiat",
            splits_file="data/udiat/splits.json"
        )

        dsc_mean = summ["dsc"]["mean"]
        dsc_median = summ["dsc"]["median"]
        hd95_mean = summ["hd95"]["mean"]
        hd95_median = summ["hd95"]["median"]

        results.append({
            "model_name": display_name,
            "regime": regime,
            "target_dataset": "UDIAT Dataset B",
            "model_size_mb": round(size_mb, 2),
            "test_dsc_mean": round(dsc_mean, 4),
            "test_dsc_median": round(dsc_median, 4),
            "test_hd95_mean_px": round(hd95_mean, 2),
            "test_hd95_median_px": round(hd95_median, 2),
        })

        print(f"{display_name:42s} | Size: {size_mb:5.2f} MB | Test DSC: {dsc_mean:.4f} (Med: {dsc_median:.4f}) | HD95: {hd95_mean:.2f} px")

    # Fine-Tuning r34 on UDIAT (10 epochs quick fine-tuning)
    print("\n--- Fine-Tuning ResNet-34 on UDIAT Train Split (100 scans) ---")
    seed_everything(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(UdiatTorchDataset(split="train"), batch_size=8, shuffle=True)
    val_loader = DataLoader(UdiatTorchDataset(split="val"), batch_size=8, shuffle=False)

    ft_model = get_model("r34", pretrained=False).to(device)
    ckpt = torch.load("models/fp32/r34_best.pt", map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ft_model.load_state_dict(ckpt["model_state_dict"])
    else:
        ft_model.load_state_dict(ckpt)
    optimizer = torch.optim.AdamW(ft_model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    for epoch in range(1, 11):
        ft_model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = ft_model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 5 == 0 or epoch == 10:
            print(f"  Epoch {epoch:2d}/10 | Loss: {total_loss/len(train_loader):.4f}")

    # Export fine-tuned ONNX with quant_pre_process
    ft_model.eval().to("cpu")
    dummy = torch.randn(1, 3, 256, 256)
    raw_onnx_path = Path("models/fp32/r34_udiat_raw.onnx")
    ft_onnx_path = Path("models/fp32/r34_udiat_finetuned.onnx")
    torch.onnx.export(
        ft_model,
        dummy,
        str(raw_onnx_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
    )
    from onnxruntime.quantization.shape_inference import quant_pre_process
    quant_pre_process(str(raw_onnx_path), str(ft_onnx_path))
    if raw_onnx_path.exists():
        raw_onnx_path.unlink()
    print(f"Exported preprocessed fine-tuned model to {ft_onnx_path}")

    # Quantize fine-tuned model
    ft_int8_path = Path("models/int8/r34_udiat_int8.onnx")
    quantize_model(
        model_name="r34",
        method="minmax",
        input_path=ft_onnx_path,
        output_path=ft_int8_path,
    )

    # Evaluate fine-tuned FP32 and INT8 on UDIAT test
    for ft_name, ft_p in [
        ("ResNet-34 FP32 (UDIAT Fine-Tuned)", ft_onnx_path),
        ("ResNet-34 INT8 (UDIAT Fine-Tuned)", ft_int8_path),
    ]:
        _, summ = evaluate_onnx_model(
            str(ft_p),
            split="test",
            data_dir="data/udiat",
            splits_file="data/udiat/splits.json"
        )
        size_mb = ft_p.stat().st_size / (1024 * 1024)
        dsc_mean = summ["dsc"]["mean"]
        dsc_median = summ["dsc"]["median"]
        hd95_mean = summ["hd95"]["mean"]
        hd95_median = summ["hd95"]["median"]

        results.append({
            "model_name": ft_name,
            "regime": "Fine-Tuned",
            "target_dataset": "UDIAT Dataset B",
            "model_size_mb": round(size_mb, 2),
            "test_dsc_mean": round(dsc_mean, 4),
            "test_dsc_median": round(dsc_median, 4),
            "test_hd95_mean_px": round(hd95_mean, 2),
            "test_hd95_median_px": round(hd95_median, 2),
        })
        print(f"{ft_name:42s} | Size: {size_mb:5.2f} MB | Test DSC: {dsc_mean:.4f} (Med: {dsc_median:.4f}) | HD95: {hd95_mean:.2f} px")

    df_out = pd.DataFrame(results)
    csv_path = out_dir / "udiat_cross_center_results.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved cross-center benchmark results to {csv_path}")
    return df_out


if __name__ == "__main__":
    run_cross_center_benchmarks()
