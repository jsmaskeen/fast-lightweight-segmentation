import argparse
import hashlib
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization.preprocess import quant_pre_process

from src.train.models import get_model
from src.data.dataset import BUSIDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export PyTorch model to ONNX Opset 17")
    parser.add_argument("--model", type=str, required=True, choices=["r34", "mbv3s", "all"], help="Model name")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    return parser.parse_args()


def export_single_model(model_name: str, opset: int = 17) -> Path:
    print(f"\n==================================================")
    print(f"Exporting {model_name} to ONNX Opset {opset}")
    print(f"==================================================")

    ckpt_path = Path("models/fp32") / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found.")

    out_dir = Path("models/fp32")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_onnx_path = out_dir / f"{model_name}.onnx"
    prep_onnx_path = out_dir / f"{model_name}_prep.onnx"

    # 1. Load PyTorch model on CPU for export
    device = torch.device("cpu")
    model = get_model(model_name, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    # 2. Export to ONNX Opset 17
    torch.onnx.export(
        model,
        dummy_input,
        str(raw_onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,  # Fixed 1x3x256x256 shape strictly
    )
    print(f"Raw ONNX exported to {raw_onnx_path}")

    # 3. Check ONNX model validity
    onnx_model = onnx.load(str(raw_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified with onnx.checker.")

    # 4. Parity check vs PyTorch (max abs diff < 1e-4)
    ort_session = ort.InferenceSession(str(raw_onnx_path), providers=["CPUExecutionProvider"])
    val_ds = BUSIDataset(split="val", transform=None)

    max_diffs = []
    num_samples = min(10, len(val_ds))
    for i in range(num_samples):
        sample = val_ds[i]
        inp_tensor = sample["image"].unsqueeze(0)  # (1, 3, 256, 256)
        with torch.no_grad():
            pt_out = model(inp_tensor).numpy()

        ort_out = ort_session.run(None, {"input": inp_tensor.numpy()})[0]
        abs_diff = np.max(np.abs(pt_out - ort_out))
        max_diffs.append(abs_diff)

    worst_diff = max(max_diffs)
    print(f"Parity check over {num_samples} validation samples: worst max abs diff = {worst_diff:.6e}")
    assert worst_diff < 1e-4, f"Parity check failed! Max diff {worst_diff} >= 1e-4"
    print(f"Parity Check PASSED: max abs logit diff < 1e-4.")

    # 5. Run quantization.preprocess to produce optimized prep model
    print(f"Running onnxruntime.quantization.preprocess...")
    quant_pre_process(
        input_model=str(raw_onnx_path),
        output_model_path=str(prep_onnx_path),
        skip_optimization=False,
    )
    print(f"Preprocessed ONNX saved to {prep_onnx_path}")

    # 6. Record SHA-256 checksums
    checksums_file = Path("models/checksums.txt")
    if checksums_file.parent.exists() and output_dir.resolve() == Path("models/fp32").resolve():
        with open(raw_onnx_path, "rb") as f:
            raw_hash = hashlib.sha256(f.read()).hexdigest()
        with open(prep_onnx_path, "rb") as f:
            prep_hash = hashlib.sha256(f.read()).hexdigest()

        with open(checksums_file, "a") as f:
            f.write(f"{raw_hash}  {raw_onnx_path.as_posix()}\n")
            f.write(f"{prep_hash}  {prep_onnx_path.as_posix()}\n")

    return prep_onnx_path


def main():
    args = parse_args()
    models = ["r34", "mbv3s"] if args.model == "all" else [args.model]
    for m in models:
        export_single_model(m, opset=args.opset)


if __name__ == "__main__":
    main()
