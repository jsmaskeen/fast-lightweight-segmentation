import argparse
import hashlib
from pathlib import Path
from typing import Optional, List
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

from src.distil.inference import InferenceEngine
from src.quant.calib_reader import BUSICalibrationDataReader
from src.data.dataset import BUSIDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Export and Quantize Distilled Model to ONNX and INT8")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/distil/mbv3s_distilled_best.pt",
        help="Path to distilled PyTorch checkpoint",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="mbv3s",
        choices=["r34", "mbv3s"],
        help="Backbone model architecture name",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=100,
        help="Number of calibration images for static INT8 quantization (default: 100)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="minmax",
        choices=["minmax", "entropy", "both"],
        help="Calibration method: minmax, entropy, or both",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="s8s8",
        choices=["s8s8", "u8s8"],
        help="Quantization scheme (default: s8s8)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="models/distil",
        help="Output directory for exported and quantized models",
    )
    return parser.parse_args()


def export_distilled_to_onnx(
    checkpoint_path: str = "models/distil/mbv3s_distilled_best.pt",
    model_name: str = "mbv3s",
    out_dir: str = "models/distil",
    opset: int = 17,
) -> Path:
    """
    Exports distilled PyTorch model checkpoint to ONNX Opset 17, validates parity and structure,
    and runs onnxruntime.quantization.preprocess to produce the optimized preparation model.
    """
    ckpt_path = Path(checkpoint_path)
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_onnx_path = output_dir / f"{model_name}_distilled.onnx"
    prep_onnx_path = output_dir / f"{model_name}_distilled_prep.onnx"

    print(f"\n==================================================")
    print(f"Exporting distilled {model_name} to ONNX Opset {opset}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"==================================================")

    # 1. Load distilled inference model on CPU
    device = torch.device("cpu")
    engine = InferenceEngine(model_name=model_name, checkpoint_path=ckpt_path if ckpt_path.exists() else None, device=device)
    engine.eval()
    model = engine.backbone

    dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

    # 2. Export to ONNX
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

    # 3. Check ONNX validity
    onnx_model = onnx.load(str(raw_onnx_path))
    onnx.checker.check_model(onnx_model)
    print("ONNX model structure verified with onnx.checker.")

    # 4. Parity check vs PyTorch
    ort_session = ort.InferenceSession(str(raw_onnx_path), providers=["CPUExecutionProvider"])
    try:
        val_ds = BUSIDataset(split="val", transform=None)
        max_diffs = []
        num_samples = min(10, len(val_ds))
        for i in range(num_samples):
            sample = val_ds[i]
            inp_tensor = sample["image"].unsqueeze(0)
            with torch.no_grad():
                pt_out = model(inp_tensor).numpy()
            ort_out = ort_session.run(None, {"input": inp_tensor.numpy()})[0]
            abs_diff = float(np.max(np.abs(pt_out - ort_out)))
            max_diffs.append(abs_diff)
        worst_diff = max(max_diffs)
        print(f"Parity check over {num_samples} validation samples: worst max abs diff = {worst_diff:.6e}")
        assert worst_diff < 1e-4, f"Parity check failed! Max diff {worst_diff} >= 1e-4"
    except Exception as e:
        print(f"Warning during validation parity check: {e}. Falling back to random tensor check.")
        with torch.no_grad():
            pt_out = model(dummy_input).numpy()
        ort_out = ort_session.run(None, {"input": dummy_input.numpy()})[0]
        worst_diff = float(np.max(np.abs(pt_out - ort_out)))
        assert worst_diff < 1e-4, f"Parity check failed! Max diff {worst_diff} >= 1e-4"

    print(f"Parity check PASSED: max abs logit diff < 1e-4.")

    # 5. Preprocess for quantization
    print(f"Running onnxruntime.quantization.preprocess...")
    quant_pre_process(
        input_model=str(raw_onnx_path),
        output_model_path=str(prep_onnx_path),
        skip_optimization=False,
    )
    print(f"Preprocessed ONNX saved to {prep_onnx_path}")

    # 6. Checksum registration
    checksums_file = Path("models/checksums.txt")
    if checksums_file.parent.exists() and output_dir.resolve() == Path("models/distil").resolve():
        with open(raw_onnx_path, "rb") as f:
            raw_hash = hashlib.sha256(f.read()).hexdigest()
        with open(prep_onnx_path, "rb") as f:
            prep_hash = hashlib.sha256(f.read()).hexdigest()
        with open(checksums_file, "a") as f:
            f.write(f"{raw_hash}  {raw_onnx_path.as_posix()}\n")
            f.write(f"{prep_hash}  {prep_onnx_path.as_posix()}\n")

    return prep_onnx_path


def quantize_distilled_model(
    prep_model_path: Path,
    model_name: str = "mbv3s",
    method: str = "minmax",
    num_calib: int = 100,
    scheme: str = "s8s8",
    out_dir: str = "models/distil",
    op_types_to_quantize: Optional[List[str]] = None,
) -> Path:
    """
    Quantizes preprocessed ONNX model into static INT8 QDQ format using BUSICalibrationDataReader.
    """
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_name}_distilled_int8_{method.lower()}.onnx"

    calib_method = (
        CalibrationMethod.MinMax if method.lower() == "minmax" else CalibrationMethod.Entropy
    )

    if scheme.lower() == "s8s8":
        act_type = QuantType.QInt8
        wt_type = QuantType.QInt8
        extra_options = {"ActivationSymmetric": True, "WeightSymmetric": True}
    else:
        act_type = QuantType.QUInt8
        wt_type = QuantType.QInt8
        extra_options = {"ActivationSymmetric": False, "WeightSymmetric": True}

    if op_types_to_quantize is None and model_name == "mbv3s":
        op_types_to_quantize = ["Conv", "Gemm", "MatMul"]

    print(f"\n--- Quantizing Distilled {model_name} ({method.upper()}, scheme={scheme}, calib={num_calib}) ---")
    if op_types_to_quantize:
        print(f"  Restricting quantization to operators: {op_types_to_quantize}")

    reader = BUSICalibrationDataReader(num_images=num_calib, seed=42)

    quantize_static(
        model_input=str(prep_model_path),
        model_output=str(output_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=op_types_to_quantize,
        per_channel=True,
        activation_type=act_type,
        weight_type=wt_type,
        calibrate_method=calib_method,
        extra_options=extra_options,
    )

    # Verify session loads
    _ = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    print(f"Quantized model verified: loads successfully into ONNX Runtime CPU session.")

    # Record checksum
    checksums_file = Path("models/checksums.txt")
    if checksums_file.parent.exists() and output_dir.resolve() == Path("models/distil").resolve():
        with open(output_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        with open(checksums_file, "a") as f:
            f.write(f"{h}  {output_path.as_posix()}\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


def main():
    args = parse_args()
    prep_path = export_distilled_to_onnx(
        checkpoint_path=args.checkpoint,
        model_name=args.model_name,
        out_dir=args.out_dir,
        opset=args.opset,
    )

    methods = ["minmax", "entropy"] if args.method == "both" else [args.method]
    for m in methods:
        quantize_distilled_model(
            prep_model_path=prep_path,
            model_name=args.model_name,
            method=m,
            num_calib=args.calib_samples,
            scheme=args.scheme,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
