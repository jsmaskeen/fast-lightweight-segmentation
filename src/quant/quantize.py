import argparse
import hashlib
from pathlib import Path
from typing import List, Optional
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    QuantFormat,
    QuantType,
    CalibrationMethod,
)

from src.quant.calib_reader import BUSICalibrationDataReader


def parse_args():
    parser = argparse.ArgumentParser(description="Static QDQ INT8 Quantization (MinMax vs Entropy)")
    parser.add_argument("--model", type=str, required=True, choices=["r34", "mbv3s", "all"], help="Model name")
    parser.add_argument("--calib-samples", type=int, default=100, help="Number of calibration images (default: 100)")
    parser.add_argument("--scheme", type=str, default="s8s8", choices=["s8s8", "u8s8"], help="Quantization scheme")
    return parser.parse_args()


def quantize_model(
    model_name: str,
    method: str = "minmax",
    num_calib: int = 100,
    scheme: str = "s8s8",
    nodes_to_exclude: Optional[List[str]] = None,
    op_types_to_quantize: Optional[List[str]] = None,
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
) -> Path:
    if input_path is not None:
        prep_model_path = Path(input_path)
    else:
        prep_model_path = Path("models/fp32") / f"{model_name}_prep.onnx"
        if not prep_model_path.exists():
            prep_model_path = Path("models/fp32") / f"{model_name}.onnx"
    if not prep_model_path.exists():
        raise FileNotFoundError(f"Neither prep nor raw ONNX found at {prep_model_path}")

    out_dir = Path("models/int8")
    out_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        output_path = out_dir / f"{model_name}_int8_{method.lower()}.onnx"

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

    # Default to Conv/Gemm/MatMul to avoid corrupting activations (HardSigmoid, Mul, Resize)
    if op_types_to_quantize is None and model_name == "mbv3s":
        op_types_to_quantize = ["Conv", "Gemm", "MatMul"]

    print(f"\n--- Quantizing {model_name} ({method.upper()}, scheme={scheme}, calib={num_calib}) ---")
    if op_types_to_quantize:
        print(f"  Restricting quantization to operators: {op_types_to_quantize}")
    if nodes_to_exclude:
        print(f"  Excluding {len(nodes_to_exclude)} nodes from quantization.")

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
        nodes_to_exclude=nodes_to_exclude or [],
        extra_options=extra_options,
    )

    # Verify model loads in ONNX Runtime
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    print(f"Quantized model verified: loads successfully into ONNX Runtime CPU session.")

    # Record checksum
    checksums_file = Path("models/checksums.txt")
    if checksums_file.parent.exists() and output_dir.resolve() == Path("models/int8").resolve():
        with open(output_path, "rb") as f:
            h = hashlib.sha256(f.read()).hexdigest()
        with open(checksums_file, "a") as f:
            f.write(f"{h}  {output_path.as_posix()}\n")

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.2f} MB)")
    return output_path


def main():
    args = parse_args()
    models = ["r34", "mbv3s"] if args.model == "all" else [args.model]

    for m in models:
        quantize_model(m, method="minmax", num_calib=args.calib_samples, scheme=args.scheme)
        quantize_model(m, method="entropy", num_calib=args.calib_samples, scheme=args.scheme)


if __name__ == "__main__":
    main()
