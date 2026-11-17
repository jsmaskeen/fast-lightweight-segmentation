import json
import os
from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import onnxruntime as ort


def profile_model_operators(
    model_path: str,
    num_warmup: int = 20,
    num_profiled: int = 100,
    num_threads: int = 4
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Mini-Study S2: Run ONNX Runtime profiler for 100 iterations.
    Extracts operator time shares and top 10 slowest nodes.
    """
    m_path = Path(model_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Model {m_path} does not exist.")

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = num_threads
    sess_opts.inter_op_num_threads = 1
    sess_opts.enable_profiling = True

    session = ort.InferenceSession(str(m_path), sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, 256, 256).astype(np.float32)

    # Warmup
    for _ in range(num_warmup):
        session.run(None, {input_name: dummy})

    # Profiled iterations
    for _ in range(num_profiled):
        session.run(None, {input_name: dummy})

    prof_file = session.end_profiling()

    # Parse JSON profile trace
    with open(prof_file, "r") as f:
        events = json.load(f)

    # Clean up profile trace file
    if os.path.exists(prof_file):
        os.remove(prof_file)

    op_durations = {}
    node_durations = {}

    for ev in events:
        if ev.get("cat") == "Node":
            dur = ev.get("dur", 0)  # microseconds
            args = ev.get("args", {})
            op_type = args.get("op_name", ev.get("name", "Unknown"))
            node_name = ev.get("name", "Unknown")

            # Standardize op category
            if "Conv" in op_type or "QLinearConv" in op_type:
                cat = "Convolution (FP32/INT8)"
            elif "Quantize" in op_type or "Dequantize" in op_type or "QDQ" in op_type:
                cat = "Quantize / Dequantize (QDQ)"
            elif "Concat" in op_type:
                cat = "Concat"
            elif "Resize" in op_type:
                cat = "Resize"
            elif "Relu" in op_type or "Sigmoid" in op_type or "HardSwish" in op_type or "Clip" in op_type:
                cat = "Activations"
            else:
                cat = "Other"

            op_durations[cat] = op_durations.get(cat, 0) + dur
            node_durations[node_name] = node_durations.get(node_name, 0) + dur

    total_time = sum(op_durations.values())
    op_records = []
    for cat, dur in op_durations.items():
        share = (dur / total_time * 100.0) if total_time > 0 else 0.0
        op_records.append({"Operator Category": cat, "Duration (us)": dur, "Time Share (%)": share})

    df_ops = pd.DataFrame(op_records).sort_values(by="Time Share (%)", ascending=False).reset_index(drop=True)

    # Top 10 nodes
    node_records = []
    for node, dur in node_durations.items():
        share = (dur / total_time * 100.0) if total_time > 0 else 0.0
        node_records.append({"Node Name": node, "Duration (us)": dur, "Time Share (%)": share})

    df_nodes = pd.DataFrame(node_records).sort_values(by="Duration (us)", ascending=False).head(10).reset_index(drop=True)

    return df_ops, df_nodes


def main():
    models = {
        "r34_fp32": "models/fp32/r34_prep.onnx",
        "r34_int8": "models/int8/r34_int8_minmax.onnx",
        "mbv3s_fp32": "models/fp32/mbv3s_prep.onnx",
        "mbv3s_int8": "models/int8/mbv3s_int8_minmax.onnx",
    }
    mixed_path = Path("models/mixed/mbv3s_mixed_best.onnx")
    if not mixed_path.exists():
        candidates = list(Path("models/mixed").glob("mbv3s_mix_k*.onnx"))
        if candidates:
            mixed_path = candidates[0]
    if mixed_path.exists():
        models["mbv3s_mixed"] = str(mixed_path)

    all_op_shares = []
    print("\n--- Mini-Study S2: Operator Profiling ---")
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found.")
            continue

        print(f"Profiling {name}...")
        df_ops, df_nodes = profile_model_operators(path)
        df_ops["Variant"] = name
        all_op_shares.append(df_ops)

        print(f"\nTop Operator Types for {name}:")
        print(df_ops[["Operator Category", "Time Share (%)"]].to_string(index=False))

    if all_op_shares:
        combined = pd.concat(all_op_shares, ignore_index=True)
        out_csv = Path("results/study_s2_operator_profile.csv")
        combined.to_csv(out_csv, index=False)
        print(f"\nStudy S2 results saved to {out_csv}")


if __name__ == "__main__":
    main()
