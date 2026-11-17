from pathlib import Path
import onnx
import onnxruntime.quantization.qdq_loss_debug as qdq_debug
import pandas as pd

from src.quant.calib_reader import BUSICalibrationDataReader


def analyze_sensitivity(
    model_name: str = "mbv3s",
    num_samples: int = 24,
    method: str = "minmax",
) -> pd.DataFrame:
    """
    Extension A Step 1-3:
    1. Weight matching between FP32 prep model and INT8 QDQ model.
    2. Activation matching on a subset of 16-32 calibration images to prevent RAM exhaustion.
    3. Rank nodes by activation error (lowest SQNR = highest error = most sensitive).
    """
    fp32_path = Path(f"models/fp32/{model_name}_prep.onnx")
    if not fp32_path.exists():
        fp32_path = Path(f"models/fp32/{model_name}.onnx")
    int8_path = Path(f"models/int8/{model_name}_int8_{method.lower()}.onnx")

    if not fp32_path.exists() or not int8_path.exists():
        raise FileNotFoundError(f"Missing {fp32_path} or {int8_path}")

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    aug_dir = Path("models/tmp_aug")
    aug_dir.mkdir(parents=True, exist_ok=True)

    aug_fp32 = aug_dir / f"{model_name}_fp32_aug.onnx"
    aug_int8 = aug_dir / f"{model_name}_int8_aug.onnx"

    print(f"\n--- Extension A: Sensitivity Analysis for {model_name} ---")

    # Step 1: Weight matching
    print("Step 1: Computing weight matching...")
    weights_match = qdq_debug.create_weight_matching(str(fp32_path), str(int8_path))
    print(f"  Matched {len(weights_match)} weight tensors.")

    # Step 2: Augment models to output intermediate activations
    print("Step 2: Augmenting models for intermediate activations...")
    qdq_debug.modify_model_output_intermediate_tensors(str(fp32_path), str(aug_fp32))
    qdq_debug.modify_model_output_intermediate_tensors(str(int8_path), str(aug_int8))

    # Collect activations on calibration images
    print(f"Collecting intermediate activations on {num_samples} calibration images...")
    reader_fp32 = BUSICalibrationDataReader(num_images=num_samples, seed=42)
    reader_int8 = BUSICalibrationDataReader(num_images=num_samples, seed=42)

    float_acts = qdq_debug.collect_activations(str(aug_fp32), reader_fp32)
    qdq_acts = qdq_debug.collect_activations(str(aug_int8), reader_int8)

    print("Matching activation tensors between FP32 and INT8 QDQ models...")
    act_match = qdq_debug.create_activation_matching(qdq_acts, float_acts)
    print(f"  Matched {len(act_match)} activation tensor pairs.")

    print("Computing SQNR error metrics...")
    act_error = qdq_debug.compute_activation_error(act_match)

    # Step 3: Map activation tensors to graph nodes and rank quantizable (Conv) nodes
    onnx_fp32 = onnx.load(str(fp32_path))
    tensor_to_node = {}
    node_to_optype = {}
    concat_inputs = set()
    for node in onnx_fp32.graph.node:
        if node.op_type == "Concat":
            concat_inputs.update(node.input)
        for out in node.output:
            tensor_to_node[out] = node.name
            node_to_optype[node.name] = node.op_type

    node_records = {}
    for t_name, err_dict in act_error.items():
        node_name = tensor_to_node.get(t_name)
        if not node_name:
            continue
        op_type = node_to_optype.get(node_name, "Unknown")
        # Lowest SQNR (most negative/smallest) indicates most noise/distortion
        sqnr_xmodel = float(err_dict.get("xmodel_err", 999.0))
        sqnr_qdq = float(err_dict.get("qdq_err", 999.0))
        primary_sqnr = sqnr_xmodel if sqnr_xmodel != 999.0 else sqnr_qdq
        is_concat_adj = any(out in concat_inputs for out in [t_name])

        if node_name not in node_records or primary_sqnr < node_records[node_name]["sqnr"]:
            node_records[node_name] = {
                "node_name": node_name,
                "op_type": op_type,
                "sqnr": primary_sqnr,
                "sqnr_xmodel": sqnr_xmodel,
                "sqnr_qdq": sqnr_qdq,
                "is_concat_adjacent": is_concat_adj,
            }

    # Filter to Conv nodes for quantization exclusion and rank by sensitivity score
    ranking_df = pd.DataFrame(list(node_records.values()))
    ranking_df = ranking_df[ranking_df["op_type"] == "Conv"].copy()

    node_obj_map = {n.name: n for n in onnx_fp32.graph.node}
    graph_inputs = {inp.name for inp in onnx_fp32.graph.input}
    graph_outputs = {out.name for out in onnx_fp32.graph.output}

    def get_sensitivity_score(row):
        # Base: lower SQNR = higher activation distortion
        score = -float(row["sqnr"])
        node = node_obj_map.get(row["node_name"])
        if node is not None:
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
            group = attrs.get("group", 1)
            is_dw = group > 1
            is_output_boundary = any(out in graph_outputs for out in node.output)
            is_input_boundary = any(inp in graph_inputs for inp in node.input)

            if is_output_boundary:
                # Output boundary generates logit segmentation contours directly
                score += 100.0
            elif is_input_boundary:
                # Input boundary extracts initial spatial texture primitives from raw image
                score += 80.0
            elif is_dw:
                # Depthwise group convolution: 9 weights/channel with zero cross-channel redundancy
                score += 50.0
        return score

    ranking_df["sensitivity_score"] = ranking_df.apply(get_sensitivity_score, axis=1)
    ranking_df = ranking_df.sort_values(by="sensitivity_score", ascending=False).reset_index(drop=True)
    ranking_df["rank"] = ranking_df.index + 1

    ranking_csv = out_dir / f"sensitivity_ranking_{model_name}.csv"
    ranking_df.to_csv(ranking_csv, index=False)
    print(f"Ranked {len(ranking_df)} quantizable Conv nodes. Saved ranking to {ranking_csv}")
    print("\nTop 15 Most Sensitive Nodes:")
    print(ranking_df[["rank", "node_name", "sqnr", "sensitivity_score"]].head(15).to_string(index=False))

    # Clean up temporary augmented models
    if aug_fp32.exists():
        aug_fp32.unlink()
    if aug_int8.exists():
        aug_int8.unlink()
    if aug_dir.exists():
        try:
            aug_dir.rmdir()
        except OSError:
            pass

    return ranking_df


if __name__ == "__main__":
    analyze_sensitivity("mbv3s", num_samples=24)
