import pandas as pd
from pathlib import Path
from src.quant.audit import evaluate_onnx_model

def update_audit():
    records = []
    models = [
        ("fp32_distilled", "models/distil/mbv3s_distilled_prep.onnx"),
        ("int8_minmax_distilled", "models/distil/mbv3s_distilled_int8_minmax.onnx"),
        ("int8_entropy_distilled", "models/distil/mbv3s_distilled_int8_entropy.onnx"),
    ]
    for v, p in models:
        df, summ = evaluate_onnx_model(p, split="test")
        d_ci = f"[{summ['dsc']['ci_lower']:.4f}, {summ['dsc']['ci_upper']:.4f}]"
        h_ci = f"[{summ['hd95']['ci_lower']:.2f}, {summ['hd95']['ci_upper']:.2f}]"
        records.append({
            "Model": "mbv3s",
            "Variant": v,
            "Split": "test",
            "Mean Dice": round(summ['dsc']['mean'], 4),
            "Dice 95% CI": d_ci,
            "Mean HD95 (px)": round(summ['hd95']['mean'], 2),
            "HD95 95% CI": h_ci,
            "Normal FPR@50px": f"{summ['normal_fpr_50px']*100:.2f}%",
            "Empty Preds": summ['empty_predictions'],
        })
    df_out = pd.DataFrame(records)
    out_path = Path("results/distil_audit_test.csv")
    df_out.to_csv(out_path, index=False)
    print("Updated results/distil_audit_test.csv successfully:")
    print(df_out.to_string(index=False))

if __name__ == "__main__":
    update_audit()
