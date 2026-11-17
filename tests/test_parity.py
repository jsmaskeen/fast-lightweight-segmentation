from pathlib import Path
import numpy as np
import pytest
import torch
import onnxruntime as ort

from src.train.models import get_model
from src.data.dataset import BUSIDataset


@pytest.mark.parametrize("model_name", ["r34", "mbv3s"])
def test_onnx_parity_if_exported(model_name):
    onnx_path = Path("models/fp32") / f"{model_name}.onnx"
    ckpt_path = Path("models/fp32") / f"{model_name}_best.pt"

    if not onnx_path.exists() or not ckpt_path.exists():
        pytest.skip(f"Model {model_name} ONNX or checkpoint not yet generated.")

    # Load PyTorch model
    device = torch.device("cpu")
    model = get_model(model_name, pretrained=False).to(device)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load ONNX Runtime session
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    val_ds = BUSIDataset(split="val", transform=None)
    for i in range(min(5, len(val_ds))):
        inp = val_ds[i]["image"].unsqueeze(0)
        with torch.no_grad():
            pt_out = model(inp).numpy()
        ort_out = session.run(None, {"input": inp.numpy()})[0]

        max_diff = np.max(np.abs(pt_out - ort_out))
        assert max_diff < 1e-4, f"Parity failed for {model_name} on sample {i}: max diff {max_diff} >= 1e-4"
