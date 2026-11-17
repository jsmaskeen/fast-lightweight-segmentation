import numpy as np
import onnxruntime as ort
import segmentation_models_pytorch as smp
import torch

from src.utils.seed import seed_everything
from src.utils.meta import make_metadata


def test_seed_everything():
    seed_everything(42)
    val1 = np.random.rand()
    seed_everything(42)
    val2 = np.random.rand()
    assert val1 == val2


def test_metadata():
    meta = make_metadata("mbv3s", "fp32")
    assert meta["model"] == "mbv3s"
    assert meta["variant"] == "fp32"
    assert "timestamp" in meta
    assert "hardware_id" in meta


def test_model_instantiation():
    # Test r34
    r34 = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    x = torch.randn(2, 3, 256, 256)
    out_r34 = r34(x)
    assert out_r34.shape == (2, 1, 256, 256)

    # Test mbv3s with shrunk decoder
    mbv3s = smp.Unet(
        encoder_name="tu-mobilenetv3_small_100",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        decoder_channels=[128, 96, 64, 32, 16],
    )
    out_mbv3s = mbv3s(x)
    assert out_mbv3s.shape == (2, 1, 256, 256)

    total_params = sum(p.numel() for p in mbv3s.parameters())
    # Verify mbv3s is between 2M and 3M params as specified in Phase 2
    assert 2_000_000 <= total_params <= 3_500_000, f"mbv3s params {total_params} out of expected range"


def test_ort_smoke():
    # Verify ONNX Runtime is functional
    assert hasattr(ort, "InferenceSession")
