import numpy as np
import torch

from src.distil.loss import KnowledgeDistillationLoss
from src.distil.inference import InferenceEngine


def test_kd_loss_computation_and_backward():
    criterion = KnowledgeDistillationLoss(temperature=2.0, alpha=0.5)

    student_logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    teacher_logits = torch.randn(2, 1, 64, 64, requires_grad=False)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = criterion(student_logits, teacher_logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert student_logits.grad is not None
    assert torch.all(torch.isfinite(student_logits.grad))


def test_inference_engine_forward_and_shapes():
    engine = InferenceEngine(model_name="mbv3s", checkpoint_path=None, device="cpu")

    dummy_input = torch.randn(2, 3, 256, 256)
    logits = engine(dummy_input)
    assert logits.shape == (2, 1, 256, 256)

    probs = engine.predict_proba(dummy_input)
    assert probs.shape == (2, 1, 256, 256)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)

    masks = engine.predict_mask(dummy_input)
    assert masks.shape == (2, 1, 256, 256)
    unique_vals = torch.unique(masks).tolist()
    assert set(unique_vals).issubset({0.0, 1.0})


def test_inference_engine_numpy_image():
    engine = InferenceEngine(model_name="mbv3s", checkpoint_path=None, device="cpu")

    dummy_img = np.random.randint(0, 256, (400, 500, 3), dtype=np.uint8)
    mask = engine.infer_single_image(dummy_img, return_binary=True)

    assert mask.shape == (400, 500)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 255})


def test_distilled_onnx_export_and_quantize(tmp_path):
    from src.distil.quantize import export_distilled_to_onnx, quantize_distilled_model
    from src.bench.harness import benchmark_single_process

    out_dir = tmp_path / "models_distil"
    prep_onnx = export_distilled_to_onnx(
        checkpoint_path="models/fp32/mbv3s_best.pt",  # Use available pretrained checkpoint
        model_name="mbv3s",
        out_dir=str(out_dir),
        opset=17,
    )
    assert prep_onnx.exists()

    int8_onnx = quantize_distilled_model(
        prep_model_path=prep_onnx,
        model_name="mbv3s",
        method="minmax",
        num_calib=5,
        scheme="s8s8",
        out_dir=str(out_dir),
    )
    assert int8_onnx.exists()

    # Benchmark single process smoke check
    bench_res = benchmark_single_process(str(int8_onnx), num_warmup=2, num_timed=5, num_threads=2)
    assert bench_res["fps"] > 0.0
    assert bench_res["p50_ms"] > 0.0

