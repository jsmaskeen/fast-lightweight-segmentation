# Environment and Hardware Documentation

## Host Development & Training Workstation
- **OS**: Windows 11 (AMD64)
- **CPU**: AMD Ryzen 7 7840HS (8 cores, 16 threads, Zen 4)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop GPU (8,188 MiB VRAM)
- **Driver Version**: 592.82
- **CUDA Version**: 13.1 (Driver supported) / 12.4 (PyTorch runtime: `torch==2.6.0+cu124`)
- **Python**: 3.12.12 (managed via `uv`, virtual environment `.venv/`)
- **Key Libraries**:
  - `torch`: 2.6.0+cu124
  - `torchvision`: 0.21.0+cu124
  - `segmentation-models-pytorch`: 0.6.0
  - `timm`: 1.0.31
  - `albumentations`: 2.0.8
  - `onnx`: 1.23.2
  - `onnxruntime`: 1.31.1

## Edge Target: Raspberry Pi 4 (Model B)
*(To be populated during Phase 7 setup over SSH)*
- **SoC**: Broadcom BCM2711, Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.5GHz / 1.8GHz
- **RAM**: Pending connection
- **OS**: Raspberry Pi OS 64-bit / Linux kernel version: Pending
- **CPU Governor**: `performance` (enforced via `cpufreq-set`)
- **ORT Version**: Pending
- **Throttle Status**: Monitored via `vcgencmd get_throttled`
