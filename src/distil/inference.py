from pathlib import Path
from typing import Optional, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import cv2

from src.train.models import get_model


class InferenceEngine(nn.Module):
    """
    Distilled Inference Model Module designed purely for deployment and inference.
    Features:
    - Pre-trained / distilled student backbone wrapped in evaluation mode.
    - Embedded or standalone probability sigmoid activation and thresholding.
    - Direct support for raw numpy images (e.g. from OpenCV or clinical video stream)
      as well as torch.Tensor batches.
    - Optimized memory footprint: frozen weights, no grad overhead, trace/export ready.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        model_name: str = "mbv3s",
        checkpoint_path: Optional[Union[str, Path]] = None,
        config_path: str = "configs/model.yaml",
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.threshold = float(threshold)
        self.model_name = model_name

        # Initialize student model architecture
        self.backbone = get_model(model_name, pretrained=False, config_path=config_path)

        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                self.backbone.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

        self.backbone.to(self.device)
        self.backbone.eval()

        # Freeze all parameters for inference-only usage
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pure inference forward pass. Returns raw segmentation logits.
        Input shape: (B, 3, H, W)
        Output shape: (B, 1, H, W)
        """
        return self.backbone(x)

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting probability maps [0.0, 1.0]."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting binary masks (0 or 1 float) based on threshold."""
        probs = self.predict_proba(x)
        return (probs >= self.threshold).float()

    def preprocess_image(self, img_bgr: np.ndarray, target_size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """
        Preprocesses a BGR or grayscale numpy image into the standard normalized tensor:
        - Resizes to (256, 256)
        - Converts to RGB
        - Replicates 1-channel to 3 channels if needed
        - Applies ImageNet mean/std normalization
        - Converts to (1, 3, H, W) torch float tensor on target device.
        """
        if len(img_bgr.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        elif img_bgr.shape[2] == 1:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        img_norm = (img_resized.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        return tensor

    @torch.no_grad()
    def infer_single_image(
        self,
        img: np.ndarray,
        return_binary: bool = True,
    ) -> np.ndarray:
        """
        End-to-end inference for a single numpy image (e.g. from video/stream or file):
        Returns binary mask (uint8: 0 or 255) or float probability map (0.0 to 1.0)
        at the original input image resolution.
        """
        orig_h, orig_w = img.shape[:2]
        tensor = self.preprocess_image(img, target_size=(256, 256))
        probs = self.predict_proba(tensor).squeeze().cpu().numpy()

        # Resize probability map back to original dimensions
        prob_resized = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        if return_binary:
            binary_mask = (prob_resized >= self.threshold).astype(np.uint8) * 255
            return binary_mask
        return prob_resized


def load_distilled_inference_model(
    checkpoint_path: str = "models/distil/mbv3s_distilled_best.pt",
    model_name: str = "mbv3s",
    device: Optional[str] = None,
) -> InferenceEngine:
    """Convenience helper to load a distilled inference engine."""
    return InferenceEngine(
        model_name=model_name,
        checkpoint_path=checkpoint_path,
        device=device,
    )
