import json
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


def get_train_transforms(image_size: int = 256) -> A.Compose:
    """Standard augmentation pipeline for training."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            translate_percent=(-0.05, 0.05),
            p=0.5,
            border_mode=cv2.BORDER_REFLECT_101
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5
        ),
        A.GaussNoise(
            std_range=(0.05, 0.15),
            p=0.3
        ),
    ])


class BUSIDataset(Dataset):
    """
    BUSI Dataset loader:
    - Replicates 1-channel grayscale to 3 channels for pretrained ImageNet encoders.
    - Normalizes with standard ImageNet mean ([0.485, 0.456, 0.406]) and std ([0.229, 0.224, 0.225]).
    - Converts binary masks to single-channel float tensors in [0.0, 1.0].
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        splits_file: str = "data/splits.json",
        split: str = "train",
        data_dir: str = "data/processed",
        transform: Optional[Callable] = None,
    ):
        super().__init__()
        self.split = split
        self.data_dir = Path(data_dir)
        self.transform = transform

        with open(splits_file, "r") as f:
            splits_data = json.load(f)

        if split not in splits_data:
            raise ValueError(f"Unknown split '{split}'. Available: {list(splits_data.keys())}")

        self.image_ids = splits_data[split]
        self.images_dir = self.data_dir / "images"
        self.masks_dir = self.data_dir / "masks"

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image_id = self.image_ids[idx]
        img_path = self.images_dir / f"{image_id}.png"
        mask_path = self.masks_dir / f"{image_id}.png"

        # Load 256x256 grayscale image and replicate to 3 channels
        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise FileNotFoundError(f"Missing image file {img_path}")
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        # Load 256x256 binary mask
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Missing mask file {mask_path}")

        # Binarize mask (0 or 1)
        mask_binary = (mask > 127).astype(np.float32)

        # Apply spatial/intensity augmentations if requested
        if self.transform is not None:
            augmented = self.transform(image=img_rgb, mask=mask_binary)
            img_rgb = augmented["image"]
            mask_binary = (augmented["mask"] > 0.5).astype(np.float32)

        # Normalize image to [0, 1] then ImageNet mean/std
        img_norm = (img_rgb.astype(np.float32) / 255.0 - self.MEAN) / self.STD
        # Transpose from (H, W, C) to (C, H, W)
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).float()

        # Mask tensor of shape (1, H, W)
        mask_tensor = torch.from_numpy(mask_binary).unsqueeze(0).float()

        # Parse class label from image_id
        if "benign" in image_id:
            class_label = "benign"
        elif "malignant" in image_id:
            class_label = "malignant"
        else:
            class_label = "normal"

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "image_id": image_id,
            "class_label": class_label,
        }
