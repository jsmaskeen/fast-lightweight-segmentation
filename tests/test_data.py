import json
from pathlib import Path
import cv2
import numpy as np
import torch

from src.data.dataset import BUSIDataset


def test_raw_dataset_counts():
    raw_dir = Path("data/raw/Dataset_BUSI_with_GT")
    benign_imgs = [f for f in (raw_dir / "benign").glob("*.png") if "_mask" not in f.name]
    malignant_imgs = [f for f in (raw_dir / "malignant").glob("*.png") if "_mask" not in f.name]
    normal_imgs = [f for f in (raw_dir / "normal").glob("*.png") if "_mask" not in f.name]

    assert len(benign_imgs) == 437, f"Expected 437 benign images, got {len(benign_imgs)}"
    assert len(malignant_imgs) == 210, f"Expected 210 malignant images, got {len(malignant_imgs)}"
    assert len(normal_imgs) == 133, f"Expected 133 normal images, got {len(normal_imgs)}"
    assert len(benign_imgs) + len(malignant_imgs) == 647, "Expected 647 lesion images"


def test_mask_bitwise_or_synthetic():
    mask1 = np.zeros((100, 100), dtype=np.uint8)
    mask2 = np.zeros((100, 100), dtype=np.uint8)

    mask1[20:50, 20:50] = 255
    mask2[40:70, 40:70] = 255

    merged = np.bitwise_or(mask1, mask2)
    # Check overlap region
    assert np.all(merged[20:50, 20:50] == 255)
    assert np.all(merged[40:70, 40:70] == 255)
    # Check disjoint non-active region
    assert np.all(merged[80:90, 80:90] == 0)


def test_splits_integrity():
    splits_file = Path("data/splits.json")
    assert splits_file.exists(), "splits.json does not exist"

    with open(splits_file, "r") as f:
        splits = json.load(f)

    train_ids = set(splits["train"])
    val_ids = set(splits["val"])
    test_ids = set(splits["test"])

    # Disjointness check
    assert len(train_ids.intersection(val_ids)) == 0, "Train and Val overlap!"
    assert len(train_ids.intersection(test_ids)) == 0, "Train and Test overlap!"
    assert len(val_ids.intersection(test_ids)) == 0, "Val and Test overlap!"

    # Normal images strictly excluded from training
    train_normal = [img_id for img_id in train_ids if "normal" in img_id]
    assert len(train_normal) == 0, f"Found normal images in train split: {train_normal}"

    # Verify normal images are split between val and test
    val_normal = [img_id for img_id in val_ids if "normal" in img_id]
    test_normal = [img_id for img_id in test_ids if "normal" in img_id]
    assert len(val_normal) + len(test_normal) == 133, f"Expected 133 total normal images, got {len(val_normal) + len(test_normal)}"


def test_processed_masks_are_binary():
    mask_files = list(Path("data/processed/masks").glob("*.png"))
    assert len(mask_files) > 0, "No processed mask files found"

    # Sample 30 random masks
    sample_indices = np.random.choice(len(mask_files), min(30, len(mask_files)), replace=False)
    for idx in sample_indices:
        mask = cv2.imread(str(mask_files[idx]), cv2.IMREAD_GRAYSCALE)
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255}), f"Mask {mask_files[idx].name} contains non-binary values: {unique_vals}"


def test_busi_dataset_loader():
    train_ds = BUSIDataset(split="train")
    val_ds = BUSIDataset(split="val")
    test_ds = BUSIDataset(split="test")

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    sample = train_ds[0]
    assert sample["image"].shape == (3, 256, 256)
    assert sample["mask"].shape == (1, 256, 256)
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32
    # Verify mask values are in {0.0, 1.0}
    unique_mask_vals = torch.unique(sample["mask"]).tolist()
    assert all(v in [0.0, 1.0] for v in unique_mask_vals)


def test_qa_overlay_grid_exists():
    grid_path = Path("figures/qa_overlays_grid.png")
    assert grid_path.exists(), "QA overlay grid image does not exist"
    img = cv2.imread(str(grid_path))
    assert img is not None
    # Grid should be 4 rows x 5 cols of 256x256 images: shape (1024, 1280, 3)
    assert img.shape == (1024, 1280, 3), f"Unexpected QA grid shape: {img.shape}"
