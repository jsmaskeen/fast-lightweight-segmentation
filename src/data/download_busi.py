import shutil
from pathlib import Path


def ensure_busi_dataset(target_dir: str = "data/raw/Dataset_BUSI_with_GT") -> Path:
    target_path = Path(target_dir)
    if target_path.exists() and (target_path / "benign").exists():
        print(f"BUSI dataset already exists at {target_path.resolve()}")
        verify_raw_counts(target_path)
        return target_path

    print("Downloading BUSI dataset via kagglehub...")
    import kagglehub
    downloaded_path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")
    source_path = Path(downloaded_path) / "Dataset_BUSI_with_GT"
    if not source_path.exists():
        # Check subdirectories
        subdirs = list(Path(downloaded_path).glob("**/Dataset_BUSI_with_GT"))
        if subdirs:
            source_path = subdirs[0]
        else:
            raise FileNotFoundError(f"Could not locate Dataset_BUSI_with_GT in {downloaded_path}")

    print(f"Copying from {source_path} to {target_path}...")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target_path, dirs_exist_ok=True)
    print(f"Dataset successfully copied to {target_path.resolve()}")
    verify_raw_counts(target_path)
    return target_path


def verify_raw_counts(dataset_path: Path) -> dict:
    counts = {}
    for cls in ["benign", "malignant", "normal"]:
        cls_dir = dataset_path / cls
        files = list(cls_dir.glob("*.png"))
        images = [f for f in files if "_mask" not in f.name]
        masks = [f for f in files if "_mask" in f.name]
        counts[cls] = {"images": len(images), "masks": len(masks)}
        print(f"  [{cls}] {len(images)} images, {len(masks)} masks")

    assert counts["benign"]["images"] == 437, f"Expected 437 benign images, got {counts['benign']['images']}"
    assert counts["malignant"]["images"] == 210, f"Expected 210 malignant images, got {counts['malignant']['images']}"
    assert counts["normal"]["images"] == 133, f"Expected 133 normal images, got {counts['normal']['images']}"
    print("Raw BUSI count check PASSED: 647 lesion (437 benign, 210 malignant) + 133 normal.")
    return counts


if __name__ == "__main__":
    ensure_busi_dataset()
