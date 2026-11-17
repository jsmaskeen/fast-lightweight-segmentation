import json
import zipfile
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split


def download_and_prepare_udiat():
    """
    Downloads UDIAT Dataset B (163 scans: 110 benign, 53 malignant from Sabadell, Spain)
    and formats images and ground truth masks to standardized 256x256 grayscale/binary PNGs.
    """
    out_dir = Path("data/udiat")
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    splits_path = out_dir / "splits.json"
    if splits_path.exists():
        with open(splits_path) as f:
            existing = json.load(f)
        if len(existing.get("metadata", [])) == 163:
            print(f"UDIAT dataset already prepared at {out_dir}")
            return existing
            return json.load(f)

    print("\n--- Ingesting UDIAT Dataset B (163 scans) ---")

    # Step 1: Download original scans from Hugging Face WOOJYE/udiat
    print("Step 1: Downloading raw ultrasound scans from Hugging Face (WOOJYE/udiat)...")
    zip_path = hf_hub_download(
        repo_id="WOOJYE/udiat",
        filename="udiat_original163_negative163.zip",
        repo_type="dataset",
    )

    # Step 2: Extract images
    print("Step 2: Extracting scans...")
    extracted_images = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for f in z.namelist():
            if f.startswith("udiat_original163/") and f.endswith(".png"):
                stem = Path(f).name  # e.g., '000001.png'
                data = z.read(f)
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
                img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
                extracted_images[stem] = img_256

    print(f"  Extracted {len(extracted_images)} raw ultrasound scans.")

    # Step 3: Fetch class labels from udiat_classwise_eval.csv
    print("Step 3: Loading metadata from udiat_classwise_eval.csv...")
    csv_path = hf_hub_download(
        repo_id="WOOJYE/udiat",
        filename="udiat_classwise_eval.csv",
        repo_type="dataset",
    )
    df_meta = pd.read_csv(csv_path)
    df_meta["filename"] = df_meta["image_path"].apply(lambda x: Path(x).name)
    pos_meta = df_meta[df_meta["class_label"] != "normal"].copy()
    label_dict = dict(zip(pos_meta["filename"], pos_meta["class_label"]))
    print(f"  Loaded labels for {len(label_dict)} positive scans ({sum(v=='benign' for v in label_dict.values())} benign, {sum(v=='malignant' for v in label_dict.values())} malignant).")

    # Step 4: Download ground truth masks from public GitHub repository
    print("Step 4: Fetching ground truth segmentation masks...")
    base_raw_url = "https://raw.githubusercontent.com/shashankhalo7/Biomedical-Image--Segmentation-and-Tumour-Classification/master/BUS"

    # Save images and download masks
    records = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for fname in sorted(extracted_images.keys()):
        img_arr = extracted_images[fname]

        # Save standardized 256x256 image
        img_save_path = images_dir / fname
        cv2.imwrite(str(img_save_path), img_arr)

        # Download mask if not present
        mask_save_path = masks_dir / fname
        if not mask_save_path.exists() or mask_save_path.stat().st_size == 0:
            mask_url = f"{base_raw_url}/GT/{fname}"
            try:
                req_m = urllib.request.Request(mask_url, headers=headers)
                with urllib.request.urlopen(req_m) as resp_m:
                    mask_raw = resp_m.read()
                mask_arr = cv2.imdecode(np.frombuffer(mask_raw, np.uint8), cv2.IMREAD_GRAYSCALE)
                mask_256 = cv2.resize(mask_arr, (256, 256), interpolation=cv2.INTER_NEAREST)
                mask_256 = np.where(mask_256 > 127, 255, 0).astype(np.uint8)
                cv2.imwrite(str(mask_save_path), mask_256)
            except Exception as e:
                print(f"  Warning: failed to download mask {fname}: {e}")
                mask_256 = np.zeros((256, 256), dtype=np.uint8)
                cv2.imwrite(str(mask_save_path), mask_256)

        label = label_dict.get(fname, "benign")

        records.append({
            "stem": Path(fname).stem,
            "filename": fname,
            "image_path": str(img_save_path.relative_to(out_dir.parent.parent)),
            "mask_path": str(mask_save_path.relative_to(out_dir.parent.parent)),
            "class_label": label,
        })

    # Step 4: Stratified Train / Val / Test Split (100 train / 23 val / 40 test)
    stems = [r["stem"] for r in records]
    labels = [r["class_label"] for r in records]

    # Split into train (100) and temp (63)
    train_stems, temp_stems, train_labels, temp_labels = train_test_split(
        stems, labels, test_size=63, random_state=42, stratify=labels
    )
    # Split temp (63) into val (23) and test (40)
    val_stems, test_stems, _, _ = train_test_split(
        temp_stems, temp_labels, test_size=40, random_state=42, stratify=temp_labels
    )

    splits = {
        "train": sorted(train_stems),
        "val": sorted(val_stems),
        "test": sorted(test_stems),
        "metadata": records,
    }

    with open(splits_path, "w") as f_out:
        json.dump(splits, f_out, indent=2)

    print(f"UDIAT prepared successfully: {len(train_stems)} train, {len(val_stems)} val, {len(test_stems)} test.")
    print(f"Splits saved to {splits_path}")
    return splits


if __name__ == "__main__":
    download_and_prepare_udiat()
