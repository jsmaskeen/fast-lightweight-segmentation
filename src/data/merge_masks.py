import hashlib
import json
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

from src.utils.seed import seed_everything


def compute_dhash(image_gray: np.ndarray, hash_size: int = 8) -> np.ndarray:
    """Compute difference hash (dhash) as a boolean array."""
    resized = cv2.resize(image_gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return diff.flatten()


def find_image_masks(raw_dir: Path) -> List[dict]:
    """Scan raw BUSI dataset and pair each base scan with all its mask files."""
    records = []
    for cls in ["benign", "malignant", "normal"]:
        cls_dir = raw_dir / cls
        files = list(cls_dir.glob("*.png"))
        images = [f for f in files if "_mask" not in f.name]
        for img_path in sorted(images, key=lambda p: p.name):
            stem = img_path.stem
            # Find all mask files associated with this scan
            masks = sorted(list(cls_dir.glob(f"{stem}_mask*.png")), key=lambda p: p.name)
            records.append({
                "id": stem,
                "class": cls,
                "image_path": str(img_path),
                "mask_paths": [str(m) for m in masks],
                "num_masks": len(masks),
            })
    return records


def process_images_and_masks(
    records: List[dict],
    processed_dir: Path,
    target_size: Tuple[int, int] = (256, 256)
) -> List[dict]:
    """
    Merge multi-mask scans with bitwise OR at native resolution before resizing,
    resize masks with nearest-neighbor, then re-binarize at > 127.
    """
    img_out_dir = processed_dir / "images"
    mask_out_dir = processed_dir / "masks"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    mask_out_dir.mkdir(parents=True, exist_ok=True)

    processed_records = []
    print(f"Processing and merging masks for {len(records)} images...")

    for r in records:
        img_gray = cv2.imread(r["image_path"], cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Failed to load image {r['image_path']}")

        h_orig, w_orig = img_gray.shape

        # Step 1: Merge masks at native resolution with bitwise OR
        if len(r["mask_paths"]) == 0:
            native_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        elif len(r["mask_paths"]) == 1:
            native_mask = cv2.imread(r["mask_paths"][0], cv2.IMREAD_GRAYSCALE)
            if native_mask is None:
                native_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
        else:
            loaded_masks = []
            for mp in r["mask_paths"]:
                m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    loaded_masks.append(m)
            if loaded_masks:
                native_mask = np.bitwise_or.reduce(loaded_masks)
            else:
                native_mask = np.zeros((h_orig, w_orig), dtype=np.uint8)

        # Re-binarize native mask strictly at > 127
        native_mask = (native_mask > 127).astype(np.uint8) * 255

        # Step 2: Resize image to 256x256
        interp = cv2.INTER_AREA if (h_orig >= target_size[1] and w_orig >= target_size[0]) else cv2.INTER_LINEAR
        resized_img = cv2.resize(img_gray, target_size, interpolation=interp)

        # Step 3: Resize mask with nearest-neighbor to prevent border anti-aliasing
        resized_mask = cv2.resize(native_mask, target_size, interpolation=cv2.INTER_NEAREST)
        # Re-binarize mask after resize at > 127
        resized_mask = (resized_mask > 127).astype(np.uint8) * 255

        # Step 4: Compute hashes for deduplication
        with open(r["image_path"], "rb") as f:
            sha256_hash = hashlib.sha256(f.read()).hexdigest()
        dhash_val = compute_dhash(img_gray)

        # Save processed files
        out_img_path = img_out_dir / f"{r['id']}.png"
        out_mask_path = mask_out_dir / f"{r['id']}.png"
        cv2.imwrite(str(out_img_path), resized_img)
        cv2.imwrite(str(out_mask_path), resized_mask)

        processed_records.append({
            "id": r["id"],
            "class": r["class"],
            "num_raw_masks": r["num_masks"],
            "orig_shape": [h_orig, w_orig],
            "image_file": str(out_img_path),
            "mask_file": str(out_mask_path),
            "sha256": sha256_hash,
            "dhash": dhash_val,
        })

    return processed_records


def deduplicate_and_cluster(records: List[dict]) -> Tuple[List[dict], dict]:
    """
    Identify exact duplicates and near duplicates via perceptual dhash.
    Drop exact byte duplicates / cross-class conflicting duplicates.
    Cluster near-duplicate scans within the same class so they never leak across splits.
    """
    # 1. Exact byte duplicate detection
    sha_map = {}
    exact_duplicates_to_drop = set()
    for r in records:
        h = r["sha256"]
        if h in sha_map:
            # Duplicate found!
            prev = sha_map[h]
            print(f"Exact duplicate detected: '{r['id']}' ({r['class']}) duplicates '{prev['id']}' ({prev['class']})")
            # Drop the duplicate (e.g. malignant (145) which duplicates benign (433))
            exact_duplicates_to_drop.add(r["id"])
        else:
            sha_map[h] = r

    # 2. Check cross-class near duplicate conflicts (dhash distance <= 1)
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            r1, r2 = records[i], records[j]
            if r1["class"] != r2["class"]:
                dist = int(np.count_nonzero(r1["dhash"] != r2["dhash"]))
                if dist <= 1:
                    print(f"Cross-class near conflict: '{r1['id']}' ({r1['class']}) vs '{r2['id']}' ({r2['class']}), dist={dist}")
                    # Drop the conflicting duplicate to prevent ambiguous labels
                    if r1["class"] == "malignant" and r2["class"] == "benign":
                        exact_duplicates_to_drop.add(r1["id"])
                    elif r2["class"] == "malignant" and r1["class"] == "benign":
                        exact_duplicates_to_drop.add(r2["id"])
                    elif "normal" in (r1["class"], r2["class"]):
                        drop_id = r1["id"] if r1["class"] == "normal" else r2["id"]
                        exact_duplicates_to_drop.add(drop_id)

    filtered_records = [r for r in records if r["id"] not in exact_duplicates_to_drop]

    # 3. Cluster near duplicates within class (Hamming distance <= 2)
    # Using connected components (Union-Find)
    n = len(filtered_records)
    parent = list(range(n))

    def find(i):
        if parent[i] == i:
            return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i in range(n):
        for j in range(i + 1, n):
            if filtered_records[i]["class"] == filtered_records[j]["class"]:
                dist = int(np.count_nonzero(filtered_records[i]["dhash"] != filtered_records[j]["dhash"]))
                if dist <= 2:
                    union(i, j)

    cluster_map = {}
    for i in range(n):
        c_id = find(i)
        filtered_records[i]["cluster_id"] = f"cluster_{filtered_records[c_id]['class']}_{c_id}"
        cluster_map.setdefault(filtered_records[i]["cluster_id"], []).append(filtered_records[i]["id"])

    multi_item_clusters = {k: v for k, v in cluster_map.items() if len(v) > 1}
    print(f"Total images after dedup: {len(filtered_records)} (dropped {len(exact_duplicates_to_drop)})")
    print(f"Formed {len(cluster_map)} clusters; {len(multi_item_clusters)} contain multiple near-duplicate scans.")

    dedup_report = {
        "raw_counts": {
            "benign": sum(1 for r in records if r["class"] == "benign"),
            "malignant": sum(1 for r in records if r["class"] == "malignant"),
            "normal": sum(1 for r in records if r["class"] == "normal"),
            "total_lesion": sum(1 for r in records if r["class"] in ["benign", "malignant"]),
            "total": len(records),
        },
        "dropped_duplicates": list(exact_duplicates_to_drop),
        "clean_counts": {
            "benign": sum(1 for r in filtered_records if r["class"] == "benign"),
            "malignant": sum(1 for r in filtered_records if r["class"] == "malignant"),
            "normal": sum(1 for r in filtered_records if r["class"] == "normal"),
            "total_lesion": sum(1 for r in filtered_records if r["class"] in ["benign", "malignant"]),
            "total": len(filtered_records),
        },
        "num_clusters": len(cluster_map),
        "multi_scan_clusters": multi_item_clusters,
    }

    return filtered_records, dedup_report


def create_splits(
    records: List[dict],
    splits_file: Path,
    seed: int = 42
) -> dict:
    """
    Create stratified train/val/test splits:
    - Lesion images (benign + malignant): Stratified 70/15/15 (train/val/test), seed 42.
    - Normal images (133): Exactly 0 in train; split 50/50 into val and test.
    - Clusters of near-duplicates are assigned as atomic units to ensure zero split leakage.
    """
    if splits_file.exists():
        print(f"Splits file {splits_file} already exists. Reading existing splits.")
        with open(splits_file, "r") as f:
            return json.load(f)

    seed_everything(seed)
    rng = np.random.RandomState(seed)

    # Separate lesion vs normal
    lesion_records = [r for r in records if r["class"] in ["benign", "malignant"]]
    normal_records = [r for r in records if r["class"] == "normal"]

    # Group lesion records by cluster
    lesion_clusters = {}
    for r in lesion_records:
        lesion_clusters.setdefault(r["cluster_id"], []).append(r)

    # Classify each cluster by majority class
    benign_clusters = []
    malignant_clusters = []
    for c_id, c_records in lesion_clusters.items():
        cls = c_records[0]["class"]
        if cls == "benign":
            benign_clusters.append(c_records)
        else:
            malignant_clusters.append(c_records)

    def split_clusters(cluster_list, train_ratio=0.70, val_ratio=0.15):
        rng.shuffle(cluster_list)
        total_imgs = sum(len(c) for c in cluster_list)
        target_train = int(round(total_imgs * train_ratio))
        target_val = int(round(total_imgs * val_ratio))

        train_imgs, val_imgs, test_imgs = [], [], []
        curr_train, curr_val = 0, 0

        for c in cluster_list:
            c_ids = [r["id"] for r in c]
            if curr_train + len(c) <= target_train:
                train_imgs.extend(c_ids)
                curr_train += len(c)
            elif curr_val + len(c) <= target_val:
                val_imgs.extend(c_ids)
                curr_val += len(c)
            else:
                test_imgs.extend(c_ids)

        return train_imgs, val_imgs, test_imgs

    b_train, b_val, b_test = split_clusters(benign_clusters, 0.70, 0.15)
    m_train, m_val, m_test = split_clusters(malignant_clusters, 0.70, 0.15)

    train_split = b_train + m_train
    val_split = b_val + m_val
    test_split = b_test + m_test

    # Normal images: 0 in train, 50% in val, 50% in test
    normal_clusters = {}
    for r in normal_records:
        normal_clusters.setdefault(r["cluster_id"], []).append(r)
    n_cluster_list = list(normal_clusters.values())
    rng.shuffle(n_cluster_list)

    target_n_val = int(round(len(normal_records) * 0.50))
    n_val, n_test = [], []
    curr_n_val = 0
    for c in n_cluster_list:
        c_ids = [r["id"] for r in c]
        if curr_n_val + len(c) <= target_n_val:
            n_val.extend(c_ids)
            curr_n_val += len(c)
        else:
            n_test.extend(c_ids)

    val_split.extend(n_val)
    test_split.extend(n_test)

    # Sort for deterministic output
    train_split = sorted(train_split)
    val_split = sorted(val_split)
    test_split = sorted(test_split)

    splits_data = {
        "metadata": {
            "seed": seed,
            "train_ratio": 0.70,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "normal_train_count": 0,
            "total_train": len(train_split),
            "total_val": len(val_split),
            "total_test": len(test_split),
        },
        "train": train_split,
        "val": val_split,
        "test": test_split,
    }

    splits_file.parent.mkdir(parents=True, exist_ok=True)
    with open(splits_file, "w") as f:
        json.dump(splits_data, f, indent=2)

    print("Splits created successfully:")
    print(f"  Train: {len(train_split)} (lesions only, 0 normal)")
    print(f"  Val:   {len(val_split)} ({len(b_val) + len(m_val)} lesions + {len(n_val)} normal)")
    print(f"  Test:  {len(test_split)} ({len(b_test) + len(m_test)} lesions + {len(n_test)} normal)")

    return splits_data


def generate_qa_overlays(
    records: List[dict],
    processed_dir: Path,
    output_grid_path: Path,
    num_samples: int = 20,
    seed: int = 42
) -> None:
    """Generate 20 QA overlay images with ground truth contours and combine into a 4x5 grid."""
    seed_everything(seed)
    rng = np.random.RandomState(seed)

    # Select samples ensuring representation: multi-mask images, benign, malignant, normal
    multimask = [r for r in records if r["num_raw_masks"] > 1]
    benign_single = [r for r in records if r["class"] == "benign" and r["num_raw_masks"] == 1]
    malignant_single = [r for r in records if r["class"] == "malignant" and r["num_raw_masks"] == 1]
    normal_single = [r for r in records if r["class"] == "normal"]

    selected = []
    # Take up to 6 multi-mask examples
    selected.extend(list(rng.choice(multimask, min(6, len(multimask)), replace=False)))
    # Take 6 benign
    selected.extend(list(rng.choice(benign_single, 6, replace=False)))
    # Take 5 malignant
    selected.extend(list(rng.choice(malignant_single, 5, replace=False)))
    # Take 3 normal
    selected.extend(list(rng.choice(normal_single, 3, replace=False)))

    qa_dir = output_grid_path.parent / "qa_overlays"
    qa_dir.mkdir(parents=True, exist_ok=True)

    grid_images = []
    for idx, r in enumerate(selected):
        img_path = processed_dir / "images" / f"{r['id']}.png"
        mask_path = processed_dir / "masks" / f"{r['id']}.png"

        img = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay = img.copy()
        # Draw ground truth contours in green
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Label info
        label = f"{r['id']} [{r['class']}] m={r['num_raw_masks']}"
        cv2.putText(overlay, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.63, (0, 255, 255), 2, cv2.LINE_AA)

        out_qa_file = qa_dir / f"qa_{idx:02d}_{r['id']}.png"
        cv2.imwrite(str(out_qa_file), overlay)
        grid_images.append(overlay)

    # Form a 4 rows x 5 cols composite grid
    rows = []
    for r in range(4):
        row_imgs = grid_images[r * 5 : (r + 1) * 5]
        rows.append(np.hstack(row_imgs))
    grid = np.vstack(rows)

    output_grid_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_grid_path), grid)
    print(f"Saved {num_samples} QA overlays to {qa_dir.resolve()} and composite grid to {output_grid_path.resolve()}")


def main():
    raw_dir = Path("data/raw/Dataset_BUSI_with_GT")
    processed_dir = Path("data/processed")
    splits_file = Path("data/splits.json")
    dedup_report_file = Path("data/dedup_report.json")
    qa_grid_file = Path("figures/qa_overlays_grid.png")

    records = find_image_masks(raw_dir)
    print(f"Found {len(records)} raw scans in {raw_dir}")

    processed_records = process_images_and_masks(records, processed_dir, target_size=(256, 256))

    clean_records, dedup_report = deduplicate_and_cluster(processed_records)

    with open(dedup_report_file, "w") as f:
        json.dump(dedup_report, f, indent=2)
    print(f"Saved deduplication report to {dedup_report_file}")

    splits = create_splits(clean_records, splits_file, seed=42)

    generate_qa_overlays(clean_records, processed_dir, qa_grid_file, num_samples=20, seed=42)
    print("Phase 1 preprocessing and split generation completed.")


if __name__ == "__main__":
    main()
