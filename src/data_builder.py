"""
MVTec AD Dataset Preprocessing Pipeline
========================================
Handles grayscale → RGB conversion (required for CLIP ViT), resize to 336x336,
stratified 80/20 train-test split, and JSONL export for LLaVA instruction-tuning.

Usage:
    python src/data_builder.py --data_dir data/raw/mvtec_ad --output_dir data/processed
"""

import os
import json
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit


CLIP_RESOLUTION = (336, 336)

INSTRUCTION_PROMPT = (
    "<image>\n"
    "Với tư cách là kỹ sư KCS, hãy phân tích bề mặt linh kiện trong ảnh này."
)

# Known MVTec AD category names for auto-discovery
MVTEC_CATEGORIES = {
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
}


def auto_discover_mvtec(search_root="/kaggle/input"):
    """
    Auto-discover MVTec AD root directory under Kaggle input.
    Searches for directories containing known MVTec category folders
    (e.g. 'bottle', 'cable') with the expected train/good/ structure.
    """
    search_root = Path(search_root)
    if not search_root.exists():
        return None

    # Walk up to 3 levels deep to find MVTec root
    for depth1 in sorted(search_root.iterdir()):
        if not depth1.is_dir():
            continue

        # Check if this directory directly contains MVTec categories
        children = {d.name for d in depth1.iterdir() if d.is_dir()}
        if len(children & MVTEC_CATEGORIES) >= 3:
            print(f"  Auto-discovered MVTec AD root: {depth1}")
            return depth1

        # Check one level deeper
        for depth2 in sorted(depth1.iterdir()):
            if not depth2.is_dir():
                continue
            grandchildren = {d.name for d in depth2.iterdir() if d.is_dir()}
            if len(grandchildren & MVTEC_CATEGORIES) >= 3:
                print(f"  Auto-discovered MVTec AD root: {depth2}")
                return depth2

    return None


def convert_to_rgb(image_path, output_path):
    """
    Ensure image is 3-channel RGB for CLIP Vision Encoder compatibility.
    MVTec AD contains both RGB textures and grayscale metal surfaces.
    Feeding a single-channel image to CLIP crashes with a tensor shape mismatch.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    img = cv2.resize(img, CLIP_RESOLUTION, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(output_path), img)
    return True


def scan_mvtec_directory(data_dir):
    """
    Scan MVTec AD directory structure.
    Expected layout:
        data_dir/
        ├── <category>/          (e.g. bottle, cable, metal_nut, ...)
        │   ├── train/good/      (normal samples)
        │   └── test/
        │       ├── good/        (normal test)
        │       └── <defect>/    (broken_large, scratch, ...)
    """
    data_dir = Path(data_dir)
    samples = []

    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        # Train split: all "good" images
        good_train = category_dir / "train" / "good"
        if good_train.exists():
            for img_path in sorted(good_train.glob("*.*")):
                if img_path.suffix.lower() in (".png", ".jpg", ".bmp"):
                    samples.append({
                        "path": str(img_path),
                        "category": category,
                        "label": 0,
                        "defect_type": "good",
                    })

        # Test split: "good" and various defect types
        test_dir = category_dir / "test"
        if test_dir.exists():
            for defect_dir in sorted(test_dir.iterdir()):
                if not defect_dir.is_dir():
                    continue
                label = 0 if defect_dir.name == "good" else 1
                for img_path in sorted(defect_dir.glob("*.*")):
                    if img_path.suffix.lower() in (".png", ".jpg", ".bmp"):
                        samples.append({
                            "path": str(img_path),
                            "category": category,
                            "label": label,
                            "defect_type": defect_dir.name,
                        })

    return pd.DataFrame(samples)


def format_label(row):
    """Generate Vietnamese instruction-tuning label text."""
    if row["label"] == 1:
        return (
            f"Phát hiện lỗi trên bề mặt linh kiện (loại: {row['defect_type']}). "
            f"Loại sản phẩm: {row['category']}. Yêu cầu loại bỏ."
        )
    return (
        f"Bề mặt linh kiện sạch, không phát hiện khuyết tật. "
        f"Loại sản phẩm: {row['category']}. Đạt tiêu chuẩn KCS."
    )


def export_jsonl(df, jsonl_path, image_output_dir):
    """Write instruction-tuning JSONL and copy preprocessed images."""
    os.makedirs(image_output_dir, exist_ok=True)
    records = []

    for idx, row in df.iterrows():
        img_name = f"{row['category']}_{row['defect_type']}_{idx:05d}.png"
        out_img = os.path.join(image_output_dir, img_name)

        if not convert_to_rgb(row["path"], out_img):
            continue

        record = {
            "id": f"{row['category']}_{idx:05d}",
            "image": img_name,
            "conversations": [
                {"from": "human", "value": INSTRUCTION_PROMPT},
                {"from": "gpt", "value": format_label(row)},
            ],
        }
        records.append(record)

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  → {len(records)} samples exported to {jsonl_path}")
    return len(records)


def build_dataset(data_dir, output_dir):
    """
    Full preprocessing pipeline:
    1. Scan MVTec AD directory
    2. Stratified 80/20 split (preserves defect ratio in both sets)
    3. Convert grayscale → RGB, resize to 336×336
    4. Export train.jsonl + test.jsonl
    """
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)

    # Auto-discover MVTec AD root on Kaggle if provided path doesn't exist
    if not data_dir.exists():
        print(f"Path '{data_dir}' not found. Attempting auto-discovery...")
        discovered = auto_discover_mvtec("/kaggle/input")
        if discovered:
            data_dir = discovered
        else:
            print(
                "ERROR: Could not auto-discover MVTec AD dataset.\n"
                "Please check your Kaggle input and specify --data_dir manually.\n"
                "Run in a Kaggle cell:  !find /kaggle/input -maxdepth 3 -name 'bottle' -type d"
            )
            return

    print(f"Scanning MVTec AD directory: {data_dir}")
    df = scan_mvtec_directory(data_dir)

    if len(df) == 0:
        print(
            "WARNING: No images found. Verify data_dir points to the MVTec AD root.\n"
            "Expected structure: data_dir/<category>/train/good/ and test/<defect_type>/"
        )
        return

    n_good = (df["label"] == 0).sum()
    n_defect = (df["label"] == 1).sum()
    print(f"Found {len(df)} images ({n_good} good, {n_defect} defect) "
          f"across {df['category'].nunique()} categories")

    # Stratified split: 80% train, 20% test
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(splitter.split(df, df["label"]))

    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    train_defect_ratio = df_train["label"].mean()
    test_defect_ratio = df_test["label"].mean()
    print(f"Stratified split: Train={len(df_train)} (defect {train_defect_ratio:.1%}), "
          f"Test={len(df_test)} (defect {test_defect_ratio:.1%})")

    os.makedirs(output_dir, exist_ok=True)
    print("\nExporting training set...")
    export_jsonl(df_train, output_dir / "train.jsonl", output_dir / "images" / "train")
    print("Exporting test set...")
    export_jsonl(df_test, output_dir / "test.jsonl", output_dir / "images" / "test")

    print(f"\nDataset ready at {output_dir}/")
    print("Next: zip the processed/ folder and upload to Kaggle Datasets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess MVTec AD dataset for LLaVA QLoRA fine-tuning"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/raw/mvtec_ad",
        help="Path to MVTec AD root directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/processed",
        help="Output directory for JSONL and processed images"
    )
    args = parser.parse_args()
    build_dataset(args.data_dir, args.output_dir)
