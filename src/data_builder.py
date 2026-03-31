"""
MVTec AD Dataset Preprocessing Pipeline (Visual Grounding Edition)
===================================================================
Converts MVTec AD into a supervised instruction-tuning format for LLaVA
with bounding box annotations extracted from ground_truth masks.

Key features:
  - Smart Crop: Extracts a 336x336 window centered on the defect region
  - Random Jitter: Prevents the model from memorizing center-biased patterns
  - Bbox Extraction: Uses cv2.findContours on binary masks
  - Fallback: Good (defect-free) samples use standard full-image resize

Usage:
    python src/data_builder.py --data_dir data/raw/mvtec_ad --output_dir data/processed
"""

import os
import json
import random
import argparse
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


CLIP_RESOLUTION = (336, 336)
CROP_SIZE = 336

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


# ─── Bounding Box Extraction from Ground Truth Masks ──────────────────────────


def extract_bbox_from_mask(mask_path):
    """
    Extract a single bounding box (union of all defect regions) from a binary mask.

    Args:
        mask_path: Path to the ground_truth mask PNG (white=defect, black=background).

    Returns:
        (ymin, xmin, ymax, xmax) in original image coordinates, or None if mask is
        empty or file doesn't exist.
    """
    if mask_path is None or not os.path.exists(str(mask_path)):
        return None

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None

    # Binarize: threshold at 128 to handle anti-aliased edges
    _, binary = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Union bounding box across all contours
    all_points = np.concatenate(contours)
    x, y, w, h = cv2.boundingRect(all_points)

    # Return in (ymin, xmin, ymax, xmax) format
    return (y, x, y + h, x + w)


# ─── Smart Crop with Random Jitter ───────────────────────────────────────────


def smart_crop_with_jitter(image, bbox, crop_size=CROP_SIZE, jitter_ratio=0.15):
    """
    Extract a crop_size x crop_size window from the full-resolution image that
    is guaranteed to contain the entire bounding box, with random positional jitter
    to prevent center-bias memorization.

    Args:
        image: Full-resolution image (H, W, 3) numpy array.
        bbox: (ymin, xmin, ymax, xmax) in original image coordinates.
        crop_size: Target crop dimension (default 336).
        jitter_ratio: Fraction of available slack used for random offset (0.0-1.0).

    Returns:
        (cropped_image, new_bbox) where new_bbox is in crop-local coords clipped to [0, crop_size-1].
    """
    img_h, img_w = image.shape[:2]
    ymin, xmin, ymax, xmax = bbox

    # If the defect bbox is larger than crop_size, resize the image so it fits
    bbox_h = ymax - ymin
    bbox_w = xmax - xmin

    if bbox_h > crop_size or bbox_w > crop_size:
        # Scale down so the largest bbox dimension fits within 80% of crop_size
        scale = (crop_size * 0.8) / max(bbox_h, bbox_w)
        new_h, new_w = int(img_h * scale), int(img_w * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ymin, xmin = int(ymin * scale), int(xmin * scale)
        ymax, xmax = int(ymax * scale), int(xmax * scale)
        img_h, img_w = new_h, new_w

    # Compute the valid range for the top-left corner of the crop window
    # The crop must fully contain the bbox
    crop_y_max_start = ymin  # crop top edge must be at or above ymin
    crop_y_min_start = max(0, ymax - crop_size)  # crop bottom edge must be at or below ymax
    crop_x_max_start = xmin
    crop_x_min_start = max(0, xmax - crop_size)

    # Clamp to image boundaries
    crop_y_min_start = max(crop_y_min_start, 0)
    crop_y_max_start = min(crop_y_max_start, img_h - crop_size)
    crop_x_min_start = max(crop_x_min_start, 0)
    crop_x_max_start = min(crop_x_max_start, img_w - crop_size)

    # Ensure valid range
    crop_y_min_start = min(crop_y_min_start, crop_y_max_start)
    crop_x_min_start = min(crop_x_min_start, crop_x_max_start)

    # Apply random jitter within the valid range
    if crop_y_max_start > crop_y_min_start:
        slack_y = crop_y_max_start - crop_y_min_start
        jitter_y = int(slack_y * jitter_ratio)
        crop_y = random.randint(
            crop_y_min_start + jitter_y,
            crop_y_max_start - jitter_y
        ) if jitter_y < slack_y // 2 else random.randint(crop_y_min_start, crop_y_max_start)
    else:
        crop_y = crop_y_min_start

    if crop_x_max_start > crop_x_min_start:
        slack_x = crop_x_max_start - crop_x_min_start
        jitter_x = int(slack_x * jitter_ratio)
        crop_x = random.randint(
            crop_x_min_start + jitter_x,
            crop_x_max_start - jitter_x
        ) if jitter_x < slack_x // 2 else random.randint(crop_x_min_start, crop_x_max_start)
    else:
        crop_x = crop_x_min_start

    # Handle edge case: image smaller than crop_size
    crop_y = max(0, min(crop_y, img_h - crop_size))
    crop_x = max(0, min(crop_x, img_w - crop_size))

    # If image is smaller than crop_size, pad with zeros
    if img_h < crop_size or img_w < crop_size:
        padded = np.zeros((max(img_h, crop_size), max(img_w, crop_size), 3), dtype=image.dtype)
        padded[:img_h, :img_w] = image
        image = padded
        crop_y = max(0, crop_y)
        crop_x = max(0, crop_x)

    # Extract the crop
    cropped = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]

    # Recalculate bbox in crop-local coordinates and clip to [0, crop_size-1]
    new_ymin = np.clip(ymin - crop_y, 0, crop_size - 1)
    new_xmin = np.clip(xmin - crop_x, 0, crop_size - 1)
    new_ymax = np.clip(ymax - crop_y, 0, crop_size - 1)
    new_xmax = np.clip(xmax - crop_x, 0, crop_size - 1)

    new_bbox = (int(new_ymin), int(new_xmin), int(new_ymax), int(new_xmax))
    return cropped, new_bbox


# ─── Image Processing ────────────────────────────────────────────────────────


def ensure_rgb(img):
    """Convert image to 3-channel RGB for CLIP Vision Encoder compatibility."""
    if img is None:
        return None
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def process_good_image(image_path, output_path):
    """Random crop a defect-free image to 336x336 to match defect texture scale."""
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False
    img = ensure_rgb(img)
    
    h, w = img.shape[:2]
    # Random crop để giữ nguyên scale chi tiết vật liệu giống như ảnh defect
    if h > CROP_SIZE or w > CROP_SIZE:
        y = random.randint(0, max(0, h - CROP_SIZE))
        x = random.randint(0, max(0, w - CROP_SIZE))
        img = img[y:y+CROP_SIZE, x:x+CROP_SIZE]
        
    # Resize nếu ảnh gốc nhỏ hơn 336 (Edge case)
    if img.shape[0] != CROP_SIZE or img.shape[1] != CROP_SIZE:
        img = cv2.resize(img, CLIP_RESOLUTION, interpolation=cv2.INTER_AREA)
        
    cv2.imwrite(str(output_path), img)
    return True


def process_defect_image(image_path, mask_path, output_path):
    """
    Smart-crop a defective image around the defect region with jitter.

    Returns:
        (success: bool, bbox: tuple or None) — bbox in crop-local coordinates.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return False, None

    img = ensure_rgb(img)
    bbox = extract_bbox_from_mask(mask_path)

    if bbox is None:
        # Fallback: mask missing or empty — resize to 336x336 without bbox
        img = cv2.resize(img, CLIP_RESOLUTION, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(output_path), img)
        return True, None

    cropped, new_bbox = smart_crop_with_jitter(img, bbox)

    # Ensure output is exactly crop_size x crop_size
    if cropped.shape[0] != CROP_SIZE or cropped.shape[1] != CROP_SIZE:
        cropped = cv2.resize(cropped, CLIP_RESOLUTION, interpolation=cv2.INTER_AREA)

    cv2.imwrite(str(output_path), cropped)
    return True, new_bbox


# ─── Directory Scanning ──────────────────────────────────────────────────────


def resolve_mask_path(data_dir, category, defect_type, image_stem):
    """
    Resolve the ground_truth mask path for a given test image.
    Tries common MVTec naming conventions: {stem}_mask.png, then {stem}.png.
    """
    gt_dir = Path(data_dir) / category / "ground_truth" / defect_type

    candidates = [
        gt_dir / f"{image_stem}_mask.png",
        gt_dir / f"{image_stem}.png",
    ]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def scan_mvtec_directory(data_dir):
    """
    Scan MVTec AD directory structure including ground_truth masks.
    Expected layout:
        data_dir/
        ├── <category>/
        │   ├── train/good/
        │   ├── test/
        │   │   ├── good/
        │   │   └── <defect>/
        │   └── ground_truth/
        │       └── <defect>/   (binary masks)
    """
    data_dir = Path(data_dir)
    samples = []

    for category_dir in sorted(data_dir.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name

        # Train split: all "good" images (no masks available)
        good_train = category_dir / "train" / "good"
        if good_train.exists():
            for img_path in sorted(good_train.glob("*.*")):
                if img_path.suffix.lower() in (".png", ".jpg", ".bmp"):
                    samples.append({
                        "path": str(img_path),
                        "category": category,
                        "label": 0,
                        "defect_type": "good",
                        "mask_path": None,
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
                        mask_path = None
                        if label == 1:
                            mask_path = resolve_mask_path(
                                data_dir, category, defect_dir.name, img_path.stem
                            )
                        samples.append({
                            "path": str(img_path),
                            "category": category,
                            "label": label,
                            "defect_type": defect_dir.name,
                            "mask_path": mask_path,
                        })

    return pd.DataFrame(samples)


# ─── Label Formatting ────────────────────────────────────────────────────────


def format_label(row, bbox=None):
    """
    Generate instruction-tuning label text with optional bounding box.

    Args:
        row: DataFrame row with 'label', 'defect_type', 'category'.
        bbox: (ymin, xmin, ymax, xmax) in crop-local coords, or None.
    """
    if row["label"] == 1:
        ymin, xmin, ymax, xmax = bbox
        return f"Detected [{row['defect_type']}] at [{ymin}, {xmin}, {ymax}, {xmax}]."
    return "Passed QA. No defects detected."


# ─── JSONL Export ─────────────────────────────────────────────────────────────


def export_jsonl(df, jsonl_path, image_output_dir, is_train=True):
    """Write instruction-tuning JSONL with bbox annotations and preprocessed images."""
    os.makedirs(image_output_dir, exist_ok=True)
    records = []
    bbox_count = 0

    for idx, row in df.iterrows():
        img_name = f"{row['category']}_{row['defect_type']}_{idx:05d}.png"
        out_img = os.path.join(image_output_dir, img_name)

        bbox = None
        if row["label"] == 1:
            if not row.get("mask_path"):
                continue  # Skip if no ground truth mask exists
            if is_train:
                success, bbox = process_defect_image(row["path"], row["mask_path"], out_img)
                if not success or bbox is None:
                    continue  # Skip if bbox extraction fails
            else:
                # Test set evaluation image MUST NOT BE CROPPED to preserve sliding window context
                img = cv2.imread(str(row["path"]), cv2.IMREAD_UNCHANGED)
                img = ensure_rgb(img)
                cv2.imwrite(out_img, img)
                bbox = extract_bbox_from_mask(row["mask_path"])
                if bbox is None: continue
            bbox_count += 1
        else:
            if is_train:
                if not process_good_image(row["path"], out_img):
                    continue
            else:
                # Test set good image MUST NOT BE CROPPED
                img = cv2.imread(str(row["path"]), cv2.IMREAD_UNCHANGED)
                img = ensure_rgb(img)
                cv2.imwrite(out_img, img)

        prompt = (
            "<image>\n"
            f"Act as a Quality Assurance Engineer, analyze the surface of the "
            f"[{row['category']}] component in this image. "
            f"If a defect is found, report its type and bounding box coordinates "
            f"[ymin, xmin, ymax, xmax]."
        )

        record = {
            "id": f"{row['category']}_{idx:05d}",
            "image": img_name,
            "conversations": [
                {"from": "human", "value": prompt},
                {"from": "gpt", "value": format_label(row, bbox)},
            ],
        }

        # Store ground truth bbox for evaluation
        if bbox is not None:
            record["gt_bbox"] = list(bbox)
        record["gt_class"] = row["defect_type"]

        records.append(record)

    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"  -> {len(records)} samples exported to {jsonl_path} "
          f"({bbox_count} with bounding boxes)")
    return len(records)


# ─── Main Pipeline ────────────────────────────────────────────────────────────


def build_dataset(data_dir, output_dir):
    """
    Full preprocessing pipeline:
    1. Scan MVTec AD directory (including ground_truth masks)
    2. Stratified 80/20 split (preserves defect ratio in both sets)
    3. Smart crop defective images around bbox with jitter; resize good images
    4. Export train.jsonl + test.jsonl with bbox annotations
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
    n_masks = df["mask_path"].notna().sum()
    print(f"Found {len(df)} images ({n_good} good, {n_defect} defect) "
          f"across {df['category'].nunique()} categories")
    print(f"  Ground truth masks found: {n_masks}/{n_defect} defective samples")

    # Stratified split by *defect type* to prevent leakage of rare classes
    df_train, df_test = train_test_split(
        df, test_size=0.20, random_state=42, stratify=df["defect_type"]
    )

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_defect_ratio = df_train["label"].mean()
    test_defect_ratio = df_test["label"].mean()
    print(f"Stratified split: Train={len(df_train)} (defect {train_defect_ratio:.1%}), "
          f"Test={len(df_test)} (defect {test_defect_ratio:.1%})")

    os.makedirs(output_dir, exist_ok=True)
    print("\nExporting training set...")
    export_jsonl(df_train, output_dir / "train.jsonl", output_dir / "images" / "train", is_train=True)
    print("Exporting test set...")
    export_jsonl(df_test, output_dir / "test.jsonl", output_dir / "images" / "test", is_train=False)

    print(f"\nDataset ready at {output_dir}/")
    print("Next: zip the processed/ folder and upload to Kaggle Datasets.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess MVTec AD dataset for LLaVA QLoRA fine-tuning (Visual Grounding)"
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
