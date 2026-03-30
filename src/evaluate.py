"""
Evaluation pipeline for LLaVA industrial defect detection (Visual Grounding).
Supports two modes:
  --baseline : Zero-shot evaluation (base model without fine-tuning)
  default    : Fine-tuned evaluation (base model + merged LoRA adapter)

Evaluation criteria (Visual Grounding):
  A prediction is True Positive only if:
    1. Defect class matches ground truth, AND
    2. IoU(predicted_bbox, gt_bbox) > 0.5
  Good samples: TP if model correctly predicts no defect.

All results are automatically saved to /kaggle/working/results/:
  - eval_baseline.json  OR  eval_finetuned.json   : Full metrics
  - eval_baseline.csv   OR  eval_finetuned.csv    : Item-wise table
  - eval_baseline_samples.json                     : Raw predictions

Usage:
    !python src/evaluate.py --baseline --test_data /path/to/processed
    !python src/evaluate.py --model_dir /path/to/lora_weights --test_data /path/to/processed
"""

import os
import re
import csv
import json
import time
import torch
import argparse
import numpy as np
from datetime import datetime
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel

RESULTS_DIR = "/kaggle/working/results"

# IoU threshold for a detection to count as True Positive
IOU_THRESHOLD = 0.5


# ─── Model Loading ────────────────────────────────────────────────────────────


def load_base_model(base_model_id="llava-hf/llava-1.5-7b-hf"):
    """Load base LLaVA model for zero-shot evaluation."""
    print(f"Loading base model (zero-shot): {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    model.eval()
    return processor, model


def load_finetuned_model(model_dir, base_model_id="llava-hf/llava-1.5-7b-hf"):
    """Load base model and merge trained LoRA adapter."""

    # Auto-detect if user interrupted training and weights are in a checkpoint subdir
    if not os.path.exists(os.path.join(model_dir, "adapter_config.json")):
        print(f"[WARN] adapter_config.json not found in {model_dir}.")
        subdirs = [os.path.join(model_dir, d) for d in os.listdir(model_dir) if d.startswith("checkpoint")]
        if subdirs:
            latest_checkpoint = max(subdirs, key=lambda x: int(x.split("-")[-1]))
            print(f"[SYSTEM] Auto-redirecting to latest checkpoint: {latest_checkpoint}")
            model_dir = latest_checkpoint
        else:
            raise FileNotFoundError(f"No LoRA adapter or checkpoints found in {model_dir}")

    print(f"Loading base model: {base_model_id}")
    processor = AutoProcessor.from_pretrained(base_model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    print(f"Merging LoRA adapter from: {model_dir}")
    model = PeftModel.from_pretrained(model, model_dir)
    model = model.merge_and_unload()
    model.eval()
    return processor, model


# ─── Bounding Box Parsing & IoU ──────────────────────────────────────────────


def parse_bbox(text):
    """
    Extract bounding box coordinates [ymin, xmin, ymax, xmax] from model output.

    Supports formats:
        "at [45, 120, 200, 280]"
        "[45, 120, 200, 280]"

    Returns:
        (ymin, xmin, ymax, xmax) as integers, or None if no bbox found.
    """
    pattern = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
    match = re.search(pattern, text)
    if match:
        return tuple(int(x) for x in match.groups())
    return None


def parse_defect_class(text):
    """
    Extract defect class name from model output.

    Supports formats:
        "Defect detected: [scratch] at ..."
        "Defect detected: [broken_large] at ..."

    Returns:
        Class name string, or None if no match.
    """
    pattern = r"Defect detected:\s*\[(\w+)\]"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) for two axis-aligned bounding boxes.

    Args:
        box_a: (ymin, xmin, ymax, xmax)
        box_b: (ymin, xmin, ymax, xmax)

    Returns:
        IoU value in [0.0, 1.0].
    """
    y1 = max(box_a[0], box_b[0])
    x1 = max(box_a[1], box_b[1])
    y2 = min(box_a[2], box_b[2])
    x2 = min(box_a[3], box_b[3])

    inter_area = max(0, y2 - y1) * max(0, x2 - x1)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union_area = area_a + area_b - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


# ─── Response Classification ─────────────────────────────────────────────────


def classify_response(text):
    """Parse model response into binary label: 1=defect, 0=good."""
    text = text.lower()
    defect_kw = ["defect", "crack", "scratch", "dent", "contamination",
                 "reject", "anomal", "broken", "damage"]
    good_kw = ["clean", "passed qa", "good", "pass", "normal", "no defect"]

    d = sum(1 for kw in defect_kw if kw in text)
    g = sum(1 for kw in good_kw if kw in text)
    return 1 if d > g else 0


# ─── Evaluation ──────────────────────────────────────────────────────────────


def run_evaluation(processor, model, test_data_dir, label="", is_baseline=True):
    """
    Run inference on the test JSONL and compute visual grounding metrics.

    Evaluation uses a strict criterion:
      - For defective samples: TP requires correct class AND IoU > 0.5
      - For good samples: TP requires correct "no defect" prediction
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "baseline" if is_baseline else "finetuned"
    start_time = time.time()

    test_jsonl = os.path.join(test_data_dir, "test.jsonl")
    image_dir = os.path.join(test_data_dir, "images", "test")

    with open(test_jsonl, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    print(f"\nRunning {label} evaluation on {len(samples)} test samples...\n")

    y_true_all, y_pred_all = [], []
    # Strict predictions: accounts for IoU + class match
    y_pred_strict = []
    category_metrics = defaultdict(lambda: {"y_true": [], "y_pred": [], "y_pred_strict": []})
    sample_predictions = []
    iou_scores = []  # Track IoU for defective samples with bbox

    for item in tqdm(samples, desc=f"{label} Inference"):
        img_path = os.path.join(image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        gt_text = item["conversations"][1]["value"]
        y_t = classify_response(gt_text)
        y_true_all.append(y_t)

        # Parse ground truth bbox and class from JSONL
        gt_bbox = tuple(item["gt_bbox"]) if item.get("gt_bbox") else None
        gt_class = item.get("gt_class", "good")

        # Parse category from id (e.g. "metal_nut_00124" -> "metal_nut")
        cat_name = "_".join(item["id"].split("_")[:-1])
        category_metrics[cat_name]["y_true"].append(y_t)

        prompt = (
            f"USER: <image>\n"
            f"Act as a Quality Assurance Engineer, analyze the surface of the "
            f"[{cat_name}] component in this image. "
            f"If a defect is found, report its type and bounding box coordinates "
            f"[ymin, xmin, ymax, xmax].\n"
            f"ASSISTANT:"
        )
        inputs = processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(model.device, torch.float16)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = processor.decode(output[0], skip_special_tokens=True)

        if "ASSISTANT:" in response:
            response = response.split("ASSISTANT:")[-1].strip()

        # Basic binary classification
        y_p = classify_response(response)
        y_pred_all.append(y_p)
        category_metrics[cat_name]["y_pred"].append(y_p)

        # Strict visual grounding evaluation
        pred_bbox = parse_bbox(response)
        pred_class = parse_defect_class(response)
        iou = 0.0
        strict_correct = False

        if y_t == 0:
            # Good sample: strict TP if model correctly says no defect
            strict_correct = (y_p == 0)
            y_p_strict = 0 if strict_correct else 1
        elif y_t == 1:
            # Defective sample: require class match + IoU > threshold
            if y_p == 1 and gt_bbox is not None and pred_bbox is not None:
                iou = compute_iou(pred_bbox, gt_bbox)
                iou_scores.append(iou)
                class_match = (pred_class == gt_class) if pred_class else False
                strict_correct = class_match and (iou > IOU_THRESHOLD)
            elif y_p == 1 and gt_bbox is None:
                # No ground truth bbox available - fall back to class-only check
                class_match = (pred_class == gt_class) if pred_class else False
                strict_correct = class_match
            y_p_strict = 0 if not strict_correct else 1

        y_pred_strict.append(y_p_strict)
        category_metrics[cat_name]["y_pred_strict"].append(y_p_strict)

        # Save raw prediction for audit
        sample_predictions.append({
            "id": item["id"],
            "category": cat_name,
            "image": item["image"],
            "ground_truth": "defect" if y_t == 1 else "good",
            "gt_class": gt_class,
            "gt_bbox": list(gt_bbox) if gt_bbox else None,
            "prediction": "defect" if y_p == 1 else "good",
            "pred_class": pred_class,
            "pred_bbox": list(pred_bbox) if pred_bbox else None,
            "iou": round(iou, 4) if gt_bbox and pred_bbox else None,
            "strict_correct": strict_correct,
            "model_response": response[:300],
        })

    elapsed = time.time() - start_time

    # ── Global Metrics (Basic Classification) ──
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    recall_val = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(
        y_true_all, y_pred_all, target_names=["Good", "Defect"], digits=4, zero_division=0
    )

    # ── Strict Grounding Metrics (Class + IoU > 0.5) ──
    f1_strict = f1_score(y_true_all, y_pred_strict, average="macro")
    precision_strict = precision_score(y_true_all, y_pred_strict, average="macro", zero_division=0)
    recall_strict = recall_score(y_true_all, y_pred_strict, average="macro", zero_division=0)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    # ── Print to console ──
    print("\n" + "=" * 60)
    print(f"  {label} OVERALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Samples  : {len(y_true_all)}")
    print(f"  --- Basic Classification ---")
    print(f"  F1 (Macro)     : {f1:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall_val:.4f}")
    print(f"  --- Visual Grounding (Class + IoU>{IOU_THRESHOLD}) ---")
    print(f"  F1 (Strict)    : {f1_strict:.4f}")
    print(f"  Precision (S)  : {precision_strict:.4f}")
    print(f"  Recall (S)     : {recall_strict:.4f}")
    print(f"  Mean IoU       : {mean_iou:.4f} (over {len(iou_scores)} bbox pairs)")
    print(f"  Duration       : {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"\n{report}")
    print("Confusion Matrix (basic):")
    print(cm)

    print("\n" + "=" * 60)
    print(f"  {label} ITEM-WISE F1-SCORE")
    print("=" * 60)
    print(f"  {'Item Type':<20} | {'Samples':<10} | {'F1 Basic':<10} | {'F1 Strict':<10}")
    print("-" * 60)

    cat_f1s = {}
    cat_f1s_strict = {}
    for cat, data in sorted(category_metrics.items()):
        cat_y_true = data["y_true"]
        cat_y_pred = data["y_pred"]
        cat_y_pred_s = data["y_pred_strict"]
        cat_f1 = f1_score(cat_y_true, cat_y_pred, average="macro")
        cat_f1_s = f1_score(cat_y_true, cat_y_pred_s, average="macro")
        cat_f1s[cat] = round(cat_f1, 4)
        cat_f1s_strict[cat] = round(cat_f1_s, 4)
        print(f"  {cat:<20} | {len(cat_y_true):<10} | {cat_f1:.4f}     | {cat_f1_s:.4f}")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # AUTO-SAVE ALL RESULTS TO DISK
    # ══════════════════════════════════════════════════════════════

    # 1. Full evaluation report (JSON) - Machine readable
    json_path = os.path.join(RESULTS_DIR, f"eval_{tag}.json")
    eval_report = {
        "timestamp": datetime.now().isoformat(),
        "mode": label,
        "duration_seconds": round(elapsed, 1),
        "total_samples": len(y_true_all),
        "iou_threshold": IOU_THRESHOLD,
        "overall_basic": {
            "f1_macro": round(f1, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall_val, 4),
        },
        "overall_strict": {
            "f1_macro": round(f1_strict, 4),
            "precision_macro": round(precision_strict, 4),
            "recall_macro": round(recall_strict, 4),
            "mean_iou": round(mean_iou, 4),
            "bbox_pairs_evaluated": len(iou_scores),
        },
        "confusion_matrix": {
            "true_good_pred_good": int(cm[0][0]) if cm.shape[0] > 0 else 0,
            "true_good_pred_defect": int(cm[0][1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
            "true_defect_pred_good": int(cm[1][0]) if cm.shape[0] > 1 else 0,
            "true_defect_pred_defect": int(cm[1][1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
        },
        "item_wise_f1_basic": cat_f1s,
        "item_wise_f1_strict": cat_f1s_strict,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"\n[LOG] Saved evaluation report -> {json_path}")

    # 2. Item-wise CSV table (for easy import to Excel/Google Sheets)
    csv_path = os.path.join(RESULTS_DIR, f"eval_{tag}_itemwise.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["item_type", "samples", "f1_basic", "f1_strict"])
        for cat, data in sorted(category_metrics.items()):
            writer.writerow([cat, len(data["y_true"]), cat_f1s[cat], cat_f1s_strict[cat]])
        writer.writerow(["OVERALL", len(y_true_all), round(f1, 4), round(f1_strict, 4)])
    print(f"[LOG] Saved item-wise CSV   -> {csv_path}")

    # 3. Raw sample predictions (JSON) - Full audit trail
    samples_path = os.path.join(RESULTS_DIR, f"eval_{tag}_samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(sample_predictions, f, indent=2, ensure_ascii=False)
    print(f"[LOG] Saved {len(sample_predictions)} sample predictions -> {samples_path}")

    print(f"\n{'='*60}")
    print(f"  ALL RESULTS SAVED TO: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    return {
        "f1_macro": f1,
        "f1_strict": f1_strict,
        "mean_iou": mean_iou,
        "cat_f1s": cat_f1s,
        "precision": precision,
        "recall": recall_val,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", action="store_true",
                        help="Run zero-shot baseline (no LoRA)")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to LoRA adapter (ignored if --baseline)")
    parser.add_argument("--test_data", type=str, required=True,
                        help="Path to processed dataset")
    parser.add_argument("--base_model_id", type=str,
                        default="llava-hf/llava-1.5-7b-hf")
    args = parser.parse_args()

    if args.baseline:
        processor, model = load_base_model(args.base_model_id)
        run_evaluation(processor, model, args.test_data,
                       label="ZERO-SHOT BASELINE", is_baseline=True)
    else:
        if not args.model_dir:
            parser.error("--model_dir is required when not using --baseline")
        processor, model = load_finetuned_model(args.model_dir, args.base_model_id)
        run_evaluation(processor, model, args.test_data,
                       label="FINE-TUNED MODEL", is_baseline=False)
