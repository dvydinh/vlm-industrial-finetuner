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
from huggingface_hub import login
from collections import defaultdict
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

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
        "Detected [scratch] at ..."
        "Detected [broken_large] at ..."

    Returns:
        Class name string, or None if no match.
    """
    pattern = r"Detected\s*\[(\w+)\]"
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    return None


def normalize_bbox(bbox, is_baseline=False):
    """
    Normalize bounding box coordinates to relative float values [0.0, 1.0].
    - Baseline models intrinsically answer on a [0, 999] scale.
    - Ground Truth & Finetuned models use [0, 335] scale.
    """
    if bbox is None:
        return None
    # Data pipeline natively targets 999 scale now for both models
    scale = 1000.0
    return tuple(v / scale for v in bbox)


def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union (IoU) for two Bounding Boxes.

    Args:
        box_a: (ymin, xmin, ymax, xmax) normalized in [0.0, 1.0]
        box_b: (ymin, xmin, ymax, xmax) normalized in [0.0, 1.0]

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
    # 1. Priority: If model outputs fine-tuned format (contains "Detected [class]")
    parsed_class = parse_defect_class(text)
    if parsed_class is not None:
        return 1
        
    # 2. If model clearly states no defect found
    text_lower = text.lower()
    if "passed qa" in text_lower or "no defect" in text_lower:
        return 0

    # 3. Fallback: Used for Baseline Zero-shot (extended keyword counting)
    defect_kw = ["defect", "crack", "scratch", "dent", "contamination",
                 "reject", "anomal", "broken", "damage", "color", "cut", "hole", "thread", "print", "wire"]
    good_kw = ["clean", "good", "pass", "normal", "perfect"]

    d = sum(1 for kw in defect_kw if kw in text_lower)
    g = sum(1 for kw in good_kw if kw in text_lower)
    
    return 1 if d > g else 0


# ─── Evaluation ──────────────────────────────────────────────────────────────


def sliding_window_inference(image, model, processor, prompt, is_baseline=False, crop_size=336, stride=224):
    """
    Perform batched sliding window inference on a potentially high-resolution image.
    Uses Non-Maximum Suppression (NMS) on bounding boxes.
    Returns array of detected defect dicts.
    """
    img_w, img_h = image.size
    
    crops = []
    coords = []
    
    for y_start in range(0, img_h, stride):
        for x_start in range(0, img_w, stride):
            y_end = min(img_h, y_start + crop_size)
            x_end = min(img_w, x_start + crop_size)
            
            if y_end - y_start < crop_size and img_h >= crop_size:
                y_start = img_h - crop_size
            if x_end - x_start < crop_size and img_w >= crop_size:
                x_start = img_w - crop_size
                
            crop_img = image.crop((x_start, y_start, x_start + crop_size, y_start + crop_size))
            crops.append(crop_img)
            coords.append((y_start, x_start))
            
    global_defects = []
    
    batch_size = 8
    for i in range(0, len(crops), batch_size):
        batch_crops = crops[i:i+batch_size]
        batch_coords = coords[i:i+batch_size]
        
        inputs = processor(
            text=[prompt] * len(batch_crops), images=batch_crops, return_tensors="pt"
        ).to(model.device, torch.float16)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            
        responses = processor.batch_decode(outputs, skip_special_tokens=True)
        
        # Memory Management explicitly decoupling refs
        del inputs, outputs
        torch.cuda.empty_cache()
        
        for j, response in enumerate(responses):
            if "ASSISTANT:" in response:
                response = response.split("ASSISTANT:")[-1].strip()
                
            y_start, x_start = batch_coords[j]
            y_p = classify_response(response)
            if y_p == 1:
                local_bbox = parse_bbox(response)
                local_class = parse_defect_class(response)
                
                if local_bbox is not None:
                    norm_local = normalize_bbox(local_bbox, is_baseline=is_baseline)
                    ly1, lx1, ly2, lx2 = [n * crop_size for n in norm_local]
                    gy1, gx1, gy2, gx2 = ly1 + y_start, lx1 + x_start, ly2 + y_start, lx2 + x_start
                    global_defects.append({
                        "class": local_class,
                        "box": (gy1 / img_h, gx1 / img_w, gy2 / img_h, gx2 / img_w),
                        "response": response
                    })
                    
    if not global_defects:
        return []
        
    # MVTec AD natively maps multiple fragmented defect polygons into a single global bounding box
    # If we evaluate via NMS, chunked segments will fail the global IoU metric comparison!
    # Therefore, we strictly union the partial boxes to form the explicit target boundaries frame natively.
    y1 = min(d["box"][0] for d in global_defects)
    x1 = min(d["box"][1] for d in global_defects)
    y2 = max(d["box"][2] for d in global_defects)
    x2 = max(d["box"][3] for d in global_defects)
    
    class_counts = {}
    for d in global_defects:
        if d["class"]:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1
    dom_class = max(class_counts, key=class_counts.get) if class_counts else None
    
    return [{
        "class": dom_class,
        "box": (y1, x1, y2, x2)
    }]


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

    global HAS_WANDB
    # Initialize wandb earlier to log incrementally
    if HAS_WANDB:
        try:
            wandb.init(
                project="vlm-industrial-finetuner",
                name=f"eval-{tag}-{datetime.now().strftime('%Y%m%d-%H%M')}",
                config={
                    "mode": tag,
                    "iou_threshold": IOU_THRESHOLD,
                    "total_samples": len(samples)
                }
            )
        except Exception as e:
            print(f"[WANDB] Failed to initialize: {e}")
            HAS_WANDB = False

    y_true_all, y_pred_all = [], []
    category_metrics = defaultdict(lambda: {"TP": 0, "FP": 0, "FN": 0, "TN": 0})
    sample_predictions = []
    iou_scores = []  # Track IoU for defective samples with bbox
    
    global_TP, global_FP, global_FN, global_TN = 0, 0, 0, 0
    
    # ── MLOPS: AUTO-RESUME LOGIC ──
    samples_backup_path = os.path.join(RESULTS_DIR, f"eval_{tag}_samples_backup.json")
    evaluated_ids = set()

    if os.path.exists(samples_backup_path):
        print(f"\n[INFO] Found backup file at {samples_backup_path}. Resuming evaluation...\n")
        try:
            with open(samples_backup_path, "r", encoding="utf-8") as bf:
                sample_predictions = json.load(bf)
            for p in sample_predictions:
                evaluated_ids.add(p["id"])
                
                cat_name = p.get("category", "_".join(p["id"].split("_")[:-1]))
                y_t = 1 if p["ground_truth"] == "defect" else 0
                y_p = 1 if p["prediction"] == "defect" else 0
                y_true_all.append(y_t)
                y_pred_all.append(y_p)
                
                cat_metrics = category_metrics[cat_name]
                if p.get("TP"):
                    global_TP += 1; cat_metrics["TP"] += 1
                if p.get("FP"):
                    global_FP += 1; cat_metrics["FP"] += 1
                if p.get("FN"):
                    global_FN += 1; cat_metrics["FN"] += 1
                if p.get("TN"):
                    global_TN += 1; cat_metrics["TN"] += 1
                
                if p.get("iou") is not None:
                    iou_scores.append(p["iou"])
            print(f"[INFO] Successfully loaded {len(evaluated_ids)} evaluated samples from backup.")
        except Exception as e:
            print(f"[WARNING] Could not load backup: {e}")

    start_eval_time = time.time()
    MAX_KAGGLE_RUNTIME = 11.5 * 3600  # 11.5 hours in seconds

    for item in tqdm(samples, desc=f"{label} Inference"):
        # ── MLOPS: Graceful Timeout Strategy ──
        if time.time() - start_eval_time > MAX_KAGGLE_RUNTIME:
            print(f"\n[MLOPS] ⏳ WARNING: Reached 11.5 hours of continuous Kaggle runtime!")
            print(f"[MLOPS] 🛑 Gracefully stopping evaluation to allow Kaggle to successfully export the output dataset.")
            print(f"[MLOPS] ♻️ Please start a new session using Notebook 1.2 to resume from this point.\n")
            break
        if item["id"] in evaluated_ids:
            continue

        img_path = os.path.join(image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size

        gt_text = item["conversations"][1]["value"]
        y_t = classify_response(gt_text)
        y_true_all.append(y_t)

        # Parse ground truth bbox and class from JSONL
        gt_bbox = tuple(item["gt_bbox"]) if item.get("gt_bbox") else None
        gt_class = item.get("gt_class", "good")

        # Parse category from id (e.g. "metal_nut_00124" -> "metal_nut")
        cat_name = "_".join(item["id"].split("_")[:-1])
        prompt = (
            f"USER: <image>\n"
            f"Act as a Quality Assurance Engineer, analyze the surface of the "
            f"[{cat_name}] component in this image patch. "
            f"If a defect is found, report its type and bounding box coordinates "
            f"[ymin, xmin, ymax, xmax].\n"
            f"ASSISTANT:"
        )
        # Updated: Batched Sliding Window Inference system
        pred_defects = sliding_window_inference(
            image, model, processor, prompt, is_baseline=is_baseline
        )

        y_p = 1 if len(pred_defects) > 0 else 0
        y_pred_all.append(y_p)
        
        is_tp = False
        is_fn = False
        fp_count = 0
        iou = 0.0
        
        # Object Detection logical constraint solver
        if y_t == 1:
            if gt_bbox is not None:
                norm_gt = (
                    gt_bbox[0] / img_h,
                    gt_bbox[1] / img_w,
                    gt_bbox[2] / img_h,
                    gt_bbox[3] / img_w
                )
                for pd in pred_defects:
                    curr_iou = compute_iou(pd["box"], norm_gt)
                    iou_scores.append(curr_iou)
                    if curr_iou > IOU_THRESHOLD and pd["class"] == gt_class:
                        is_tp = True
                        iou = max(iou, curr_iou)
            else:
                for pd in pred_defects:
                    if pd["class"] == gt_class:
                        is_tp = True
                        
            if is_tp:
                global_TP += 1; category_metrics[cat_name]["TP"] += 1
                fp_count = max(0, len(pred_defects) - 1)
            else:
                global_FN += 1; category_metrics[cat_name]["FN"] += 1
                is_fn = True
                fp_count = len(pred_defects)
                
            global_FP += fp_count; category_metrics[cat_name]["FP"] += fp_count
            
        else: # Good sample
            if len(pred_defects) == 0:
                global_TN += 1; category_metrics[cat_name]["TN"] += 1
            else:
                fp_count = len(pred_defects)
                global_FP += fp_count; category_metrics[cat_name]["FP"] += fp_count

        # Save raw prediction for audit
        sample_predictions.append({
            "id": item["id"],
            "category": cat_name,
            "image": item["image"],
            "ground_truth": "defect" if y_t == 1 else "good",
            "gt_class": gt_class,
            "gt_bbox": list(gt_bbox) if gt_bbox else None,
            "prediction": "defect" if y_p == 1 else "good",
            "model_response": f"Defect: {pred_class} at {norm_pred}" if y_p == 1 else "Good",
        })

        # --- CONTINUOUS SAVING & WANDB LOGGING ---
        idx = len(y_true_all)
        if idx % 50 == 0 or idx == len(samples):
            # Compute current rolling macro f1
            current_f1 = f1_score(y_true_all, y_pred_all, average="macro")
            
            p_s = global_TP / (global_TP + global_FP) if (global_TP + global_FP) > 0 else 0
            r_s = global_TP / (global_TP + global_FN) if (global_TP + global_FN) > 0 else 0
            current_f1_strict = 2 * p_s * r_s / (p_s + r_s) if (p_s + r_s) > 0 else 0
            
            current_mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

            if HAS_WANDB and wandb.run is not None:
                wandb.log({
                    "step": idx,
                    "rolling/f1_basic_macro": current_f1,
                    "rolling/f1_strict_binary": current_f1_strict,
                    "rolling/mean_iou": current_mean_iou
                })

            # Save local backups continuously
            samples_backup_path = os.path.join(RESULTS_DIR, f"eval_{tag}_samples_backup.json")
            with open(samples_backup_path, "w", encoding="utf-8") as f:
                json.dump(sample_predictions, f, indent=2, ensure_ascii=False)

            # Mirror to WANDB Cloud periodically to prevent Kaggle crash erasure
            if HAS_WANDB and wandb.run is not None:
                try:
                    artifact = wandb.Artifact(
                        name=f"run-{wandb.run.id}-eval-{tag}",
                        type="eval-backup"
                    )
                    artifact.add_file(samples_backup_path)
                    wandb.run.log_artifact(artifact, aliases=[f"step_{idx}", "latest"])
                except Exception:
                    pass # silent fail to prevent inference loop crash

    elapsed = time.time() - start_time

    # ── Global Metrics (Basic Classification) ──
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    recall_val = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(
        y_true_all, y_pred_all, target_names=["Good", "Defect"], digits=4, zero_division=0
    )

    # ── Strict Grounding Object Detection Metrics (TP + IoU > 0.5) ──
    def calc_metrics(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        idx_f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, idx_f1

    precision_strict, recall_strict, f1_strict = calc_metrics(global_TP, global_FP, global_FN)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    # ── Print to console ──
    print("\n" + "=" * 60)
    print(f"  {label} OVERALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Samples  : {len(y_true_all)}")
    print(f"  --- Basic Classification (Macro) ---")
    print(f"  F1 (Macro)     : {f1:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall_val:.4f}")
    print(f"  --- Strict Object Detection (Defect Class Only, IoU>{IOU_THRESHOLD}) ---")
    print(f"  F1 (Binary)    : {f1_strict:.4f}")
    print(f"  Precision (S)  : {precision_strict:.4f}  [TP: {global_TP}, FP: {global_FP}]")
    print(f"  Recall (S)     : {recall_strict:.4f}  [FN: {global_FN}]")
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

    cat_f1s_strict = {}
    for cat, data in sorted(category_metrics.items()):
        cat_p, cat_r, cat_f1_s = calc_metrics(data["TP"], data["FP"], data["FN"])
        cat_f1s_strict[cat] = round(cat_f1_s, 4)
        print(f"  {cat:<20} | {data['TP']+data['FP']+data['FN']+data['TN']:<10} | {-1.0:10.4f} | {cat_f1_s:.4f}")
    print("=" * 60)

    # ══════════════════════════════════════════════════════════════
    # AUTO-SAVE ALL RESULTS TO DISK
    # ══════════════════════════════════════════════════════════════

    # 1. Full evaluation report (JSON) - Machine readable
    json_path = os.path.join(RESULTS_DIR, f"eval_{tag}.json")
    
    # Pre-calculate category dictionary
    cat_f1_basic = {}
    
    # Calculate macro basic F1s for categories if needed, but we'll use binary for both now
    for cat, data in sorted(category_metrics.items()):
        # Basic binary mapping logic
        tp_b = data["TP"] + (data["FP"] if IOU_THRESHOLD == 0 else 0) # Fallback heuristic
        # Actual strict is already in cat_f1s_strict
        cat_f1_basic[cat] = -1.0 
        
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
            "f1_binary": round(f1_strict, 4),
            "precision_binary": round(precision_strict, 4),
            "recall_binary": round(recall_strict, 4),
            "mean_iou": round(mean_iou, 4),
            "bbox_pairs_evaluated": len(iou_scores),
            "total_TP": global_TP,
            "total_FP": global_FP,
            "total_FN": global_FN,
        },
        "confusion_matrix_basic": {
            "true_good_pred_good": int(cm[0][0]) if cm.shape[0] > 0 else 0,
            "true_good_pred_defect": int(cm[0][1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
            "true_defect_pred_good": int(cm[1][0]) if cm.shape[0] > 1 else 0,
            "true_defect_pred_defect": int(cm[1][1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
        },
        "item_wise_f1_strict_binary": cat_f1s_strict,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"\n[LOG] Saved evaluation report -> {json_path}")
    
    # 2. Item-wise CSV table (for easy import to Excel/Google Sheets)
    csv_path = os.path.join(RESULTS_DIR, f"eval_{tag}_itemwise.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["item_type", "samples", "f1_strict_binary", "TP", "FP", "FN"])
        for cat, data in sorted(category_metrics.items()):
            samples_cnt = data["TP"] + data["FP"] + data["FN"] + data["TN"]
            writer.writerow([cat, samples_cnt, cat_f1s_strict[cat], data["TP"], data["FP"], data["FN"]])
        writer.writerow(["OVERALL", len(y_true_all), round(f1_strict, 4), global_TP, global_FP, global_FN])
    print(f"[LOG] Saved item-wise CSV   -> {csv_path}")

    # 3. Raw sample predictions (JSON) - Full audit trail
    samples_path = os.path.join(RESULTS_DIR, f"eval_{tag}_samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(sample_predictions, f, indent=2, ensure_ascii=False)
    print(f"[LOG] Saved {len(sample_predictions)} sample predictions -> {samples_path}")

    print(f"\n{'='*60}")
    print(f"  ALL RESULTS SAVED TO: {RESULTS_DIR}")
    print(f"{'='*60}\n")
    
    # ══════════════════════════════════════════════════════════════
    # WANDB CONTINUOUS LOGGING & RESCUE ARTIFACTS
    # ══════════════════════════════════════════════════════════════
    if HAS_WANDB and wandb.run is not None:
        print("\n  [WANDB] Syncing evaluation metrics & artifacts to cloud...")
        try:
            # Log primary metrics
            wandb.log({
                "eval/f1_macro_basic": f1,
                "eval/precision_basic": precision,
                "eval/recall_basic": recall_val,
                "eval/f1_macro_strict": f1_strict,
                "eval/mean_iou": mean_iou
            })
            
            # Pack and upload results folder as artifact
            artifact = wandb.Artifact(
                name=f"eval_{tag}_results",
                type="evaluation-results",
                description=f"CSV & JSON results for {tag} evaluation phase"
            )
            artifact.add_file(csv_path)
            artifact.add_file(json_path)
            if os.path.exists(samples_path):
                artifact.add_file(samples_path)
                
            wandb.run.log_artifact(artifact)
            wandb.finish()
            print("  [WANDB] Sync complete! Your metrics and reports are safe.")
        except Exception as e:
            print(f"  [WANDB] Failed to log gracefully: {e}")

    return {
        "f1_macro_basic": f1,
        "f1_strict_binary": f1_strict,
        "mean_iou": mean_iou,
        "cat_f1s_strict": cat_f1s_strict,
        "precision_basic": precision,
        "recall_basic": recall_val,
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
