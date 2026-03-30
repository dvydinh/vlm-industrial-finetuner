"""
Evaluation pipeline for LLaVA industrial defect detection.
Supports two modes:
  --baseline : Zero-shot evaluation (base model without fine-tuning)
  default    : Fine-tuned evaluation (base model + merged LoRA adapter)

All results are automatically saved to /kaggle/working/results/:
  - eval_baseline.json  OR  eval_finetuned.json   : Full metrics
  - eval_baseline.csv   OR  eval_finetuned.csv    : Item-wise table
  - eval_baseline_samples.json                     : Raw predictions

Usage:
    !python src/evaluate.py --baseline --test_data /path/to/processed
    !python src/evaluate.py --model_dir /path/to/lora_weights --test_data /path/to/processed
"""

import os
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


def classify_response(text):
    """Parse model response into binary label: 1=defect, 0=good."""
    text = text.lower()
    defect_kw = ["lỗi", "nứt", "crack", "xước", "scratch", "loại bỏ",
                 "defect", "khuyết", "dent", "contamination", "reject",
                 "anomal", "broken", "damage"]
    good_kw = ["sạch", "đạt", "không phát hiện", "good", "pass",
               "tiêu chuẩn", "normal", "no defect", "clean"]

    d = sum(1 for kw in defect_kw if kw in text)
    g = sum(1 for kw in good_kw if kw in text)
    return 1 if d > g else 0


def run_evaluation(processor, model, test_data_dir, label="", is_baseline=True):
    """Run inference on the test JSONL and compute metrics. Auto-saves all results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tag = "baseline" if is_baseline else "finetuned"
    start_time = time.time()

    test_jsonl = os.path.join(test_data_dir, "test.jsonl")
    image_dir = os.path.join(test_data_dir, "images", "test")

    with open(test_jsonl, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    print(f"\nRunning {label} evaluation on {len(samples)} test samples...\n")

    y_true_all, y_pred_all = [], []
    category_metrics = defaultdict(lambda: {"y_true": [], "y_pred": []})
    sample_predictions = []  # Raw predictions for audit trail

    for item in tqdm(samples, desc=f"{label} Inference"):
        img_path = os.path.join(image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        gt_text = item["conversations"][1]["value"]
        y_t = classify_response(gt_text)
        y_true_all.append(y_t)

        # Parse category from id (e.g. "metal_nut_00124" -> "metal_nut")
        cat_name = "_".join(item["id"].split("_")[:-1])
        category_metrics[cat_name]["y_true"].append(y_t)

        prompt = (
            f"USER: <image>\n"
            f"Với tư cách là kỹ sư KCS, hãy phân tích bề mặt linh kiện [{cat_name}] trong ảnh này.\n"
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

        y_p = classify_response(response)
        y_pred_all.append(y_p)
        category_metrics[cat_name]["y_pred"].append(y_p)

        # Save raw prediction for audit
        sample_predictions.append({
            "id": item["id"],
            "category": cat_name,
            "image": item["image"],
            "ground_truth": "defect" if y_t == 1 else "good",
            "prediction": "defect" if y_p == 1 else "good",
            "correct": y_t == y_p,
            "model_response": response[:300],  # Truncate for readability
        })

    elapsed = time.time() - start_time

    # ── Global Metrics ──
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(
        y_true_all, y_pred_all, target_names=["Good", "Defect"], digits=4, zero_division=0
    )

    # ── Print to console ──
    print("\n" + "=" * 60)
    print(f"  {label} OVERALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Samples  : {len(y_true_all)}")
    print(f"  Overall F1 (Ma): {f1:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"  Duration       : {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print(f"\n{report}")
    print("Confusion Matrix:")
    print(cm)

    print("\n" + "=" * 60)
    print(f"  {label} ITEM-WISE F1-SCORE (MACRO)")
    print("=" * 60)
    print(f"  {'Item Type':<20} | {'Samples':<10} | {'F1-Score':<10}")
    print("-" * 46)

    cat_f1s = {}
    for cat, data in sorted(category_metrics.items()):
        cat_y_true = data["y_true"]
        cat_y_pred = data["y_pred"]
        cat_f1 = f1_score(cat_y_true, cat_y_pred, average="macro")
        cat_f1s[cat] = round(cat_f1, 4)
        print(f"  {cat:<20} | {len(cat_y_true):<10} | {cat_f1:.4f}")
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
        "overall": {
            "f1_macro": round(f1, 4),
            "precision_macro": round(precision, 4),
            "recall_macro": round(recall, 4),
        },
        "confusion_matrix": {
            "true_good_pred_good": int(cm[0][0]) if cm.shape[0] > 0 else 0,
            "true_good_pred_defect": int(cm[0][1]) if cm.shape[0] > 0 and cm.shape[1] > 1 else 0,
            "true_defect_pred_good": int(cm[1][0]) if cm.shape[0] > 1 else 0,
            "true_defect_pred_defect": int(cm[1][1]) if cm.shape[0] > 1 and cm.shape[1] > 1 else 0,
        },
        "item_wise_f1": cat_f1s,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(eval_report, f, indent=2, ensure_ascii=False)
    print(f"\n[LOG] Saved evaluation report -> {json_path}")

    # 2. Item-wise CSV table (for easy import to Excel/Google Sheets)
    csv_path = os.path.join(RESULTS_DIR, f"eval_{tag}_itemwise.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["item_type", "samples", "f1_score"])
        for cat, data in sorted(category_metrics.items()):
            writer.writerow([cat, len(data["y_true"]), cat_f1s[cat]])
        writer.writerow(["OVERALL", len(y_true_all), round(f1, 4)])
    print(f"[LOG] Saved item-wise CSV   -> {csv_path}")

    # 3. Raw sample predictions (JSON) - Full audit trail
    samples_path = os.path.join(RESULTS_DIR, f"eval_{tag}_samples.json")
    with open(samples_path, "w", encoding="utf-8") as f:
        json.dump(sample_predictions, f, indent=2, ensure_ascii=False)
    print(f"[LOG] Saved {len(sample_predictions)} sample predictions -> {samples_path}")

    print(f"\n{'='*60}")
    print(f"  ALL RESULTS SAVED TO: {RESULTS_DIR}")
    print(f"{'='*60}\n")

    return {"f1_macro": f1, "cat_f1s": cat_f1s, "precision": precision, "recall": recall}


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
