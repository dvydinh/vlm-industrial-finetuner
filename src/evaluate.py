"""
Evaluation pipeline for LLaVA industrial defect detection.
Supports two modes:
  --baseline : Zero-shot evaluation (base model without fine-tuning)
  default    : Fine-tuned evaluation (base model + merged LoRA adapter)

Usage:
    # Zero-shot baseline
    !python src/evaluate.py --baseline --test_data /path/to/processed

    # Fine-tuned evaluation
    !python src/evaluate.py --model_dir /path/to/lora_weights --test_data /path/to/processed
"""

import os
import json
import torch
import argparse
import numpy as np
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


def run_evaluation(processor, model, test_data_dir, label=""):
    """Run inference on the test JSONL and compute metrics."""
    test_jsonl = os.path.join(test_data_dir, "test.jsonl")
    image_dir = os.path.join(test_data_dir, "images", "test")

    with open(test_jsonl, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    print(f"\nRunning {label} evaluation on {len(samples)} test samples...\n")

    y_true_all, y_pred_all = [], []
    category_metrics = defaultdict(lambda: {"y_true": [], "y_pred": []})

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
        
        y_p = classify_response(response)
        y_pred_all.append(y_p)
        category_metrics[cat_name]["y_pred"].append(y_p)

    # Global Metrics
    f1 = f1_score(y_true_all, y_pred_all, average="macro")
    precision = precision_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    recall = recall_score(y_true_all, y_pred_all, average="macro", zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all)
    report = classification_report(
        y_true_all, y_pred_all, target_names=["Good", "Defect"], digits=4, zero_division=0
    )

    print("\n" + "=" * 60)
    print(f"  {label} OVERALL EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Samples  : {len(y_true_all)}")
    print(f"  Overall F1 (Ma): {f1:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
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
        cat_f1s[cat] = cat_f1
        print(f"  {cat.capitalize():<20} | {len(cat_y_true):<10} | {cat_f1:.4f}")
    print("=" * 60)

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
        run_evaluation(processor, model, args.test_data, label="ZERO-SHOT BASELINE")
    else:
        if not args.model_dir:
            parser.error("--model_dir is required when not using --baseline")
        processor, model = load_finetuned_model(args.model_dir, args.base_model_id)
        run_evaluation(processor, model, args.test_data, label="FINE-TUNED MODEL")
