"""
Evaluation & Inference Pipeline
================================
Loads the base LLaVA 1.5-7B model, merges the trained LoRA adapter,
runs inference on the held-out test split, and computes:
  - F1-Score (macro)
  - Precision / Recall
  - Confusion Matrix

Usage (on Kaggle):
    !python vlm-industrial-finetuner/src/evaluate.py \
        --model_dir /kaggle/working/lora_weights \
        --test_data /kaggle/input/<dataset-name>
"""

import os
import json
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import PeftModel


def load_merged_model(model_dir, base_model_id="llava-hf/llava-1.5-7b-hf"):
    """
    Load the base model and merge the LoRA adapter weights.
    After merging, PEFT is no longer needed at inference time.
    The adapter itself is only ~30-50 MB (vs ~14 GB for the full model).
    """
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
    """
    Parse free-form Vietnamese model response into binary label.
    Returns 1 (defect) or 0 (good).
    """
    text = text.lower()
    defect_kw = ["lỗi", "nứt", "crack", "xước", "scratch", "loại bỏ", "defect", "khuyết"]
    good_kw = ["sạch", "đạt", "không phát hiện", "good", "pass", "tiêu chuẩn"]

    d_score = sum(1 for kw in defect_kw if kw in text)
    g_score = sum(1 for kw in good_kw if kw in text)
    return 1 if d_score > g_score else 0


def evaluate(model_dir, test_data_dir, base_model_id="llava-hf/llava-1.5-7b-hf"):
    """Run full evaluation on the held-out test JSONL."""
    processor, model = load_merged_model(model_dir, base_model_id)

    test_jsonl = os.path.join(test_data_dir, "test.jsonl")
    image_dir = os.path.join(test_data_dir, "images", "test")

    with open(test_jsonl, "r", encoding="utf-8") as fh:
        samples = [json.loads(line) for line in fh if line.strip()]

    print(f"\nRunning inference on {len(samples)} test samples...\n")

    y_true, y_pred = [], []

    for item in tqdm(samples, desc="Evaluating"):
        img_path = os.path.join(image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        # Ground truth
        gt_text = item["conversations"][1]["value"]
        y_true.append(classify_response(gt_text))

        # Model inference
        prompt = (
            "USER: <image>\n"
            "Với tư cách là kỹ sư KCS, hãy phân tích bề mặt linh kiện này.\n"
            "ASSISTANT:"
        )
        inputs = processor(
            text=prompt, images=image, return_tensors="pt"
        ).to(model.device, torch.float16)

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        response = processor.decode(output[0], skip_special_tokens=True)
        y_pred.append(classify_response(response))

    # ── Metrics ────────────────────────────────────────────────────────────────
    f1 = f1_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true, y_pred, target_names=["Good", "Defect"], digits=4
    )

    print("\n" + "=" * 60)
    print("  FINE-TUNED MODEL — EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total Samples  : {len(y_true)}")
    print(f"  F1 (Macro)     : {f1:.4f}")
    print(f"  Precision      : {precision:.4f}")
    print(f"  Recall         : {recall:.4f}")
    print(f"\n{report}")
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)

    return {
        "f1_macro": round(f1, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "confusion_matrix": cm.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned LLaVA on MVTec AD test split"
    )
    parser.add_argument(
        "--model_dir", type=str, required=True,
        help="Path to LoRA adapter weights directory"
    )
    parser.add_argument(
        "--test_data", type=str, required=True,
        help="Path to processed dataset (contains test.jsonl and images/test/)"
    )
    parser.add_argument(
        "--base_model_id", type=str, default="llava-hf/llava-1.5-7b-hf"
    )
    args = parser.parse_args()

    evaluate(args.model_dir, args.test_data, args.base_model_id)
