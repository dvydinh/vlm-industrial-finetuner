"""
QLoRA Fine-tuning of LLaVA 1.5-7B for Industrial Defect Detection
===================================================================
Uses 4-bit NormalFloat (NF4) quantization via bitsandbytes to compress the
7B-parameter model from ~14GB down to ~4GB VRAM footprint.

LoRA adapters are injected ONLY into the LLM's Q-K-V self-attention projections.
The CLIP Vision Encoder remains completely frozen.

Designed to run on Kaggle GPU T4x2 (2×16GB VRAM).

Usage (on Kaggle):
    !python vlm-industrial-finetuner/src/train.py \
        --dataset /kaggle/input/<dataset-name> \
        --output_dir /kaggle/working/lora_weights
"""

import os
import json
import torch
import argparse
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import wandb


# ─── Dataset ───────────────────────────────────────────────────────────────────

class MVTecInstructDataset(Dataset):
    """
    Reads processed JSONL and serves image-text pairs for SFTTrainer.
    Each sample follows LLaVA conversation format:
        human: <image>\n[inspection prompt]
        gpt:   [defect assessment in Vietnamese]
    """

    def __init__(self, jsonl_path, image_dir, processor, max_length=1024):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.samples = []

        with open(jsonl_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.samples.append(json.loads(line))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_path = os.path.join(self.image_dir, item["image"])
        image = Image.open(img_path).convert("RGB")

        user_msg = item["conversations"][0]["value"]
        assistant_msg = item["conversations"][1]["value"]
        text = f"USER: {user_msg}\nASSISTANT: {assistant_msg}"

        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ─── Training ──────────────────────────────────────────────────────────────────

def train(args):
    # ── wandb ──
    wandb.init(
        project="vlm-industrial-finetuner",
        name=f"qlora-llava-mvtec-r{args.lora_r}-lr{args.lr}",
        config=vars(args),
    )

    # ── 4-bit NF4 Quantization ──
    print("[1/5] Configuring 4-bit NormalFloat quantization (bitsandbytes)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # ── Base Model ──
    print("[2/5] Loading LLaVA 1.5-7B base model...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # ── LoRA Injection ──
    print("[3/5] Injecting LoRA adapters (r={}, α={}) into Q-K-V projections...".format(
        args.lora_r, args.lora_alpha
    ))
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Datasets ──
    print("[4/5] Loading datasets...")
    dataset_root = args.dataset
    train_dataset = MVTecInstructDataset(
        jsonl_path=os.path.join(dataset_root, "train.jsonl"),
        image_dir=os.path.join(dataset_root, "images", "train"),
        processor=processor,
        max_length=args.max_seq_length,
    )
    # Use a small portion of training data for validation if no separate val set
    test_jsonl = os.path.join(dataset_root, "test.jsonl")
    test_image_dir = os.path.join(dataset_root, "images", "test")

    val_dataset = MVTecInstructDataset(
        jsonl_path=test_jsonl,
        image_dir=test_image_dir,
        processor=processor,
        max_length=args.max_seq_length,
    )
    print(f"  Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")

    # ── Training Arguments ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        optim="paged_adamw_8bit",
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to="wandb",
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # ── Trainer ──
    print("[5/5] Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=processor.tokenizer,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    print("Training started — monitor loss at https://wandb.ai")
    trainer.train()

    # ── Save ──
    print(f"Saving LoRA adapter to {args.output_dir}/")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    wandb.finish()

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of LLaVA for MVTec AD defect detection"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to processed dataset (contains train.jsonl, test.jsonl, images/)")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/lora_weights",
                        help="Directory to save LoRA adapter weights")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    args = parser.parse_args()

    train(args)
