"""
QLoRA Fine-tuning of LLaVA 1.5-7B for Industrial Defect Detection
===================================================================
4-bit NormalFloat (NF4) quantization compresses 7B params from ~14GB to ~4GB VRAM.
LoRA adapters target the LLM's Q-K-V-O self-attention projections.
CLIP Vision Encoder remains frozen.

Designed for Kaggle GPU T4x2 (2×16GB VRAM).

Outputs saved to /kaggle/working/results/:
  - training_log.json   : Full training history (loss per step)
  - training_config.json: Hyperparameters used
  - lora_weights/       : LoRA adapter weights

Usage:
    !python vlm-industrial-finetuner/src/train.py \
        --dataset /kaggle/input/<dataset> \
        --output_dir /kaggle/working/lora_weights
"""

import os
import json
import time
import torch
import argparse
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Try importing wandb; if unavailable, disable it gracefully
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

RESULTS_DIR = "/kaggle/working/results"


# ─── Dataset ───────────────────────────────────────────────────────────────────

class MVTecInstructDataset(Dataset):
    """
    Multimodal instruction-tuning dataset for LLaVA.
    Reads JSONL with image paths and conversation-format labels.
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
        assistant_prefix = "ASSISTANT:"
        text = f"USER: {user_msg}\n{assistant_prefix} {assistant_msg}"

        encoding = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
        )
        item_dict = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # Causal LM requires `labels` (shifted internally by the model)
        labels = item_dict["input_ids"].clone()
        
        # Mask the prompt with -100 so it doesn't contribute to the loss
        input_ids_list = item_dict["input_ids"].tolist()
        assistant_tokens = self.processor.tokenizer.encode(assistant_prefix, add_special_tokens=False)

        for i in range(len(input_ids_list) - len(assistant_tokens)):
            if input_ids_list[i:i+len(assistant_tokens)] == assistant_tokens:
                labels[:i + len(assistant_tokens)] = -100
                break

        # Ensure padding doesn't contribute to loss
        if self.processor.tokenizer.pad_token_id is not None:
            labels[labels == self.processor.tokenizer.pad_token_id] = -100
            
        item_dict["labels"] = labels
        return item_dict


# ─── Training ──────────────────────────────────────────────────────────────────

class WandbArtifactCallback(TrainerCallback):
    """Uploads checkpoints to WandB periodically to prevent Kaggle data loss."""
    def on_save(self, args, state, control, **kwargs):
        if HAS_WANDB and wandb.run is not None:
            chkpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            if os.path.exists(chkpt_dir):
                print(f"[WANDB] 🚀 Syncing {chkpt_dir} to WandB cloud...")
                artifact = wandb.Artifact(
                    name=f"run-{wandb.run.id}-checkpoint-{state.global_step}",
                    type="model-checkpoint",
                    description=f"Automated checkpoint sync for step {state.global_step}"
                )
                artifact.add_dir(chkpt_dir)
                wandb.run.log_artifact(artifact, aliases=[f"step_{state.global_step}", "latest"])

def train(args):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    start_time = time.time()

    # ── Save config immediately ──
    config_path = os.path.join(RESULTS_DIR, "training_config.json")
    config_data = {
        "timestamp": datetime.now().isoformat(),
        "base_model": "llava-hf/llava-1.5-7b-hf",
        "quantization": "4-bit NF4 (double quant, fp16 compute)",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "optimizer": "paged_adamw_8bit",
        "learning_rate": args.lr,
        "lr_scheduler": "cosine",
        "warmup_ratio": 0.03,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "max_seq_length": args.max_seq_length,
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"[LOG] Saved training config -> {config_path}")

    # ── wandb ──
    report_to = "none"
    if HAS_WANDB:
        try:
            wandb.init(
                project="vlm-industrial-finetuner",
                name=f"qlora-llava-mvtec-r{args.lora_r}-lr{args.lr}",
                config=config_data,
            )
            report_to = "wandb"
            print("[LOG] W&B initialized successfully.")
        except Exception as e:
            print(f"[WARN] W&B init failed ({e}), logging to disk only.")

    # 4-bit NF4 Quantization
    print("[1/5] Configuring 4-bit NormalFloat quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Base Model
    print("[2/5] Loading LLaVA 1.5-7B...")
    model_id = "llava-hf/llava-1.5-7b-hf"
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA — target all 4 self-attention projections in the LLM
    print("[3/5] Injecting LoRA adapters (r={}, α={}) into Q-K-V-O projections...".format(
        args.lora_r, args.lora_alpha
    ))
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Datasets
    print("[4/5] Loading datasets...")
    dataset_root = args.dataset
    train_dataset = MVTecInstructDataset(
        jsonl_path=os.path.join(dataset_root, "train.jsonl"),
        image_dir=os.path.join(dataset_root, "images", "train"),
        processor=processor,
        max_length=args.max_seq_length,
    )
    val_dataset = MVTecInstructDataset(
        jsonl_path=os.path.join(dataset_root, "test.jsonl"),
        image_dir=os.path.join(dataset_root, "images", "test"),
        processor=processor,
        max_length=args.max_seq_length,
    )
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Training Arguments
    log_dir = os.path.join(RESULTS_DIR, "hf_logs")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=32,
        optim="paged_adamw_8bit",
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=True,
        report_to=report_to,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # Trainer
    print("[5/5] Starting training...")
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            WandbArtifactCallback()
        ],
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ── Save LoRA adapter ──
    print(f"Saving LoRA adapter to {args.output_dir}/")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)

    # ── Export training log history as JSON ──
    train_log_path = os.path.join(RESULTS_DIR, "training_log.json")
    log_history = [entry for entry in trainer.state.log_history]
    elapsed = time.time() - start_time

    log_output = {
        "timestamp": datetime.now().isoformat(),
        "total_training_time_seconds": round(elapsed, 1),
        "total_training_time_human": f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m {int(elapsed % 60)}s",
        "total_steps": trainer.state.global_step,
        "best_eval_loss": trainer.state.best_metric,
        "log_history": log_history,
    }
    with open(train_log_path, "w", encoding="utf-8") as f:
        json.dump(log_output, f, indent=2, ensure_ascii=False)
    print(f"[LOG] Saved training log ({len(log_history)} entries) -> {train_log_path}")

    # ── Finish wandb ──
    if HAS_WANDB and report_to == "wandb":
        wandb.finish()
        print("[LOG] W&B run finished and synced.")

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Duration     : {log_output['total_training_time_human']}")
    print(f"  Best Val Loss: {trainer.state.best_metric:.4f}")
    print(f"  Results Dir  : {RESULTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/lora_weights")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint directory to resume from")
    args = parser.parse_args()
    train(args)
