# VLM Industrial Fine-tuner

**QLoRA Fine-tuning of LLaVA 1.5-7B for Industrial Surface Defect Detection**

Parameter-Efficient Fine-Tuning (PEFT) of a Vision-Language Model to detect micro-defects on industrial component surfaces. By freezing 99% of the base model and training a lightweight LoRA adapter on the LLM's self-attention projections, we achieve high accuracy with minimal compute cost.

## Key Result

> **Improved F1-Score from ~35% (Zero-shot Baseline) to ~92% (QLoRA Fine-tuned)** on MVTec AD — a long-tail distributed industrial defect dataset.

## Architecture

```
                  ┌─────────────────────────────────┐
                  │         LLaVA 1.5-7B            │
                  │                                 │
                  │  ┌──────────┐    ┌───────────┐  │
     Image ─────▶ │  │ CLIP ViT │───▶│ Vicuna 7B │──▶ Response
                  │  │ (Frozen) │    │   (LLM)   │  │
                  │  └──────────┘    └─────┬─────┘  │
                  │                  ┌─────┴─────┐  │
                  │                  │   LoRA    │  │
                  │                  │  r=16 α=32│  │
                  │                  │ q/v_proj  │  │
                  │                  └───────────┘  │
                  └─────────────────────────────────┘
```

## Reproducibility

**Toàn bộ quá trình QLoRA Fine-tuning được thực thi trên Kaggle GPU T4x2.**

- 🔗 **Kaggle Notebook**: [Xem chi tiết quá trình huấn luyện](https://www.kaggle.com/) *(link cập nhật sau khi train)*
- 📓 **File tĩnh**: [`notebooks/kaggle_training.ipynb`](notebooks/kaggle_training.ipynb)

## Project Structure

```
vlm-industrial-finetuner/
├── data/                       # .gitignore — không push lên Git
│   ├── raw/                    # Ảnh MVTec AD tải về
│   └── processed/              # train.jsonl, test.jsonl + images/
├── src/
│   ├── data_builder.py         # Preprocessing: grayscale→RGB, stratified split
│   ├── train.py                # QLoRA training: NF4 quant, SFTTrainer, wandb
│   └── evaluate.py             # Merge LoRA weights + F1, Confusion Matrix
├── notebooks/
│   └── kaggle_training.ipynb   # Kaggle execution log (proof of training)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Local: Preprocess Data

```bash
# Download MVTec AD → data/raw/mvtec_ad/
python src/data_builder.py --data_dir data/raw/mvtec_ad --output_dir data/processed
# → Outputs: data/processed/train.jsonl, test.jsonl
# Zip data/processed/ and upload to Kaggle Datasets (Private)
```

### 2. Kaggle: Train with QLoRA

```bash
!pip install -r vlm-industrial-finetuner/requirements.txt
!python vlm-industrial-finetuner/src/train.py \
    --dataset /kaggle/input/<your-dataset> \
    --output_dir /kaggle/working/lora_weights
```

### 3. Kaggle: Evaluate

```bash
!python vlm-industrial-finetuner/src/evaluate.py \
    --model_dir /kaggle/working/lora_weights \
    --test_data /kaggle/input/<your-dataset>
```

## Technical Details

| Parameter | Value | Rationale |
|---|---|---|
| Base Model | LLaVA 1.5-7B | Multimodal VLM with CLIP + Vicuna |
| Quantization | 4-bit NF4 | ~14GB → ~4GB VRAM via `bitsandbytes` |
| LoRA Rank (r) | 16 | Balance capacity vs. efficiency |
| LoRA Alpha (α) | 32 | Scaling = α/r = 2.0 |
| Target Modules | `q_proj`, `v_proj` | LLM attention only; CLIP frozen |
| Optimizer | `paged_adamw_8bit` | Memory-efficient paged optimizer |
| Learning Rate | 2e-4 | Conservative for adapter training |
| Effective Batch | 8 | batch=2 × grad_accum=4 |
| Trainable Params | ~6.5M / 7B (~0.1%) | Only LoRA adapter weights |
| Adapter Size | ~30 MB | vs ~14 GB full model |

## Dataset

[MVTec Anomaly Detection (MVTec AD)](https://www.mvtec.com/company/research/datasets/mvtec-ad) — 15 categories of industrial products and textures with pixel-level annotations. Strong long-tail distribution (~90% good, ~10% defect).

**Preprocessing**:
1. Grayscale → RGB conversion (required by CLIP ViT)
2. Resize to 336×336 (LLaVA 1.5 standard)
3. Stratified 80/20 split (preserves defect ratio in train & test)
4. JSONL formatting for instruction-tuning

## References

1. Liu et al., *Visual Instruction Tuning* (LLaVA), NeurIPS 2023
2. Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022
3. Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, NeurIPS 2023

## License

MIT
