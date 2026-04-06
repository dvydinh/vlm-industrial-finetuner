# Industrial VLM: Parameter-Efficient Fine-Tuning for Anomaly Detection

> **Quick Pitch**: Optimizing LLaVA-1.5-7B via QLoRA on the MVTec AD dataset to perform automated, pixel-aware industrial surface defect detection under strict free-tier GPU constraints.

![Project Status](https://img.shields.io/badge/Status-Completed-success) ![Framework](https://img.shields.io/badge/Framework-HuggingFace-yellow) ![Model](https://img.shields.io/badge/Base_Model-LLaVA_1.5_7B-blue)

---

## 1. Executive Summary & KPIs

This project implements a multimodal instruction-tuning pipeline designed for automated quality control (KCS). The core objective is transforming a conversational language model into a strict, coordinate-aware defect localizer capable of precise visual grounding on industrial surfaces.

### Final Results versus Zero-Shot Baseline
* **Recall (Defect Detection)**: **97.6%** (246/252 critical defects captured) - completely eliminating the 0.0% failure rate observed in the zero-shot baseline.
* **Strict Localization (IoU > 0.5)**: Achieved 45 perfect bounding box localizations with a mean IoU soaring from 0.00 to **0.3554**.
* **Basic F1-Score (Macro)**: **0.6007** - High sensitivity achieved, with precision structurally constrained by intentional hard negative mining sub-sampling during the data pipeline rendering phase.

**Live tracking and loss convergence analytics are publicly available on Weights & Biases:**
🚀 **[View W&B Analytics Dashboard](https://wandb.ai/dvydinh/vlm-industrial-finetuner)**

---

## 2. Architecture & Hardware Constraint Management

Training a multimodal 7B-parameter architecture natively demands ~112GB of VRAM. This repository engineers a pipeline explicitly designed to conform to a **single NVIDIA Tesla T4 (15GB VRAM)**.

### Technical Stack
* **Representation**: CLIP Vision Encoder (ViT-L/14) cross-aligned to Vicuna-7B.
* **Compression**: 4-bit NormalFloat (NF4) double quantization via `bitsandbytes`.
* **Adaptation**: Low-Rank Adaptation (LoRA) specifically targeting the self-attention manifold (`q_proj`, `k_proj`, `v_proj`, `o_proj`) with an atypically high mathematical rank (`r=64`, `alpha=128`). This high-rank injection preserves the dense high-frequency topological data required to classify microscopic industrial defects.
* **Optimizer**: `paged_adamw_8bit` integrated with mixed-precision `fp16` computing to mitigate VRAM out-of-memory spikes.

---

## 3. Data Engineering: ETL Pipeline for MVTec AD

The MVTec AD Dataset is inherently designed for unsupervised discriminative learning. This pipeline (`src/data_builder.py`) structurally refactors it into a unified instruction-tuning schema.

### Resolution & Spatial Tuning
Industrial defects cannot be identified mathematically under aggressive 224x224 interpolation. To resolve this:
1. **Sliding Window NMS Inference**: Original 1024×1024 topographies are dynamically tessellated into overlapping **336×336** inference macro-blocks.
2. **Hard Negative Mining**: To prevent catastrophic Kaggle storage failure (30GB disk limit), non-defective background crops were heavily sub-sampled. While this safely averted storage exhaustion, the constrained negative sampling ratio directly accounts for the over-sensitivity (False Positive bias) observed during validation.
3. **Stratified Splitting**: Strict `sklearn` splitting algorithmically anchors the long-tail rare defects proportionally across both the training and testing manifolds.

### Instruction-Tuning Structure (JSONL)
```json
{
  "id": "metal_nut_00124",
  "image": "metal_nut_scratch_00124.png",
  "gt_bbox": [120, 45, 150, 80],
  "gt_class": "scratch",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nAct as a Quality Assurance Engineer, analyze the surface of the [metal_nut] component in this image. If a defect is found, report its type and bounding box coordinates [ymin, xmin, ymax, xmax]."
    },
    {
      "from": "gpt",
      "value": "Detected [scratch] at [120, 45, 150, 80]."
    }
  ]
}
```

---

## 4. Evaluation Criteria

The system circumvents inaccurate string-matching by deploying a mathematical Visual Grounding engine (`src/evaluate.py`). A response is classified as a True Positive exclusively if:
1. The predicted defect terminology matches the ground truth.
2. The exact coordinate boundaries intersect the ground-truth map with an **Intersection over Union (IoU) > 0.5**.

---

## 5. Reproducibility

The entire data preparation, baseline metric validation, iterative parameter tuning, and continuous cloud-based evaluation workflows are decoupled into three sequentially independent Python kernels.

```bash
# 1. Initialize workspace
git clone https://github.com/dvydinh/vlm-industrial-finetuner.git && cd vlm-industrial-finetuner
pip install -r requirements.txt

# 2. Execute execution units natively
# -> notebooks/1_baseline_evaluation.ipynb (Baseline Inference Pipeline)
# -> notebooks/2_qlora_training.ipynb (Distributed QLoRA Compilation)
# -> notebooks/3_finetune_evaluation.ipynb (Cloud Model Recovery & Final Eval)
```
