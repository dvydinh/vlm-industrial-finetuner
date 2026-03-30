# Industrial-VLM: Parameter-Efficient Fine-Tuning for Anomaly Detection

> **Quick Pitch**: Optimizing LLaVA-1.5-7B via QLoRA on the MVTec AD dataset to perform automated, pixel-aware industrial surface defect detection under strict VRAM constraints.

---

## 1. Architecture & Core Concept

This project implements a multimodal instruction-tuning pipeline designed for automated quality control (KCS).

### Application Architecture
1. **Vision Encoding**: Input images are processed by the **CLIP Vision Encoder** (ViT-L/14) to extract dense spatial embeddings.
2. **Cross-Modal Alignment**: A **Projection Layer** maps visual tokens into the LLM's text embedding space.
3. **Reasoning & Generation**: The **Vicuna-7B LLM** acts as the cognitive engine to classify defects based on the aligned visual prompts.

### Technical Rationale
- **Why QLoRA?** Training a 7B-parameter model requires ~112GB of VRAM in pure FP32. By leveraging **4-bit NormalFloat (NF4)** quantization (via `bitsandbytes`), we compress the base LLM footprint to ~4GB, allowing the entire training pipeline to run on a single NVIDIA Tesla T4 (15GB VRAM) on Kaggle.
- **Why LoRA?** Instead of full fine-tuning, Low-Rank Adaptation (LoRA) injects trainable rank decomposition matrices ($\Delta W = BA$) into the Transformer's self-attention layers. This freezes 99% of the LLM and reduces trainable parameters to less than 20M, preventing catastrophic forgetting while drastically accelerating training speeds.

---

## 2. Data Engineering: ETL Pipeline for MVTec AD

The original [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad) is designed strictly for *unsupervised* generative anomaly detection. 
To convert this into a supervised instruction-tuning format suitable for LLaVA, an ETL pipeline (`src/data_builder.py`) was engineered:

- **Extract**: Scans the highly imbalanced Kaggle dataset structure spanning 15 domains (textures and distinct objects).
- **Transform**:
  - **Color Space Unification**: 3 domains (Grid, Screw, Zipper) are natively grayscale. These cause tensor-shape mismatch crashes in the RGB-only CLIP ViT. The pipeline forcefully converts all 1-channel images to 3-channel RGB (`cv2.cvtColor`).
  - **Resolution Downsampling**: Native 1024×1024 images are aggressively resized to **336×336** to match LLaVA's maximum context window, cutting dataset size from 5GB to ~400MB.
  - **Stratified Splitting**: Applies a strict 80/20 train/test split utilizing `StratifiedShuffleSplit`. This guarantees that the severe long-tail distribution of rare defects (e.g., *scratch*, *contamination*) is proportionally represented in both the training manifold and the evaluation set.
- **Load**: Exports structured `train.jsonl` and `test.jsonl` files natively compatible with HuggingFace `datasets`.

### Instruction-Tuning Format Example (JSONL)
```json
{
  "id": "metal_nut_00124",
  "image": "metal_nut_scratch_00124.png",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nVới tư cách là kỹ sư KCS, hãy phân tích bề mặt linh kiện trong ảnh này."
    },
    {
      "from": "gpt",
      "value": "Phát hiện lỗi trên bề mặt linh kiện (loại: scratch). Loại sản phẩm: metal_nut. Yêu cầu loại bỏ."
    }
  ]
}
```

---

## 3. Training Configuration

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Base Model** | `llava-hf/llava-1.5-7b-hf` | Baseline VLM pre-trained on generic concepts. |
| **Quantization** | `4-bit (NF4)` | Double quantization enabled, `fp16` compute dtype. |
| **LoRA Target Modules** | `q_proj`, `k_proj`, `v_proj`, `o_proj` | Targets all 4 self-attention projections for max capacity. |
| **LoRA Rank ($r$)** | `32` | Increased from 16 to capture complex industrial defect features. |
| **LoRA Alpha ($\alpha$)** | `64` | Maintains the standard $\alpha/r = 2.0$ scaling factor. |
| **Optimizer** | `paged_adamw_8bit` | Essential for preventing VRAM OOM spikes during backward passes. |
| **Learning Rate** | `2e-5` | Cosine decay schedule with 3% warmup steps. |
| **Hardware Constraint** | `1x NVIDIA Tesla T4` | Trained natively on Kaggle free-tier GPU. |

---

## 4. Experiment Tracking & MLOps

> **W&B Dashboard**: [Insert WandB Link Here]

*(Insert screenshot of training and validation loss curves here after completion)*

---

## 5. Results & Evaluation

> **⚠️ Performance Disclaimer**: Sự suy giảm F1-Score trên các lỗi vi mô (micro-defects) phần lớn đến từ giới hạn Context Window 336px của mô hình CLIP. Việc ép ảnh chụp công nghiệp MVTec 1024x1024 xuống độ phân giải cố định 336x336 bằng interpolation đã làm xóa sổ hoàn toàn các vết xước siêu vi. Tương lai cần áp dụng Random Crop và Sliding Window để khắc phục.

Evaluation is conducted by running inference homogeneously across the entire test set but strictly computing **Item-Wise F1-Score (Macro)** to explicitly measure the model's capability to generalize and recognize defects accurately per material.

| Item Type | Zero-shot F1 | QLoRA F1 |
| :--- | :--- | :--- |
| Metal Nut | [Pending]% | [Pending]% |
| Cable | [Pending]% | [Pending]% |
| Leather | [Pending]% | [Pending]% |
| Bottle | [Pending]% | [Pending]% |
| Hazelnut | [Pending]% | [Pending]% |
| Pill | [Pending]% | [Pending]% |
| Transistor | [Pending]% | [Pending]% |
| Zipper | [Pending]% | [Pending]% |
| Carpet | [Pending]% | [Pending]% |
| Grid | [Pending]% | [Pending]% |
| Tile | [Pending]% | [Pending]% |
| Wood | [Pending]% | [Pending]% |
| Capsule | [Pending]% | [Pending]% |
| Screw | [Pending]% | [Pending]% |
| Toothbrush | [Pending]% | [Pending]% |
| **Overall (Trung bình)** | **[Pending]%** | **[Pending]%** |

### Qualitative Results
*(Insert Table of 3 validation samples here: Good vs Correct Defect vs Edge-case Defect)*

---

## 6. Reproducibility

**Kaggle Notebook**: [Insert Kaggle Link Here]

To reproduce the zero-shot baseline or fine-tuned evaluation locally or on a cloud instance:

```bash
# 1. Clone repository
git clone https://github.com/dvydinh/vlm-industrial-finetuner.git && cd vlm-industrial-finetuner

# 2. Install minimal dependencies
pip install -r requirements.txt

# 3. Run evaluation (assuming processed MVTec AD is in data/processed)
python src/evaluate.py --baseline --test_data data/processed
```
