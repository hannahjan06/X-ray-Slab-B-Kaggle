# Grand X‑Ray Slam (Division B)

Ensemble CNNs for multi‑label chest X‑ray classification. Built for the **Grand X‑Ray Slam: Division B** Kaggle competition.

> **Validation**: AUC **0.91**, Weighted F1 **0.89**
> **Rank**: 67 (first Kaggle comp)
> **Backbones**: ResNet50 · EfficientNetB3 · DenseNet121 (ensemble)

---

## Table of Contents

* [Overview](#overview)
* [Dataset](#dataset)
* [Labels](#labels)
* [Approach](#approach)
* [Training Pipeline](#training-pipeline)
* [Metrics](#metrics)
* [Environment](#environment)
* [Quickstart](#quickstart)
* [Inference](#inference)
* [Reproducibility Notes](#reproducibility-notes)
* [Results & Screenshots](#results--screenshots)
* [Acknowledgements](#acknowledgements)
* [License](#license)

---

## Overview

This project tackles **multi‑label classification** of chest X‑rays using an **ensemble of three CNN backbones**. Each model predicts logits for the 14 thoracic findings; the ensemble averages the logits to produce the final prediction. Training is done in TensorFlow/Keras with mixed precision and AdamW.

* **Goal:** robust, general multi‑label predictions across 14 findings.
* **Why ensemble?** Different architectures learn complementary features; averaging stabilizes performance and reduces variance.

---

## Dataset

* **Source:** NIH Chest X‑Ray dataset (Kaggle competition bundle)
* **Scale:** **100K+** images (train split prepared via provided CSV)
* **Input size:** 224×224 (this repo) — easily adjustable
* **Preprocessing:** resize, light normalization (Keras preprocess), optional grayscale as 3‑channel

> Broken images listed in notebook are filtered out prior to training.

---

## Labels

The task is **multi‑label** across these **14** classes:

1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Enlarged Cardiomediastinum
6. Fracture
7. Lung Lesion
8. Lung Opacity
9. No Finding
10. Pleural Effusion
11. Pleural Other
12. Pneumonia
13. Pneumothorax
14. Support Devices

---

## Approach

**Backbones** (Keras Applications):

* **ResNet50** (no top)
* **EfficientNetB3** (no top, ImageNet weights)
* **DenseNet121** (no top, ImageNet weights)

**Heads (per‑model):** GlobalAveragePooling → Dropout(0.3) → Dense(14, logits)

**Ensemble:** Average of per‑model logits → Sigmoid at evaluation time.

**Loss:** Weighted Binary Cross‑Entropy with class weights derived from label prevalence.
**Optimizer:** AdamW (weight decay 1e‑6 to 1e‑7), gradient clip 1.0.
**Mixed precision:** enabled.
**Early stopping & LR scheduling:** on `val_auc` and `val_loss` with ReduceLROnPlateau.

> A fine‑tuning pass unfreezes the top ~50 layers of each backbone at a very low LR for stability.

---

## Training Pipeline

**Augmentations (train):**

* Horizontal flip, small rotations (via `rot90`), brightness ±0.1, contrast 0.85–1.15
* Targeted extra aug for under‑represented classes (optional saturation/hue/flip‑up‑down, light noise)

**Splits:** 80/20 train/validation (random state 42).
**Batch sizes:** 64 (pretrain), 32 (fine‑tune).
**Epochs:** 10 (pretrain), 5 (fine‑tune) with early stopping.

---

## Metrics

* **Primary:** `AUC` (from logits)
* **Also tracked:** Binary Accuracy (0.5 threshold in‑training)

**Validation (best):**

* **AUC:** **0.91**
* **Weighted F1:** **0.89** (computed post‑hoc on validation predictions)

> For deployment, compute **per‑class thresholds** by sweeping 0.05–0.95 to maximize F1 for each label (improves macro performance vs global 0.5).

---

## Environment

* **Runtime:** Kaggle Notebook (GPU enabled)
* **GPU:** NVIDIA Tesla **T4** (~15 GB)
* **Python:** 3.11.13
* **TensorFlow:** 2.x (mixed precision)
* **Key libs:** `tensorflow`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `opencv-python`

---

## Quickstart

> These steps mirror the Kaggle notebook; adapt paths if running locally.

1. **Prepare CSVs & paths**

   * Load `train2.csv` and create absolute `Image_path` entries.
   * Drop non‑model columns: `Image_name`, `Study`, `Patient_ID`.
   * Fill missing `Age` (mean) and `Sex` (mode).

2. **Build datasets**

   * Use the provided TF data pipeline with `decode_and_process_train/val`.

3. **Train ensemble**

   * Stage 1: train all three backbones + heads with early stopping.
   * Save best weights to `ensemble_model.weights.h5`.

4. **Fine‑tune**

   * Rebuild ensemble, load weights, unfreeze top ~50 layers per backbone.
   * Train with LR `5e‑6`, save to `final_finetuned_ensemble.weights.h5`.

---

## Inference

Use the helper to get class probabilities (sigmoid applied to logits):

```python
probs = predict_ensemble([
    "/path/to/xray1.jpg",
    "/path/to/xray2.jpg",
])  # shape: [N, 14]
```

> Apply per‑class thresholds for binary decisions; calibrate with temperature scaling if you need better probability quality.

---

## Reproducibility Notes

* Set seeds for NumPy/TF if deterministic runs are required.
* Mixed precision may introduce tiny numeric drift; monitor AUC/F1 on the same validation split.
* Ensure consistent preprocessing (same Keras `preprocess_input` used in train/val/infer).

---

## Results & Screenshots

<img width="1472" height="704" alt="image" src="https://github.com/user-attachments/assets/83e9904b-87f4-4c3f-99de-01828f68b32e" />
<img width="1200" height="1200" alt="Grand X-Ray Slam (Division B)" src="https://github.com/user-attachments/assets/0189176e-0281-4475-9d76-be7d71879fb9" />

---

## Acknowledgements

* NIH Chest X‑Ray dataset maintainers and the Kaggle competition organizers.
* Keras Applications for backbone architectures.
