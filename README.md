<h1 align="center">Grand X-Ray Slam: Division B</h1>
<p align="center">
  Multi-Label Chest X-Ray Classification using Ensemble CNNs
</p>

<p align="center">
  <img src="https://img.shields.io/github/last-commit/hannahjan06/X-ray-Slab-B-Kaggle" alt="last commit">
  <img src="https://img.shields.io/github/languages/top/hannahjan06/X-ray-Slab-B-Kaggle?color=1b4b9c&label=top%20language" alt="Python">
  <img src="https://img.shields.io/github/languages/count/hannahjan06/X-ray-Slab-B-Kaggle?label=languages" alt="languages">
</p>

<p align="center">
  Built using:
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Google%20Colab-FFC107?style=for-the-badge&logo=googlecolab&logoColor=black">
  <img src="https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3572A5?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
</p>

---

## Overview

This project implements an ensemble deep learning system for multi-label chest X-ray classification as part of the Grand X-Ray Slam Division B Kaggle competition. Multiple convolutional backbones are trained independently and combined through logit-level averaging to improve generalization, stability, and robustness across 14 thoracic diagnostic labels.

Performance is evaluated using AUC and weighted F1, aligned with competition scoring. Mixed precision, data augmentation, class weighting, and staged fine-tuning are used to optimize results.

---

## Model Summary

* **Model Type:** Multi-label medical image classifier
* **Backbones:** ResNet50, EfficientNetB3, DenseNet121 (ensemble)
* **Input Resolution:** 224×224
* **Validation Performance:**

  * AUC: 0.91
  * Weighted F1: 0.89
* **Rank:** 67 (first Kaggle competition entry)

---

## Dataset

* **Source:** NIH Chest X-Ray dataset (Kaggle-provided competition bundle)
* **Scale:** 100K+ images
* **Format:** DICOM to processed JPEG/PNG
* **Preprocessing:** resize, normalization, optional grayscale replication

---

## Labels (14 Classes)

Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices.

---

## Training Strategy

* Class-weighted binary cross entropy
* AdamW optimizer with weight decay
* Staged fine-tuning with partial unfreezing
* Data augmentation to address imbalance
* Mixed precision for efficiency

---

## Environment

* GPU: NVIDIA Tesla T4 (Kaggle)
* Python: 3.11
* Frameworks: TensorFlow/Keras, NumPy, scikit-learn, OpenCV

---

## Quickstart

```bash
pip install -r requirements.txt
```

Train models using the included notebooks or scripts.
Paths may be adapted for Kaggle, local, or Colab environments.

---

## Inference Example

```python
probs = model.predict(img_batch)  # returns shape [N, 14]
```

Post-processing uses label-specific thresholding for binary decisions.

---

## Reproducibility Notes

* Use consistent preprocessing across training and validation.
* Set seeds where deterministic runs are required.
* Precision mode may introduce minor floating-point variability.

---

## Results

Inference examples, metrics graphs, and performance screenshots are available in the repository.

---

## Acknowledgements

* NIH ChestX-ray dataset
* Kaggle competition organizers
* TensorFlow/Keras model zoo backbones
