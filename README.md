# Chest X-Ray Classification (Kaggle rank 67th | 91% Accuracy)

**Multi-Label Disease Classification using Ensemble Deep Learning**  
*Kaggle Competition: NIH Chest X-Ray Dataset | Ranked 67th Globally*

---

## Project Overview
Built an **ensemble of 3 CNN architectures** (ResNet50 + EfficientNetB3 + DenseNet121) achieving **91% AUC** on **14 chest X-ray disease classes**.  

- **Dataset**: 30K+ NIH Chest X-Ray images  
- **Classes**: Pneumonia, COVID, Atelectasis, Cardiomegaly, 11+ diseases  
- **Rank**: **67th rank**
- **Used in**: Ministry AI Project Application  

---

## Results
| Metric     | Score | Benchmark     |
|------------|-------|---------------|
| **AUC**    | **0.91** | Top 15%   |
| **Accuracy** | 91%   | Outperformed 85% teams |
| **F1-Score** | 0.89  | Multi-label weighted |

---

## Technical Stack
Deep Learning: TensorFlow 2.15, Keras  
Architectures: ResNet50 + EfficientNetB3 + DenseNet121 (Ensemble)  
Data: 30K+ NIH Chest X-Rays (224x224)  
Augmentation: Rotation, Brightness, Noise, Flips  
Optimization: AdamW, Mixed Precision, GPU (15GB limit)  
Libraries: NumPy, Pandas, OpenCV, Matplotlib, Scikit-learn  

---

## Key Features

### 1. Smart Data Pipeline
- GPU-Optimized: 15GB memory limit, XLA compilation  
- Augmentation: Class-balanced (extra for rare diseases)  
- Preprocessing: ResNet50 normalization, bilinear resize  

### 2. Weighted Loss Function
Handles class imbalance (20% threshold)  
Stable multi-label weighted BCE loss  

### 3. Ensemble Architecture
Input (224x224x3) → [ResNet50] [EfficientNetB3] [DenseNet121] → Average Layer → 14 Disease Probabilities  

### 4. Production Callbacks
- Early Stopping (patience=3)  
- Learning Rate Reduction (factor=0.5)  
- Best Weights Checkpoint  
- Training Time: 10 epochs (~2.5 hrs on GPU)  

---

## Training Results
Epoch 1/10: auc=0.85, val_auc=0.87, loss=0.42  
Epoch 5/10: auc=0.90, val_auc=0.91, loss=0.28 ← BEST  
Epoch 10/10: auc=0.93, val_auc=0.90, loss=0.25  

---

## Class Performance
Top Performers: Pneumonia (94%), No Finding (93%)  
Challenging: Pleural Other (78%), Lung Lesion (82%)  

---

## What I Learned
- Ensemble Methods: +4% AUC gain over single model  
- Class Imbalance: Weighted BCE + targeted augmentation  
- Production DL: Mixed precision, GPU optimization, callbacks  
- Multi-label: 14 simultaneous disease predictions
