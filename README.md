# Real-Time Domain Adaptation for Semantic Segmentation

## 📌 Overview

This project investigates **domain adaptation** techniques to improve the performance of **real-time semantic segmentation models** when transferring from synthetic datasets (GTA5) to real-world data (Cityscapes). We focus on comparing a classical architecture (**DeepLabV2**) and a real-time network (**BiSeNet**), applying **data augmentation** and **adversarial learning** to mitigate domain shift.

---

## 📚 Contents

* [Introduction](#introduction)
* [Datasets](#datasets)
* [Models](#models)
* [Domain Adaptation Techniques](#domain-adaptation-techniques)
* [Results](#results)
* [Conclusion](#conclusion)

---

## 🚀 Introduction

Semantic segmentation is a vital task in computer vision, essential for applications like autonomous driving and surveillance. However, models trained on synthetic data often underperform on real-world images due to domain shift. This project explores **real-time domain adaptation**, aiming to close the performance gap while keeping inference fast and efficient.

---

## 🗂️ Datasets

### 📷 Cityscapes (Target Domain)

* Real-world urban street scenes
* 5,000 finely annotated images from 50 cities
* Resolution: **2048×1024**
* 19 semantic classes

### 🎮 GTA5 (Source Domain)

* Synthetic dataset from *Grand Theft Auto V*
* 24,966 images with pixel-wise annotations
* Resolution: **1914×1052**
* Class labels compatible with Cityscapes

---

## 🧠 Models

### 🔹 DeepLabV2 (Classical)

* Uses **Atrous Spatial Pyramid Pooling (ASPP)** and **CRFs**
* High accuracy, but slower inference

### 🔸 BiSeNet (Real-Time)

* Dual-path design:

  * **Spatial Path**: Preserves detail
  * **Context Path**: Captures semantic context
* **Feature Fusion Module**: Merges the two paths
* Designed for **real-time** applications

---

## 🔁 Domain Adaptation Techniques

### 1. **Data Augmentation**

Enhances model generalization:

* `HorizontalFlip`
* `ColorJitter`
* `GaussianBlur`
* `RandomResizedCrop`

### 2. **Adversarial Learning**

Uses an output-space discriminator:

* Aligns predicted segmentation maps between source and target domains
* Helps the model learn domain-invariant features

---

## 📊 Results

| Method                                | mIoU (%)  | Inference Time (s) |
| ------------------------------------- | --------- | ------------------ |
| **Classical Network**                 |           |                    |
| DeepLabV2 (Cityscapes)                | **50.09** | 0.037              |
| **Real-Time Network**                 |           |                    |
| BiSeNet (Cityscapes)                  | 48.04     | **0.013**          |
| BiSeNet (GTA5 → Cityscapes)           | 20.61     | 0.013              |
| **Domain Adaptation**                 |           |                    |
| BiSeNet + Data Augmentation           | 27.43     | 0.013              |
| BiSeNet + Adversarial Learning (Ours) | **30.44** | 0.013              |

### ✅ Key Takeaways

* **+33%** mIoU gain with data augmentation alone
* **+47%** total gain using data augmentation + adversarial learning
* **Real-time performance preserved** (0.013s inference)
* BiSeNet is **2.8× faster** than DeepLabV2, with competitive accuracy post-adaptation

---

## 🧾 Conclusion

Our work demonstrates that combining **data augmentation** with **adversarial learning** can effectively reduce domain shift in real-time semantic segmentation. The adapted BiSeNet achieves significant improvements in accuracy without sacrificing inference speed. While a performance gap remains compared to models trained directly on real data, this approach offers a promising direction for deploying efficient segmentation systems in real-world scenarios.

---

## 👥 Authors

* **Tommaso Mazzarini** – Politecnico di Torino – [tommaso.mazzarini@studenti.polito.it](mailto:tommaso.mazzarini@studenti.polito.it)
* **Leonardo Merelli** – Politecnico di Torino – [leonardo.merelli@studenti.polito.it](mailto:leonardo.merelli@studenti.polito.it)
* **Giovanni Stinà** – Politecnico di Torino – [giovanni.stina@studenti.polito.it](mailto:giovanni.stina@studenti.polito.it)
