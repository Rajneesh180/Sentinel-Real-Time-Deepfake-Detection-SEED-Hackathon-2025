## **Sentinel : Real Time DeepFake Detection — SEED Hackathon by [@Rajneesh180](https://github.com/Rajneesh180) & [@Lord-Alpha-dark](https://github.com/Lord-Alpha-dark)** 

## Hackathon Details:

[SEED Hackathon Page](https://seedglobaleducation.com/seed-hackathon/2025/showcase-interest/?utm_campaign=24542873-SEED%20Hackathon%202025&utm_source=SEED106)

## Deepfake Detection — Full Pipeline Documentation

A complete overview of the system we used to train, evaluate, and deploy our deepfake-classification models. This document highlights the architecture, preprocessing pipeline, training strategy, and inference workflow in a fresh structure.

---

## 📘 Table of Contents

1. [Introduction](#introduction)
2. [High-Level Model Flow](#high-level-model-flow)
3. [Face Detection Strategy](#face-detection-strategy)
4. [Image Dimensions & Cropping](#image-dimensions--cropping)
5. [Model Architecture & Encoders](#model-architecture--encoders)
6. [Frame Aggregation Logic](#frame-aggregation-logic)
7. [Augmentation Pipeline](#augmentation-pipeline)
8. [Environment & Docker Setup](#environment--docker-setup)
9. [Dataset Preparation Steps](#dataset-preparation-steps)
10. [Training Configuration](#training-configuration)
11. [Monitoring & Logs](#monitoring--logs)
12. [Inference & Submission](#inference--submission)
13. [Pretrained Weights](#pretrained-weights)
14. [Hardware Notes](#hardware-notes)

---

## Introduction

This repository implements a **frame-level deepfake detection system**, heavily optimized for performance on the DFDC dataset. Rather than relying on temporal models or video-sequence networks, the design emphasizes **independent frame classification followed by a robust aggregation step**, which empirically surpassed more complex approaches.

---

## High-Level Model Flow

```
Video → Dynamic Face Detector → Cropped Frames  
      → Resize + Augment → EfficientNet Encoder  
      → Frame-level Predictions → Confidence-based Fusion  
      → Final Deepfake Score
```

The pipeline was designed for reproducibility, GPU efficiency, and compatibility with DFDC inference constraints.

---

## Face Detection Strategy

MTCNN served as the backbone face detector due to its consistent speed inside kernel limits.
Even though S3FD offers better robustness, available PyTorch builds lack compatible licenses, ruling it out.

**Adaptive detector input resolution:**

| Video Width (max side) | Scale Applied |
| ---------------------- | ------------- |
| < 300 px               | ×2 upscale    |
| 300–1000 px            | unchanged     |
| > 1000 px              | ×0.5          |
| > 1900 px              | ×0.33         |

This prevents unnecessary computational load while retaining detection accuracy.

---

## Image Dimensions & Cropping

### Input Resolution

* Initial experiments with many encoders showed **EfficientNet** performing significantly better.
* Early trials started with **EfficientNet-B4**, hence the adoption of its default size: **380×380**.
* Even when upgrading to B7, resolution remained the same due to GPU memory constraints.

### Cropping Margin

Every detected face was expanded by **30% on all sides** to preserve essential context, helpful for handling subtle deepfake artifacts.

---

# ## Model Architecture & Encoders

The final ensemble relies on **EfficientNet-B7** pretrained using the **Noisy Student** technique:

> “Self-Training with Noisy Student improves ImageNet classification”
> [https://arxiv.org/abs/1911.04252](https://arxiv.org/abs/1911.04252)

This encoder delivered the strongest results across all experiments.

---

## Frame Aggregation Logic

Each video contributes **32 sampled frames**.

A confidence-weighted fusion strategy performs significantly better than average pooling:

```python
def confident_strategy(pred, t=0.8):
    import numpy as np
    pred = np.array(pred)
    sz = len(pred)
    f_high = np.count_nonzero(pred > t)

    if f_high > sz // 2.5 and f_high > 11:
        return np.mean(pred[pred > t])
    if np.count_nonzero(pred < 0.2) > 0.9 * sz:
        return np.mean(pred[pred < 0.2])
    return np.mean(pred)
```

The heuristic effectively handles videos dominated by strong fake cues or overwhelmingly real cues.

---

## Augmentation Pipeline

Augmentations are intentionally aggressive to counter dataset bias. Most transforms come from **Albumentations**, supplemented with a custom isotropic resize operator.

```python
Compose([
    ImageCompression(60, 100, p=0.5),
    GaussNoise(p=0.1),
    GaussianBlur(3, p=0.05),
    HorizontalFlip(),
    OneOf([
        IsotropicResize(max_side=size, ...),
        IsotropicResize(max_side=size, ...),
        IsotropicResize(max_side=size, ...),
    ], p=1),
    PadIfNeeded(size, size),
    OneOf([RandomBrightnessContrast(), FancyPCA(), HueSaturationValue()], p=0.7),
    ToGray(p=0.2),
    ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=10, p=0.5)
])
```

Additional robustness was gained using **cutout-style masking**, simulating occlusions and compression artifacts.

---

## Environment & Docker Setup

### Build

```bash
docker build -t deepfake-detector .
```

### Run

```bash
docker run --runtime=nvidia --ipc=host --rm \
  --volume <DATA_ROOT>:/dataset -it deepfake-detector
```

All dependencies (including CUDA, Python packages, and system libs) come preconfigured in the `Dockerfile`.

---

# ## Dataset Preparation Steps

After you download DFDC data, ensure directories follow the standard structure:

```
DATA_ROOT/
    dfdc_train_000/
    dfdc_train_001/
    ...
```

A single script **`preprocess_data.sh`** orchestrates the entire preparation workflow:

---

### **(1) Detect Faces**

```
python preprocessing/detect_original_faces.py --root-dir DATA_ROOT
```

Outputs: `bboxes/`

### **(2) Extract Face Crops**

```
python preprocessing/extract_crops.py --root-dir DATA_ROOT --crops-dir crops
```

Outputs: `crops/`

### **(3) Generate Facial Landmarks**

```
python preprocessing/generate_landmarks.py --root-dir DATA_ROOT
```

Outputs: `landmarks/`

### **(4) SSIM Difference Masks**

```
python preprocessing/generate_diffs.py --root-dir DATA_ROOT
```

Outputs: `diffs/`

### **(5) Cross-validation Folds**

```
python preprocessing/generate_folds.py --root-dir DATA_ROOT --out folds.csv
```

16 folds are created, with a few reserved for validation.

---

## Training Configuration

Training five B7 models (with five seeds) is automated in:

```
train.sh
```

* Checkpoints are stored every epoch.
* Fake and real losses are tracked separately for stability.

---

## Monitoring & Logs

Plot loss curves using:

```
python plot_loss.py --log-file logs/<log_file>
```

---

## Inference & Submission

Inference replicates the same logic used in the competition kernel:

```
python predict_folder.py
```

For ensemble submission:

```
./predict_submission.sh <video_directory> <output_csv>
```

Example:

```
./predict_submission.sh /mnt/data/test_videos submission.csv
```

---

## Pretrained Weights

Download all model weights before building Docker:

```
download_weights.sh
```

Weights are stored in the `weights/` directory.

---

