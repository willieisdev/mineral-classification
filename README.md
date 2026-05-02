# MUMDMC2025 Explainable Mineral Classification Pipeline


A comprehensive machine learning pipeline for mineral classification with state-of-the-art explainability (XAI) techniques. Compares 6 different models (2 classical + 4 deep learning) and provides visual explanations for their decisions.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset Structure](#dataset-structure)
- [Models](#models)
- [Explainability Methods](#explainability-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Outputs](#outputs)
- [Results](#results)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This pipeline addresses the MUMDMC2025 mineral classification challenge by implementing and comparing multiple approaches:
- **Classical ML**: Decision Trees and K-Nearest Neighbors
- **Deep Learning**: EfficientNet-B0, ResNet-50, VGG-16, and DenseNet-121

**Unique Value Proposition**: Unlike standard classification pipelines, this implementation focuses on **explainability** - helping geologists and researchers understand *why* each model makes its predictions.

## ✨ Features

- **Multi-Model Comparison**: Evaluate 6 different models on the same dataset
- **Two-Phase Fine-Tuning**: Optimal training strategy for pretrained CNNs
- **Comprehensive XAI Suite**:
  - Feature importance maps (Decision Trees)
  - LIME explanations (KNN)
  - Grad-CAM++ visualizations (All DL models)
- **Automatic Dataset Discovery**: Handles nested folder structures automatically
- **Rich Visualization Suite**: Confusion matrices, per-class metrics, comparison grids
- **Reproducible Results**: Fixed random seeds and stratified splits

## 📁 Dataset Structure

The pipeline expects the following nested directory structure:
Dataset/
├── ClassName1/
│ ├── 14022024_1stShape/
│ │ ├── CN1/
│ │ │ ├── image1.PNG
│ │ │ └── image2.PNG
│ │ └── CN2/
│ │ └── image3.PNG
│ └── 15022024_2ndShape/
│ └── ...
├── ClassName2/
│ └── ...
└── ClassNameN/
└── ...


**Supported image formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

## 🤖 Models

### Classical Machine Learning
| Model | Type | Feature Space | XAI Method |
|-------|------|---------------|------------|
| Decision Tree | Tree-based | 150×150×3 flattened (67,500 features) | Feature importance maps |
| K-Nearest Neighbors | Distance-based | 150×150×3 flattened | LIME (superpixel explanations) |

### Deep Learning (Pretrained on ImageNet)
| Model | Parameters | Input Size | XAI Method |
|-------|------------|------------|-------------|
| EfficientNet-B0 | 5.3M | 224×224 | Grad-CAM++ |
| ResNet-50 | 25.6M | 224×224 | Grad-CAM++ |
| VGG-16 | 138M | 224×224 | Grad-CAM++ |
| DenseNet-121 | 8.0M | 224×224 | Grad-CAM++ |

**Training Strategy**:
1. **Phase 1** (10 epochs): Train only classification head (frozen backbone)
2. **Phase 2** (15 epochs): Full fine-tuning with lower learning rate

## 🔍 Explainability Methods

### Decision Tree - Feature Importance
- Extracts global feature importances from the tree
- Reshapes weights back to image space (150×150)
- Creates heatmap overlays showing important pixel regions

### KNN - LIME (Local Interpretable Model-agnostic Explanations)
- Segments image into superpixels
- Perturbs segments to measure impact on prediction
- Highlights positive and negative contributors

### Deep Learning - Grad-CAM++
- Computes gradient-weighted activation maps
- Identifies spatial regions that most influence classification
- Compares attention across all 4 DL models simultaneously

## 🚀 Installation

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mineral-classification.git
cd mumdmc2025-mineral-classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

```
## Quick Start
# Run the complete pipeline
python mineral_classification_pipeline.py

# Or in a Jupyter notebook
from mineral_classification_pipeline import main
main()

## Acknowledgemnts

MUMDMC2025 Dataset Creators

PyTorch and timm libraries for pretrained models

Grad-CAM++ and LIME libraries for XAI implementations
