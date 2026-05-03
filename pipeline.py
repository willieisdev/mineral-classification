#!/usr/bin/env python3
"""
MUMDMC2025 Explainable Mineral Classification Pipeline
Kaggle Notebook — williamsadajiagbane/mumdmc2025
Dataset path: /kaggle/input/mumdmc2025/Cropped_Images/

Models:
  Classical  : Decision Tree, K-Nearest Neighbors
  Pretrained : EfficientNet-B0, ResNet-50, VGG-16, DenseNet-121

XAI:
  Decision Tree  : Feature importance map (reshaped to image space)
  KNN            : LIME image explainer
  All DL models  : Grad-CAM++
"""

# ─── 0. INSTALL DEPENDENCIES ────────────────────────────────────────────────
# Run this cell first on Kaggle
# !pip install -q timm pytorch-grad-cam lime

# ─── 1. IMPORTS ─────────────────────────────────────────────────────────────
import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from collections import Counter
from PIL import Image

# Sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc,
                             classification_report)
from sklearn.preprocessing import label_binarize

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm

# XAI
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from lime import lime_image
from skimage.segmentation import mark_boundaries

import cv2
warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")

# ─── 2. CONFIGURATION ────────────────────────────────────────────────────────
DATASET_PATH   = "/kaggle/input/datasets/williamsadajiagbane/mumdmc2025/MUMDMC2025_DataSet/Original_Images"
OUTPUT_PATH    = "/kaggle/working/outputs"
IMG_SIZE       = 224          # standard for all pretrained models
BATCH_SIZE     = 32
EPOCHS_PHASE1  = 10           # frozen backbone
EPOCHS_PHASE2  = 15           # full fine-tune
LR_HEAD        = 1e-3
LR_FINETUNE    = 1e-5
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE   = 42
NUM_XAI_SAMPLES = 3           # samples per class for XAI visualisations

os.makedirs(OUTPUT_PATH, exist_ok=True)
print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_PATH}")

# ─── 3. DATASET CLASS (PyTorch) ──────────────────────────────────────────────
class MineralDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels      = labels
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class RawTestDataset:
    """Simplified dataset for Grad-CAM without transforms"""
    def __init__(self, paths, labels):
        self.image_paths = paths
        self.labels      = labels
    
    def __len__(self):
        return len(self.image_paths)

# ─── 4. DATA LOADING UTILITIES ───────────────────────────────────────────────
def load_dataset(dataset_path):
    """
    Handles the nested structure:
      Cropped_Images/<ClassName>/CN/<CrystalNumber>/ﬁle.PNG
 
    The class label is always the first subdirectory under dataset_path.
    os.walk recurses all the way down to find the actual image files.
    """
    image_paths, labels = [], []
 
    # Top-level subdirs are the mineral class names
    class_names = sorted([
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    ])
    print(f"Classes found: {class_names}")
 
    VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
 
    for class_name in class_names:
        class_root = os.path.join(dataset_path, class_name)
        class_count = 0
 
        # Walk recursively through date-stamped subfolders e.g. 14022024_1stShape/
        for dirpath, _, filenames in os.walk(class_root):
            for fname in filenames:
                if Path(fname).suffix.lower() in VALID_EXT:
                    image_paths.append(os.path.join(dirpath, fname))
                    labels.append(class_name)
                    class_count += 1
 
        print(f"  {class_name}: {class_count} images")
 
    print(f"\nTotal images loaded: {len(image_paths)}")
    return image_paths, labels, class_names


def build_sklearn_features(image_paths, size=(150, 150)):
    """
    Flatten images to numpy feature vectors for classical ML.
    Normalises pixel values to [0, 1].
    """
    features = []
    raw_images = []  # keep RGB uint8 for XAI visualisations
    for path in image_paths:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw_images.append(cv2.resize(img, size))
        flat = cv2.resize(img, size).flatten().astype(np.float32) / 255.0
        features.append(flat)
    return np.array(features), raw_images


def get_dl_transforms(img_size=224, augment=False):
    """
    Returns torchvision transforms.
    Mild augmentation for training; deterministic for validation/test.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        return transforms.Compose([
            transforms.Resize((img_size + 16, img_size + 16)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

# ─── 5. CLASSICAL ML PIPELINE ────────────────────────────────────────────────
class ClassicalMLPipeline:
    def __init__(self, X, y, class_names, raw_images):
        self.X           = X
        self.y           = y
        self.class_names = class_names
        self.raw_images  = raw_images
        self.le          = LabelEncoder()
        self.y_enc       = self.le.fit_transform(y)
        self.results     = {}

        (self.X_train, self.X_test,
         self.y_train, self.y_test,
         self.idx_train, self.idx_test) = train_test_split(
            X, self.y_enc, np.arange(len(X)),
            test_size=0.2, random_state=RANDOM_STATE, stratify=self.y_enc
        )
        print(f"Classical ML — Train: {len(self.X_train)} | Test: {len(self.X_test)}")

    def train(self):
        models = {
            "DecisionTree": DecisionTreeClassifier(random_state=RANDOM_STATE),
            "KNN":          KNeighborsClassifier(n_neighbors=5, metric='euclidean'),
        }
        for name, model in models.items():
            t0 = time.time()
            model.fit(self.X_train, self.y_train)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = model.predict(self.X_test)
            test_time = time.time() - t0

            self.results[name] = {
                "model":      model,
                "y_pred":     y_pred,
                "train_time": train_time,
                "test_time":  test_time,
                "accuracy":   accuracy_score(self.y_test, y_pred),
                "precision":  precision_score(self.y_test, y_pred, average='weighted'),
                "recall":     recall_score(self.y_test, y_pred, average='weighted'),
                "f1":         f1_score(self.y_test, y_pred, average='weighted'),
            }
            print(f"{name}: Acc={self.results[name]['accuracy']:.4f} | "
                  f"F1={self.results[name]['f1']:.4f}")

    # ── XAI: Decision Tree feature importance map ────────────────────────────
    def explain_dt(self):
        """
        Reshape feature importances back to 150×150×3 space.
        Average across RGB channels → grayscale importance map.
        One map per class is not directly available from DT's global importances,
        so we show the global map overlaid on one sample per class.
        """
        model = self.results["DecisionTree"]["model"]
        importances = model.feature_importances_  # (150*150*3,)
        imp_map = importances.reshape(150, 150, 3).mean(axis=2)  # (150,150)

        fig, axes = plt.subplots(
            2, len(self.class_names),
            figsize=(4 * len(self.class_names), 8)
        )
        fig.suptitle("Decision Tree — Feature Importance Map per Class",
                     fontsize=14, fontweight='bold')

        for ci, cname in enumerate(self.class_names):
            # Pick first correctly predicted test sample for this class
            class_idx = self.le.transform([cname])[0]
            candidates = [
                i for i in range(len(self.y_test))
                if self.y_test[i] == class_idx
                and self.results["DecisionTree"]["y_pred"][i] == class_idx
            ]
            raw_idx = self.idx_test[candidates[0]] if candidates else 0
            sample_img = self.raw_images[raw_idx]  # (150,150,3) uint8

            axes[0, ci].imshow(sample_img)
            axes[0, ci].set_title(cname)
            axes[0, ci].axis('off')

            axes[1, ci].imshow(sample_img, alpha=0.5)
            axes[1, ci].imshow(imp_map, cmap='hot', alpha=0.6)
            axes[1, ci].set_title("Importance\nOverlay")
            axes[1, ci].axis('off')

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PATH}/xai_dt_importance.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ DT XAI saved.")

    # ── XAI: LIME for KNN ────────────────────────────────────────────────────
    def explain_knn_lime(self, n_samples=500):
        """
        LIME image explainer for KNN.
        Generates superpixel-level importance masks.
        """
        knn_model = self.results["KNN"]["model"]
        img_size  = (150, 150)

        def knn_predict_fn(images):
            """Wrapper: list of HWC uint8 → KNN probability array."""
            feats = []
            for img in images:
                img_resized = cv2.resize(img, img_size)
                feats.append(img_resized.flatten().astype(np.float32) / 255.0)
            return knn_model.predict_proba(np.array(feats))

        explainer = lime_image.LimeImageExplainer(random_state=RANDOM_STATE)

        fig, axes = plt.subplots(
            3, len(self.class_names),
            figsize=(4 * len(self.class_names), 10)
        )
        fig.suptitle("KNN — LIME Explanations per Class",
                     fontsize=14, fontweight='bold')
        row_labels = ["Original", "LIME Positive", "LIME Boundary"]

        for ci, cname in enumerate(self.class_names):
            class_idx = self.le.transform([cname])[0]
            candidates = [
                i for i in range(len(self.y_test))
                if self.y_test[i] == class_idx
                and self.results["KNN"]["y_pred"][i] == class_idx
            ]
            raw_idx    = self.idx_test[candidates[0]] if candidates else 0
            sample_img = self.raw_images[raw_idx]  # uint8 HWC

            explanation = explainer.explain_instance(
                sample_img,
                knn_predict_fn,
                top_labels=len(self.class_names),
                hide_color=0,
                num_samples=n_samples
            )

            # Positive superpixels for the true class
            temp, mask = explanation.get_image_and_mask(
                class_idx, positive_only=True, num_features=5,
                hide_rest=False
            )

            axes[0, ci].imshow(sample_img)
            axes[0, ci].set_title(cname)
            axes[0, ci].axis('off')

            axes[1, ci].imshow(temp)
            axes[1, ci].set_title("Positive")
            axes[1, ci].axis('off')

            axes[2, ci].imshow(mark_boundaries(temp / 255.0, mask))
            axes[2, ci].set_title("Boundaries")
            axes[2, ci].axis('off')

        for ri, rl in enumerate(row_labels):
            axes[ri, 0].set_ylabel(rl, fontsize=11, rotation=90)

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_PATH}/xai_knn_lime.png", dpi=300, bbox_inches='tight')
        plt.show()
        print("✅ KNN LIME XAI saved.")

# ─── 6. DEEP LEARNING PIPELINE ───────────────────────────────────────────────
DL_MODELS = {
    "EfficientNet-B0": "efficientnet_b0",
    "ResNet-50":       "resnet50",
    "VGG-16":          "vgg16",
    "DenseNet-121":    "densenet121",
}

def build_model(model_key, num_classes):
    """Load pretrained model from timm with replaced classifier head."""
    model = timm.create_model(
        DL_MODELS[model_key],
        pretrained=True,
        num_classes=num_classes
    )
    return model.to(DEVICE)

def load_saved_model(model_key, pth_path, num_classes):
    """Load a saved model checkpoint."""
    model = timm.create_model(
        DL_MODELS[model_key],
        pretrained=False,
        num_classes=num_classes
    )
    model.load_state_dict(
        torch.load(pth_path, map_location=DEVICE)
    )
    model = model.to(DEVICE)
    model.eval()
    print(f"✅ Loaded {model_key} from {pth_path}")
    return model

def freeze_backbone(model, model_key):
    """Freeze all layers except the classifier head."""
    head_names = {"efficientnet_b0": "classifier",
                  "resnet50":        "fc",
                  "vgg16":           "head",
                  "densenet121":     "classifier"}
    head = head_names.get(DL_MODELS[model_key], "classifier")
    for name, param in model.named_parameters():
        if not name.startswith(head):
            param.requires_grad = False

def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += images.size(0)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images = images.to(DEVICE)
        outputs = model(images)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
    return (np.array(all_labels), np.array(all_preds),
            np.array(all_probs))

def train_dl_model(model_key, train_loader, val_loader, num_classes):
    """Two-phase fine-tuning: head only → full network."""
    print(f"\n{'='*50}")
    print(f"Training: {model_key}")
    print(f"{'='*50}")

    model    = build_model(model_key, num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Phase 1 — head only
    freeze_backbone(model, model_key)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PHASE1
    )
    print("Phase 1: Training head only...")
    for epoch in range(EPOCHS_PHASE1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS_PHASE1} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    # Phase 2 — full fine-tune
    unfreeze_all(model)
    optimizer = optim.AdamW(model.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS_PHASE2
    )
    print("Phase 2: Full fine-tuning...")
    best_acc, best_state = 0.0, None
    for epoch in range(EPOCHS_PHASE2):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion)
        y_true, y_pred, _ = evaluate(model, val_loader)
        val_acc = accuracy_score(y_true, y_pred)
        scheduler.step()
        if val_acc > best_acc:
            best_acc   = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{EPOCHS_PHASE2} | "
                  f"Loss: {loss:.4f} | Val Acc: {val_acc:.4f} | Best: {best_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"✅ {model_key} done. Best val acc: {best_acc:.4f}")
    return model

# ─── 7. GRAD-CAM++ XAI ───────────────────────────────────────────────────────
def get_target_layer(model, model_key):
    """Return the last convolutional layer for each architecture."""
    mapping = {
        "EfficientNet-B0": model.blocks[-1],
        "ResNet-50":       model.layer4[-1],
        "VGG-16":          model.features[-1],
        "DenseNet-121":    model.features.denseblock4.denselayer16.conv2,
    }
    return mapping[model_key]

def gradcam_explain_all_models(trained_models, test_dataset,
                               class_names, n_samples=NUM_XAI_SAMPLES):
    """
    Big comparison figure:
      Rows    = mineral classes
      Columns = Original + one column per DL model
    """
    n_classes = len(class_names)
    n_models  = len(trained_models)
    n_cols    = 1 + n_models  # original + gradcam per model

    val_transform = get_dl_transforms(IMG_SIZE, augment=False)

    # Collect one sample per class
    class_samples = {c: [] for c in range(n_classes)}
    for idx in range(len(test_dataset)):
        img_path, label = test_dataset.image_paths[idx], test_dataset.labels[idx]
        if len(class_samples[label]) < n_samples:
            class_samples[label].append(img_path)
        if all(len(v) >= n_samples for v in class_samples.values()):
            break

    fig, axes = plt.subplots(
        n_classes * n_samples, n_cols,
        figsize=(n_cols * 3.5, n_classes * n_samples * 3.5)
    )
    col_titles = ["Original"] + list(trained_models.keys())
    for ci, ct in enumerate(col_titles):
        axes[0, ci].set_title(ct, fontsize=11, fontweight='bold', pad=8)

    for ci, cname in enumerate(class_names):
        for si in range(n_samples):
            row = ci * n_samples + si
            img_path = class_samples[ci][si]
            pil_img  = Image.open(img_path).convert("RGB")
            orig_arr = np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))) / 255.0

            # Column 0: original
            axes[row, 0].imshow(orig_arr)
            axes[row, 0].set_ylabel(f"{cname}\n#{si+1}", fontsize=9)
            axes[row, 0].axis('off')

            tensor = val_transform(pil_img).unsqueeze(0).to(DEVICE)

            for mi, (model_key, model) in enumerate(trained_models.items()):
                model.eval()
                target_layer = get_target_layer(model, model_key)
                cam = GradCAMPlusPlus(
                    model=model,
                    target_layers=[target_layer],
                )
                grayscale_cam = cam(
                    input_tensor=tensor,
                    targets=[ClassifierOutputTarget(ci)]
                )[0]
                rgb_img = np.float32(orig_arr)
                vis     = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                axes[row, mi + 1].imshow(vis)
                axes[row, mi + 1].axis('off')

    plt.suptitle("Grad-CAM++ Comparison: All Pretrained Models × All Mineral Classes",
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/xai_gradcam_comparison.png",
                dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Grad-CAM++ comparison figure saved.")

# ─── 8. RESULTS & REPORTING ──────────────────────────────────────────────────
def compile_results_table(classical_results, dl_results):
    rows = []
    for name, r in classical_results.items():
        rows.append({
            "Model":     name,
            "Type":      "Classical",
            "Accuracy":  round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1-Score":  round(r["f1"],        4),
        })
    for name, r in dl_results.items():
        rows.append({
            "Model":     name,
            "Type":      "Deep Learning",
            "Accuracy":  round(r["accuracy"],  4),
            "Precision": round(r["precision"], 4),
            "Recall":    round(r["recall"],    4),
            "F1-Score":  round(r["f1"],        4),
        })
    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    df.to_csv(f"{OUTPUT_PATH}/all_model_results.csv", index=False)
    print("\n📊 FINAL RESULTS TABLE")
    print(df.to_string(index=False))
    return df

def plot_combined_metrics(results_df):
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors  = {"Classical": "#4C72B0", "Deep Learning": "#DD8452"}

    for ax, metric in zip(axes, metrics):
        for _, row in results_df.iterrows():
            ax.barh(row["Model"], row[metric],
                    color=colors[row["Type"]], edgecolor='white')
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric)
        ax.axvline(x=0.876, color='red', linestyle='--',
                   linewidth=1.2, label='Paper KNN baseline')
        ax.legend(fontsize=8)
        ax.grid(axis='x', alpha=0.3)

    plt.suptitle("All Models — Performance Metrics vs. Paper Baseline",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/all_model_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_all_confusion_matrices(classical_results, dl_results,
                                class_names, y_test_classical, y_test_dl, dl_predictions):
    all_results = (
        [(n, r["y_pred"], "Classical") for n, r in classical_results.items()] +
        [(n, dl_predictions[n], "Deep Learning") for n in dl_results]
    )
    n = len(all_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4.5))
    axes = axes.ravel()

    for i, (name, y_pred, mtype) in enumerate(all_results):
        y_true = y_test_classical if mtype == "Classical" else y_test_dl
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i])
        axes[i].set_title(f"{name}\n({mtype})")
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/all_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_per_class_metrics(all_reports, class_names, output_path):
    """Plot per-class precision, recall, and F1-score heatmaps."""
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metrics_to_plot = ['precision', 'recall', 'f1-score']

    for ax, (model_name, report) in zip(axes, all_reports.items()):
        data = pd.DataFrame({
            cls: {m: report[cls][m] for m in metrics_to_plot}
            for cls in class_names
        }).T

        sns.heatmap(
            data, annot=True, fmt='.3f',
            cmap='YlOrRd', vmin=0.7, vmax=1.0,
            ax=ax, linewidths=0.5
        )
        ax.set_title(f"{model_name}", fontweight='bold')
        ax.set_xlabel("Metric")
        ax.set_ylabel("Mineral Class")

    plt.suptitle("Per-Class Precision, Recall and F1-Score — All Models",
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{output_path}/per_class_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()

# ─── 9. MAIN EXECUTION ───────────────────────────────────────────────────────
def main():
    print("🚀 MUMDMC2025 Explainable Mineral Classification")
    print("=" * 60)

    # ── 9.1 Load data ─────────────────────────────────────────────────────
    image_paths, labels, class_names = load_dataset(DATASET_PATH)
    num_classes = len(class_names)
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)

    # ── 9.2 Train/test split ──────────────────────────────────────────────
    (train_paths, test_paths,
     train_labels, test_labels) = train_test_split(
        image_paths, labels_enc,
        test_size=0.2, random_state=RANDOM_STATE, stratify=labels_enc
    )

    # ── 9.3 Classical ML ──────────────────────────────────────────────────
    print("\n[1/4] Building classical ML features (150×150 flatten)...")
    X_all, raw_all = build_sklearn_features(image_paths)

    classical_pipeline = ClassicalMLPipeline(
        X_all, [class_names[i] for i in labels_enc], class_names, raw_all
    )
    classical_pipeline.train()

    # ── 9.4 Deep learning DataLoaders ─────────────────────────────────────
    print("\n[2/4] Building PyTorch DataLoaders...")
    train_dataset = MineralDataset(train_paths, train_labels,
                                   transform=get_dl_transforms(IMG_SIZE, augment=True))
    test_dataset  = MineralDataset(test_paths,  test_labels,
                                   transform=get_dl_transforms(IMG_SIZE, augment=False))
    
    # Keep raw test dataset for Grad-CAM (needs original PIL paths)
    test_dataset_raw = RawTestDataset(test_paths, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # ── 9.5 Train all DL models ───────────────────────────────────────────
    print("\n[3/4] Training pretrained models (two-phase fine-tuning)...")
    trained_models = {}
    dl_results     = {}
    dl_predictions = {}

    for model_key in DL_MODELS:
        model = train_dl_model(model_key, train_loader, test_loader, num_classes)
        y_true, y_pred, y_prob = evaluate(model, test_loader)

        trained_models[model_key] = model
        dl_predictions[model_key] = y_pred
        dl_results[model_key] = {
            "accuracy":  accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall":    recall_score(y_true, y_pred, average='weighted'),
            "f1":        f1_score(y_true, y_pred, average='weighted'),
            "y_true":    y_true,
            "y_prob":    y_prob,
        }
        print(f"  {model_key}: "
              f"Acc={dl_results[model_key]['accuracy']:.4f} | "
              f"F1={dl_results[model_key]['f1']:.4f}")

        # Save model checkpoint
        torch.save(model.state_dict(),
                   f"{OUTPUT_PATH}/{model_key.replace('-','_').replace(' ','_')}.pth")

    # ── 9.6 XAI ───────────────────────────────────────────────────────────
    print("\n[4/4] Generating XAI explanations...")

    # Decision Tree
    classical_pipeline.explain_dt()

    # KNN LIME
    classical_pipeline.explain_knn_lime(n_samples=500)

    # Grad-CAM++ for all DL models
    gradcam_explain_all_models(
        trained_models, test_dataset_raw, class_names, n_samples=NUM_XAI_SAMPLES
    )

    # ── 9.7 Per-class metrics and reporting ───────────────────────────────
    # Generate classification reports
    all_reports = {}
    for model_name, y_pred in {**classical_pipeline.results, **dl_predictions}.items():
        y_true = (classical_pipeline.y_test 
                  if model_name in ["DecisionTree", "KNN"] 
                  else dl_results[model_name]["y_true"])
        
        print(f"\n{'='*55}")
        print(f"Classification Report — {model_name}")
        print(f"{'='*55}")
        report = classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4
        )
        print(report)
        
        report_dict = classification_report(
            y_true, y_pred,
            target_names=class_names,
            digits=4,
            output_dict=True
        )
        all_reports[model_name] = report_dict
    
    # Plot per-class metrics
    plot_per_class_metrics(all_reports, class_names, OUTPUT_PATH)
    
    # Save per-class metrics to CSV
    rows = []
    for model_name, report in all_reports.items():
        for cls in class_names:
            rows.append({
                "Model":     model_name,
                "Class":     cls,
                "Precision": round(report[cls]["precision"], 4),
                "Recall":    round(report[cls]["recall"],    4),
                "F1-Score":  round(report[cls]["f1-score"],  4),
                "Support":   int(report[cls]["support"]),
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUTPUT_PATH}/per_class_metrics.csv", index=False)
    print("\n✅ Per-class metrics saved.")

    # ── 9.8 Final results tables and plots ────────────────────────────────
    results_df = compile_results_table(
        classical_pipeline.results, dl_results
    )
    plot_combined_metrics(results_df)
    plot_all_confusion_matrices(
        classical_pipeline.results, dl_results,
        class_names,
        classical_pipeline.y_test,
        dl_results[list(dl_results.keys())[0]]["y_true"],
        dl_predictions
    )

    print("\n🎉 Pipeline complete!")
    print(f"📁 All outputs saved to: {OUTPUT_PATH}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_PATH)):
        size = os.path.getsize(f"{OUTPUT_PATH}/{f}") // 1024
        print(f"  {f}  ({size} KB)")

if __name__ == "__main__":
    main()
