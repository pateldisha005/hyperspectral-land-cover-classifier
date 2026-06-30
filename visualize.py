"""
Visualization module for land cover classification results.

Produces:
  - False-colour RGB composite of the input image
  - NDVI heatmap
  - Predicted vs. ground-truth land cover maps (side by side)
  - Per-class accuracy bar chart
  - Optional interactive Folium map (if lat/lon bounds are known)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix
import seaborn as sns

CLASS_NAMES  = ["Water", "Forest", "Urban", "Agriculture", "Bare Soil"]
CLASS_COLORS = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e", "#bcbd22"]
CMAP = ListedColormap(CLASS_COLORS)


def plot_rgb(image: np.ndarray, save_path="outputs/rgb_composite.png"):
    """RGB composite using Red (B4), Green (B3), Blue (B2) bands."""
    import os; os.makedirs("outputs", exist_ok=True)
    rgb = image[:, :, [2, 1, 0]]  # R, G, B bands
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title("False-Colour RGB Composite (B4-B3-B2)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_ndvi(image: np.ndarray, save_path="outputs/ndvi.png"):
    """NDVI = (NIR - Red) / (NIR + Red). Higher = denser vegetation."""
    import os; os.makedirs("outputs", exist_ok=True)
    NIR, RED = image[:, :, 6], image[:, :, 2]
    ndvi = (NIR - RED) / (NIR + RED + 1e-8)
    plt.figure(figsize=(7, 6))
    im = plt.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    plt.colorbar(im, label="NDVI")
    plt.title("NDVI — Normalized Difference Vegetation Index")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_classification(pred_map: np.ndarray, true_map: np.ndarray = None,
                        save_path="outputs/classification_map.png"):
    """Side-by-side predicted vs. ground truth land cover maps."""
    import os; os.makedirs("outputs", exist_ok=True)
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(CLASS_COLORS, CLASS_NAMES)]

    if true_map is not None:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        axes[0].imshow(true_map, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
        axes[0].set_title("Ground Truth")
        axes[0].axis("off")
        axes[1].imshow(pred_map, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
        axes[1].set_title("Predicted")
        axes[1].axis("off")
        fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10)
    else:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(pred_map, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title("Predicted Land Cover")
        ax.axis("off")
        fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save_path="outputs/confusion_matrix.png"):
    import os; os.makedirs("outputs", exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def plot_feature_importance(clf, n_bands=12, save_path="outputs/feature_importance.png"):
    import os; os.makedirs("outputs", exist_ok=True)
    band_names = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A",
                  "B09", "B10", "B11", "B12", "NDVI", "NDWI", "NDBI", "EVI"]
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 5))
    plt.bar(range(len(importances)), importances[indices], color="#2ca02c")
    plt.xticks(range(len(importances)), [band_names[i] for i in indices], rotation=45)
    plt.title("Random Forest Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")
