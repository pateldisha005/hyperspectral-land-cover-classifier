"""
Entry point — runs the full pipeline:
  1. Generate synthetic Sentinel-2-style data (or load real data)
  2. Train Random Forest classifier
  3. Predict land cover map
  4. Produce all visualizations in outputs/
"""

import numpy as np
from download_data import create_synthetic_sentinel2
from classifier import train, predict_map
from visualize import plot_rgb, plot_ndvi, plot_classification, plot_confusion_matrix, plot_feature_importance
from sklearn.model_selection import train_test_split


def main():
    print("=" * 50)
    print("  Sentinel-2 Land Cover Classifier")
    print("=" * 50)

    # Step 1 — Data
    print("\n[1/4] Preparing data...")
    image, labels = create_synthetic_sentinel2()

    # Step 2 — Train
    print("\n[2/4] Training classifier...")
    clf, scaler, image_feat, labels = train()

    # Step 3 — Predict full map
    print("\n[3/4] Predicting land cover map...")
    pred_map = predict_map(image, clf, scaler)

    # Step 4 — Visualize
    print("\n[4/4] Generating visualizations...")
    plot_rgb(image)
    plot_ndvi(image)
    plot_classification(pred_map, labels)
    plot_feature_importance(clf)

    # Confusion matrix on held-out pixels
    from classifier import compute_spectral_indices, flatten_pixels
    from sklearn.preprocessing import StandardScaler

    img_feat = compute_spectral_indices(image)
    X, y = flatten_pixels(img_feat, labels)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    y_pred = clf.predict(X_test_scaled)
    plot_confusion_matrix(y_test, y_pred)

    print("\nDone! All outputs saved to outputs/")
    print("Open outputs/ to see the classification maps and analysis plots.")


if __name__ == "__main__":
    main()
