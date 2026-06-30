# Hyperspectral Land Cover Classifier

Classifies land cover types (water, forest, urban, agriculture, bare soil) from Sentinel-2 multispectral satellite imagery using a Random Forest trained on spectral signatures and derived indices (NDVI, NDWI, NDBI, EVI).

Built as an extension of hyperspectral data processing work from the FINCH CubeSat project at the University of Toronto.

## What it does

- Loads 12-band Sentinel-2-style imagery (or generates a realistic synthetic dataset offline)
- Computes spectral indices: NDVI (vegetation), NDWI (water), NDBI (built-up), EVI (enhanced vegetation)
- Trains a Random Forest pixel classifier on labelled pixels
- Produces a full land cover map with per-class accuracy metrics
- Outputs RGB composites, NDVI heatmaps, confusion matrix, and feature importance plots

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

All outputs are saved to `outputs/`.

## Real Sentinel-2 data

1. Go to [Copernicus Browser](https://browser.dataspace.copernicus.eu/) (free, no account needed for search)
2. Select a cloud-free scene → download **L2A** product (surface reflectance)
3. Place the `.tif` band files in `data/sentinel2/`
4. Update `BAND_PATHS` in `classifier.py` to point to your files

## Project structure

```
download_data.py   — data download + synthetic data generation
classifier.py      — feature engineering, model training, inference
visualize.py       — all plots and maps
main.py            — end-to-end pipeline entry point
outputs/           — generated figures (git-ignored)
models/            — saved model weights (git-ignored)
```

## Key technical decisions

**Why Random Forest?** Interpretable, handles correlated spectral bands well, and fast to train — making it practical for a satellite scene without a GPU. The feature importance plot directly shows which bands and indices drive the classification.

**Why spectral indices?** Raw reflectance values alone don't fully separate classes. NDVI separates vegetation from non-vegetation; NDWI isolates water; NDBI highlights urban surfaces. Together with raw bands, they give the classifier a richer signal.

**Scalability:** The pixel-wise approach scales linearly with image size. For very large scenes, the image can be tiled and processed in chunks.

## Sample outputs

After running `main.py` you'll find in `outputs/`:
- `rgb_composite.png` — true-colour satellite view
- `ndvi.png` — vegetation index heatmap
- `classification_map.png` — predicted vs. ground truth side by side
- `confusion_matrix.png` — per-class accuracy breakdown
- `feature_importance.png` — which spectral bands matter most

## Dependencies

`numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `rasterio` (for real GeoTIFF data), `scipy`, `pandas`
