"""
Download sample Sentinel-2 data for land cover classification.

We use the EarthPy library's sample datasets which include Landsat/Sentinel-style
multispectral data — no account required. Alternatively, instructions are provided
for downloading real Sentinel-2 data from ESA Copernicus.
"""

import os
import urllib.request
import zipfile
import numpy as np

DATA_DIR = "data"


def download_sample_data():
    """
    Downloads a small multispectral sample dataset for demonstration.
    This uses NASA Earthdata's publicly accessible sample files.
    No login required.
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    # Using EarthPy's openly hosted sample Landsat 8 data (similar band structure to Sentinel-2)
    # Source: https://earthpy.readthedocs.io/en/latest/
    url = "https://ndownloader.figshare.com/files/12907582"
    zip_path = os.path.join(DATA_DIR, "cold-springs-landsat.zip")

    if not os.path.exists(zip_path):
        print("Downloading sample multispectral dataset (~40MB)...")
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete.")
    else:
        print("Data already downloaded.")

    extract_dir = os.path.join(DATA_DIR, "cold-springs-landsat")
    if not os.path.exists(extract_dir):
        print("Extracting...")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        print("Extraction complete.")

    return extract_dir


def create_synthetic_sentinel2(save_dir="data/synthetic"):
    """
    Creates a synthetic Sentinel-2-like multispectral image for offline use.

    Generates 12 bands matching Sentinel-2 band layout:
      B2 (Blue), B3 (Green), B4 (Red), B5-B7 (Red Edge),
      B8 (NIR), B8A (NIR narrow), B11, B12 (SWIR)

    Land cover classes are spatially clustered so the classifier
    has a realistic signal to learn from.
    """
    os.makedirs(save_dir, exist_ok=True)
    rng = np.random.default_rng(42)

    H, W = 256, 256
    n_bands = 12

    # Class map: 0=water, 1=forest, 2=urban, 3=agriculture, 4=bare soil
    class_map = np.zeros((H, W), dtype=np.uint8)
    class_map[0:60, :] = 0       # water
    class_map[60:130, 0:120] = 1  # forest
    class_map[60:130, 120:] = 3   # agriculture
    class_map[130:200, :] = 2     # urban
    class_map[200:, :] = 4        # bare soil

    # Spectral signatures per class (12 bands, normalised reflectance)
    # Based on typical Sentinel-2 values from literature
    signatures = {
        0: [0.05, 0.06, 0.04, 0.03, 0.03, 0.03, 0.02, 0.02, 0.03, 0.03, 0.02, 0.01],  # water
        1: [0.03, 0.07, 0.04, 0.12, 0.25, 0.35, 0.38, 0.38, 0.35, 0.34, 0.15, 0.08],  # forest
        2: [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],  # urban
        3: [0.05, 0.09, 0.06, 0.18, 0.38, 0.42, 0.44, 0.44, 0.40, 0.39, 0.20, 0.12],  # agriculture
        4: [0.12, 0.14, 0.15, 0.17, 0.20, 0.22, 0.23, 0.23, 0.22, 0.22, 0.20, 0.18],  # bare soil
    }

    image = np.zeros((H, W, n_bands), dtype=np.float32)
    noise_std = 0.015

    for cls, sig in signatures.items():
        mask = class_map == cls
        for b, val in enumerate(sig):
            image[:, :, b][mask] = val + rng.normal(0, noise_std, mask.sum())

    image = np.clip(image, 0, 1)

    np.save(os.path.join(save_dir, "image.npy"), image)
    np.save(os.path.join(save_dir, "labels.npy"), class_map)
    print(f"Synthetic data saved to {save_dir}/")
    print(f"  Image shape: {image.shape}  (H x W x Bands)")
    print(f"  Classes: 0=water, 1=forest, 2=urban, 3=agriculture, 4=bare_soil")
    return image, class_map


if __name__ == "__main__":
    print("=== Data Preparation ===")
    print("Generating synthetic Sentinel-2-style data (works fully offline)...")
    create_synthetic_sentinel2()
    print("\nTo use real Sentinel-2 imagery instead:")
    print("  1. Go to https://browser.dataspace.copernicus.eu/")
    print("  2. Search for a cloud-free scene")
    print("  3. Download L2A product (surface reflectance, already atmospherically corrected)")
    print("  4. Place .tif band files in data/sentinel2/")
    print("  5. Update BAND_PATHS in classifier.py to point to your files")
