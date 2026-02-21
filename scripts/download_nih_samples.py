#!/usr/bin/env python3
"""
Download sample CXR images from the NIH ChestX-ray14 dataset.

The NIH Clinical Center ChestX-ray14 dataset is public domain (CC0 1.0).
Source: https://nihcc.app.box.com/v/ChestXray-NIHCC

Usage:
    # Using Kaggle API (recommended — fastest, uses cached dataset):
    pip install kaggle
    python scripts/download_nih_samples.py --kaggle

    # Or manually: download from NIH, extract, and run:
    python scripts/download_nih_samples.py --source /path/to/extracted/images/

    # Or on Kaggle notebooks, images are already at:
    /kaggle/input/nih-chest-xrays/images/

This script copies 3 sample images per pathology category (33 total)
into data/demo/{Category}/ for the ENGRAM training interface.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# NIH ChestX-ray14 label mapping to ENGRAM categories
NIH_CATEGORIES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",       # Maps to "Pleural Effusion"
    "Fracture",
    "Infiltration",   # Maps to "Lung Opacity"
    "No Finding",
    "Pneumonia",
    "Pneumothorax",
]

NIH_TO_ENGRAM = {
    "Effusion": "Pleural Effusion",
    "Infiltration": "Lung Opacity",
}

ENGRAM_CATEGORIES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Fracture", "Lung Opacity", "No Finding", "Pleural Effusion",
    "Pneumonia", "Pneumothorax", "Support Devices",
]

SAMPLES_PER_CATEGORY = 3
DATA_DIR = Path(__file__).parent.parent / "data" / "demo"


def parse_nih_labels(csv_path: str) -> dict[str, list[str]]:
    """Parse Data_Entry_2017.csv → {category: [image_filenames]}."""
    category_images: dict[str, list[str]] = {c: [] for c in NIH_CATEGORIES}

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels = row["Finding Labels"].split("|")
            for label in labels:
                label = label.strip()
                if label in category_images:
                    category_images[label].append(row["Image Index"])

    # Also find "Support Devices" (Hernia is rare, use images with multiple findings)
    return category_images


def copy_samples(
    image_dir: str,
    csv_path: str | None = None,
    label_map: dict[str, list[str]] | None = None,
):
    """Copy sample images from NIH dataset into ENGRAM demo directory."""
    image_dir = Path(image_dir)

    if csv_path and not label_map:
        label_map = parse_nih_labels(csv_path)

    if not label_map:
        print("Error: Need either --csv or pre-parsed label map.")
        sys.exit(1)

    copied = 0
    for nih_cat, filenames in label_map.items():
        engram_cat = NIH_TO_ENGRAM.get(nih_cat, nih_cat)
        if engram_cat not in ENGRAM_CATEGORIES:
            continue

        cat_dir = DATA_DIR / engram_cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Pick random samples
        available = [f for f in filenames if (image_dir / f).exists()]
        if not available:
            print(f"  Warning: No images found for {engram_cat}")
            continue

        samples = random.sample(available, min(SAMPLES_PER_CATEGORY, len(available)))

        for i, filename in enumerate(samples, 1):
            src = image_dir / filename
            dst = cat_dir / f"case_{i:03d}.png"
            shutil.copy2(str(src), str(dst))
            copied += 1

        print(f"  {engram_cat}: {len(samples)} images copied")

    # Support Devices — use No Finding as fallback if not available
    sd_dir = DATA_DIR / "Support Devices"
    if not any(sd_dir.glob("*.png")):
        sd_dir.mkdir(parents=True, exist_ok=True)
        nf_files = list((DATA_DIR / "No Finding").glob("*.png"))
        for i, f in enumerate(nf_files[:SAMPLES_PER_CATEGORY], 1):
            shutil.copy2(str(f), str(sd_dir / f"case_{i:03d}.png"))
        print(f"  Support Devices: {min(SAMPLES_PER_CATEGORY, len(nf_files))} images (fallback)")

    print(f"\nDone. {copied} images copied to {DATA_DIR}")


def setup_from_kaggle():
    """Download NIH ChestX-ray14 dataset via Kaggle API."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("Install kaggle: pip install kaggle")
        print("Then set up API credentials: https://www.kaggle.com/docs/api")
        sys.exit(1)

    tmp_dir = os.path.join(tempfile.gettempdir(), "nih_cxr")
    print("Downloading NIH ChestX-ray14 metadata...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "nih-chest-xrays/data",
         "-f", "Data_Entry_2017.csv", "-p", tmp_dir, "--force"],
        check=True,
    )
    print("Downloading sample images (this may take a while)...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "nih-chest-xrays/sample",
         "-p", os.path.join(tmp_dir, "images"), "--force", "--unzip"],
        check=True,
    )

    csv_path = os.path.join(tmp_dir, "Data_Entry_2017.csv")
    if Path(csv_path).exists():
        copy_samples(os.path.join(tmp_dir, "images"), csv_path=csv_path)
    else:
        print(f"Could not find {csv_path}. Download manually from NIH.")


def main():
    parser = argparse.ArgumentParser(description="Download sample CXRs for ENGRAM demo")
    parser.add_argument("--kaggle", action="store_true", help="Use Kaggle API to download")
    parser.add_argument("--source", type=str, help="Path to extracted NIH images directory")
    parser.add_argument("--csv", type=str, help="Path to Data_Entry_2017.csv")
    args = parser.parse_args()

    if args.kaggle:
        setup_from_kaggle()
    elif args.source:
        if not args.csv:
            # Look for CSV in standard locations
            for candidate in [
                Path(args.source).parent / "Data_Entry_2017.csv",
                Path(args.source) / "Data_Entry_2017.csv",
            ]:
                if candidate.exists():
                    args.csv = str(candidate)
                    break
        if not args.csv:
            print("Provide --csv path to Data_Entry_2017.csv")
            sys.exit(1)
        copy_samples(args.source, csv_path=args.csv)
    else:
        print("Usage:")
        print("  python scripts/download_nih_samples.py --kaggle")
        print("  python scripts/download_nih_samples.py --source /path/to/images/ --csv /path/to/Data_Entry_2017.csv")
        print()
        print("On Kaggle notebooks, images are at: /kaggle/input/nih-chest-xrays/images/")


if __name__ == "__main__":
    main()
