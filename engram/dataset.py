"""
ENGRAM Dataset Management
Handles medical image datasets (CheXpert, MIMIC-CXR, or custom).
Creates Card objects with FSRS states for the training system.
"""

from __future__ import annotations

import csv
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from .fsrs6 import Card, FSRSState


# CheXpert pathology labels
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# Teaching-friendly category names
CATEGORY_DESCRIPTIONS = {
    "No Finding": "Normal chest X-ray with no significant findings",
    "Enlarged Cardiomediastinum": "Widened mediastinal silhouette suggesting vascular or lymph pathology",
    "Cardiomegaly": "Enlarged cardiac silhouette (CTR > 0.5)",
    "Lung Opacity": "Opacification in lung fields — could indicate infection, fluid, or mass",
    "Lung Lesion": "Discrete lesion visible in lung parenchyma",
    "Edema": "Pulmonary edema — fluid in the lungs, often from heart failure",
    "Consolidation": "Dense opacification suggesting alveolar filling (pneumonia, hemorrhage)",
    "Pneumonia": "Infectious consolidation pattern",
    "Atelectasis": "Partial or complete lung collapse",
    "Pneumothorax": "Air in the pleural space — can be life-threatening",
    "Pleural Effusion": "Fluid collection in the pleural space",
    "Pleural Other": "Other pleural abnormality",
    "Fracture": "Bone fracture visible on X-ray",
    "Support Devices": "Lines, tubes, and devices (ETT, central line, chest tube, etc.)",
}


@dataclass
class DatasetCase:
    """A single case from the dataset."""
    image_path: str
    labels: dict[str, float]  # label -> confidence (1.0, 0.0, or -1.0 uncertain)
    patient_id: str = ""
    study_id: str = ""
    view: str = "frontal"     # frontal or lateral


def load_chexpert_csv(
    csv_path: str,
    image_root: str = "",
    max_cases: int = 0,
    frontal_only: bool = True,
) -> list[DatasetCase]:
    """
    Load CheXpert dataset from CSV.
    Handles uncertainty labels: -1 -> treated as positive (U-Ones strategy).
    """
    cases = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Filter to frontal views
            if frontal_only and row.get("Frontal/Lateral", "Frontal") != "Frontal":
                continue

            path = row.get("Path", "")
            if image_root:
                path = os.path.join(image_root, path)

            labels = {}
            for label in CHEXPERT_LABELS:
                val = row.get(label, "")
                if val == "" or val == " ":
                    labels[label] = 0.0
                else:
                    v = float(val)
                    labels[label] = 1.0 if v == -1.0 else v  # U-Ones strategy

            cases.append(DatasetCase(
                image_path=path,
                labels=labels,
                patient_id=row.get("Patient", ""),
                study_id=row.get("Study", ""),
                view="frontal" if row.get("Frontal/Lateral") == "Frontal" else "lateral",
            ))

            if max_cases and len(cases) >= max_cases:
                break

    return cases


def create_cards_from_cases(
    cases: list[DatasetCase],
    categories: list[str] | None = None,
    max_per_category: int = 50,
) -> list[Card]:
    """
    Create ENGRAM cards from dataset cases.
    Each card represents a diagnostic challenge for one finding category.
    """
    if categories is None:
        categories = [l for l in CHEXPERT_LABELS if l != "No Finding"]

    cards_by_cat: dict[str, list[Card]] = {cat: [] for cat in categories}

    for case in cases:
        positive_labels = [l for l in categories if case.labels.get(l, 0.0) == 1.0]

        for label in positive_labels:
            if len(cards_by_cat.get(label, [])) >= max_per_category:
                continue

            card = Card(
                card_id=str(uuid.uuid4())[:8],
                category=label,
                image_path=case.image_path,
                label=label,
                fsrs=FSRSState(),
            )
            cards_by_cat[label].append(card)

    # Also add normal cases
    normal_cases = [c for c in cases if c.labels.get("No Finding", 0.0) == 1.0]
    for case in normal_cases[:max_per_category]:
        card = Card(
            card_id=str(uuid.uuid4())[:8],
            category="No Finding",
            image_path=case.image_path,
            label="No Finding",
            fsrs=FSRSState(),
        )
        cards_by_cat.setdefault("No Finding", []).append(card)

    # Flatten
    all_cards = []
    for cat_cards in cards_by_cat.values():
        all_cards.extend(cat_cards)

    return all_cards


def load_demo_dataset(demo_dir: str) -> list[Card]:
    """
    Load a small demo dataset from a local directory.
    Expects: demo_dir/{category_name}/image_001.jpg, etc.
    """
    cards = []
    demo_path = Path(demo_dir)

    if not demo_path.exists():
        return cards

    for category_dir in sorted(demo_path.iterdir()):
        if not category_dir.is_dir():
            continue

        category = category_dir.name
        for img_file in sorted(category_dir.glob("*.jpg")) + sorted(category_dir.glob("*.png")):
            card = Card(
                card_id=str(uuid.uuid4())[:8],
                category=category,
                image_path=str(img_file),
                label=category,
                fsrs=FSRSState(),
            )
            cards.append(card)

    return cards
