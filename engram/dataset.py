"""ENGRAM Dataset Management"""
from __future__ import annotations
import os, uuid
from dataclasses import dataclass
from pathlib import Path
from .fsrs6 import Card, FSRSState

@dataclass
class DatasetCase:
    image_path: str; labels: dict[str, float]; view: str = "frontal"

CATEGORY_DESCRIPTIONS = {
    "Cardiomegaly": "Enlarged cardiac silhouette, CTR > 0.5",
    "Pneumothorax": "Visceral pleural line, absent lung markings",
    "Pleural Effusion": "Blunting of costophrenic angle, meniscus sign",
    "Consolidation": "Dense opacity with air bronchograms",
    "Lung Opacity": "Patchy airspace opacity",
    "Atelectasis": "Volume loss, elevated hemidiaphragm",
    "Edema": "Bilateral perihilar haziness, Kerley B lines",
    "Pneumonia": "Focal consolidation with air bronchograms",
    "Fracture": "Cortical disruption of rib",
    "No Finding": "No acute cardiopulmonary abnormality",
    "Support Devices": "ETT tip 3cm above carina",
}

def load_demo_dataset(demo_dir: str) -> list[Card]:
    cards = []
    demo_path = Path(demo_dir)
    if not demo_path.exists(): return cards
    for cat_dir in sorted(d for d in demo_path.iterdir() if d.is_dir()):
        for img in list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")):
            cards.append(Card(str(uuid.uuid4())[:8], cat_dir.name, str(img), cat_dir.name, FSRSState()))
    return cards