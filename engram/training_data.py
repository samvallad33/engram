"""
ENGRAM FSRS-Weighted Training Data Generator
Generates curriculum-ordered training examples for LoRA fine-tuning.
"""
from __future__ import annotations
import json, random
from dataclasses import dataclass, field
from engram.mock_engine import CLINICAL_DATA, DEFAULT_CLINICAL

CATEGORY_DIFFICULTY = {
    "No Finding": 2.5, "Cardiomegaly": 3.2, "Support Devices": 3.8, "Fracture": 5.5,
    "Pneumothorax": 5.8, "Pleural Effusion": 6.0, "Consolidation": 6.2, "Lung Opacity": 6.5,
    "Pneumonia": 7.0, "Edema": 7.5, "Atelectasis": 8.2,
}

@dataclass
class TrainingExample:
    category: str; skill_level: str; student_answer: str; correct_findings: list[str]; missed_findings: list[str]; score: float; fsrs_difficulty: float; fsrs_weight: float; example_type: str = "grading"; messages: list[dict] = field(default_factory=list)

def generate_grading_example(cat: str) -> TrainingExample:
    data = CLINICAL_DATA.get(cat, DEFAULT_CLINICAL)
    found = [f for f in data["findings"] if random.random() < 0.5]
    missed = [f for f in data["findings"] if f not in found]
    
    score = min(1.0, max(0.0, len(found) / max(1, len(data["findings"])) + random.uniform(-0.05, 0.05)))
    d = CATEGORY_DIFFICULTY.get(cat, 5.0)
    w = 0.5 + 1.5 * (d / 10.0)

    comp = json.dumps({"score": round(score,3), "correct_findings": found, "missed_findings": missed, "false_positives": [], "explanation": f"Assessment: {'Excellent' if score > 0.7 else 'Needs Work'}"}, indent=2)
    msgs = [{"role": "user", "content": f"Grade this. Category: {cat}\nFindings: {data['findings']}\nStudent: I see {found}"}, {"role": "assistant", "content": f"```json\n{comp}\n```"}]
    
    return TrainingExample(cat, "mixed", f"I see {found}", found, missed, score, d, w, "grading", msgs)

def generate_curriculum_dataset(n_examples: int = 1000, seed: int = 42) -> list[TrainingExample]:
    random.seed(seed)
    weights = [CATEGORY_DIFFICULTY.get(c, 5.0) for c in CLINICAL_DATA.keys()]
    cats = random.choices(list(CLINICAL_DATA.keys()), weights=weights, k=n_examples)
    
    examples = [generate_grading_example(c) for c in cats]
    examples.sort(key=lambda x: (x.fsrs_difficulty, random.random())) # Easy to Hard
    return examples