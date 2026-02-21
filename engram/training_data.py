"""
ENGRAM FSRS-Weighted Training Data Generator
Generates curriculum-ordered training examples for LoRA fine-tuning MedGemma.

Innovation: Uses FSRS-6 difficulty signals to weight training examples.
Cases that students struggle with (high D, many lapses) get higher training
weight, teaching MedGemma to be better at explaining exactly what's hard.

This is the co-evolutionary data flywheel: human memory science drives
machine learning optimization.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

from .mock_engine import CLINICAL_DATA, DEFAULT_CLINICAL


# ─── Student Skill Levels ──────────────────────────────────────

SKILL_LEVELS = {
    "novice": {
        "finding_rate": 0.15,
        "uses_jargon": False,
        "makes_errors": True,
        "score_range": (0.05, 0.25),
    },
    "beginner": {
        "finding_rate": 0.35,
        "uses_jargon": False,
        "makes_errors": True,
        "score_range": (0.20, 0.45),
    },
    "intermediate": {
        "finding_rate": 0.55,
        "uses_jargon": True,
        "makes_errors": True,
        "score_range": (0.40, 0.65),
    },
    "advanced": {
        "finding_rate": 0.80,
        "uses_jargon": True,
        "makes_errors": False,
        "score_range": (0.65, 0.85),
    },
    "expert": {
        "finding_rate": 0.95,
        "uses_jargon": True,
        "makes_errors": False,
        "score_range": (0.80, 1.00),
    },
}


# ─── FSRS-6 Difficulty Per Category ────────────────────────────
# Based on radiology education literature: satisfaction-of-search
# error rates, diagnostic miss rates, and inter-reader variability.

CATEGORY_DIFFICULTY = {
    "No Finding": 2.5,
    "Cardiomegaly": 3.2,
    "Support Devices": 3.8,
    "Fracture": 5.5,
    "Pneumothorax": 5.8,
    "Pleural Effusion": 6.0,
    "Consolidation": 6.2,
    "Lung Opacity": 6.5,
    "Pneumonia": 7.0,
    "Edema": 7.5,
    "Atelectasis": 8.2,
}


# ─── Data Generation ───────────────────────────────────────────

@dataclass
class TrainingExample:
    """A single FSRS-weighted training example."""
    category: str
    skill_level: str
    student_answer: str
    correct_findings: list[str]
    missed_findings: list[str]
    score: float
    fsrs_difficulty: float
    fsrs_weight: float
    example_type: str = "grading"
    messages: list[dict] = field(default_factory=list)


def _generate_student_response(
    category: str, skill_level: str,
) -> tuple[str, list[str], list[str]]:
    """Generate a simulated student response at a given skill level."""
    data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
    config = SKILL_LEVELS[skill_level]
    all_findings = data["findings"]

    found, missed = [], []
    for finding in all_findings:
        if random.random() < config["finding_rate"]:
            found.append(finding)
        else:
            missed.append(finding)

    parts = []
    if not found:
        if config["makes_errors"]:
            wrong_cats = [c for c in CLINICAL_DATA if c != category]
            wrong_cat = random.choice(wrong_cats)
            wrong_data = CLINICAL_DATA.get(wrong_cat, DEFAULT_CLINICAL)
            parts.append(f"This looks like it could be {wrong_cat.lower()}.")
            parts.append(f"I think I see {wrong_data['findings'][0].lower()}.")
        else:
            parts.append("I'm having difficulty identifying specific findings.")
            parts.append("The lung fields appear somewhat abnormal.")
    else:
        for f in found:
            if config["uses_jargon"]:
                parts.append(f"I identify {f.lower()}.")
            else:
                simplified = f.lower()
                simplified = simplified.replace("opacification", "white area")
                simplified = simplified.replace("costophrenic", "lower angle")
                parts.append(f"I see {simplified}.")

        if config["uses_jargon"] and len(found) >= 2:
            loc = random.choice(data["locations"])
            parts.append(f"Location: {loc.lower()}.")
            diff = random.choice(data["differentials"])
            parts.append(f"Differential includes {diff.lower()}.")

    return " ".join(parts), found, missed


def generate_grading_example(
    category: str, skill_level: str | None = None,
) -> TrainingExample:
    """Generate one FSRS-weighted diagnostic grading training example."""
    if skill_level is None:
        skill_level = random.choice(list(SKILL_LEVELS.keys()))

    data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
    config = SKILL_LEVELS[skill_level]

    student_answer, found, missed = _generate_student_response(category, skill_level)

    total = len(data["findings"])
    raw_score = len(found) / total if total > 0 else 0.0
    score = max(config["score_range"][0], min(config["score_range"][1], raw_score))
    score = round(score + random.uniform(-0.05, 0.05), 3)
    score = max(0.0, min(1.0, score))

    fsrs_d = CATEGORY_DIFFICULTY.get(category, 5.0)
    fsrs_weight = 0.5 + 1.5 * (fsrs_d / 10.0)

    teaching = random.choice(data["teaching"])

    if score >= 0.7:
        assessment = "Excellent"
    elif score >= 0.4:
        assessment = "Partial — key findings missed"
    else:
        assessment = "Needs significant improvement"

    explanation_parts = [f"**Assessment: {assessment}**"]
    if found:
        explanation_parts.append(
            f"You correctly identified: {', '.join(f[:50] for f in found)}."
        )
    if missed:
        explanation_parts.append(
            f"You missed: {', '.join(m[:50] for m in missed)}."
        )
    explanation_parts.append(f"\n**Teaching point:** {teaching}")

    completion = json.dumps({
        "score": score,
        "correct_findings": [f[:60] for f in found],
        "missed_findings": [m[:60] for m in missed],
        "false_positives": [],
        "explanation": "\n\n".join(explanation_parts),
    }, indent=2)

    prompt = (
        "You are an attending radiologist grading a medical student's "
        "interpretation of a chest X-ray.\n\n"
        f"**Category:** {category}\n"
        f"**Key findings:** {', '.join(f[:60] for f in data['findings'])}\n"
        f"**Student's answer:** {student_answer}\n\n"
        "Grade the student's response. Output ONLY valid JSON with: "
        "score (0-1), correct_findings, missed_findings, false_positives, "
        "explanation."
    )

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"```json\n{completion}\n```"},
    ]

    return TrainingExample(
        category=category,
        skill_level=skill_level,
        student_answer=student_answer,
        correct_findings=[f[:60] for f in found],
        missed_findings=[m[:60] for m in missed],
        score=score,
        fsrs_difficulty=fsrs_d,
        fsrs_weight=fsrs_weight,
        example_type="grading",
        messages=messages,
    )


def generate_socratic_example(category: str) -> TrainingExample:
    """Generate a Socratic mode training example."""
    data = CLINICAL_DATA.get(category, DEFAULT_CLINICAL)
    skill_level = random.choice(["novice", "beginner", "intermediate"])
    student_answer, found, missed = _generate_student_response(category, skill_level)

    if missed:
        target = missed[0]
        key_terms = [w for w in target.lower().split() if len(w) > 4][:2]
        hint = " or ".join(key_terms) if key_terms else category.lower()
        question = (
            f"You've made some observations, but consider: what about "
            f"{hint}? What specific feature would help confirm or exclude "
            f"{category.lower()}?"
        )
    else:
        question = (
            f"Good observations. Can you think of any differential "
            f"diagnoses or additional findings that would change management?"
        )

    prompt = (
        "You are a radiology professor using the Socratic method. "
        f"A student described a chest X-ray (category: {category}).\n\n"
        f"Student's answer: {student_answer}\n\n"
        "Ask ONE probing question that guides them toward findings they "
        "missed. Do not reveal the answer directly."
    )

    response = (
        f"**Socratic Question:**\n\n{question}\n\n"
        f"*Think about this before seeing the full answer.*"
    )

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    fsrs_d = CATEGORY_DIFFICULTY.get(category, 5.0)

    return TrainingExample(
        category=category,
        skill_level=skill_level,
        student_answer=student_answer,
        correct_findings=[f[:60] for f in found],
        missed_findings=[m[:60] for m in missed],
        score=0.5,
        fsrs_difficulty=fsrs_d,
        fsrs_weight=0.5 + 1.5 * (fsrs_d / 10.0),
        example_type="socratic",
        messages=messages,
    )


def generate_contrastive_example(cat_a: str, cat_b: str) -> TrainingExample:
    """Generate a contrastive pair grading training example."""
    data_a = CLINICAL_DATA.get(cat_a, DEFAULT_CLINICAL)
    data_b = CLINICAL_DATA.get(cat_b, DEFAULT_CLINICAL)

    key_a = data_a["findings"][0][:50] if data_a["findings"] else cat_a
    key_b = data_b["findings"][0][:50] if data_b["findings"] else cat_b

    prompt = (
        "You are a radiology professor teaching differential diagnosis.\n\n"
        f"A student is comparing two chest X-rays: one showing {cat_a}, "
        f"the other showing {cat_b}.\n\n"
        f"The student says: 'Both images show white areas in the lungs. "
        f"I'm not sure how to tell them apart.'\n\n"
        "Explain the KEY distinguishing features between these two conditions "
        "on chest X-ray."
    )

    response = (
        f"**Key Distinction: {cat_a} vs {cat_b}**\n\n"
        f"**{cat_a}:** {key_a}. {random.choice(data_a['teaching'])}\n\n"
        f"**{cat_b}:** {key_b}. {random.choice(data_b['teaching'])}\n\n"
        f"The critical differentiator is to look at the overall pattern, "
        f"distribution, and associated findings."
    )

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]

    fsrs_d = max(
        CATEGORY_DIFFICULTY.get(cat_a, 5.0),
        CATEGORY_DIFFICULTY.get(cat_b, 5.0),
    )

    return TrainingExample(
        category=f"{cat_a}_vs_{cat_b}",
        skill_level="intermediate",
        student_answer="Both images show white areas in the lungs.",
        correct_findings=[],
        missed_findings=[],
        score=0.5,
        fsrs_difficulty=fsrs_d,
        fsrs_weight=0.5 + 1.5 * (fsrs_d / 10.0),
        example_type="contrastive",
        messages=messages,
    )


def generate_curriculum_dataset(
    n_examples: int = 1000,
    include_socratic: bool = True,
    include_contrastive: bool = True,
    seed: int = 42,
) -> list[TrainingExample]:
    """
    Generate a full FSRS-weighted curriculum dataset.

    Harder categories (higher FSRS difficulty) get proportionally more
    training examples — the co-evolutionary flywheel principle.
    """
    random.seed(seed)

    categories = list(CLINICAL_DATA.keys())
    examples: list[TrainingExample] = []

    # Distribute examples weighted by FSRS difficulty
    total_d = sum(CATEGORY_DIFFICULTY.get(c, 5.0) for c in categories)

    for category in categories:
        d = CATEGORY_DIFFICULTY.get(category, 5.0)
        n_cat = max(5, int(n_examples * 0.65 * d / total_d))

        # Grading examples across all skill levels
        for _ in range(n_cat):
            examples.append(generate_grading_example(category))

        # Socratic examples for harder categories
        if include_socratic and d >= 5.0:
            n_soc = max(2, n_cat // 4)
            for _ in range(n_soc):
                examples.append(generate_socratic_example(category))

    # Contrastive pairs
    if include_contrastive:
        contrastive_pairs = [
            ("Consolidation", "Atelectasis"),
            ("Pleural Effusion", "Lung Opacity"),
            ("Cardiomegaly", "Edema"),
            ("Pneumonia", "Consolidation"),
            ("Edema", "Pneumonia"),
            ("Pneumothorax", "No Finding"),
        ]
        for cat_a, cat_b in contrastive_pairs:
            for _ in range(5):
                examples.append(generate_contrastive_example(cat_a, cat_b))

    # Pad to target
    while len(examples) < n_examples:
        cat = random.choice(categories)
        examples.append(generate_grading_example(cat))

    # Sort by FSRS difficulty (curriculum: easy → hard)
    examples.sort(key=lambda x: (x.fsrs_difficulty, random.random()))

    return examples[:n_examples]


def examples_to_jsonl(examples: list[TrainingExample], path: str) -> None:
    """Save training examples as JSONL for HuggingFace datasets."""
    with open(path, "w") as f:
        for ex in examples:
            record = {
                "messages": ex.messages,
                "category": ex.category,
                "skill_level": ex.skill_level,
                "example_type": ex.example_type,
                "fsrs_difficulty": ex.fsrs_difficulty,
                "fsrs_weight": ex.fsrs_weight,
            }
            f.write(json.dumps(record) + "\n")
