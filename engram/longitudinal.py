"""
ENGRAM Longitudinal CXR Comparison Training
Teaches students to detect interval changes between prior and current imaging.

This is what radiologists do every day: "Compared to prior study from [date],
the pleural effusion has [increased/decreased/resolved]."

MedGemma 1.5 achieves 65.7% macro accuracy on MS-CXR-T longitudinal imaging
(+5% over v1.0, per official model card).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .medgemma import FeedbackResult


@dataclass
class LongitudinalCase:
    """A pair of images for interval change detection."""
    prior_path: str
    current_path: str
    category: str
    change_type: str  # "improved", "worsened", "stable", "new", "resolved"
    description: str


# ─── Clinical knowledge base for interval changes ─────────────────

LONGITUDINAL_DATA = {
    "Cardiomegaly": {
        "worsened": [
            "The cardiac silhouette has further enlarged compared to the prior study. "
            "The cardiothoracic ratio has increased from approximately 0.55 to 0.65. "
            "New bilateral pleural effusions suggest decompensated heart failure.",
            "Interval increase in heart size with new pulmonary venous congestion. "
            "Upper lobe cephalization is now present, suggesting worsening failure.",
        ],
        "improved": [
            "The cardiac silhouette has decreased in size compared to the prior study. "
            "Previously noted pulmonary vascular congestion has improved. "
            "Small bilateral effusions have largely resolved.",
        ],
        "stable": [
            "The cardiac silhouette remains stably enlarged, unchanged from prior. "
            "No new pulmonary edema or pleural effusion.",
        ],
    },
    "Pleural Effusion": {
        "worsened": [
            "The right pleural effusion has significantly increased in size. "
            "Previously only blunting the costophrenic angle, it now extends to "
            "the mid-hemithorax with associated compressive atelectasis.",
            "Bilateral pleural effusions have worsened. The left-sided effusion is new "
            "since the prior study. Right-sided effusion is larger.",
        ],
        "improved": [
            "The bilateral pleural effusions have decreased in volume. "
            "The right costophrenic angle is now visible, previously obscured. "
            "Improvement may be related to diuresis.",
        ],
        "resolved": [
            "Previously noted right pleural effusion has completely resolved. "
            "Both costophrenic angles are now sharp. "
            "Lungs are clear with no residual effusion.",
        ],
    },
    "Pneumothorax": {
        "worsened": [
            "The previously small right apical pneumothorax has enlarged, "
            "now extending to the mid-thorax. No tension physiology yet, "
            "but close monitoring is recommended.",
        ],
        "improved": [
            "The right pneumothorax has decreased in size following chest tube placement. "
            "The lung has partially re-expanded. Small residual pneumothorax remains.",
        ],
        "resolved": [
            "The previously noted right pneumothorax has completely resolved. "
            "The right lung is fully expanded. Chest tube has been removed.",
        ],
    },
    "Consolidation": {
        "worsened": [
            "The right lower lobe consolidation has progressed, now involving the "
            "right middle lobe as well. Air bronchograms are more prominent. "
            "Consider treatment failure or superinfection.",
        ],
        "improved": [
            "The right lower lobe consolidation has decreased in density and extent. "
            "Partial clearing suggests response to antibiotic therapy. "
            "Small residual opacity remains.",
        ],
        "new": [
            "New left lower lobe consolidation not present on prior study. "
            "Dense opacity with air bronchograms concerning for pneumonia. "
            "Clinical correlation for new infection recommended.",
        ],
    },
    "Edema": {
        "worsened": [
            "Worsening bilateral pulmonary edema compared to prior. "
            "New Kerley B lines and peribronchial cuffing are now present. "
            "Bilateral pleural effusions have increased.",
        ],
        "improved": [
            "Significant improvement in pulmonary edema since prior study. "
            "Previously noted bilateral airspace disease has largely cleared. "
            "Vascular redistribution has resolved.",
        ],
    },
    "Support Devices": {
        "new": [
            "Interval placement of endotracheal tube with tip projecting 4cm above "
            "the carina — within normal position. New right internal jugular central "
            "venous catheter with tip in the proximal SVC.",
        ],
        "stable": [
            "Lines and tubes are unchanged in position. ETT tip remains appropriately "
            "positioned. Central venous catheter tip at the cavoatrial junction.",
        ],
    },
}

# Default for categories not in the knowledge base
DEFAULT_LONGITUDINAL = {
    "worsened": ["Findings have progressed compared to prior study."],
    "improved": ["Findings have improved compared to prior study."],
    "stable": ["Findings are stable compared to prior study."],
    "new": ["New finding not present on prior study."],
    "resolved": ["Previously noted finding has resolved."],
}


def generate_longitudinal_question(category: str, change_type: str) -> str:
    """Generate a clinical question about interval changes."""
    time_intervals = [
        "24 hours", "48 hours", "3 days", "1 week", "2 weeks", "1 month",
    ]
    interval = random.choice(time_intervals)

    contexts = {
        "worsened": f"This patient had a chest X-ray {interval} ago (shown as the prior image).",
        "improved": f"This patient was treated and had a follow-up CXR {interval} later.",
        "stable": f"Routine follow-up imaging obtained {interval} after the prior study.",
        "new": f"Follow-up chest X-ray obtained {interval} after a normal prior study.",
        "resolved": f"Post-treatment follow-up imaging {interval} after initial diagnosis.",
    }

    return (
        f"**Longitudinal Comparison — {category}**\n\n"
        f"{contexts.get(change_type, contexts['stable'])}\n\n"
        f"**Prior image** is shown on the left. **Current image** is on the right.\n\n"
        f"**Task:**\n"
        f"1. Compare the two images systematically\n"
        f"2. Identify ALL interval changes\n"
        f"3. Describe whether findings have improved, worsened, or are stable\n"
        f"4. Comment on any new findings\n"
        f"5. Recommend follow-up if appropriate"
    )


def get_longitudinal_feedback(
    category: str,
    change_type: str,
    student_answer: str,
) -> FeedbackResult:
    """Grade a student's interval change detection response (mock mode)."""
    cat_data = LONGITUDINAL_DATA.get(category, DEFAULT_LONGITUDINAL)
    changes = cat_data.get(change_type, cat_data.get("stable", ["Stable findings."]))
    ground_truth = random.choice(changes)

    answer_lower = student_answer.lower() if student_answer else ""

    # Score based on change detection keywords
    change_keywords = {
        "worsened": ["worsen", "increase", "progress", "larger", "new", "more"],
        "improved": ["improve", "decrease", "better", "smaller", "less", "clear"],
        "stable": ["stable", "unchanged", "no change", "same", "persist"],
        "new": ["new", "interval", "not present", "appeared", "developed"],
        "resolved": ["resolv", "clear", "absent", "no longer", "gone"],
    }

    keywords = change_keywords.get(change_type, [])
    matches = sum(1 for kw in keywords if kw in answer_lower)

    if not student_answer.strip():
        score = 0.0
    elif matches >= 2:
        score = 0.80 + random.uniform(0, 0.20)
    elif matches >= 1:
        score = 0.50 + random.uniform(0, 0.20)
    else:
        score = 0.10 + random.uniform(0, 0.20)

    correct = [kw for kw in keywords if kw in answer_lower]
    missed = [kw for kw in keywords if kw not in answer_lower]

    return FeedbackResult(
        score=min(1.0, score),
        correct_findings=correct,
        missed_findings=missed[:3],
        false_positives=[],
        explanation=(
            f"**Interval Change:** {change_type.upper()}\n\n"
            f"**Expert Assessment:**\n{ground_truth}\n\n"
            f"**Teaching Point:**\n"
            f"When comparing studies, use a systematic approach:\n"
            f"1. First confirm patient identity and study dates\n"
            f"2. Compare overall lung volumes and heart size\n"
            f"3. Look at each anatomical region side-by-side\n"
            f"4. Use specific language: 'increased', 'decreased', 'new', 'resolved'\n"
            f"5. Quantify changes when possible (e.g., 'effusion now extends to mid-thorax')"
        ),
        box_iou=score * 0.7,
    )


def create_longitudinal_pairs(
    cards_by_category: dict[str, list],
) -> list[LongitudinalCase]:
    """
    Create longitudinal training pairs from existing cards.
    Pairs images from the same category as prior/current studies.
    """
    pairs = []
    change_types = ["worsened", "improved", "stable", "new", "resolved"]

    for category, cards in cards_by_category.items():
        if len(cards) < 2:
            continue

        # Create pairs from different images in same category
        for i in range(0, len(cards) - 1, 2):
            change_type = random.choice(change_types)
            cat_data = LONGITUDINAL_DATA.get(category, DEFAULT_LONGITUDINAL)

            # Only use change types that have data
            available_types = [ct for ct in change_types if ct in cat_data]
            if available_types:
                change_type = random.choice(available_types)

            descriptions = cat_data.get(change_type, ["Interval change detected."])

            pairs.append(LongitudinalCase(
                prior_path=cards[i].image_path,
                current_path=cards[i + 1].image_path,
                category=category,
                change_type=change_type,
                description=random.choice(descriptions),
            ))

    return pairs
