"""ENGRAM Longitudinal CXR Comparison Training"""
from __future__ import annotations
import random
from dataclasses import dataclass
from .medgemma import FeedbackResult

@dataclass
class LongitudinalCase:
    prior_path: str; current_path: str; category: str; change_type: str; description: str

def generate_longitudinal_question(cat: str, change_type: str) -> str:
    return f"**Longitudinal Comparison — {cat}**\n\nCompare the prior study (left) to the current study (right). Identify ALL interval changes ({change_type})."

def get_longitudinal_feedback(cat: str, change: str, ans: str) -> FeedbackResult:
    ans_lower = (ans or "").lower()
    kw_map = {"worsened": ["worsen", "increase", "new"], "improved": ["improve", "decrease", "clear"], "stable": ["stable", "unchanged", "same"], "new": ["new", "interval", "develop"], "resolved": ["resolv", "clear", "gone"]}
    kws = kw_map.get(change, [])
    
    matches = sum(1 for k in kws if k in ans_lower)
    score = min(1.0, (0.8 if matches >= 2 else (0.5 if matches == 1 else 0.1)) + random.uniform(0, 0.2))
    
    return FeedbackResult(score, [k for k in kws if k in ans_lower], [k for k in kws if k not in ans_lower][:3], [], f"**Interval Change:** {change.upper()}", score * 0.7)

def create_longitudinal_pairs(cards_by_category: dict) -> list[LongitudinalCase]:
    pairs = []
    for cat, cards in cards_by_category.items():
        if len(cards) >= 2:
            change = random.choice(["worsened", "improved", "stable", "new", "resolved"])
            for i in range(0, len(cards)-1, 2):
                pairs.append(LongitudinalCase(cards[i].image_path, cards[i+1].image_path, cat, change, "Interval change."))
    return pairs