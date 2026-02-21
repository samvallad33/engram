"""ENGRAM Student State Management"""
from __future__ import annotations
import json, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from .fsrs6 import Card, FSRSState, LearningState, Rating, ReviewResult, analyze_blind_spots, BlindSpot

@dataclass
class ReviewLog:
    card_id: str; category: str; grade: int; score: float; box_iou: float; retrievability: float; interval: int
    timestamp: float = field(default_factory=time.time)
    confidence: int = 0; search_completeness: float = 0.0; found_findings: int = 0
    total_findings: int = 0; gestalt_score: float = 0.0; contrastive_pair: str = ""

@dataclass
class StudentState:
    student_id: str; name: str = "Student"
    cards: dict[str, Card] = field(default_factory=dict)
    review_history: list[ReviewLog] = field(default_factory=list)
    total_sessions: int = 0; total_reviews: int = 0
    created_at: float = field(default_factory=time.time); last_session: float = field(default_factory=time.time)

    def add_card(self, card: Card): self.cards[card.card_id] = card
    def get_blind_spots(self) -> list[BlindSpot]: return analyze_blind_spots(list(self.cards.values()))

    def record_review(self, card: Card, grade: Rating, res: ReviewResult, score=0.0, box_iou=0.0, conf=0, comp=0.0, found=0, tot=0, gest=0.0, pair=""):
        card.fsrs = res.state
        card.times_shown += 1
        if grade != Rating.Again: card.times_correct += 1
        self.review_history.append(ReviewLog(card.card_id, card.category, grade.value, score, box_iou, res.retrievability, res.interval, confidence=conf, search_completeness=comp, found_findings=found, total_findings=tot, gestalt_score=gest, contrastive_pair=pair))
        self.total_reviews += 1
        self.cards[card.card_id] = card

    def get_session_stats(self) -> dict:
        recent = self.review_history[-50:]
        if not recent: return {}
        return {"total_reviews": self.total_reviews, "avg_score": sum(r.score for r in recent)/len(recent), "avg_box_iou": sum(r.box_iou for r in recent)/len(recent), "categories_practiced": len(set(r.category for r in recent))}

    def calibration_per_category(self) -> dict:
        cats = {}
        for r in [r for r in self.review_history if r.confidence > 0]: cats.setdefault(r.category, []).append((r.confidence/5.0, r.score))
        return {cat: {"calibration_gap": sum(c for c,_ in pairs)/len(pairs) - sum(s for _,s in pairs)/len(pairs), "overconfident": (sum(c for c,_ in pairs)/len(pairs) - sum(s for _,s in pairs)/len(pairs)) > 0.15} for cat, pairs in cats.items()}

    def save(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        for cid, c in data["cards"].items(): c["fsrs"]["state"] = c["fsrs"]["state"].value # Enum fix
        with open(path, "w", encoding="utf-8") as f: json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> StudentState:
        with open(path, encoding="utf-8") as f: data = json.load(f)
        student = cls(data["student_id"], data.get("name", "Student"), total_sessions=data.get("total_sessions", 0), total_reviews=data.get("total_reviews", 0))
        for cid, c in data.get("cards", {}).items():
            f_data = c.get("fsrs", {})
            f_data["state"] = LearningState(f_data.get("state", 0))
            student.cards[cid] = Card(c["card_id"], c["category"], c.get("image_path",""), c.get("label",""), FSRSState(**f_data), c.get("times_shown",0), c.get("times_correct",0))
        student.review_history = [ReviewLog(**r) for r in data.get("review_history", [])]
        return student