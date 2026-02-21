"""
ENGRAM Student State Management
Tracks per-student learning progress, card states, and session history.
Persists to JSON for offline usage.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from .fsrs6 import (
    Card, FSRSState, LearningState, Rating,
    ReviewResult, analyze_blind_spots, BlindSpot,
)


@dataclass
class ReviewLog:
    """Record of a single review event."""
    card_id: str
    category: str
    grade: int                  # Rating value (1-4)
    score: float                # MedGemma score (0-1)
    box_iou: float              # Spatial accuracy (0-1)
    retrievability: float       # R at review time
    interval: int               # Next interval (days)
    timestamp: float = field(default_factory=time.time)
    # v0.4 fields — default to zero for backward compatibility
    confidence: int = 0              # F1: self-reported confidence (1-5, 0=not collected)
    search_completeness: float = 0.0 # F3: fraction of findings found (0-1)
    found_findings: int = 0          # F3: count of findings found
    total_findings: int = 0          # F3: total findings in case
    gestalt_score: float = 0.0       # F4: System 1 rapid impression score
    contrastive_pair: str = ""       # F5: paired category name


@dataclass
class StudentState:
    """Complete state for one student's learning journey."""
    student_id: str
    name: str = "Student"
    cards: dict[str, Card] = field(default_factory=dict)
    review_history: list[ReviewLog] = field(default_factory=list)
    total_sessions: int = 0
    total_reviews: int = 0
    created_at: float = field(default_factory=time.time)
    last_session: float = field(default_factory=time.time)

    def add_card(self, card: Card) -> None:
        self.cards[card.card_id] = card

    def get_card(self, card_id: str) -> Card | None:
        return self.cards.get(card_id)

    def record_review(
        self,
        card: Card,
        grade: Rating,
        result: ReviewResult,
        score: float = 0.0,
        box_iou: float = 0.0,
        confidence: int = 0,
        search_completeness: float = 0.0,
        found_findings: int = 0,
        total_findings: int = 0,
        gestalt_score: float = 0.0,
        contrastive_pair: str = "",
    ) -> None:
        """Record a review and update card state."""
        card.fsrs = result.state
        card.times_shown += 1
        if grade != Rating.Again:
            card.times_correct += 1

        self.review_history.append(ReviewLog(
            card_id=card.card_id,
            category=card.category,
            grade=grade.value,
            score=score,
            box_iou=box_iou,
            retrievability=result.retrievability,
            interval=result.interval,
            confidence=confidence,
            search_completeness=search_completeness,
            found_findings=found_findings,
            total_findings=total_findings,
            gestalt_score=gestalt_score,
            contrastive_pair=contrastive_pair,
        ))
        self.total_reviews += 1
        self.cards[card.card_id] = card

    def get_blind_spots(self) -> list[BlindSpot]:
        """Analyze diagnostic blind spots across all categories."""
        return analyze_blind_spots(list(self.cards.values()))

    def get_session_stats(self) -> dict:
        """Get aggregate stats for the current session."""
        if not self.review_history:
            return {
                "total_reviews": 0,
                "avg_score": 0.0,
                "avg_box_iou": 0.0,
                "categories_practiced": 0,
            }

        recent = self.review_history[-50:]  # Last 50 reviews
        cats = set(r.category for r in recent)

        return {
            "total_reviews": self.total_reviews,
            "avg_score": sum(r.score for r in recent) / len(recent),
            "avg_box_iou": sum(r.box_iou for r in recent) / len(recent),
            "categories_practiced": len(cats),
            "strongest": max(cats, key=lambda c: self._cat_avg(c, recent)) if cats else None,
            "weakest": min(cats, key=lambda c: self._cat_avg(c, recent)) if cats else None,
        }

    def _cat_avg(self, category: str, reviews: list[ReviewLog]) -> float:
        cat_reviews = [r for r in reviews if r.category == category]
        if not cat_reviews:
            return 0.0
        return sum(r.score for r in cat_reviews) / len(cat_reviews)

    def calibration_per_category(self) -> dict[str, dict]:
        """Compute confidence calibration per category.
        Returns {category: {mean_confidence, mean_accuracy, calibration_gap, overconfident, n_reviews}}.
        """
        cats: dict[str, list[tuple[int, float]]] = {}
        for r in self.review_history:
            if r.confidence > 0:
                cats.setdefault(r.category, []).append((r.confidence, r.score))
        result = {}
        for cat, pairs in cats.items():
            confs = [c / 5.0 for c, _ in pairs]  # Normalize to 0-1
            accs = [s for _, s in pairs]
            mean_conf = sum(confs) / len(confs)
            mean_acc = sum(accs) / len(accs)
            gap = mean_conf - mean_acc  # Positive = overconfident
            result[cat] = {
                "mean_confidence": mean_conf,
                "mean_accuracy": mean_acc,
                "calibration_gap": gap,
                "overconfident": gap > 0.15,
                "n_reviews": len(pairs),
            }
        return result

    def search_completeness_per_category(self) -> dict[str, float]:
        """Average search completeness per category."""
        cats: dict[str, list[float]] = {}
        for r in self.review_history:
            if r.total_findings > 0:
                cats.setdefault(r.category, []).append(r.search_completeness)
        return {cat: sum(v) / len(v) for cat, v in cats.items()}

    def dual_process_stats(self) -> dict[str, dict]:
        """Per-category System 1 vs System 2 comparison.
        Returns {category: {mean_gestalt, mean_analytical, gestalt_gap, weak_pattern_recognition}}.
        """
        cats: dict[str, list[tuple[float, float]]] = {}
        for r in self.review_history:
            if r.gestalt_score > 0:
                cats.setdefault(r.category, []).append((r.gestalt_score, r.score))
        result = {}
        for cat, pairs in cats.items():
            g_avg = sum(g for g, _ in pairs) / len(pairs)
            a_avg = sum(a for _, a in pairs) / len(pairs)
            gap = a_avg - g_avg
            result[cat] = {
                "mean_gestalt": g_avg,
                "mean_analytical": a_avg,
                "gestalt_gap": gap,
                "weak_pattern_recognition": gap > 0.3,
            }
        return result

    def discrimination_per_pair(self) -> dict[str, float]:
        """Average score per contrastive category pair."""
        pairs: dict[str, list[float]] = {}
        for r in self.review_history:
            if r.contrastive_pair:
                pairs.setdefault(r.contrastive_pair, []).append(r.score)
        return {p: sum(v) / len(v) for p, v in pairs.items()}

    def save(self, path: str | Path) -> None:
        """Save student state to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "student_id": self.student_id,
            "name": self.name,
            "total_sessions": self.total_sessions,
            "total_reviews": self.total_reviews,
            "created_at": self.created_at,
            "last_session": self.last_session,
            "cards": {},
            "review_history": [],
        }

        for cid, card in self.cards.items():
            data["cards"][cid] = {
                "card_id": card.card_id,
                "category": card.category,
                "image_path": card.image_path,
                "label": card.label,
                "times_shown": card.times_shown,
                "times_correct": card.times_correct,
                "created_at": card.created_at,
                "fsrs": {
                    "difficulty": card.fsrs.difficulty,
                    "stability": card.fsrs.stability,
                    "state": card.fsrs.state.value,
                    "reps": card.fsrs.reps,
                    "lapses": card.fsrs.lapses,
                    "last_review": card.fsrs.last_review,
                    "scheduled_days": card.fsrs.scheduled_days,
                },
            }

        for rev in self.review_history[-500:]:  # Keep last 500
            data["review_history"].append({
                "card_id": rev.card_id,
                "category": rev.category,
                "grade": rev.grade,
                "score": rev.score,
                "box_iou": rev.box_iou,
                "retrievability": rev.retrievability,
                "interval": rev.interval,
                "timestamp": rev.timestamp,
                "confidence": rev.confidence,
                "search_completeness": rev.search_completeness,
                "found_findings": rev.found_findings,
                "total_findings": rev.total_findings,
                "gestalt_score": rev.gestalt_score,
                "contrastive_pair": rev.contrastive_pair,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> StudentState:
        """Load student state from JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        student = cls(
            student_id=data["student_id"],
            name=data.get("name", "Student"),
            total_sessions=data.get("total_sessions", 0),
            total_reviews=data.get("total_reviews", 0),
            created_at=data.get("created_at", time.time()),
            last_session=data.get("last_session", time.time()),
        )

        for cid, cdata in data.get("cards", {}).items():
            fsrs_data = cdata.get("fsrs", {})
            card = Card(
                card_id=cdata["card_id"],
                category=cdata["category"],
                image_path=cdata.get("image_path", ""),
                label=cdata.get("label", ""),
                times_shown=cdata.get("times_shown", 0),
                times_correct=cdata.get("times_correct", 0),
                created_at=cdata.get("created_at", time.time()),
                fsrs=FSRSState(
                    difficulty=fsrs_data.get("difficulty", 5.0),
                    stability=fsrs_data.get("stability", 0.0),
                    state=LearningState(fsrs_data.get("state", 0)),
                    reps=fsrs_data.get("reps", 0),
                    lapses=fsrs_data.get("lapses", 0),
                    last_review=fsrs_data.get("last_review", 0.0),
                    scheduled_days=fsrs_data.get("scheduled_days", 0),
                ),
            )
            student.cards[cid] = card

        for rdata in data.get("review_history", []):
            student.review_history.append(ReviewLog(
                card_id=rdata["card_id"],
                category=rdata["category"],
                grade=rdata["grade"],
                score=rdata.get("score", 0.0),
                box_iou=rdata.get("box_iou", 0.0),
                retrievability=rdata.get("retrievability", 0.0),
                interval=rdata.get("interval", 0),
                timestamp=rdata.get("timestamp", 0.0),
                confidence=rdata.get("confidence", 0),
                search_completeness=rdata.get("search_completeness", 0.0),
                found_findings=rdata.get("found_findings", 0),
                total_findings=rdata.get("total_findings", 0),
                gestalt_score=rdata.get("gestalt_score", 0.0),
                contrastive_pair=rdata.get("contrastive_pair", ""),
            ))

        return student
