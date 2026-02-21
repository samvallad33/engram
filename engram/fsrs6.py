"""
ENGRAM FSRS-6: Spaced Repetition Scheduler
Ported from Vestige's Rust implementation.
"""
from __future__ import annotations
import math, time
from dataclasses import dataclass, field
from enum import IntEnum

FSRS6_WEIGHTS: list[float] = [
    0.2120, 1.2931, 2.3065, 8.2956, 6.4133, 0.8334, 3.0194, 0.0010,
    1.8722, 0.1666, 0.7960, 1.4835, 0.0614, 0.2629, 1.6483, 0.6014,
    1.8729, 0.5425, 0.0912, 0.0658, 0.1542,
]

class Rating(IntEnum): Again = 1; Hard = 2; Good = 3; Easy = 4
class LearningState(IntEnum): New = 0; Learning = 1; Review = 2; Relearning = 3

@dataclass
class FSRSState:
    difficulty: float = 5.0
    stability: float = 0.0
    state: LearningState = LearningState.New
    reps: int = 0
    lapses: int = 0
    last_review: float = 0.0
    scheduled_days: int = 0

@dataclass
class ReviewResult:
    state: FSRSState; retrievability: float; interval: int; is_lapse: bool

@dataclass
class Card:
    card_id: str; category: str; image_path: str = ""; label: str = ""
    fsrs: FSRSState = field(default_factory=FSRSState)
    times_shown: int = 0; times_correct: int = 0
    created_at: float = field(default_factory=time.time)

def clamp(value: float, lo: float, hi: float) -> float: return max(lo, min(hi, value))
def forgetting_factor(w20: float = 0.1542) -> float: return math.pow(0.9, -1.0 / w20) - 1.0

def retrievability(stability: float, elapsed_days: float, w20: float = 0.1542) -> float:
    if stability <= 0.0: return 0.0
    if elapsed_days <= 0.0: return 1.0
    return clamp(math.pow(1.0 + forgetting_factor(w20) * elapsed_days / stability, -w20), 0.0, 1.0)

# [Mathematical functions remain identical to preserve proven tests]
def initial_stability(grade: Rating, w=FSRS6_WEIGHTS) -> float: return max(0.1, w[grade.value - 1])
def initial_difficulty(grade: Rating, w=FSRS6_WEIGHTS) -> float: return clamp(w[4] - math.exp(w[5] * (grade.value - 1)) + 1.0, 1.0, 10.0)
def next_difficulty(d: float, grade: Rating, w=FSRS6_WEIGHTS) -> float:
    d_new = d - w[6] * (grade.value - 3) * ((10.0 - d) / 9.0)
    return clamp(w[7] * initial_difficulty(Rating.Easy, w) + (1.0 - w[7]) * d_new, 1.0, 10.0)

def next_stability_recall(s: float, d: float, r: float, grade: Rating, w=FSRS6_WEIGHTS) -> float:
    mult = math.exp(w[8]) * (11.0 - d) * math.pow(s, -w[9]) * (math.exp(w[10] * (1.0 - r)) - 1.0) * (w[15] if grade == Rating.Hard else 1.0) * (w[16] if grade == Rating.Easy else 1.0)
    return clamp(s * (mult + 1.0), 0.1, 36500.0)

def next_stability_lapse(s: float, d: float, r: float, w=FSRS6_WEIGHTS) -> float:
    s_f = w[11] * math.pow(d, -w[12]) * (math.pow(s + 1.0, w[13]) - 1.0) * math.exp(w[14] * (1.0 - r))
    return clamp(min(max(s_f, s / math.exp(w[17] * w[18])), s), 0.1, 36500.0)

def next_stability_same_day(s: float, grade: Rating, w=FSRS6_WEIGHTS) -> float:
    new_s = s * math.exp(w[17] * (grade.value - 3.0 + w[18])) * math.pow(s, -w[19])
    return clamp(max(new_s, 1.0) if grade.value >= 2 else new_s, 0.1, 36500.0)

def next_interval(s: float, ret: float = 0.9, w20: float = 0.1542) -> int:
    if s <= 0.0 or ret >= 1.0: return 0
    return clamp(int(round((s / forgetting_factor(w20)) * (math.pow(ret, -1.0 / w20) - 1.0))), 1, 36500)

def interval_modifier_for_overconfidence(gap: float) -> float: return clamp(1.0 - gap, 0.5, 1.0) if gap > 0.1 else 1.0
def search_completeness_modifier(comp: float) -> float: return clamp(0.5 + (comp - 0.3) * 1.0, 0.5, 1.0)

class FSRS6Scheduler:
    def __init__(self, weights=None, desired_retention=0.9, max_interval=365):
        self.w = weights or FSRS6_WEIGHTS.copy()
        self.retention = desired_retention
        self.max_interval = max_interval

    def review(self, state: FSRSState, grade: Rating, elapsed_days: float | None = None) -> ReviewResult:
        now = time.time()
        elapsed = elapsed_days if elapsed_days is not None else max(0.0, (now - state.last_review) / 86400.0 if state.last_review else 0.0)
        r = 1.0 if state.state == LearningState.New else retrievability(state.stability, elapsed, self.w[20])
        
        new = FSRSState(difficulty=state.difficulty, stability=state.stability, state=state.state, reps=state.reps, lapses=state.lapses)
        is_lapse = False

        if state.state == LearningState.New:
            new.stability, new.difficulty = initial_stability(grade, self.w), initial_difficulty(grade, self.w)
            new.state = LearningState.Learning if grade in (Rating.Again, Rating.Hard) else LearningState.Review
            new.reps = 1 if grade != Rating.Again else 0
        elif elapsed < 1.0:
            new.stability, new.difficulty = next_stability_same_day(state.stability, grade, self.w), next_difficulty(state.difficulty, grade, self.w)
            if grade != Rating.Again: new.reps += 1
        elif grade == Rating.Again:
            is_lapse = state.state in (LearningState.Review, LearningState.Relearning)
            new.stability, new.difficulty = next_stability_lapse(state.stability, state.difficulty, r, self.w), next_difficulty(state.difficulty, grade, self.w)
            new.reps = 0
            if is_lapse: new.lapses += 1; new.state = LearningState.Relearning
        else:
            new.stability, new.difficulty = next_stability_recall(state.stability, state.difficulty, r, grade, self.w), next_difficulty(state.difficulty, grade, self.w)
            new.reps += 1
            if state.state in (LearningState.Learning, LearningState.Relearning): new.state = LearningState.Review

        new.scheduled_days = min(next_interval(new.stability, self.retention, self.w[20]), self.max_interval)
        new.last_review = now
        return ReviewResult(state=new, retrievability=r, interval=new.scheduled_days, is_lapse=is_lapse)

    def get_due_cards(self, cards: list[Card], now: float | None = None) -> list[Card]:
        now = now or time.time()
        scored = [(0.0 if c.fsrs.state == LearningState.New else retrievability(c.fsrs.stability, (now - c.fsrs.last_review) / 86400.0, self.w[20]), c) for c in cards]
        return [c for _, c in sorted(scored, key=lambda x: x[0])]

@dataclass
class BlindSpot:
    category: str; retention: float; stability: float; difficulty: float; total_reviews: int; total_lapses: int; mastery_level: str

def analyze_blind_spots(cards: list[Card], now: float | None = None) -> list[BlindSpot]:
    now = now or time.time()
    cats = {}
    for c in cards: cats.setdefault(c.category, []).append(c)

    spots = []
    for cat, items in cats.items():
        rets = [0.0 if c.fsrs.state == LearningState.New else retrievability(c.fsrs.stability, (now - c.fsrs.last_review)/86400.0) for c in items]
        avg_r = sum(rets) / len(items) if items else 0.0
        level = "mastered" if avg_r >= 0.9 else ("strong" if avg_r >= 0.75 else ("developing" if avg_r >= 0.55 else ("weak" if avg_r >= 0.3 else "danger")))
        spots.append(BlindSpot(cat, avg_r, sum(c.fsrs.stability for c in items)/len(items), sum(c.fsrs.difficulty for c in items)/len(items), sum(c.fsrs.reps for c in items), sum(c.fsrs.lapses for c in items), level))
    
    return sorted(spots, key=lambda s: s.retention)