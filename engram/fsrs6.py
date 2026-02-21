"""
ENGRAM FSRS-6: Spaced Repetition Scheduler
Ported from Vestige's Rust implementation.

FSRS-6 is a 21-parameter spaced repetition algorithm using power-law
forgetting curves. It tracks per-item Stability (S) and Difficulty (D)
to optimally schedule reviews.

Key improvements over SM-2 (Anki):
- Power law decay (matches human memory better than exponential)
- 21 trainable parameters vs fixed constants
- Same-day review handling (w17-w19)
- Personalizable decay parameter (w20)
- Difficulty mean reversion
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum


# ─── FSRS-6 Default Parameters (w0-w20) ───────────────────────────
# Trained on hundreds of millions of reviews by open-spaced-repetition project
FSRS6_WEIGHTS: list[float] = [
    0.2120,   # w0:  Initial stability for Again
    1.2931,   # w1:  Initial stability for Hard
    2.3065,   # w2:  Initial stability for Good
    8.2956,   # w3:  Initial stability for Easy
    6.4133,   # w4:  Initial difficulty base
    0.8334,   # w5:  Initial difficulty grade modifier
    3.0194,   # w6:  Difficulty delta
    0.0010,   # w7:  Difficulty mean reversion rate
    1.8722,   # w8:  Stability increase base
    0.1666,   # w9:  Stability saturation exponent
    0.7960,   # w10: Retrievability influence on stability
    1.4835,   # w11: Forget stability base
    0.0614,   # w12: Forget difficulty influence
    0.2629,   # w13: Forget stability influence
    1.6483,   # w14: Forget retrievability influence
    0.6014,   # w15: Hard penalty
    1.8729,   # w16: Easy bonus
    0.5425,   # w17: Same-day review base (NEW in FSRS-6)
    0.0912,   # w18: Same-day review grade modifier (NEW in FSRS-6)
    0.0658,   # w19: Same-day review stability influence (NEW in FSRS-6)
    0.1542,   # w20: Forgetting curve decay (PERSONALIZABLE)
]

# ─── Constants ─────────────────────────────────────────────────────
MAX_DIFFICULTY = 10.0
MIN_DIFFICULTY = 1.0
MAX_STABILITY = 36500.0   # 100 years
MIN_STABILITY = 0.1
DEFAULT_RETENTION = 0.9   # 90% target recall
DEFAULT_DECAY = 0.1542    # w20


# ─── Rating ────────────────────────────────────────────────────────
class Rating(IntEnum):
    Again = 1   # Complete failure
    Hard = 2    # Recalled with difficulty
    Good = 3    # Recalled with some effort
    Easy = 4    # Instant recall


# ─── Learning State ────────────────────────────────────────────────
class LearningState(IntEnum):
    New = 0
    Learning = 1
    Review = 2
    Relearning = 3


# ─── FSRS State ────────────────────────────────────────────────────
@dataclass
class FSRSState:
    difficulty: float = 5.0
    stability: float = 0.0
    state: LearningState = LearningState.New
    reps: int = 0
    lapses: int = 0
    last_review: float = 0.0          # Unix timestamp
    scheduled_days: int = 0

    def copy(self) -> FSRSState:
        return FSRSState(
            difficulty=self.difficulty,
            stability=self.stability,
            state=self.state,
            reps=self.reps,
            lapses=self.lapses,
            last_review=self.last_review,
            scheduled_days=self.scheduled_days,
        )


# ─── Review Result ─────────────────────────────────────────────────
@dataclass
class ReviewResult:
    state: FSRSState
    retrievability: float
    interval: int
    is_lapse: bool


# ─── Card (for ENGRAM: one diagnostic concept) ────────────────────
@dataclass
class Card:
    """Represents a single diagnostic concept/case for a student."""
    card_id: str
    category: str                     # e.g., "pneumothorax", "cardiomegaly"
    image_path: str = ""
    label: str = ""
    fsrs: FSRSState = field(default_factory=FSRSState)
    times_shown: int = 0
    times_correct: int = 0
    created_at: float = field(default_factory=time.time)


# ─── Core Algorithm Functions ──────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def forgetting_factor(w20: float = DEFAULT_DECAY) -> float:
    """Calculate the factor used in the power-law forgetting curve."""
    return math.pow(0.9, -1.0 / w20) - 1.0


def retrievability(
    stability: float,
    elapsed_days: float,
    w20: float = DEFAULT_DECAY,
) -> float:
    """
    Power-law forgetting curve: R(t) = (1 + factor * t/S)^(-w20)

    Returns probability of recall (0.0 to 1.0).
    """
    if stability <= 0.0:
        return 0.0
    if elapsed_days <= 0.0:
        return 1.0
    factor = forgetting_factor(w20)
    r = math.pow(1.0 + factor * elapsed_days / stability, -w20)
    return _clamp(r, 0.0, 1.0)


def initial_stability(grade: Rating, w: list[float] = FSRS6_WEIGHTS) -> float:
    """S0(G) = w[G-1]. Initial stability indexed from first 4 weights."""
    return max(MIN_STABILITY, w[grade.value - 1])


def initial_difficulty(grade: Rating, w: list[float] = FSRS6_WEIGHTS) -> float:
    """D0(G) = w4 - exp(w5 * (G - 1)) + 1"""
    d = w[4] - math.exp(w[5] * (grade.value - 1)) + 1.0
    return _clamp(d, MIN_DIFFICULTY, MAX_DIFFICULTY)


def next_difficulty(
    d: float,
    grade: Rating,
    w: list[float] = FSRS6_WEIGHTS,
) -> float:
    """
    D' = w7 * D0(4) + (1 - w7) * (D + delta * (10-D)/9)
    Mean reversion toward initial Easy difficulty.
    """
    delta = -w[6] * (grade.value - 3)
    d_new = d + delta * ((10.0 - d) / 9.0)
    d0_easy = initial_difficulty(Rating.Easy, w)
    d_final = w[7] * d0_easy + (1.0 - w[7]) * d_new
    return _clamp(d_final, MIN_DIFFICULTY, MAX_DIFFICULTY)


def next_stability_recall(
    s: float,
    d: float,
    r: float,
    grade: Rating,
    w: list[float] = FSRS6_WEIGHTS,
) -> float:
    """
    S' after successful recall:
    S' = S * (exp(w8) * (11-D) * S^(-w9) * (exp(w10*(1-R)) - 1) * HP * EB + 1)
    """
    hard_penalty = w[15] if grade == Rating.Hard else 1.0
    easy_bonus = w[16] if grade == Rating.Easy else 1.0

    multiplier = (
        math.exp(w[8])
        * (11.0 - d)
        * math.pow(s, -w[9])
        * (math.exp(w[10] * (1.0 - r)) - 1.0)
        * hard_penalty
        * easy_bonus
    )
    new_s = s * (multiplier + 1.0)
    return _clamp(new_s, MIN_STABILITY, MAX_STABILITY)


def next_stability_lapse(
    s: float,
    d: float,
    r: float,
    w: list[float] = FSRS6_WEIGHTS,
) -> float:
    """
    S' after lapse (forgetting):
    S_f = w11 * D^(-w12) * ((S+1)^w13 - 1) * exp(w14*(1-R))
    Constraint: S_f <= S (post-lapse can't exceed pre-lapse)
    """
    s_f = (
        w[11]
        * math.pow(d, -w[12])
        * (math.pow(s + 1.0, w[13]) - 1.0)
        * math.exp(w[14] * (1.0 - r))
    )
    s_min = s / math.exp(w[17] * w[18])  # FSRS-6 minimum stability floor
    s_f = max(s_f, s_min)
    s_f = min(s_f, s)  # Post-lapse stability cannot exceed pre-lapse
    return _clamp(s_f, MIN_STABILITY, MAX_STABILITY)


def next_stability_same_day(
    s: float,
    grade: Rating,
    w: list[float] = FSRS6_WEIGHTS,
) -> float:
    """
    Same-day review (NEW in FSRS-6):
    S'(S,G) = S * exp(w17 * (G - 3 + w18)) * S^(-w19)
    """
    new_s = s * math.exp(w[17] * (grade.value - 3.0 + w[18])) * math.pow(s, -w[19])
    if grade.value >= 2:  # Hard, Good, Easy — don't let same-day review decrease below 1.0
        new_s = max(new_s, 1.0)
    return _clamp(new_s, MIN_STABILITY, MAX_STABILITY)


def next_interval(
    stability: float,
    desired_retention: float = DEFAULT_RETENTION,
    w20: float = DEFAULT_DECAY,
) -> int:
    """
    Calculate days until next review:
    t = (S / factor) * (R^(-1/w20) - 1)
    """
    if stability <= 0.0:
        return 0
    if desired_retention >= 1.0:
        return 0
    if desired_retention <= 0.0:
        return int(MAX_STABILITY)

    factor = forgetting_factor(w20)
    t = (stability / factor) * (math.pow(desired_retention, -1.0 / w20) - 1.0)
    return max(1, min(int(round(t)), int(MAX_STABILITY)))


def fuzz_interval(interval: int, seed: int = 0) -> int:
    """Add small random offset to prevent review clustering."""
    if interval <= 2:
        return interval
    fuzz_range = max(1, int(interval * 0.05))
    random_val = ((seed * 1103515245 + 12345) % 32768)
    offset = (random_val % (2 * fuzz_range + 1)) - fuzz_range
    return max(1, interval + offset)


# ─── Interval Modifiers (v0.4 cognitive features) ─────────────────

def interval_modifier_for_overconfidence(calibration_gap: float) -> float:
    """Shorten intervals for overconfident categories.
    calibration_gap > 0 means student is overconfident.
    Returns modifier in [0.5, 1.0] to multiply against interval.
    """
    if calibration_gap <= 0.1:
        return 1.0
    return max(0.5, 1.0 - calibration_gap)


def search_completeness_modifier(completeness: float) -> float:
    """Shorten intervals when student misses findings (satisfaction of search).
    completeness < 0.5 means student found fewer than half the findings.
    Returns modifier in [0.5, 1.0].
    """
    if completeness >= 0.8:
        return 1.0
    if completeness < 0.3:
        return 0.5
    return 0.5 + (completeness - 0.3) * (0.5 / (0.8 - 0.3))  # Linear 0.5→1.0 over 0.3→0.8


# ─── FSRS-6 Scheduler ─────────────────────────────────────────────

class FSRS6Scheduler:
    """
    Complete FSRS-6 scheduler ported from Vestige.
    Manages review scheduling with the 21-parameter algorithm.
    """

    def __init__(
        self,
        weights: list[float] | None = None,
        desired_retention: float = DEFAULT_RETENTION,
        max_interval: int = 365,
        enable_fuzz: bool = True,
    ):
        self.w = weights or FSRS6_WEIGHTS.copy()
        self.desired_retention = desired_retention
        self.max_interval = max_interval
        self.enable_fuzz = enable_fuzz

    def review(
        self,
        state: FSRSState,
        grade: Rating,
        elapsed_days: float | None = None,
    ) -> ReviewResult:
        """
        Process a review and return the new state + next interval.
        This is the main entry point for the scheduler.
        """
        now = time.time()

        # Calculate elapsed days
        if elapsed_days is None:
            if state.last_review > 0:
                elapsed_days = (now - state.last_review) / 86400.0
            else:
                elapsed_days = 0.0

        # Calculate current retrievability
        if state.state == LearningState.New:
            r = 1.0
        else:
            r = retrievability(state.stability, elapsed_days, self.w[20])

        # Check if same-day review
        is_same_day = elapsed_days < 1.0 and state.state != LearningState.New

        # Route to handler
        if state.state == LearningState.New:
            new_state, is_lapse = self._handle_first_review(grade), False
        elif is_same_day:
            new_state, is_lapse = self._handle_same_day(state, grade), False
        elif grade == Rating.Again:
            is_learned = state.state in (LearningState.Review, LearningState.Relearning)
            new_state = self._handle_lapse(state, r, grade)
            is_lapse = is_learned
        else:
            new_state, is_lapse = self._handle_recall(state, r, grade), False

        # Calculate interval
        interval = next_interval(new_state.stability, self.desired_retention, self.w[20])
        interval = min(interval, self.max_interval)

        if self.enable_fuzz and interval > 2:
            interval = fuzz_interval(interval, seed=int(now))

        new_state.last_review = now
        new_state.scheduled_days = interval

        return ReviewResult(
            state=new_state,
            retrievability=r,
            interval=interval,
            is_lapse=is_lapse,
        )

    def _handle_first_review(self, grade: Rating) -> FSRSState:
        s = initial_stability(grade, self.w)
        d = initial_difficulty(grade, self.w)
        if grade in (Rating.Again, Rating.Hard):
            state = LearningState.Learning
        else:
            state = LearningState.Review
        return FSRSState(
            difficulty=d,
            stability=s,
            state=state,
            reps=1 if grade != Rating.Again else 0,
            lapses=0,
        )

    def _handle_same_day(self, state: FSRSState, grade: Rating) -> FSRSState:
        new = state.copy()
        new.stability = next_stability_same_day(state.stability, grade, self.w)
        new.difficulty = next_difficulty(state.difficulty, grade, self.w)
        if grade != Rating.Again:
            new.reps += 1
        return new

    def _handle_recall(self, state: FSRSState, r: float, grade: Rating) -> FSRSState:
        new = state.copy()
        new.stability = next_stability_recall(state.stability, state.difficulty, r, grade, self.w)
        new.difficulty = next_difficulty(state.difficulty, grade, self.w)
        new.reps += 1
        if state.state == LearningState.Learning:
            new.state = LearningState.Review
        elif state.state == LearningState.Relearning and grade in (Rating.Good, Rating.Easy):
            new.state = LearningState.Review
        return new

    def _handle_lapse(self, state: FSRSState, r: float, grade: Rating) -> FSRSState:
        new = state.copy()
        new.stability = next_stability_lapse(state.stability, state.difficulty, r, self.w)
        new.difficulty = next_difficulty(state.difficulty, grade, self.w)
        new.reps = 0
        if state.state in (LearningState.Review, LearningState.Relearning):
            new.lapses += 1
            new.state = LearningState.Relearning
        elif state.state == LearningState.Learning:
            new.state = LearningState.Learning
        return new

    def get_due_cards(self, cards: list[Card], now: float | None = None) -> list[Card]:
        """
        Return cards sorted by urgency (lowest retrievability first).
        Cards with R below desired_retention are due for review.
        """
        if now is None:
            now = time.time()

        scored: list[tuple[float, Card]] = []
        for card in cards:
            if card.fsrs.state == LearningState.New:
                scored.append((0.0, card))  # New cards first
                continue

            elapsed = (now - card.fsrs.last_review) / 86400.0
            r = retrievability(card.fsrs.stability, elapsed, self.w[20])
            scored.append((r, card))

        # Sort: lowest retrievability first (most urgent)
        scored.sort(key=lambda x: x[0])
        return [card for _, card in scored]


# ─── Blind Spot Analysis (ENGRAM-specific) ─────────────────────────

@dataclass
class BlindSpot:
    """A diagnostic category where the student is weak."""
    category: str
    retention: float          # Current estimated retention (0-1)
    stability: float          # FSRS stability (days)
    difficulty: float         # FSRS difficulty (1-10)
    total_reviews: int
    total_lapses: int
    mastery_level: str        # "danger", "weak", "developing", "strong", "mastered"


def analyze_blind_spots(cards: list[Card], now: float | None = None) -> list[BlindSpot]:
    """
    Analyze student's diagnostic blind spots using FSRS-6 state.
    Groups cards by category and computes aggregate retention.
    """
    if now is None:
        now = time.time()

    categories: dict[str, list[Card]] = {}
    for card in cards:
        if card.category not in categories:
            categories[card.category] = []
        categories[card.category].append(card)

    spots: list[BlindSpot] = []
    for cat, cat_cards in categories.items():
        retentions = []
        total_reps = 0
        total_lapses = 0
        avg_stability = 0.0
        avg_difficulty = 0.0

        for c in cat_cards:
            if c.fsrs.state == LearningState.New:
                retentions.append(0.0)
            else:
                elapsed = (now - c.fsrs.last_review) / 86400.0
                r = retrievability(c.fsrs.stability, elapsed)
                retentions.append(r)
            total_reps += c.fsrs.reps
            total_lapses += c.fsrs.lapses
            avg_stability += c.fsrs.stability
            avg_difficulty += c.fsrs.difficulty

        n = len(cat_cards)
        avg_r = sum(retentions) / n if n > 0 else 0.0
        avg_stability /= n if n > 0 else 1
        avg_difficulty /= n if n > 0 else 1

        if avg_r >= 0.90:
            level = "mastered"
        elif avg_r >= 0.75:
            level = "strong"
        elif avg_r >= 0.55:
            level = "developing"
        elif avg_r >= 0.30:
            level = "weak"
        else:
            level = "danger"

        spots.append(BlindSpot(
            category=cat,
            retention=avg_r,
            stability=avg_stability,
            difficulty=avg_difficulty,
            total_reviews=total_reps,
            total_lapses=total_lapses,
            mastery_level=level,
        ))

    # Sort: lowest retention first (biggest blind spots)
    spots.sort(key=lambda s: s.retention)
    return spots
