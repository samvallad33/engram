"""
ENGRAM v0.4.0 Test Suite
Tests all core modules: FSRS-6, MedASR, Longitudinal, CXR Foundation, Mock Engine,
plus v0.4.0 features: Confidence Calibration, Socratic Mode, Satisfaction of Search,
Dual-Process Training, Contrastive Case Pairs, HeAR Integration.
"""

import math
import os
import tempfile

import numpy as np
from PIL import Image

# ─── FSRS-6 Core Tests ──────────────────────────────────────────

from engram.fsrs6 import (
    FSRS6_WEIGHTS,
    Card,
    FSRSState,
    FSRS6Scheduler,
    LearningState,
    Rating,
    forgetting_factor,
    initial_difficulty,
    initial_stability,
    retrievability,
)


def test_fsrs6_weights_count():
    """FSRS-6 must have exactly 21 parameters (w0-w20)."""
    assert len(FSRS6_WEIGHTS) == 21


def test_initial_stability():
    """S0(G) should increase with better grades."""
    s_again = initial_stability(Rating.Again)
    s_hard = initial_stability(Rating.Hard)
    s_good = initial_stability(Rating.Good)
    s_easy = initial_stability(Rating.Easy)
    assert s_again < s_hard < s_good < s_easy
    assert abs(s_again - 0.2120) < 0.001
    assert abs(s_easy - 8.2956) < 0.001


def test_initial_difficulty():
    """D0(G) should decrease with better grades (easier = lower D)."""
    d_again = initial_difficulty(Rating.Again)
    d_easy = initial_difficulty(Rating.Easy)
    assert d_again > d_easy
    assert 1.0 <= d_again <= 10.0
    assert 1.0 <= d_easy <= 10.0


def test_retrievability_fresh():
    """R = 1.0 when elapsed_days = 0."""
    assert retrievability(10.0, 0.0) == 1.0


def test_retrievability_decays():
    """Retrievability should decrease over time (power-law)."""
    r1 = retrievability(10.0, 1.0)
    r10 = retrievability(10.0, 10.0)
    r100 = retrievability(10.0, 100.0)
    r1000 = retrievability(10.0, 1000.0)
    assert r1 > r10 > r100 > r1000
    assert r1000 < 0.5  # Must forget eventually


def test_retrievability_stability_effect():
    """Higher stability = slower forgetting."""
    r_low_s = retrievability(1.0, 5.0)
    r_high_s = retrievability(100.0, 5.0)
    assert r_high_s > r_low_s


def test_forgetting_factor():
    """Factor should be positive and finite."""
    f = forgetting_factor()
    assert f > 0
    assert math.isfinite(f)


def test_scheduler_new_card():
    """First review of a new card should transition to Learning state."""
    scheduler = FSRS6Scheduler()
    state = FSRSState()
    assert state.state == LearningState.New
    result = scheduler.review(state, Rating.Good)
    assert result.state.state != LearningState.New
    assert result.state.stability > 0
    assert result.interval >= 1


def test_scheduler_lapse():
    """Rating Again on a Review card should create a lapse."""
    scheduler = FSRS6Scheduler()
    state = FSRSState()
    # First review: New -> Learning/Review
    result1 = scheduler.review(state, Rating.Good)
    # Second review: should be in Review
    result2 = scheduler.review(result1.state, Rating.Good, elapsed_days=3.0)
    # Third review with Again after time has passed: should be a lapse
    result3 = scheduler.review(result2.state, Rating.Again, elapsed_days=1.5)
    assert result3.is_lapse is True
    assert result3.state.lapses >= 1


def test_scheduler_same_day_not_lapse():
    """Same-day Again should NOT count as a lapse (FSRS-6 same-day handling)."""
    scheduler = FSRS6Scheduler()
    state = FSRSState()
    result1 = scheduler.review(state, Rating.Good)
    # Immediate Again (same day, 0 elapsed)
    result2 = scheduler.review(result1.state, Rating.Again)
    assert result2.is_lapse is False


def test_scheduler_due_cards():
    """get_due_cards should prioritize new cards and overdue cards."""
    scheduler = FSRS6Scheduler()
    cards = [
        Card(card_id="new1", category="Pneumothorax"),
        Card(card_id="new2", category="Cardiomegaly"),
    ]
    due = scheduler.get_due_cards(cards)
    assert len(due) == 2  # Both new cards are due


# ─── MedASR Tests ────────────────────────────────────────────────

from engram.medasr import MockMedASREngine, TranscriptionResult


def test_mock_medasr_load():
    """MockMedASREngine should be immediately loaded."""
    engine = MockMedASREngine()
    assert engine._loaded is True


def test_mock_medasr_transcribe():
    """Mock transcription should return a TranscriptionResult."""
    engine = MockMedASREngine()
    result = engine.transcribe("/fake/audio.wav")
    assert isinstance(result, TranscriptionResult)
    assert len(result.text) > 0
    assert result.confidence > 0


def test_mock_medasr_array():
    """Mock transcription from array should handle different lengths."""
    engine = MockMedASREngine()
    # Short array -> empty
    short = np.zeros(100, dtype=np.float32)
    result_short = engine.transcribe_array(short)
    assert result_short.text == ""

    # Long array -> mock transcription
    long = np.random.randn(16000).astype(np.float32)
    result_long = engine.transcribe_array(long)
    assert len(result_long.text) > 0
    assert result_long.confidence > 0


def test_mock_medasr_tuple_input():
    """Mock transcription should handle (sample_rate, array) tuple."""
    engine = MockMedASREngine()
    audio = np.random.randn(32000).astype(np.float32)
    result = engine.transcribe_array((16000, audio))
    assert isinstance(result, TranscriptionResult)
    assert "2.0s" in result.text  # 32000 / 16000 = 2.0s


# ─── Longitudinal Tests ─────────────────────────────────────────

from engram.longitudinal import (
    DEFAULT_LONGITUDINAL,
    LONGITUDINAL_DATA,
    LongitudinalCase,
    create_longitudinal_pairs,
    generate_longitudinal_question,
    get_longitudinal_feedback,
)


def test_longitudinal_data_categories():
    """Should have clinical knowledge for major categories."""
    expected = {"Cardiomegaly", "Pleural Effusion", "Pneumothorax",
                "Consolidation", "Edema", "Support Devices"}
    assert expected == set(LONGITUDINAL_DATA.keys())


def test_generate_longitudinal_question():
    """Should generate a question with category and instructions."""
    q = generate_longitudinal_question("Cardiomegaly", "worsened")
    assert "Cardiomegaly" in q
    assert "Prior" in q or "prior" in q
    assert "Current" in q or "current" in q


def test_longitudinal_feedback_good_answer():
    """Student who uses correct keywords should score well."""
    from engram.medgemma import FeedbackResult
    result = get_longitudinal_feedback(
        "Cardiomegaly", "worsened",
        "The cardiac silhouette has worsened and increased in size. "
        "New progression of findings.",
    )
    assert isinstance(result, FeedbackResult)
    assert result.score >= 0.5  # Should detect keywords


def test_longitudinal_feedback_empty_answer():
    """Empty answer should score 0."""
    result = get_longitudinal_feedback("Cardiomegaly", "worsened", "")
    assert result.score == 0.0


def test_longitudinal_feedback_wrong_answer():
    """Answer with no correct keywords should score low."""
    result = get_longitudinal_feedback(
        "Cardiomegaly", "worsened",
        "The lungs appear normal with no significant findings.",
    )
    assert result.score < 0.5


def test_create_longitudinal_pairs():
    """Should create pairs from cards with 2+ images per category."""
    cards = {
        "Cardiomegaly": [
            Card(card_id="c1", category="Cardiomegaly", image_path="/fake/1.png"),
            Card(card_id="c2", category="Cardiomegaly", image_path="/fake/2.png"),
        ],
        "Pneumothorax": [
            Card(card_id="p1", category="Pneumothorax", image_path="/fake/3.png"),
        ],  # Only 1 — should be skipped
    }
    pairs = create_longitudinal_pairs(cards)
    assert len(pairs) == 1  # Only Cardiomegaly has 2
    assert pairs[0].category == "Cardiomegaly"
    assert isinstance(pairs[0], LongitudinalCase)


def test_default_longitudinal_coverage():
    """Default should cover all change types."""
    for ct in ["worsened", "improved", "stable", "new", "resolved"]:
        assert ct in DEFAULT_LONGITUDINAL


# ─── CXR Foundation Tests ───────────────────────────────────────

from engram.cxr_foundation import CXRFoundationRetriever, CXRSimilarCase


def test_cxr_foundation_init():
    """CXRFoundationRetriever should initialize without loading model."""
    retriever = CXRFoundationRetriever()
    assert retriever._loaded is False
    assert retriever.index is None


def test_cxr_foundation_search_empty():
    """Search on empty index should return empty list."""
    retriever = CXRFoundationRetriever()
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    results = retriever.search(img)
    assert results == []


def test_cxr_similar_case_dataclass():
    """CXRSimilarCase should hold expected fields."""
    case = CXRSimilarCase(
        image_path="/fake/img.png",
        category="Pneumothorax",
        similarity=0.95,
        card_id="abc123",
    )
    assert case.similarity == 0.95
    assert case.category == "Pneumothorax"


# ─── Mock Engine Tests ───────────────────────────────────────────

from engram.mock_engine import MockMedGemmaEngine
from engram.medgemma import BoundingBox, FeedbackResult


def test_mock_engine_load():
    """Mock engine should load without errors."""
    engine = MockMedGemmaEngine()
    engine.load()
    assert engine._loaded is True


def test_mock_engine_grade():
    """Mock engine should return FeedbackResult."""
    engine = MockMedGemmaEngine()
    engine.load()
    img = Image.new("RGB", (512, 512), color=(0, 0, 0))
    result = engine.grade_response(img, "enlarged heart shadow", "Cardiomegaly")
    assert isinstance(result, FeedbackResult)
    assert 0 <= result.score <= 1


def test_mock_engine_boxes():
    """Mock engine should return bounding boxes for known categories."""
    engine = MockMedGemmaEngine()
    engine.load()
    img = Image.new("RGB", (512, 512), color=(0, 0, 0))
    boxes = engine.localize_findings(img, "Cardiomegaly")
    assert len(boxes) > 0
    assert all(isinstance(b, BoundingBox) for b in boxes)


def test_bounding_box_to_pixel():
    """BoundingBox.to_pixel should convert normalized coords to pixels."""
    box = BoundingBox(y0=0, x0=0, y1=1000, x1=1000, label="test")
    py0, px0, py1, px1 = box.to_pixel(512, 512)
    assert py0 == 0
    assert px0 == 0
    assert py1 == 512
    assert px1 == 512


# ─── Student State Tests ────────────────────────────────────────

from engram.student import StudentState, ReviewLog


def test_student_state_create():
    """StudentState should initialize with empty cards and history."""
    student = StudentState(student_id="test1", name="Test Student")
    assert student.student_id == "test1"
    assert len(student.cards) == 0
    assert len(student.review_history) == 0


def test_student_state_add_card():
    """Student should be able to add cards."""
    student = StudentState(student_id="test2", name="Test")
    card = Card(card_id="c1", category="Pneumothorax")
    student.add_card(card)
    assert "c1" in student.cards


def test_student_state_save_load():
    """Student state should round-trip through JSON."""
    student = StudentState(student_id="test3", name="Saver")
    card = Card(card_id="c1", category="Cardiomegaly")
    student.add_card(card)

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        student.save(path)
        loaded = StudentState.load(path)
        assert loaded.student_id == "test3"
        assert "c1" in loaded.cards
        assert loaded.cards["c1"].category == "Cardiomegaly"
    finally:
        os.unlink(path)


# ─── Blindspot Tests ────────────────────────────────────────────

from engram.blindspot import render_blindspot_html, render_session_stats_html


def test_blindspot_empty():
    """Empty blindspot list should render without errors."""
    html = render_blindspot_html([])
    assert isinstance(html, str)


def test_session_stats_html():
    """Session stats should render with given data."""
    html = render_session_stats_html({
        "total_reviews": 10,
        "avg_score": 0.75,
        "avg_box_iou": 0.6,
        "categories_practiced": 3,
    })
    assert "10" in html
    assert isinstance(html, str)


# ─── Dataset Tests ──────────────────────────────────────────────

from engram.dataset import CATEGORY_DESCRIPTIONS, load_demo_dataset


def test_category_descriptions():
    """Should have descriptions for standard CheXpert categories."""
    assert "Cardiomegaly" in CATEGORY_DESCRIPTIONS
    assert "Pneumothorax" in CATEGORY_DESCRIPTIONS
    assert len(CATEGORY_DESCRIPTIONS) >= 10


def test_load_demo_dataset():
    """Should load cards from demo directory."""
    demo_dir = os.path.join(os.path.dirname(__file__), "..", "data", "demo")
    if os.path.exists(demo_dir):
        cards = load_demo_dataset(demo_dir)
        assert len(cards) > 0
        assert all(isinstance(c, Card) for c in cards)


# ─── Integration Test ───────────────────────────────────────────

def test_full_review_cycle():
    """End-to-end: create student, load card, review, verify state update."""
    scheduler = FSRS6Scheduler()
    student = StudentState(student_id="int1", name="Integration")
    card = Card(card_id="int_c1", category="Pneumothorax", image_path="/fake.png")
    student.add_card(card)

    # First review
    result = scheduler.review(card.fsrs, Rating.Good)
    assert result.state.stability > 0
    assert result.interval >= 1

    # Second review (simulate passage of time)
    result2 = scheduler.review(result.state, Rating.Hard, elapsed_days=2.0)
    assert result2.state.reps >= 2

    # Third review — Easy
    result3 = scheduler.review(result2.state, Rating.Easy, elapsed_days=5.0)
    assert result3.state.stability > result2.state.stability  # Easy should boost stability


def test_version():
    """Version should be 0.4.0."""
    from engram import __version__
    assert __version__ == "0.4.0"


# ═══════════════════════════════════════════════════════════════
# v0.4.0 Feature Tests (32 tests)
# ═══════════════════════════════════════════════════════════════

from engram.fsrs6 import (
    interval_modifier_for_overconfidence,
    search_completeness_modifier,
)
from engram.blindspot import render_calibration_chart_html
from engram.mock_engine import MockMedGemmaEngine
from engram.hear import MockHeAREngine, HeAREngine, LUNG_SOUNDS, LungSoundCase


# ─── F1: Confidence Calibration ───────────────────────────────

def test_review_log_confidence_field():
    """ReviewLog should support confidence field with default 0."""
    from engram.student import ReviewLog
    log = ReviewLog(card_id="c1", category="Cardiomegaly", grade=3,
                    score=0.8, box_iou=0.5, retrievability=0.9, interval=3)
    assert log.confidence == 0
    log2 = ReviewLog(card_id="c1", category="Cardiomegaly", grade=3,
                     score=0.8, box_iou=0.5, retrievability=0.9, interval=3,
                     confidence=4)
    assert log2.confidence == 4


def test_overconfidence_modifier_no_gap():
    """Small calibration gap should return modifier 1.0."""
    assert interval_modifier_for_overconfidence(0.05) == 1.0
    assert interval_modifier_for_overconfidence(0.1) == 1.0


def test_overconfidence_modifier_large_gap():
    """Large calibration gap should reduce interval."""
    mod = interval_modifier_for_overconfidence(0.4)
    assert 0.5 <= mod < 1.0


def test_overconfidence_modifier_clamp():
    """Modifier should never go below 0.5."""
    assert interval_modifier_for_overconfidence(0.9) == 0.5
    assert interval_modifier_for_overconfidence(1.0) == 0.5


def test_calibration_per_category():
    """Student should compute calibration data per category."""
    student = StudentState(student_id="cal1", name="CalTest")
    card = Card(card_id="cal_c1", category="Edema", image_path="/fake.png")
    student.add_card(card)

    scheduler = FSRS6Scheduler()
    result = scheduler.review(card.fsrs, Rating.Good)
    student.record_review(card, Rating.Good, result, score=0.5, confidence=5)
    student.record_review(card, Rating.Good, result, score=0.6, confidence=5)

    cal = student.calibration_per_category()
    assert "Edema" in cal
    assert cal["Edema"]["overconfident"] is True
    assert cal["Edema"]["mean_confidence"] > 0


def test_calibration_chart_empty():
    """Empty calibration data should render without errors."""
    html = render_calibration_chart_html({})
    assert isinstance(html, str)
    assert "No confidence data" in html


def test_calibration_chart_with_data():
    """Calibration chart should render with real data."""
    data = {
        "Edema": {
            "mean_confidence": 0.9,
            "mean_accuracy": 0.5,
            "calibration_gap": 0.4,
            "overconfident": True,
            "n_reviews": 5,
        }
    }
    html = render_calibration_chart_html(data)
    assert "Edema" in html
    assert "OVERCONFIDENT" in html


# ─── F2: Socratic Mode ───────────────────────────────────────

def test_socratic_question_generation():
    """Mock engine should generate Socratic questions."""
    eng = MockMedGemmaEngine()
    eng.load()
    q = eng.generate_socratic_question(None, "I see nothing", "Consolidation")
    assert isinstance(q, str)
    assert len(q) > 10


def test_socratic_followup():
    """Mock engine should generate Socratic followup."""
    eng = MockMedGemmaEngine()
    eng.load()
    f = eng.generate_socratic_followup("", "bronchial breath sounds", "Consolidation")
    assert isinstance(f, str)
    assert len(f) > 10


def test_socratic_covers_all_categories():
    """Socratic templates should cover common categories."""
    eng = MockMedGemmaEngine()
    eng.load()
    for cat in ["Cardiomegaly", "Pneumothorax", "Consolidation", "Edema", "No Finding"]:
        q = eng.generate_socratic_question(None, "", cat)
        assert isinstance(q, str)


# ─── F3: Satisfaction of Search ───────────────────────────────

def test_search_completeness_grading():
    """Should grade search completeness for student answer."""
    eng = MockMedGemmaEngine()
    eng.load()
    found, missed, score = eng.grade_search_completeness(
        "enlarged cardiac silhouette, pulmonary congestion",
        "Cardiomegaly",
    )
    assert isinstance(found, list)
    assert isinstance(missed, list)
    assert 0.0 <= score <= 1.0


def test_search_completeness_empty_answer():
    """Empty answer should find nothing."""
    eng = MockMedGemmaEngine()
    eng.load()
    found, missed, score = eng.grade_search_completeness("", "Cardiomegaly")
    assert len(found) == 0
    assert score == 0.0


def test_search_completeness_modifier_high():
    """High completeness should not reduce interval."""
    assert search_completeness_modifier(0.9) == 1.0
    assert search_completeness_modifier(1.0) == 1.0


def test_search_completeness_modifier_low():
    """Low completeness should reduce interval."""
    mod = search_completeness_modifier(0.2)
    assert mod == 0.5


def test_search_completeness_modifier_mid():
    """Mid completeness should partially reduce interval."""
    mod = search_completeness_modifier(0.5)
    assert 0.5 < mod < 1.0


def test_review_log_search_fields():
    """ReviewLog should support search completeness fields."""
    from engram.student import ReviewLog
    log = ReviewLog(card_id="c1", category="Edema", grade=3, score=0.8,
                    box_iou=0.5, retrievability=0.9, interval=3,
                    search_completeness=0.67, found_findings=2, total_findings=3)
    assert log.search_completeness == 0.67
    assert log.found_findings == 2
    assert log.total_findings == 3


# ─── F4: Dual-Process Training ───────────────────────────────

def test_gestalt_grading():
    """Mock engine should grade gestalt impressions."""
    eng = MockMedGemmaEngine()
    eng.load()
    score = eng.grade_gestalt("enlarged heart", "Cardiomegaly")
    assert 0.0 <= score <= 1.0
    assert score > 0  # Should match "enlarged" or "heart"


def test_gestalt_empty_answer():
    """Empty gestalt answer should score 0."""
    eng = MockMedGemmaEngine()
    eng.load()
    score = eng.grade_gestalt("", "Cardiomegaly")
    assert score == 0.0


def test_gestalt_wrong_category():
    """Gestalt for wrong category keywords should score low."""
    eng = MockMedGemmaEngine()
    eng.load()
    score = eng.grade_gestalt("completely normal lungs", "Consolidation")
    assert score < 0.5


def test_review_log_gestalt_field():
    """ReviewLog should support gestalt_score field."""
    from engram.student import ReviewLog
    log = ReviewLog(card_id="c1", category="Edema", grade=3, score=0.8,
                    box_iou=0.5, retrievability=0.9, interval=3,
                    gestalt_score=0.7)
    assert log.gestalt_score == 0.7


# ─── F5: Contrastive Case Pairs ──────────────────────────────

def test_contrastive_pairs_defined():
    """Mock engine should have contrastive pairs."""
    eng = MockMedGemmaEngine()
    eng.load()
    assert len(eng.CONTRASTIVE_PAIRS) >= 4


def test_contrastive_question_generation():
    """Should generate contrastive question for a pair."""
    eng = MockMedGemmaEngine()
    eng.load()
    q = eng.generate_contrastive_question("Consolidation", "Atelectasis")
    assert isinstance(q, str)
    assert len(q) > 10


def test_contrastive_grading():
    """Should grade contrastive response."""
    eng = MockMedGemmaEngine()
    eng.load()
    result = eng.grade_contrastive(
        "consolidation has air bronchograms, atelectasis has volume loss",
        "Consolidation", "Atelectasis",
    )
    assert hasattr(result, "score")
    assert 0.0 <= result.score <= 1.0


def test_review_log_contrastive_field():
    """ReviewLog should support contrastive_pair field."""
    from engram.student import ReviewLog
    log = ReviewLog(card_id="c1", category="Edema", grade=3, score=0.8,
                    box_iou=0.5, retrievability=0.9, interval=3,
                    contrastive_pair="Consolidation vs Atelectasis")
    assert log.contrastive_pair == "Consolidation vs Atelectasis"


# ─── F6: HeAR Integration ────────────────────────────────────

def test_lung_sounds_coverage():
    """LUNG_SOUNDS should cover all standard categories."""
    assert "Cardiomegaly" in LUNG_SOUNDS
    assert "Pneumothorax" in LUNG_SOUNDS
    assert "No Finding" in LUNG_SOUNDS
    assert len(LUNG_SOUNDS) >= 10


def test_lung_sound_data_structure():
    """Each lung sound entry should have required fields."""
    for cat, data in LUNG_SOUNDS.items():
        assert "sound" in data, f"Missing 'sound' for {cat}"
        assert "location" in data, f"Missing 'location' for {cat}"
        assert "character" in data, f"Missing 'character' for {cat}"
        assert "correlation" in data, f"Missing 'correlation' for {cat}"


def test_mock_hear_generate_sound():
    """MockHeAREngine should generate synthetic audio."""
    eng = MockHeAREngine()
    eng.load()
    sr, audio = eng.generate_lung_sound("Consolidation")
    assert sr == 16000
    assert len(audio) > 0
    assert audio.dtype == np.float32


def test_mock_hear_classify():
    """MockHeAREngine should classify audio into categories."""
    eng = MockHeAREngine()
    eng.load()
    audio = np.random.randn(16000).astype(np.float32)
    result = eng.classify_lung_sound(audio)
    assert isinstance(result, dict)
    assert len(result) >= 10
    assert abs(sum(result.values()) - 1.0) < 0.01


def test_mock_hear_lung_sound_case():
    """MockHeAREngine should return LungSoundCase dataclass."""
    eng = MockHeAREngine()
    eng.load()
    case = eng.get_lung_sound_case("Consolidation")
    assert isinstance(case, LungSoundCase)
    assert case.category == "Consolidation"
    assert "bronchial" in case.sound_type


def test_hear_engine_stub():
    """HeAREngine should exist and have same interface as mock."""
    eng = HeAREngine()
    eng.load()
    sr, audio = eng.generate_lung_sound("Cardiomegaly")
    assert sr == 16000
    assert len(audio) > 0


# ─── Cross-Feature Integration ───────────────────────────────

def test_student_save_load_v04_fields():
    """Student state should persist all v0.4 fields."""
    from engram.student import ReviewLog
    student = StudentState(student_id="v04", name="V04Test")
    card = Card(card_id="v04_c1", category="Edema", image_path="/fake.png")
    student.add_card(card)

    scheduler = FSRS6Scheduler()
    result = scheduler.review(card.fsrs, Rating.Good)
    student.record_review(
        card, Rating.Good, result, score=0.8,
        confidence=4, search_completeness=0.67,
        found_findings=2, total_findings=3,
        gestalt_score=0.7, contrastive_pair="Edema vs Cardiomegaly",
    )

    # Save and reload
    path = tempfile.NamedTemporaryFile(suffix=".json", delete=False).name
    try:
        student.save(path)
        loaded = StudentState.load(path)
        log = loaded.review_history[0]
        assert log.confidence == 4
        assert log.search_completeness == 0.67
        assert log.found_findings == 2
        assert log.total_findings == 3
        assert log.gestalt_score == 0.7
        assert log.contrastive_pair == "Edema vs Cardiomegaly"
    finally:
        os.unlink(path)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
