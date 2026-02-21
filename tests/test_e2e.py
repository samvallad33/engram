"""
ENGRAM v0.4.0 End-to-End Tests
Tests the COMPLETE learning loop through app.py functions.

Day 6: Full learning loop verification.
"""

import os
import tempfile

import numpy as np
from PIL import Image


from engram.fsrs6 import (
    Card, FSRSState, FSRS6Scheduler, LearningState, Rating,
)
from engram.medasr import MockMedASREngine
from engram.mock_engine import MockMedGemmaEngine
from engram.medgemma import BoundingBox, FeedbackResult
from engram.longitudinal import (
    create_longitudinal_pairs, generate_longitudinal_question,
    get_longitudinal_feedback, LongitudinalCase,
)
from engram.cxr_foundation import CXRFoundationRetriever
from engram.student import StudentState
from engram.blindspot import render_blindspot_html, render_session_stats_html
from engram.dataset import CATEGORY_DESCRIPTIONS, load_demo_dataset


# ─── Helpers ────────────────────────────────────────────────────

DEMO_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "demo")


def _create_test_image(path: str, color=(128, 128, 128)):
    """Create a test image at the given path."""
    img = Image.new("RGB", (256, 256), color=color)
    img.save(path)
    return path


def _has_demo_images():
    """Check if demo images exist."""
    return os.path.exists(DEMO_DIR) and any(
        os.path.isdir(os.path.join(DEMO_DIR, d))
        for d in os.listdir(DEMO_DIR)
    )


# ─── E2E Test 1: Complete Training Session ─────────────────────

def test_e2e_complete_training_session():
    """
    Full learning loop:
    Start session → Load cards → Get next case → Submit answer →
    Get AI feedback → FSRS-6 update → Verify state change → Next case
    """
    scheduler = FSRS6Scheduler(desired_retention=0.9)
    engine = MockMedGemmaEngine()
    engine.load()

    # 1. Create student
    student = StudentState(student_id="e2e_1", name="E2E Tester")
    assert len(student.cards) == 0

    # 2. Load demo dataset
    if _has_demo_images():
        cards = load_demo_dataset(DEMO_DIR)
        assert len(cards) > 0, "Demo images exist but no cards loaded"
    else:
        # Create temporary images for testing
        with tempfile.TemporaryDirectory() as tmpdir:
            categories = ["Cardiomegaly", "Pneumothorax", "Pleural Effusion"]
            cards = []
            for i, cat in enumerate(categories):
                cat_dir = os.path.join(tmpdir, cat)
                os.makedirs(cat_dir)
                for j in range(2):
                    path = os.path.join(cat_dir, f"test_{j}.png")
                    _create_test_image(path)
                    cards.append(Card(
                        card_id=f"{cat}_{j}", category=cat, image_path=path,
                    ))

    # 3. Add cards to student
    for card in cards:
        student.add_card(card)
    assert len(student.cards) > 0

    # 4. Get due cards (all should be due — they're New)
    all_cards = list(student.cards.values())
    due = scheduler.get_due_cards(all_cards)
    assert len(due) > 0, "No due cards found"

    # 5. Pick first card, generate question
    card = due[0]
    assert card.fsrs.state == LearningState.New

    try:
        img = Image.open(card.image_path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (256, 256))

    question = engine.generate_question(img, card.category)
    assert len(question) > 0, "Question should not be empty"

    # 6. Submit answer — get AI feedback
    student_answer = "I see an enlarged cardiac silhouette suggestive of cardiomegaly"
    feedback = engine.grade_response(img, student_answer, card.category)
    assert isinstance(feedback, FeedbackResult)
    assert 0.0 <= feedback.score <= 1.0

    # 7. Get bounding boxes
    boxes = engine.localize_findings(img, card.category)
    assert isinstance(boxes, list)

    # 8. FSRS-6 review
    result = scheduler.review(card.fsrs, Rating.Good)
    assert result.state.state != LearningState.New
    assert result.state.stability > 0
    assert result.interval >= 1

    # 9. Record review in student state
    student.record_review(card, Rating.Good, result, score=feedback.score)
    assert student.total_reviews == 1
    assert len(student.review_history) == 1
    assert student.review_history[0].category == card.category
    assert student.review_history[0].score == feedback.score

    # 10. Verify card state was updated
    updated_card = student.cards[card.card_id]
    assert updated_card.fsrs.state != LearningState.New
    assert updated_card.times_shown == 1

    # 11. Get blind spots
    spots = student.get_blind_spots()
    assert isinstance(spots, list)

    # 12. Get session stats
    stats = student.get_session_stats()
    assert stats["total_reviews"] == 1
    assert stats["categories_practiced"] >= 1

    # 13. Render HTML outputs (no crash)
    blindspot_html = render_blindspot_html(spots)
    assert isinstance(blindspot_html, str)
    stats_html = render_session_stats_html(stats)
    assert isinstance(stats_html, str)


# ─── E2E Test 2: Multi-Review Learning Curve ──────────────────

def test_e2e_multi_review_learning_curve():
    """
    Simulate a full study session: 10 reviews across 3 categories.
    Verify FSRS-6 states evolve correctly and student stats accumulate.
    """
    scheduler = FSRS6Scheduler(desired_retention=0.9)
    engine = MockMedGemmaEngine()
    engine.load()
    student = StudentState(student_id="e2e_2", name="Multi Reviewer")

    # Create cards
    categories = ["Cardiomegaly", "Pneumothorax", "Pleural Effusion"]
    for i, cat in enumerate(categories):
        for j in range(3):
            student.add_card(Card(
                card_id=f"{cat}_{j}", category=cat,
                image_path=f"/fake/{cat}_{j}.png",
            ))

    # Simulate 10 reviews with varying grades
    grades = [
        Rating.Good, Rating.Hard, Rating.Easy, Rating.Again,
        Rating.Good, Rating.Good, Rating.Hard, Rating.Easy,
        Rating.Good, Rating.Again,
    ]
    img = Image.new("RGB", (256, 256))

    reviewed_cards = list(student.cards.values())
    for i, grade in enumerate(grades):
        card = reviewed_cards[i % len(reviewed_cards)]
        feedback = engine.grade_response(img, "test answer", card.category)

        elapsed = 1.0 + i * 0.5 if card.fsrs.state != LearningState.New else 0.0
        result = scheduler.review(card.fsrs, grade, elapsed_days=elapsed)
        student.record_review(card, grade, result, score=feedback.score)

    # Verify accumulated state
    assert student.total_reviews == 10
    assert len(student.review_history) == 10

    stats = student.get_session_stats()
    assert stats["total_reviews"] == 10
    assert stats["categories_practiced"] == 3
    assert 0.0 <= stats["avg_score"] <= 1.0

    # Verify FSRS-6 states diverged per card
    stabilities = [c.fsrs.stability for c in student.cards.values() if c.fsrs.stability > 0]
    assert len(stabilities) > 0, "Some cards should have non-zero stability"

    # Verify blind spots include all reviewed categories
    spots = student.get_blind_spots()
    spot_categories = {s.category for s in spots}
    for cat in categories:
        assert cat in spot_categories, f"Category {cat} missing from blind spots"


# ─── E2E Test 3: MedASR Voice Dictation Pipeline ─────────────

def test_e2e_medasr_pipeline():
    """
    Full voice pipeline: Record audio → MedASR transcribe → Feed to grading.
    Tests mock mode end-to-end.
    """
    asr = MockMedASREngine()
    engine = MockMedGemmaEngine()
    engine.load()

    # Simulate Gradio audio input (sample_rate, numpy_array)
    sample_rate = 16000
    duration_sec = 3.0
    audio_array = np.random.randn(int(sample_rate * duration_sec)).astype(np.float32)
    gradio_audio = (sample_rate, audio_array)

    # Step 1: Transcribe
    result = asr.transcribe_array(gradio_audio)
    assert len(result.text) > 0
    assert result.confidence > 0
    assert "3.0s" in result.text  # Should mention duration

    # Step 2: Feed transcription to grading
    img = Image.new("RGB", (256, 256))
    feedback = engine.grade_response(img, result.text, "Cardiomegaly")
    assert isinstance(feedback, FeedbackResult)
    assert 0.0 <= feedback.score <= 1.0

    # Step 3: Verify boxes work alongside
    boxes = engine.localize_findings(img, "Cardiomegaly")
    assert len(boxes) > 0


# ─── E2E Test 4: Longitudinal Comparison Mode ────────────────

def test_e2e_longitudinal_mode():
    """
    Full longitudinal flow: Generate pair → Show question → Submit answer →
    Grade → FSRS-6 update.
    """
    scheduler = FSRS6Scheduler()
    student = StudentState(student_id="e2e_4", name="Longitudinal Tester")

    # Need 2+ images per category for pairs
    with tempfile.TemporaryDirectory() as tmpdir:
        cards_by_cat = {}
        for cat in ["Cardiomegaly", "Pleural Effusion"]:
            cards_by_cat[cat] = []
            for j in range(3):
                path = os.path.join(tmpdir, f"{cat}_{j}.png")
                _create_test_image(path)
                card = Card(card_id=f"{cat}_{j}", category=cat, image_path=path)
                cards_by_cat[cat].append(card)
                student.add_card(card)

        # Step 1: Create longitudinal pairs
        pairs = create_longitudinal_pairs(cards_by_cat)
        assert len(pairs) >= 2, f"Expected 2+ pairs, got {len(pairs)}"

        # Step 2: Pick a case
        case = pairs[0]
        assert isinstance(case, LongitudinalCase)
        assert os.path.exists(case.prior_path)
        assert os.path.exists(case.current_path)

        # Step 3: Generate question
        question = generate_longitudinal_question(case.category, case.change_type)
        assert case.category in question
        assert "Prior" in question or "prior" in question

        # Step 4: Load images
        prior_img = Image.open(case.prior_path).convert("RGB")
        current_img = Image.open(case.current_path).convert("RGB")
        assert prior_img.size[0] > 0
        assert current_img.size[0] > 0

        # Step 5: Submit answer and get feedback
        answer = "The cardiac silhouette has worsened and increased in size"
        feedback = get_longitudinal_feedback(case.category, case.change_type, answer)
        assert isinstance(feedback, FeedbackResult)
        assert 0.0 <= feedback.score <= 1.0
        assert len(feedback.explanation) > 0

        # Step 6: FSRS-6 update on a related card
        cat_cards = [c for c in student.cards.values() if c.category == case.category]
        card = cat_cards[0]
        result = scheduler.review(card.fsrs, Rating.Good)
        student.record_review(card, Rating.Good, result, score=feedback.score)

        assert student.total_reviews == 1
        assert card.fsrs.state != LearningState.New


# ─── E2E Test 5: Student State Persistence Round-Trip ─────────

def test_e2e_state_persistence():
    """
    Full round-trip: Create student → Do reviews → Save → Load → Verify
    all state was preserved including FSRS-6 states and review history.
    """
    scheduler = FSRS6Scheduler()
    student = StudentState(student_id="e2e_5", name="Persistence Test")

    # Add cards and do some reviews
    for cat in ["Pneumothorax", "Cardiomegaly"]:
        card = Card(card_id=f"p_{cat}", category=cat, image_path=f"/fake/{cat}.png")
        student.add_card(card)
        result = scheduler.review(card.fsrs, Rating.Good)
        student.record_review(card, Rating.Good, result, score=0.75)

    # Save
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        student.save(path)

        # Load
        loaded = StudentState.load(path)

        # Verify everything persisted
        assert loaded.student_id == "e2e_5"
        assert loaded.name == "Persistence Test"
        assert loaded.total_reviews == 2
        assert len(loaded.review_history) == 2
        assert len(loaded.cards) == 2

        # Verify FSRS-6 states persisted
        for cid, card in loaded.cards.items():
            assert card.fsrs.stability > 0, f"Card {cid} stability should be > 0"
            assert card.fsrs.state != LearningState.New, f"Card {cid} should not be New"
            assert card.times_shown == 1

        # Verify review history
        for rev in loaded.review_history:
            assert rev.grade == Rating.Good.value
            assert rev.score == 0.75
            assert rev.interval >= 1
    finally:
        os.unlink(path)


# ─── E2E Test 6: CXR Foundation Retrieval (Mock) ─────────────

def test_e2e_cxr_foundation_retrieval():
    """CXR Foundation retriever initializes and handles empty state gracefully."""
    retriever = CXRFoundationRetriever()
    assert retriever._loaded is False
    assert retriever.index is None

    # Search without loading should return empty (no crash)
    img = Image.new("RGB", (224, 224))
    results = retriever.search(img)
    assert results == []


# ─── E2E Test 7: Gradio App Builds ───────────────────────────

def test_e2e_gradio_app_builds():
    """The Gradio app should build without errors."""
    # Import the build function
    from app import build_app
    app = build_app()
    assert app is not None


# ─── E2E Test 8: Full Session Cycle with Export ──────────────

def test_e2e_session_export():
    """Full session → review → export → verify exported data."""
    scheduler = FSRS6Scheduler()
    student = StudentState(student_id="e2e_8", name="Exporter")
    engine = MockMedGemmaEngine()
    engine.load()

    # Do a complete session
    categories = ["Cardiomegaly", "Pneumothorax", "Pleural Effusion",
                   "Consolidation", "Edema"]
    img = Image.new("RGB", (256, 256))

    for cat in categories:
        card = Card(card_id=f"exp_{cat}", category=cat, image_path=f"/fake/{cat}.png")
        student.add_card(card)

        # Generate question
        q = engine.generate_question(img, cat)
        assert len(q) > 0

        # Grade
        fb = engine.grade_response(img, "test answer", cat)

        # FSRS-6
        result = scheduler.review(card.fsrs, Rating.Good)
        student.record_review(card, Rating.Good, result, score=fb.score)

    # Export
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        student.save(path)

        # Verify file exists and has content
        assert os.path.getsize(path) > 100

        # Load and verify
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["student_id"] == "e2e_8"
        assert len(data["cards"]) == 5
        assert len(data["review_history"]) == 5
        assert data["total_reviews"] == 5

        # Verify FSRS states in export
        for cid, cdata in data["cards"].items():
            assert cdata["fsrs"]["stability"] > 0
            assert cdata["fsrs"]["state"] != LearningState.New.value
    finally:
        os.unlink(path)


# ─── E2E Test 9: Lapse Recovery Flow ────────────────────────

def test_e2e_lapse_recovery():
    """
    Simulate student forgetting: Good → Good → Again (lapse) → Recovery.
    Verify FSRS-6 handles the lapse correctly and schedules recovery.
    """
    scheduler = FSRS6Scheduler()
    student = StudentState(student_id="e2e_9", name="Lapse Tester")

    card = Card(card_id="lapse_1", category="Pneumothorax", image_path="/fake.png")
    student.add_card(card)

    # Review 1: Good (New → Learning/Review)
    r1 = scheduler.review(card.fsrs, Rating.Good)
    student.record_review(card, Rating.Good, r1, score=0.75)
    s1 = r1.state.stability
    assert s1 > 0

    # Review 2: Good (stabilize further)
    r2 = scheduler.review(r1.state, Rating.Good, elapsed_days=3.0)
    student.record_review(card, Rating.Good, r2, score=0.80)
    s2 = r2.state.stability
    assert s2 > s1  # Stability should increase

    # Review 3: Again (LAPSE — next day, simulating forgetting)
    r3 = scheduler.review(r2.state, Rating.Again, elapsed_days=1.5)
    student.record_review(card, Rating.Again, r3, score=0.10)
    assert r3.is_lapse is True
    assert r3.state.lapses >= 1
    assert r3.state.stability < s2  # Stability drops on lapse

    # Review 4: Recovery — Good
    r4 = scheduler.review(r3.state, Rating.Good, elapsed_days=0.5)
    student.record_review(card, Rating.Good, r4, score=0.70)

    # Verify full history
    assert student.total_reviews == 4
    grades = [rev.grade for rev in student.review_history]
    assert grades == [3, 3, 1, 3]  # Good, Good, Again, Good


# ─── E2E Test 10: All Categories Coverage ───────────────────

def test_e2e_all_categories():
    """Verify mock engine handles ALL 11 CheXpert categories."""
    engine = MockMedGemmaEngine()
    engine.load()
    img = Image.new("RGB", (256, 256))

    expected_categories = [
        "Cardiomegaly", "Pneumothorax", "Pleural Effusion",
        "Consolidation", "Lung Opacity", "Edema",
        "Pneumonia", "Atelectasis", "Support Devices",
        "Fracture", "No Finding",
    ]

    for cat in expected_categories:
        # Question generation
        q = engine.generate_question(img, cat)
        assert len(q) > 0, f"Empty question for {cat}"

        # Grading
        fb = engine.grade_response(img, "test answer", cat)
        assert isinstance(fb, FeedbackResult), f"Bad feedback for {cat}"
        assert 0.0 <= fb.score <= 1.0, f"Bad score for {cat}"

        # Bounding boxes
        boxes = engine.localize_findings(img, cat)
        assert isinstance(boxes, list), f"Bad boxes for {cat}"
        # Some categories like "No Finding" may return empty boxes
        if cat != "No Finding":
            assert len(boxes) > 0, f"No boxes for {cat}"


# ─── E2E Test 11: Bounding Box Drawing Pipeline ─────────────

def test_e2e_bounding_box_drawing():
    """Verify bounding box drawing produces valid annotated image."""
    from app import draw_boxes_on_image

    img = Image.new("RGB", (512, 512), color=(50, 50, 50))
    boxes = [
        BoundingBox(y0=100, x0=150, y1=400, x1=450, label="cardiomegaly"),
        BoundingBox(y0=50, x0=300, y1=200, x1=500, label="pleural_effusion"),
    ]

    annotated = draw_boxes_on_image(img, boxes)
    assert annotated.size == img.size
    assert annotated.mode == "RGB"
    # Should be different from original (boxes drawn)
    assert list(annotated.get_flattened_data()) != list(img.get_flattened_data())


# ─── E2E Test 12: Blind Spot Detection Accuracy ─────────────

def test_e2e_blindspot_detection():
    """
    Student weak in Pneumothorax, strong in Cardiomegaly.
    Blind spot detection should identify the weak category.
    """
    scheduler = FSRS6Scheduler()
    student = StudentState(student_id="e2e_12", name="Blindspot Tester")

    # Cardiomegaly: reviewed well (high stability)
    card_strong = Card(card_id="strong_1", category="Cardiomegaly")
    student.add_card(card_strong)
    r1 = scheduler.review(card_strong.fsrs, Rating.Easy)
    student.record_review(card_strong, Rating.Easy, r1, score=0.95)
    r2 = scheduler.review(r1.state, Rating.Easy, elapsed_days=5.0)
    student.record_review(card_strong, Rating.Easy, r2, score=0.95)

    # Pneumothorax: reviewed poorly (low stability)
    card_weak = Card(card_id="weak_1", category="Pneumothorax")
    student.add_card(card_weak)
    r3 = scheduler.review(card_weak.fsrs, Rating.Again)
    student.record_review(card_weak, Rating.Again, r3, score=0.15)

    spots = student.get_blind_spots()
    assert len(spots) >= 2

    # Find the blind spot entries
    spot_dict = {s.category: s for s in spots}
    assert "Cardiomegaly" in spot_dict
    assert "Pneumothorax" in spot_dict

    # Cardiomegaly should have higher stability
    assert spot_dict["Cardiomegaly"].stability > spot_dict["Pneumothorax"].stability


# ─── E2E Test 13: Mixed Training + Longitudinal Session ─────

def test_e2e_mixed_session():
    """
    Simulate realistic session: regular training + longitudinal comparison
    in the same session. Verify both affect student state.
    """
    scheduler = FSRS6Scheduler()
    engine = MockMedGemmaEngine()
    engine.load()
    student = StudentState(student_id="e2e_13", name="Mixed Mode Tester")

    # Create cards with temp images
    with tempfile.TemporaryDirectory() as tmpdir:
        cards_by_cat = {}
        for cat in ["Cardiomegaly", "Pleural Effusion"]:
            cards_by_cat[cat] = []
            for j in range(2):
                path = os.path.join(tmpdir, f"{cat}_{j}.png")
                _create_test_image(path)
                card = Card(card_id=f"m_{cat}_{j}", category=cat, image_path=path)
                student.add_card(card)
                cards_by_cat[cat].append(card)

        # Phase 1: Regular training (2 reviews)
        img = Image.new("RGB", (256, 256))
        for card in list(student.cards.values())[:2]:
            fb = engine.grade_response(img, "test", card.category)
            result = scheduler.review(card.fsrs, Rating.Good)
            student.record_review(card, Rating.Good, result, score=fb.score)

        assert student.total_reviews == 2

        # Phase 2: Longitudinal comparison (1 review)
        pairs = create_longitudinal_pairs(cards_by_cat)
        if pairs:
            case = pairs[0]
            fb = get_longitudinal_feedback(case.category, case.change_type, "worsened increased")
            cat_cards = [c for c in student.cards.values() if c.category == case.category]
            if cat_cards:
                card = cat_cards[0]
                result = scheduler.review(card.fsrs, Rating.Good, elapsed_days=0.5)
                student.record_review(card, Rating.Good, result, score=fb.score)

            assert student.total_reviews == 3

    # Verify both modes contributed to stats
    stats = student.get_session_stats()
    assert stats["total_reviews"] >= 3
    assert stats["categories_practiced"] >= 1


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
