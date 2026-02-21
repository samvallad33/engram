"""
ENGRAM v0.4.0 End-to-End Tests
Tests the COMPLETE learning loop through app.py functions.
Refactored with Pytest Fixtures for maximum efficiency.
"""

import os, tempfile
import pytest
import numpy as np
from PIL import Image

from engram.fsrs6 import Card, FSRS6Scheduler, LearningState, Rating
from engram.medasr import MockMedASREngine
from engram.mock_engine import MockMedGemmaEngine
from engram.medgemma import BoundingBox, FeedbackResult
from engram.longitudinal import create_longitudinal_pairs, generate_longitudinal_question, get_longitudinal_feedback, LongitudinalCase
from engram.cxr_foundation import CXRFoundationRetriever
from engram.student import StudentState
from engram.blindspot import render_blindspot_html, render_session_stats_html
from app import format_annotated_image

# ─── FIXTURES (The S-Tier Refactor) ──────────────────────────────

@pytest.fixture(scope="module")
def test_env():
    """Provides a unified test environment with engines and temp images."""
    env = {
        "scheduler": FSRS6Scheduler(desired_retention=0.9),
        "gemma": MockMedGemmaEngine(),
        "asr": MockMedASREngine(),
        "tmpdir": tempfile.TemporaryDirectory(),
        "img_paths": []
    }
    env["gemma"].load()
    
    # Generate mock images across core categories
    for cat in ["Cardiomegaly", "Pneumothorax", "Pleural Effusion"]:
        cat_dir = os.path.join(env["tmpdir"].name, cat)
        os.makedirs(cat_dir, exist_ok=True)
        for j in range(3):
            path = os.path.join(cat_dir, f"test_{j}.png")
            Image.new("RGB", (256, 256), color=(128, 128, 128)).save(path)
            env["img_paths"].append((cat, path))
            
    yield env
    env["tmpdir"].cleanup()

@pytest.fixture
def student(test_env):
    """Provides a fresh student with loaded cards for each test."""
    s = StudentState(student_id="e2e_test", name="Tester")
    for idx, (cat, path) in enumerate(test_env["img_paths"]):
        s.add_card(Card(card_id=f"c_{idx}", category=cat, image_path=path))
    return s

# ─── E2E Tests ──────────────────────────────────────────────────

def test_e2e_complete_training_session(test_env, student):
    """Full learning loop: Next case → Grade → FSRS Update → Stats."""
    due = test_env["scheduler"].get_due_cards(list(student.cards.values()))
    card = due[0]
    img = Image.open(card.image_path).convert("RGB")
    
    # Generate & Grade
    question = test_env["gemma"].generate_question(img, card.category)
    feedback = test_env["gemma"].grade_response(img, "test answer", card.category)
    boxes = test_env["gemma"].localize_findings(img, card.category)
    
    # FSRS Update
    result = test_env["scheduler"].review(card.fsrs, Rating.Good)
    student.record_review(card, Rating.Good, result, score=feedback.score)
    
    assert student.total_reviews == 1
    assert student.cards[card.card_id].fsrs.state != LearningState.New
    assert isinstance(render_blindspot_html(student.get_blind_spots()), str)

def test_e2e_multi_review_learning_curve(test_env, student):
    """Simulate a full study session across 10 reviews."""
    grades = [Rating.Good, Rating.Hard, Rating.Easy, Rating.Again] * 3
    img = Image.new("RGB", (256, 256))
    
    cards = list(student.cards.values())
    for i, grade in enumerate(grades[:10]):
        c = cards[i % len(cards)]
        fb = test_env["gemma"].grade_response(img, "test", c.category)
        res = test_env["scheduler"].review(c.fsrs, grade, elapsed_days=1.0)
        student.record_review(c, grade, res, score=fb.score)
        
    stats = student.get_session_stats()
    assert stats["total_reviews"] == 10
    assert len(student.get_blind_spots()) > 0

def test_e2e_medasr_pipeline(test_env):
    """Voice dictation to grading pipeline."""
    audio = (16000, np.random.randn(48000).astype(np.float32))
    transcript = test_env["asr"].transcribe_array(audio).text
    
    fb = test_env["gemma"].grade_response(Image.new("RGB", (256, 256)), transcript, "Cardiomegaly")
    assert fb.score >= 0

def test_e2e_longitudinal_mode(test_env, student):
    """Longitudinal comparison pair generation and grading."""
    cards_by_cat = {}
    for c in student.cards.values():
        cards_by_cat.setdefault(c.category, []).append(c)
        
    pairs = create_longitudinal_pairs(cards_by_cat)
    case = pairs[0]
    
    question = generate_longitudinal_question(case.category, case.change_type)
    fb = get_longitudinal_feedback(case.category, case.change_type, "worsened")
    
    assert "prior" in question.lower()
    assert 0.0 <= fb.score <= 1.0

def test_e2e_state_persistence(test_env, student):
    """Save/Load roundtrip verification."""
    card = list(student.cards.values())[0]
    res = test_env["scheduler"].review(card.fsrs, Rating.Good)
    student.record_review(card, Rating.Good, res, score=0.8)
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        student.save(path)
        loaded = StudentState.load(path)
        assert loaded.total_reviews == 1
        assert loaded.cards[card.card_id].fsrs.state != LearningState.New
    finally:
        os.unlink(path)

def test_e2e_lapse_recovery(test_env, student):
    """Simulate forgetting (Again rating) and recovery."""
    card = list(student.cards.values())[0]
    
    r1 = test_env["scheduler"].review(card.fsrs, Rating.Good)
    r2 = test_env["scheduler"].review(r1.state, Rating.Good, elapsed_days=2.0)
    r3 = test_env["scheduler"].review(r2.state, Rating.Again, elapsed_days=1.5) # Lapse
    
    assert r3.is_lapse
    assert r3.state.stability < r2.state.stability

def test_e2e_bounding_box_drawing():
    """Verify format_annotated_image structure."""
    img = Image.new("RGB", (512, 512), color=(50, 50, 50))
    boxes = [BoundingBox(y0=100, x0=150, y1=400, x1=450, label="cardiomegaly")]
    
    result = format_annotated_image(img, boxes)
    assert result[0] is img
    assert result[1][0][1] == "Cardiomegaly"
    assert len(result[1][0][0]) == 4

def test_e2e_cxr_foundation_retrieval():
    """Fallback handling for CXR Retriever."""
    retriever = CXRFoundationRetriever()
    assert retriever.search(Image.new("RGB", (224, 224))) == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])