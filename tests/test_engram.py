"""
ENGRAM v0.4.0 Test Suite (Refactored)
Tests all core modules: FSRS-6, MedASR, Longitudinal, CXR Foundation, Mock Engines,
plus v0.4.0 cognitive features. 
"""

import math, os, tempfile
import pytest
import numpy as np
from PIL import Image

# ─── FIXTURES (The Senior Engineer Secret Weapon) ──────────────

@pytest.fixture
def fsrs():
    from engram.fsrs6 import FSRS6Scheduler
    return FSRS6Scheduler()

@pytest.fixture
def blank_image():
    return Image.new("RGB", (512, 512), color=(0, 0, 0))

@pytest.fixture
def mock_gemma():
    from engram.mock_engine import MockMedGemmaEngine
    eng = MockMedGemmaEngine()
    eng.load()
    return eng

@pytest.fixture
def mock_hear():
    from engram.hear import MockHeAREngine
    eng = MockHeAREngine()
    eng.load()
    return eng

# ─── FSRS-6 Core Tests ──────────────────────────────────────────

from engram.fsrs6 import (
    FSRS6_WEIGHTS, Card, FSRSState, LearningState, Rating,
    forgetting_factor, initial_difficulty, initial_stability, retrievability
)

def test_fsrs6_parameters():
    """FSRS-6 math bounds and parameter counts."""
    assert len(FSRS6_WEIGHTS) == 21
    assert math.isfinite(forgetting_factor())
    assert forgetting_factor() > 0

def test_initial_conditions():
    """S0(G) and D0(G) bounds check."""
    assert initial_stability(Rating.Again) < initial_stability(Rating.Easy)
    assert abs(initial_stability(Rating.Again) - 0.2120) < 0.001
    assert initial_difficulty(Rating.Again) > initial_difficulty(Rating.Easy)
    assert 1.0 <= initial_difficulty(Rating.Again) <= 10.0

def test_retrievability_decay():
    """Power-law forgetting verification."""
    r_curve = [retrievability(10.0, d) for d in [0.0, 1.0, 10.0, 100.0, 1000.0]]
    assert r_curve[0] == 1.0
    assert r_curve[1] > r_curve[2] > r_curve[3] > r_curve[4]
    assert r_curve[-1] < 0.5  # Must forget eventually
    assert retrievability(100.0, 5.0) > retrievability(1.0, 5.0) # Stability effect

def test_scheduler_lifecycle(fsrs):
    """Test New -> Learning -> Lapse lifecycle including same-day logic."""
    state = FSRSState()
    assert state.state == LearningState.New
    
    # First Review
    r1 = fsrs.review(state, Rating.Good)
    assert r1.state.state != LearningState.New
    
    # Same day Again (No lapse penalty)
    r2 = fsrs.review(r1.state, Rating.Again)
    assert not r2.is_lapse
    
    # Future Again (Triggers lapse)
    r3 = fsrs.review(r2.state, Rating.Good, elapsed_days=3.0)
    r4 = fsrs.review(r3.state, Rating.Again, elapsed_days=1.5)
    assert r4.is_lapse
    assert r4.state.lapses >= 1

# ─── MedASR & CXR Foundation Tests ──────────────────────────────

from engram.medasr import MockMedASREngine
from engram.cxr_foundation import CXRFoundationRetriever

def test_mock_medasr():
    engine = MockMedASREngine()
    assert engine._loaded
    assert engine.transcribe_array(np.zeros(100, dtype=np.float32)).text == ""
    assert engine.transcribe_array((16000, np.random.randn(32000).astype(np.float32))).confidence > 0

def test_cxr_foundation():
    retriever = CXRFoundationRetriever()
    assert not retriever._loaded
    assert retriever.search(Image.new("RGB", (224, 224))) == []

# ─── Longitudinal & Engine Tests ────────────────────────────────

from engram.longitudinal import generate_longitudinal_question, get_longitudinal_feedback

def test_longitudinal_logic():
    q = generate_longitudinal_question("Cardiomegaly", "worsened")
    assert all(word in q.lower() for word in ["cardiomegaly", "prior", "current"])
    
    assert get_longitudinal_feedback("Cardiomegaly", "worsened", "Worsened cardiac silhouette").score >= 0.5
    assert get_longitudinal_feedback("Cardiomegaly", "worsened", "").score < 0.4  # Low but nonzero due to random jitter

def test_mock_gemma_engine(mock_gemma, blank_image):
    assert mock_gemma._loaded
    assert 0 <= mock_gemma.grade_response(blank_image, "enlarged heart", "Cardiomegaly").score <= 1
    
    boxes = mock_gemma.localize_findings(blank_image, "Cardiomegaly")
    assert len(boxes) > 0
    assert boxes[0].to_pixel(512, 512) == (102, 153, 358, 358) # Testing Box coordinate translation

# ─── Student State Integration ──────────────────────────────────

from engram.student import StudentState, ReviewLog

def test_student_state_persistence():
    student = StudentState(student_id="t1", name="Test")
    student.add_card(Card(card_id="c1", category="Cardiomegaly"))
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        student.save(path)
        loaded = StudentState.load(path)
        assert loaded.student_id == "t1"
        assert loaded.cards["c1"].category == "Cardiomegaly"
    finally:
        os.unlink(path)

# ═══════════════════════════════════════════════════════════════
# v0.4.0 Feature Tests (Cognitive Modifiers)
# ═══════════════════════════════════════════════════════════════

from engram.fsrs6 import interval_modifier_for_overconfidence, search_completeness_modifier

def test_cognitive_modifiers():
    # F1: Confidence
    assert interval_modifier_for_overconfidence(0.05) == 1.0
    assert 0.5 <= interval_modifier_for_overconfidence(0.4) < 1.0
    assert interval_modifier_for_overconfidence(1.0) == 0.5
    
    # F3: Satisfaction of Search
    assert search_completeness_modifier(0.9) == 1.0
    assert search_completeness_modifier(0.2) == 0.5
    assert 0.5 < search_completeness_modifier(0.5) < 1.0

def test_v04_review_log_fields():
    log = ReviewLog(card_id="c1", category="Edema", grade=3, score=0.8, box_iou=0.5, 
                    retrievability=0.9, interval=3, confidence=4, search_completeness=0.67, 
                    gestalt_score=0.7, contrastive_pair="Edema vs Cardiomegaly")
    assert log.confidence == 4
    assert log.search_completeness == 0.67
    assert log.gestalt_score == 0.7
    assert log.contrastive_pair == "Edema vs Cardiomegaly"

def test_socratic_and_contrastive_modes(mock_gemma):
    # F2: Socratic
    assert len(mock_gemma.generate_socratic_question(None, "", "Consolidation")) > 10
    
    # F4 & F5: Gestalt & Contrastive
    assert mock_gemma.grade_gestalt("enlarged heart", "Cardiomegaly") > 0
    assert mock_gemma.grade_gestalt("", "Cardiomegaly") == 0
    assert len(mock_gemma.generate_contrastive_question("Consolidation", "Atelectasis")) > 10

def test_hear_engine_integration(mock_hear):
    # F6: HeAR 
    sr, audio = mock_hear.generate_lung_sound("Consolidation")
    assert sr == 16000 and len(audio) > 0
    
    classification = mock_hear.classify_lung_sound(audio)
    assert abs(sum(classification.values()) - 1.0) < 0.01

    case = mock_hear.get_lung_sound_case("Consolidation")
    assert "bronchial" in case.sound_type