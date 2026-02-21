"""
ENGRAM: FSRS-6 Adaptive Medical Visual Diagnosis Training System
Main Gradio Application — v0.4.0 (Refactored)
"""
from __future__ import annotations

import os, random, re, threading, time, uuid
from pathlib import Path

import gradio as gr
from PIL import Image

from engram.fsrs6 import FSRS6Scheduler, Rating, LearningState, retrievability
from engram.fsrs6 import interval_modifier_for_overconfidence, search_completeness_modifier
from engram.student import StudentState
from engram.dataset import CATEGORY_DESCRIPTIONS, load_demo_dataset
from engram.blindspot import render_blindspot_html, render_session_stats_html, render_calibration_chart_html
from engram.hear import LUNG_SOUNDS

# ─── Configuration ─────────────────────────────────────────────────
USE_REAL_MEDGEMMA = os.environ.get("ENGRAM_USE_MEDGEMMA", "true").lower() != "false"
DATA_DIR = Path(__file__).parent / "data"
STUDENT_DIR = Path(__file__).parent / "students"
STUDENT_DIR.mkdir(exist_ok=True)

RATING_MAP = {
    "Missed It (Again)": Rating.Again, "Struggled (Hard)": Rating.Hard,
    "Got It (Good)": Rating.Good, "Easy (Easy)": Rating.Easy,
}

# ─── Shared State & Model Manager ─────────────────────────────────
class ModelManager:
    """Consolidated lazy-loading for all 5 HAI-DEF models."""
    def __init__(self):
        self.lock = threading.Lock()
        self.scheduler = FSRS6Scheduler(desired_retention=0.9, max_interval=365)
        self.models = {"gemma": None, "asr": None, "siglip": None, "cxr": None, "hear": None}

    def get_model(self, name: str):
        if self.models[name] is not None:
            return self.models[name]
        
        with self.lock:
            if self.models[name] is not None:
                return self.models[name]
            
            print(f"[ENGRAM] Initializing {name.upper()}...")
            if USE_REAL_MEDGEMMA:
                if name == "gemma":
                    from engram.medgemma import MedGemmaEngine
                    self.models[name] = MedGemmaEngine(lora_path=os.environ.get("ENGRAM_LORA_PATH"))
                elif name == "asr":
                    from engram.medasr import MedASREngine
                    self.models[name] = MedASREngine()
                elif name == "siglip":
                    from engram.retrieval import MedSigLIPRetriever
                    self.models[name] = MedSigLIPRetriever()
                elif name == "cxr":
                    from engram.cxr_foundation import CXRFoundationRetriever
                    self.models[name] = CXRFoundationRetriever()
                elif name == "hear":
                    from engram.hear import HeAREngine
                    self.models[name] = HeAREngine()
            else:
                # Mock Fallbacks
                if name == "gemma":
                    from engram.mock_engine import MockMedGemmaEngine
                    self.models[name] = MockMedGemmaEngine()
                elif name == "asr":
                    from engram.medasr import MockMedASREngine
                    self.models[name] = MockMedASREngine()
                elif name == "hear":
                    from engram.hear import MockHeAREngine
                    self.models[name] = MockHeAREngine()
            
            if hasattr(self.models[name], 'load'):
                self.models[name].load()
            return self.models[name]

    def build_indices(self, student: StudentState):
        if not student or not student.cards: return
        paths = [c.image_path for c in student.cards.values() if c.image_path]
        cats = [c.category for c in student.cards.values() if c.image_path]
        ids = [c.card_id for c in student.cards.values() if c.image_path]
        
        if paths:
            if USE_REAL_MEDGEMMA:
                self.get_model("siglip").build_index(paths, cats, ids)
                self.get_model("cxr").build_index(paths, cats, ids)

MM = ModelManager()

def _default_state() -> dict:
    return {"student": None, "card": None, "longitudinal": None, "contrastive_pair": None, "auscultation_category": None}

def _sanitize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))[:64] or "student"

def get_or_create_student(name: str) -> StudentState:
    file = STUDENT_DIR / f"{_sanitize_name(name)}.json"
    if file.exists():
        s = StudentState.load(file)
        s.total_sessions += 1
        s.last_session = time.time()
        return s
    return StudentState(student_id=str(uuid.uuid4())[:8], name=name, total_sessions=1)

# ─── Core Logic Functions ─────────────────────────────────────────

def transcribe_audio(audio) -> str:
    return MM.get_model("asr").transcribe_array(audio).text if audio is not None else ""

def format_annotated_image(image: Image.Image, boxes: list) -> tuple:
    """Convert custom boxes to Gradio AnnotatedImage format."""
    if not boxes or not image: return (image, [])
    h, w = image.size[1], image.size[0]
    annotations = []
    for b in boxes:
        y0, x0, y1, x1 = b.to_pixel(h, w)
        annotations.append(((x0, y0, x1, y1), b.label.replace("_", " ").title()))
    return (image, annotations)

def start_session(name: str, state: dict):
    state = dict(state)
    student = get_or_create_student(name or "Student")
    state["student"] = student

    if not student.cards:
        demo_cards = load_demo_dataset(str(DATA_DIR / "demo"))
        if not demo_cards: return ((None, []), "**No demo images found.**", "", "", state)
        for c in demo_cards: student.add_card(c)
        student.save(STUDENT_DIR / f"{_sanitize_name(student.name)}.json")
    
    MM.build_indices(student)
    return get_next_case(state)

def get_next_case(state: dict):
    state = dict(state)
    student = state.get("student")
    if not student or not student.cards: return ((None, []), "No active session.", "", "", state)

    due = MM.scheduler.get_due_cards(list(student.cards.values()))
    if not due: return ((None, []), "All caught up!", "", render_session_stats_html(student.get_session_stats()), state)

    card = next((c for c in due if c.image_path), None)
    if not card: return ((None, []), "Image missing.", "", "", state)

    state["card"] = card
    img = Image.open(card.image_path).convert("RGB")
    q = MM.get_model("gemma").generate_question(img, card.category)
    return ((img, []), f"### Next Case: {card.category}\n{q}", "", render_session_stats_html(student.get_session_stats()), state)

def submit_answer(ans: str, rating: str, conf: int, state: dict):
    state = dict(state)
    student, card = state.get("student"), state.get("card")
    if not student or not card: return ((None, []), "No case.", state)

    img = Image.open(card.image_path).convert("RGB") if card.image_path else Image.new("RGB", (512, 512))
    
    # AI Processing
    engine = MM.get_model("gemma")
    truth = CATEGORY_DESCRIPTIONS.get(card.category, card.category)
    fb = engine.grade_response(img, ans, card.category, truth)
    boxes = engine.localize_findings(img, card.category)
    
    # FSRS Update
    grade = RATING_MAP.get(rating, Rating.Good)
    res = MM.scheduler.review(card.fsrs, grade)

    # F3: Satisfaction of Search Modifier
    search_comp = len(fb.correct_findings) / max(1, len(fb.correct_findings) + len(fb.missed_findings)) if fb else 1.0
    if search_comp < 0.8:
        res.interval = max(1, int(res.interval * search_completeness_modifier(search_comp)))

    # F1: Confidence Calibration Modifier
    cal_data = student.calibration_per_category().get(card.category)
    if cal_data and cal_data["overconfident"]:
        res.interval = max(1, int(res.interval * interval_modifier_for_overconfidence(cal_data["calibration_gap"])))

    # Save State
    student.record_review(card, grade, res, score=fb.score if fb else 0.5, confidence=conf)
    student.save(STUDENT_DIR / f"{_sanitize_name(student.name)}.json")
    
    annotated = format_annotated_image(img, boxes)
    md = f"### Assessment: {fb.score*100:.0f}%\n{fb.explanation if fb else ''}\n\n*Next review in {res.interval} days.*"
    
    return (annotated, md, state)

# ─── F2 & F4: Socratic & Gestalt Handlers ─────────────────────────

def submit_gestalt(ans: str, state: dict):
    card = state.get("card")
    if not card: return "No active case."
    score = MM.get_model("gemma").grade_gestalt(ans, card.category)
    return f"**System 1 (Gestalt) Score: {score:.0%}**\n\nNow do a full System 2 review."

def ask_socratic(ans: str, state: dict):
    card = state.get("card")
    if not card: return "No active case."
    return MM.get_model("gemma").generate_socratic_question(None, ans, card.category)

# ─── F5: Contrastive Case Pairs ───────────────────────────────────

def start_contrastive(state: dict):
    state = dict(state)
    student = state.get("student")
    if not student or not student.cards: return ((None, []), (None, []), "Start session first.", state)

    pairs = list(MM.get_model("gemma").CONTRASTIVE_PAIRS.keys())
    cat_a, cat_b = random.choice(pairs)
    state["contrastive_pair"] = (cat_a, cat_b)

    cards_a = [c for c in student.cards.values() if c.category == cat_a and c.image_path]
    cards_b = [c for c in student.cards.values() if c.category == cat_b and c.image_path]
    if not cards_a or not cards_b: return ((None, []), (None, []), f"Need images for {cat_a} & {cat_b}.", state)

    img_a = Image.open(random.choice(cards_a).image_path).convert("RGB")
    img_b = Image.open(random.choice(cards_b).image_path).convert("RGB")
    q = MM.get_model("gemma").generate_contrastive_question(cat_a, cat_b)
    return ((img_a, []), (img_b, []), f"### {cat_a} vs {cat_b}\n{q}", state)

def submit_contrastive(ans: str, state: dict):
    state = dict(state)
    pair = state.get("contrastive_pair")
    if not pair: return ("No active pair.", state)
    fb = MM.get_model("gemma").grade_contrastive(ans, pair[0], pair[1])
    return (f"### Discrimination Score: {fb.score:.0%}\n\n{fb.explanation}", state)

# ─── F6: HeAR Auscultation ────────────────────────────────────────

def start_auscultation(state: dict):
    state = dict(state)
    cat = random.choice(list(LUNG_SOUNDS.keys()))
    state["ausc_cat"] = cat
    sr, audio = MM.get_model("hear").generate_lung_sound(cat)
    return ((sr, audio), "Listen carefully. What do you hear? What CXR findings do you predict?", state)

def submit_auscultation(ans: str, state: dict):
    state = dict(state)
    cat = state.get("ausc_cat")
    if not cat: return ((None, []), "No active case.", state)

    student = state.get("student")
    img_path = next((c.image_path for c in student.cards.values() if c.category == cat and c.image_path), None) if student else None
    img = Image.open(img_path).convert("RGB") if img_path else Image.new("RGB", (512, 512))

    data = LUNG_SOUNDS[cat]
    fb = f"### Actual Category: {cat}\n**Sound:** {data['sound']}\n**Correlation:** {data['correlation']}"
    return ((img, []), fb, state)

# ─── Gradio UI Layout ──────────────────────────────────────────────

def build_app() -> gr.Blocks:
    with gr.Blocks(title="ENGRAM — Adaptive Medical Training") as app:
        session = gr.State(value=_default_state())
        
        gr.Markdown("# ENGRAM: FSRS-6 Adaptive Medical Visual Diagnosis Training")
        
        with gr.Tab("Training"):
            with gr.Row():
                name_input = gr.Textbox(label="Your Name", scale=2)
                start_btn = gr.Button("Start Session", variant="primary")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Replaced gr.Image with gr.AnnotatedImage for native bounding boxes
                    case_image = gr.AnnotatedImage(label="Case Image", height=420)
                    question = gr.Markdown("*Click Start Session*")
                    
                    audio_in = gr.Audio(label="Dictate (MedASR)", sources=["microphone"], type="numpy")
                    transcribe_btn = gr.Button("Transcribe")
                    
                    answer = gr.Textbox(label="Your Interpretation", lines=3)
                    with gr.Row():
                        conf = gr.Slider(1, 5, 3, step=1, label="Confidence")
                        rating = gr.Radio(list(RATING_MAP.keys()), value="Got It (Good)", label="Self-Assessment")
                    
                    with gr.Row():
                        submit_btn = gr.Button("Submit", variant="primary")
                        next_btn = gr.Button("Next Case")
                        
                    feedback = gr.Markdown()
                    
                with gr.Column(scale=1):
                    stats = gr.HTML()

                    with gr.Accordion("Socratic Mode (F2)", open=False):
                        socratic_fb = gr.Markdown()
                        socratic_btn = gr.Button("Ask Socratic Question")

                    with gr.Accordion("Dual-Process / Gestalt (F4)", open=False):
                        gestalt_fb = gr.Markdown()
                        gestalt_btn = gr.Button("Grade Gestalt (System 1)")

        with gr.Tab("Contrastive Pairs (F5)"):
            with gr.Row():
                cp_img_a = gr.AnnotatedImage(label="Case A")
                cp_img_b = gr.AnnotatedImage(label="Case B")
            cp_q = gr.Markdown("*Click to generate pair*")
            cp_ans = gr.Textbox(label="Key Differences")
            with gr.Row():
                cp_start = gr.Button("Generate Pair")
                cp_sub = gr.Button("Submit", variant="primary")
            cp_fb = gr.Markdown()

        with gr.Tab("Auscultation (F6)"):
            ausc_audio = gr.Audio(label="HeAR Lung Sound", interactive=False)
            ausc_q = gr.Markdown("*Click to listen*")
            ausc_ans = gr.Textbox(label="Predicted CXR Findings")
            with gr.Row():
                ausc_start = gr.Button("Listen to Patient")
                ausc_sub = gr.Button("Reveal CXR & Grade", variant="primary")
            ausc_img = gr.AnnotatedImage(label="Revealed CXR")
            ausc_fb = gr.Markdown()

        # Wiring — Training tab
        start_btn.click(start_session, [name_input, session], [case_image, question, answer, stats, session])
        transcribe_btn.click(transcribe_audio, [audio_in], [answer])
        submit_btn.click(submit_answer, [answer, rating, conf, session], [case_image, feedback, session])
        next_btn.click(get_next_case, [session], [case_image, question, answer, stats, session])
        socratic_btn.click(ask_socratic, [answer, session], [socratic_fb])
        gestalt_btn.click(submit_gestalt, [answer, session], [gestalt_fb])

        # Wiring — Contrastive Pairs tab
        cp_start.click(start_contrastive, [session], [cp_img_a, cp_img_b, cp_q, session])
        cp_sub.click(submit_contrastive, [cp_ans, session], [cp_fb, session])

        # Wiring — Auscultation tab
        ausc_start.click(start_auscultation, [session], [ausc_audio, ausc_q, session])
        ausc_sub.click(submit_auscultation, [ausc_ans, session], [ausc_img, ausc_fb, session])

    return app

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860)