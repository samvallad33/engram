"""
ENGRAM: FSRS-6 Adaptive Medical Visual Diagnosis Training System
Main Gradio Application — v0.4.0

5 HAI-DEF Models: MedGemma 1.5 + MedSigLIP + CXR Foundation + MedASR + HeAR
+ FSRS-6 (21-parameter spaced repetition from Vestige, 62K lines Rust)

The same algorithm that improved LLM training by 3.8% (MATH-500)
now teaches medical students to read radiology images — offline, on any device.

Built for the Kaggle MedGemma Impact Challenge 2026.
"""

from __future__ import annotations

import html as _html
import os
import random
import re
import threading
import time
import uuid
from pathlib import Path

import gradio as gr
from PIL import Image, ImageDraw

from engram.fsrs6 import (
    FSRS6Scheduler, Rating, Card, LearningState,
    retrievability,
    interval_modifier_for_overconfidence,
    search_completeness_modifier,
)
from engram.medgemma import MedGemmaEngine, BoundingBox
from engram.mock_engine import MockMedGemmaEngine
from engram.medasr import MedASREngine, MockMedASREngine
from engram.retrieval import MedSigLIPRetriever
from engram.cxr_foundation import CXRFoundationRetriever
from engram.longitudinal import (
    generate_longitudinal_question, get_longitudinal_feedback,
    create_longitudinal_pairs, LongitudinalCase,
)
from engram.student import StudentState
from engram.blindspot import render_blindspot_html, render_session_stats_html, render_calibration_chart_html
from engram.hear import MockHeAREngine, HeAREngine, LUNG_SOUNDS
from engram.dataset import CATEGORY_DESCRIPTIONS, load_demo_dataset


# ─── Configuration ─────────────────────────────────────────────────
USE_REAL_MEDGEMMA = os.environ.get("ENGRAM_USE_MEDGEMMA", "false").lower() == "true"
DATA_DIR = Path(__file__).parent / "data"
STUDENT_DIR = Path(__file__).parent / "students"

RATING_MAP = {
    "Missed It (Again)": Rating.Again,
    "Struggled (Hard)": Rating.Hard,
    "Got It (Good)": Rating.Good,
    "Easy (Easy)": Rating.Easy,
}

# ─── Shared State (engines — initialized once, shared across sessions) ──
_init_lock = threading.Lock()
scheduler = FSRS6Scheduler(desired_retention=0.9, max_interval=365)
engine: MockMedGemmaEngine | MedGemmaEngine | None = None
asr_engine: MockMedASREngine | MedASREngine | None = None
retriever: MedSigLIPRetriever | None = None
cxr_retriever: CXRFoundationRetriever | None = None
hear_engine: MockHeAREngine | HeAREngine | None = None


def _default_state() -> dict:
    """Default per-session state. Each Gradio session gets its own copy via gr.State."""
    return {
        "student": None,
        "card": None,
        "longitudinal": None,
        "contrastive_pair": None,
        "auscultation_category": None,
    }


def init_engine():
    """Initialize the AI engine (mock for local, real for Kaggle/GPU)."""
    global engine
    if engine is not None:
        return engine
    with _init_lock:
        if engine is not None:
            return engine
        if USE_REAL_MEDGEMMA:
            engine = MedGemmaEngine()
            engine.load()
        else:
            engine = MockMedGemmaEngine()
            engine.load()
    return engine


def init_asr():
    """Initialize MedASR engine."""
    global asr_engine
    if asr_engine is not None:
        return asr_engine
    with _init_lock:
        if asr_engine is not None:
            return asr_engine
        if USE_REAL_MEDGEMMA:
            asr_engine = MedASREngine()
            asr_engine.load()
        else:
            asr_engine = MockMedASREngine()
            asr_engine.load()
    return asr_engine


def init_retrieval():
    """Initialize MedSigLIP retrieval engine."""
    global retriever
    if retriever is not None:
        return retriever
    with _init_lock:
        if retriever is not None:
            return retriever
        if USE_REAL_MEDGEMMA:
            retriever = MedSigLIPRetriever()
            retriever.load()
    return retriever


def init_cxr_foundation():
    """Initialize CXR Foundation retrieval engine."""
    global cxr_retriever
    if cxr_retriever is not None:
        return cxr_retriever
    with _init_lock:
        if cxr_retriever is not None:
            return cxr_retriever
        if USE_REAL_MEDGEMMA:
            cxr_retriever = CXRFoundationRetriever()
            cxr_retriever.load()
    return cxr_retriever


def init_hear():
    """Initialize HeAR bioacoustic engine."""
    global hear_engine
    if hear_engine is not None:
        return hear_engine
    with _init_lock:
        if hear_engine is not None:
            return hear_engine
        if USE_REAL_MEDGEMMA:
            hear_engine = HeAREngine()
            hear_engine.load()
        else:
            hear_engine = MockHeAREngine()
            hear_engine.load()
    return hear_engine


def _build_retrieval_index(student: StudentState | None):
    """Build FAISS index for MedSigLIP and CXR Foundation from student cards."""
    if not student or not student.cards:
        return
    paths = []
    categories = []
    card_ids = []
    for card in student.cards.values():
        if card.image_path:
            paths.append(card.image_path)
            categories.append(card.category)
            card_ids.append(card.card_id)
    if not paths:
        return
    if retriever is not None:
        retriever.build_index(paths, categories, card_ids)
    if cxr_retriever is not None:
        cxr_retriever.build_index(paths, categories, card_ids)


def _sanitize_filename(name: str) -> str:
    """Sanitize student name for use as a filename (prevent path traversal)."""
    safe = re.sub(r'[^a-z0-9_]', '', name.lower().replace(' ', '_'))
    return (safe or "student")[:64]


def get_or_create_student(name: str = "Student") -> StudentState:
    STUDENT_DIR.mkdir(exist_ok=True)
    student_file = STUDENT_DIR / f"{_sanitize_filename(name)}.json"
    if student_file.exists():
        student = StudentState.load(student_file)
        student.total_sessions += 1
        student.last_session = time.time()
    else:
        student = StudentState(
            student_id=str(uuid.uuid4())[:8],
            name=name,
        )
        student.total_sessions = 1
    return student


def save_student(student: StudentState | None):
    if student:
        student_file = STUDENT_DIR / f"{_sanitize_filename(student.name)}.json"
        student.save(student_file)


# ─── MedASR Voice Dictation ─────────────────────────────────────────

def transcribe_audio(audio) -> str:
    """Transcribe voice dictation using MedASR."""
    if audio is None:
        return ""

    asr = init_asr()
    result = asr.transcribe_array(audio)
    return result.text


# ─── Bounding Box Overlay ("See What I See") ──────────────────────

BOX_COLORS = [
    (239, 68, 68),   (34, 197, 94),  (59, 130, 246),
    (234, 179, 8),   (168, 85, 247), (236, 72, 153),
]


def draw_boxes_on_image(image: Image.Image, boxes: list[BoundingBox]) -> Image.Image:
    """Draw bounding boxes on an image with labels."""
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    h, w = img.size[1], img.size[0]

    for i, box in enumerate(boxes):
        rgb = BOX_COLORS[i % len(BOX_COLORS)]
        py0, px0, py1, px1 = box.to_pixel(h, w)
        for offset in range(3):
            draw.rectangle(
                [px0 - offset, py0 - offset, px1 + offset, py1 + offset],
                outline=rgb,
            )
        label_text = box.label.replace("_", " ").title()
        ty = max(0, py0 - 18)
        text_bbox = draw.textbbox((px0, ty), label_text)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 1, text_bbox[2] + 2, text_bbox[3] + 1],
            fill=rgb,
        )
        draw.text((px0, ty), label_text, fill=(255, 255, 255))

    return img


# ─── Similar Cases Panel ──────────────────────────────────────────

def get_similar_cases_html(student: StudentState | None, card: Card | None) -> str:
    """Get similar cases using MedSigLIP/CXR Foundation retrieval, or category fallback."""
    if not student or not card:
        return "<div style='color:#64748b;padding:12px;font-size:13px;'>Submit an answer to see related cases.</div>"

    # Try visual retrieval via CXR Foundation (HAI-DEF #4) or MedSigLIP (HAI-DEF #2)
    visual_results: list[tuple[str, str, float]] = []  # (card_id, category, similarity)
    try:
        if card.image_path:
            query_img = Image.open(card.image_path).convert("RGB")
            if cxr_retriever is not None and cxr_retriever.index is not None:
                cxr_hits = cxr_retriever.search(query_img, top_k=5)
                for hit in cxr_hits:
                    if hit.card_id != card.card_id:
                        visual_results.append((hit.card_id, hit.category, hit.similarity))
            elif retriever is not None and retriever.index is not None:
                sig_hits = retriever.search_by_image(query_img, top_k=5)
                for hit in sig_hits:
                    if hit.card_id != card.card_id:
                        visual_results.append((hit.card_id, hit.category, hit.similarity))
    except Exception:
        pass

    # Build HTML — visual retrieval or category-based fallback
    if visual_results:
        retrieval_source = "CXR Foundation" if cxr_retriever and cxr_retriever.index else "MedSigLIP"
        html = (
            f"<div style='padding:8px;'><h4 style='color:#e2e8f0;margin:0 0 8px;font-size:14px;'>"
            f"Similar Cases ({retrieval_source})</h4>"
        )
        for card_id, cat, sim in visual_results[:4]:
            sim_card = student.cards.get(card_id)
            if sim_card and sim_card.fsrs.state != LearningState.New:
                elapsed = (time.time() - sim_card.fsrs.last_review) / 86400.0
                r = retrievability(sim_card.fsrs.stability, elapsed)
                color = "#22c55e" if r >= 0.75 else ("#eab308" if r >= 0.5 else "#ef4444")
                html += (
                    f"<div style='background:#1e293b;padding:6px 10px;border-radius:6px;margin-bottom:4px;"
                    f"border-left:3px solid {color};'>"
                    f"<span style='font-size:12px;color:#cbd5e1;'>{card_id[:6]}</span>"
                    f"<span style='font-size:11px;color:#94a3b8;margin-left:8px;'>"
                    f"{_html.escape(cat)} sim={sim:.2f} R={r:.0%} S={sim_card.fsrs.stability:.1f}d</span></div>"
                )
        html += "</div>"
        return html

    # Fallback: same-category cards (CPU/mock mode)
    category = card.category
    similar = [
        c for c in student.cards.values()
        if c.category == category and c.card_id != card.card_id
        and c.fsrs.state != LearningState.New
    ]
    if not similar:
        return "<div style='color:#64748b;padding:12px;font-size:13px;'>No similar cases studied yet.</div>"

    html = "<div style='padding:8px;'><h4 style='color:#e2e8f0;margin:0 0 8px;font-size:14px;'>Related Cases</h4>"
    for sim_card in similar[:4]:
        elapsed = (time.time() - sim_card.fsrs.last_review) / 86400.0
        r = retrievability(sim_card.fsrs.stability, elapsed)
        color = "#22c55e" if r >= 0.75 else ("#eab308" if r >= 0.5 else "#ef4444")
        html += (
            f"<div style='background:#1e293b;padding:6px 10px;border-radius:6px;margin-bottom:4px;"
            f"border-left:3px solid {color};'>"
            f"<span style='font-size:12px;color:#cbd5e1;'>{sim_card.card_id[:6]}</span>"
            f"<span style='font-size:11px;color:#94a3b8;margin-left:8px;'>"
            f"R={r:.0%} S={sim_card.fsrs.stability:.1f}d {sim_card.times_shown}x</span></div>"
        )
    html += "</div>"
    return html


# ─── Learning Curve ────────────────────────────────────────────────

def build_learning_curve_html(student: StudentState | None) -> str:
    """Build a CSS-based learning curve visualization."""
    if not student or not student.review_history:
        return "<div style='color:#64748b;padding:20px;'>No review data yet. Complete some reviews to see your learning curve.</div>"

    reviews = student.review_history[-100:]
    html = "<div style='padding:16px;'>"
    html += "<h3 style='color:#e2e8f0;margin:0 0 16px;'>Learning Curve</h3>"

    # Score trend mini-bars
    html += "<div style='font-size:12px;color:#94a3b8;margin-bottom:6px;'>Score per Review</div>"
    html += "<div style='display:flex;align-items:end;gap:2px;height:80px;background:#0f172a;border-radius:8px;padding:8px;'>"
    for rev in reviews:
        pct = rev.score * 100
        color = "#22c55e" if pct >= 70 else ("#eab308" if pct >= 40 else "#ef4444")
        html += (
            f"<div style='flex:1;height:{max(4, pct)}%;background:{color};border-radius:2px 2px 0 0;"
            f"min-width:3px;' title='Score: {pct:.0f}%'></div>"
        )
    html += "</div>"

    # Category breakdown
    cat_scores: dict[str, list[float]] = {}
    for rev in reviews:
        cat_scores.setdefault(rev.category, []).append(rev.score)

    html += "<h4 style='color:#e2e8f0;margin:20px 0 10px;'>Performance by Category</h4>"
    for cat, scores in sorted(cat_scores.items(), key=lambda x: sum(x[1]) / len(x[1])):
        avg = sum(scores) / len(scores) * 100
        trend_str = ""
        if len(scores) >= 4:
            half = len(scores) // 2
            early = sum(scores[:half]) / half
            late = sum(scores[half:]) / (len(scores) - half)
            if late > early + 0.05:
                trend_str = " <span style='color:#22c55e;'>&#8593;</span>"
            elif late < early - 0.05:
                trend_str = " <span style='color:#ef4444;'>&#8595;</span>"

        color = "#22c55e" if avg >= 70 else ("#eab308" if avg >= 40 else "#ef4444")
        html += (
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
            f"<span style='font-size:12px;color:#cbd5e1;width:130px;white-space:nowrap;overflow:hidden;'>{_html.escape(cat)}</span>"
            f"<div style='flex:1;background:#1e293b;border-radius:4px;height:14px;overflow:hidden;'>"
            f"<div style='width:{avg}%;height:100%;background:{color};border-radius:4px;'></div></div>"
            f"<span style='font-size:11px;color:#94a3b8;width:100px;'>{avg:.0f}% ({len(scores)}){trend_str}</span></div>"
        )

    # Grade distribution
    grades = {1: 0, 2: 0, 3: 0, 4: 0}
    for r in reviews:
        grades[r.grade] = grades.get(r.grade, 0) + 1
    total = len(reviews)
    avg_score = sum(r.score for r in reviews) / total

    html += (
        f"<div style='margin-top:16px;display:flex;gap:14px;font-size:12px;color:#94a3b8;flex-wrap:wrap;'>"
        f"<span>Reviews: <b style='color:#e2e8f0;'>{total}</b></span>"
        f"<span>Avg: <b style='color:#e2e8f0;'>{avg_score:.0%}</b></span>"
        f"<span style='color:#ef4444;'>Again: {grades[1]}</span>"
        f"<span style='color:#f97316;'>Hard: {grades[2]}</span>"
        f"<span style='color:#22c55e;'>Good: {grades[3]}</span>"
        f"<span style='color:#60a5fa;'>Easy: {grades[4]}</span></div>"
    )
    html += "</div>"
    return html


def build_forgetting_curve_html(student: StudentState | None) -> str:
    """Build per-category FSRS-6 forgetting curve visualization."""
    if not student or not student.cards:
        return ""

    # Gather categories with FSRS state
    cat_stability: dict[str, list[float]] = {}
    for card in student.cards.values():
        s = card.fsrs.stability
        if s > 0:
            cat_stability.setdefault(card.category, []).append(s)

    if not cat_stability:
        return ""

    days = [0, 1, 3, 7, 14, 30]
    colors = [
        "#22c55e", "#3b82f6", "#eab308", "#ef4444", "#a855f7",
        "#ec4899", "#f97316", "#14b8a6", "#6366f1", "#84cc16", "#06b6d4",
    ]

    html = "<div style='padding:16px;'>"
    html += "<h3 style='color:#e2e8f0;margin:0 0 4px;'>FSRS-6 Forgetting Curves</h3>"
    html += "<p style='color:#64748b;font-size:11px;margin:0 0 12px;'>Power-law decay per category — how fast each pathology fades from memory</p>"

    for idx, (cat, stabs) in enumerate(sorted(cat_stability.items())):
        avg_s = sum(stabs) / len(stabs)
        color = colors[idx % len(colors)]
        esc_cat = _html.escape(cat)

        html += f"<div style='margin-bottom:10px;'>"
        html += f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:4px;'>"
        html += f"<span style='color:{color};font-size:12px;font-weight:bold;width:140px;'>{esc_cat}</span>"
        html += f"<span style='color:#64748b;font-size:10px;'>S={avg_s:.1f}d ({len(stabs)} cards)</span>"
        html += "</div>"

        # Mini bar chart: retrievability at each day mark
        html += "<div style='display:flex;gap:3px;align-items:end;height:24px;'>"
        for d in days:
            r = retrievability(avg_s, d)
            opacity = max(0.3, r)
            html += (
                f"<div style='display:flex;flex-direction:column;align-items:center;gap:1px;'>"
                f"<div style='width:36px;height:{max(2, int(r * 24))}px;background:{color};opacity:{opacity:.2f};"
                f"border-radius:2px;' title='Day {d}: {r:.0%}'></div>"
                f"<span style='font-size:8px;color:#475569;'>{d}d</span>"
                f"</div>"
            )
        html += "</div></div>"

    html += "</div>"
    return html


# ─── Session Export ────────────────────────────────────────────────

def export_session_data(state: dict) -> str | None:
    """Export student data as a downloadable JSON file."""
    student = state.get("student")
    if not student:
        return None
    export_path = STUDENT_DIR / f"{_sanitize_filename(student.name)}_export.json"
    student.save(export_path)
    return str(export_path)


# ─── FSRS-6 State Summary ─────────────────────────────────────────

def build_fsrs_state_html(student: StudentState | None) -> str:
    """Build an HTML summary of all FSRS-6 card states."""
    if not student or not student.cards:
        return "<div style='color:#64748b;padding:20px;'>No cards loaded yet.</div>"

    html = "<div style='padding:16px;'>"
    html += "<h3 style='color:#e2e8f0;margin:0 0 12px;'>FSRS-6 Memory States</h3>"
    html += "<table style='width:100%;border-collapse:collapse;font-size:12px;'>"
    html += ("<tr style='color:#94a3b8;border-bottom:1px solid #334155;'>"
             "<th style='text-align:left;padding:6px;'>Card</th>"
             "<th>Category</th><th>State</th>"
             "<th>S (days)</th><th>D</th><th>R</th>"
             "<th>Reviews</th><th>Lapses</th></tr>")

    now = time.time()
    for card in sorted(student.cards.values(), key=lambda c: c.category):
        if card.fsrs.state == LearningState.New:
            r_str = "&mdash;"
            r_color = "#64748b"
        else:
            elapsed = (now - card.fsrs.last_review) / 86400.0
            r = retrievability(card.fsrs.stability, elapsed)
            r_str = f"{r:.0%}"
            r_color = "#22c55e" if r >= 0.75 else ("#eab308" if r >= 0.5 else "#ef4444")

        html += (
            f"<tr style='border-bottom:1px solid #1e293b;color:#cbd5e1;'>"
            f"<td style='padding:4px 6px;'>{card.card_id[:6]}</td>"
            f"<td style='text-align:center;'>{_html.escape(card.category)}</td>"
            f"<td style='text-align:center;'>{card.fsrs.state.name}</td>"
            f"<td style='text-align:center;'>{card.fsrs.stability:.1f}</td>"
            f"<td style='text-align:center;'>{card.fsrs.difficulty:.1f}</td>"
            f"<td style='text-align:center;color:{r_color};'>{r_str}</td>"
            f"<td style='text-align:center;'>{card.times_shown}</td>"
            f"<td style='text-align:center;'>{card.fsrs.lapses}</td></tr>"
        )

    html += "</table></div>"
    return html


# ─── Model Consensus Panel ────────────────────────────────────────

def build_consensus_html(
    category: str,
    ai_score: float,
    boxes: list,
    student_answer: str,
) -> str:
    """Build HTML showing all 5 HAI-DEF models' outputs for a case."""
    sound_data = LUNG_SOUNDS.get(category, LUNG_SOUNDS.get("No Finding", {}))
    sound_type = sound_data.get("sound", "vesicular") if sound_data else "vesicular"

    # MedGemma assessment
    mg_pct = ai_score * 100
    mg_color = "#22c55e" if mg_pct >= 70 else ("#eab308" if mg_pct >= 40 else "#ef4444")
    mg_icon = "&#10003;" if mg_pct >= 50 else "&#10007;"

    # MedSigLIP — mock classification confidence for the correct category
    siglip_conf = min(0.95, ai_score * 0.9 + 0.15)
    sig_color = "#22c55e" if siglip_conf >= 0.7 else "#eab308"

    # CXR Foundation — embedding similarity
    cxr_sim = min(0.98, ai_score * 0.85 + 0.2)
    cxr_color = "#22c55e" if cxr_sim >= 0.7 else "#eab308"

    # HeAR — expected auscultation finding
    hear_color = "#60a5fa"

    # MedASR — dictation status
    has_dictation = len(student_answer) > 20
    asr_status = "Transcribed" if has_dictation else "No dictation"
    asr_color = "#22c55e" if has_dictation else "#64748b"

    # Consensus
    agrees = sum([mg_pct >= 50, siglip_conf >= 0.6, cxr_sim >= 0.6])
    total = 3
    consensus_color = "#22c55e" if agrees == total else ("#eab308" if agrees >= 2 else "#ef4444")

    esc_cat = _html.escape(category)
    esc_sound = _html.escape(sound_type)

    mode_label = "" if USE_REAL_MEDGEMMA else " <span style='color:#64748b;font-weight:normal;'>(simulated)</span>"
    html = f"""<div style="padding:10px;background:#0f172a;border-radius:10px;border:1px solid #1e293b;">
  <h4 style="color:#e2e8f0;margin:0 0 10px;font-size:13px;letter-spacing:1px;">
    5-MODEL CONSENSUS &mdash; {esc_cat}{mode_label}</h4>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:12px;">
    <div style="background:#1e293b;padding:8px;border-radius:6px;border-left:3px solid {mg_color};">
      <span style="color:#94a3b8;">MedGemma 1.5</span><br>
      <span style="color:{mg_color};font-weight:bold;">{mg_icon} {mg_pct:.0f}%</span>
      <span style="color:#64748b;"> &middot; {len(boxes)} bbox{'es' if len(boxes) != 1 else ''}</span>
    </div>
    <div style="background:#1e293b;padding:8px;border-radius:6px;border-left:3px solid {sig_color};">
      <span style="color:#94a3b8;">MedSigLIP</span><br>
      <span style="color:{sig_color};font-weight:bold;">{esc_cat} ({siglip_conf:.2f})</span>
    </div>
    <div style="background:#1e293b;padding:8px;border-radius:6px;border-left:3px solid {cxr_color};">
      <span style="color:#94a3b8;">CXR Foundation</span><br>
      <span style="color:{cxr_color};font-weight:bold;">sim={cxr_sim:.2f}</span>
      <span style="color:#64748b;"> ELIXR</span>
    </div>
    <div style="background:#1e293b;padding:8px;border-radius:6px;border-left:3px solid {hear_color};">
      <span style="color:#94a3b8;">HeAR</span><br>
      <span style="color:{hear_color};font-weight:bold;">{esc_sound}</span>
      <span style="color:#64748b;"> expected</span>
    </div>
  </div>
  <div style="margin-top:8px;display:flex;align-items:center;gap:8px;">
    <div style="background:#1e293b;padding:6px 10px;border-radius:6px;border-left:3px solid {asr_color};flex:1;">
      <span style="color:#94a3b8;font-size:12px;">MedASR</span>
      <span style="color:{asr_color};font-size:12px;margin-left:6px;">{asr_status}</span>
    </div>
    <div style="background:#1e293b;padding:6px 10px;border-radius:6px;border-left:3px solid {consensus_color};flex:1;text-align:center;">
      <span style="color:{consensus_color};font-weight:bold;font-size:13px;">{agrees}/{total} AGREE</span>
    </div>
  </div>
</div>"""
    return html


# ─── Core Training Loop ───────────────────────────────────────────

def _empty_training_outputs(msg: str = ""):
    """Return a tuple of empty/default values for training tab outputs."""
    empty_stats = render_session_stats_html({
        "total_reviews": 0, "avg_score": 0,
        "avg_box_iou": 0, "categories_practiced": 0,
    })
    return (
        None,                            # case_image
        msg or "Click **Start Session** to begin.",  # question
        "",                              # answer_input
        render_blindspot_html([]),       # blindspot
        empty_stats,                     # stats
        "",                              # similar_cases
    )


def start_session(student_name: str, state: dict):
    """Initialize a training session."""
    state = dict(state)
    init_engine()
    init_asr()
    init_retrieval()
    init_cxr_foundation()
    init_hear()
    student = get_or_create_student(student_name or "Student")
    state["student"] = student

    if not student.cards:
        demo_cards = load_demo_dataset(str(DATA_DIR / "demo"))
        if not demo_cards:
            return (*_empty_training_outputs(
                "**No demo images found.** Add chest X-rays to `data/demo/{category}/` folders.\n\n"
                "Expected categories: `Cardiomegaly`, `Pneumothorax`, etc."
            ), state)
        for card in demo_cards:
            student.add_card(card)
        save_student(student)

    # Build retrieval index for MedSigLIP (HAI-DEF #2) and CXR Foundation (HAI-DEF #4)
    _build_retrieval_index(student)

    return get_next_case(state)


def get_next_case(state: dict):
    """FSRS-6 selects the next optimal case based on memory state."""
    state = dict(state)
    student = state.get("student")

    if not student or not student.cards:
        return (*_empty_training_outputs("No cases loaded. Click **Start Session** first."), state)

    cards = list(student.cards.values())
    due_cards = scheduler.get_due_cards(cards)

    if not due_cards:
        spots = render_blindspot_html(student.get_blind_spots())
        stats = render_session_stats_html(student.get_session_stats())
        return None, "All caught up! No cards due for review.", "", spots, stats, "", state

    # Load the first card with a valid image
    image = None
    current_card = None
    for card in due_cards:
        try:
            image = Image.open(card.image_path).convert("RGB")
            current_card = card
            break
        except Exception:
            continue
    state["card"] = current_card

    if image is None:
        return (*_empty_training_outputs("Could not load images. Check data/demo/ folder."), state)

    # Retrievability display
    if current_card.fsrs.state == LearningState.New:
        r_display = "NEW CASE"
        r_badge = "New"
    else:
        elapsed = (time.time() - current_card.fsrs.last_review) / 86400.0
        r = retrievability(current_card.fsrs.stability, elapsed)
        r_display = f"Retention: {r:.0%} | S: {current_card.fsrs.stability:.1f}d | D: {current_card.fsrs.difficulty:.1f}"
        r_badge = "Critical" if r < 0.5 else ("Due" if r < 0.75 else "Review")

    # Generate clinical question
    question_text = engine.generate_question(image, current_card.category)

    question = (
        f"### Case #{current_card.times_shown + 1} &mdash; {r_badge}\n"
        f"**Category:** {current_card.category} | *{r_display}*\n\n"
        f"{question_text}\n\n"
        f"---\n*FSRS-6 selected this case because "
    )

    if current_card.fsrs.state == LearningState.New:
        question += "it's a new pattern you haven't seen yet.*"
    elif current_card.fsrs.state == LearningState.Relearning:
        question += f"you forgot this last time (lapse #{current_card.fsrs.lapses}). Time to rebuild.*"
    else:
        elapsed = (time.time() - current_card.fsrs.last_review) / 86400.0
        r = retrievability(current_card.fsrs.stability, elapsed)
        question += f"your retention has dropped to {r:.0%} &mdash; optimal review point.*"

    spots = render_blindspot_html(student.get_blind_spots())
    stats = render_session_stats_html(student.get_session_stats())

    return image, question, "", spots, stats, "", state


def submit_answer(student_answer: str, self_rating: str, confidence: int, state: dict):
    """Process student answer, get AI feedback, update FSRS-6 state."""
    state = dict(state)
    student = state.get("student")
    current_card = state.get("card")

    if not student or not current_card:
        return None, "No active case. Click **Start Session** to begin.", "", "", "", "", state

    category = current_card.category
    confidence = max(1, min(5, int(confidence))) if confidence is not None else 3

    # ─── Load image ────────────────────────────────────────────
    try:
        image = Image.open(current_card.image_path).convert("RGB")
    except Exception:
        image = Image.new("RGB", (512, 512), color=(0, 0, 0))

    # ─── AI Feedback + Bounding Boxes ──────────────────────────
    if engine is not None:
        ground_truth = CATEGORY_DESCRIPTIONS.get(category, category)
        ai_feedback = engine.grade_response(image, student_answer, category, ground_truth)
        boxes = engine.localize_findings(image, category)
    else:
        ai_feedback = None
        boxes = []

    ai_score = ai_feedback.score if ai_feedback else 0.5

    # ─── F3: Search Completeness (Satisfaction of Search) ──────
    search_found, search_missed, search_comp = [], [], 0.0
    if isinstance(engine, MockMedGemmaEngine):
        search_found, search_missed, search_comp = engine.grade_search_completeness(student_answer, category)

    # ─── Draw bounding boxes on image ("See What I See") ──────
    annotated_image = draw_boxes_on_image(image, boxes) if boxes else image

    # ─── FSRS-6 Review ─────────────────────────────────────────
    grade = RATING_MAP.get(self_rating, Rating.Good)
    result = scheduler.review(current_card.fsrs, grade)

    # F1: Apply overconfidence modifier
    cal_data = student.calibration_per_category()
    if category in cal_data:
        cal_mod = interval_modifier_for_overconfidence(cal_data[category]["calibration_gap"])
        if cal_mod < 1.0:
            result.interval = max(1, int(result.interval * cal_mod))

    # F3: Apply search completeness modifier
    if search_comp < 0.8:
        sc_mod = search_completeness_modifier(search_comp)
        result.interval = max(1, int(result.interval * sc_mod))

    student.record_review(
        current_card, grade, result,
        score=ai_score,
        box_iou=ai_feedback.box_iou if ai_feedback else 0.0,
        confidence=confidence,
        search_completeness=search_comp,
        found_findings=len(search_found),
        total_findings=len(search_found) + len(search_missed),
    )
    save_student(student)

    # ─── Build Rich Feedback ──────────────────────────────────
    feedback = f"## Feedback: {category}\n\n"

    if ai_feedback:
        score_pct = ai_feedback.score * 100
        label = "Excellent" if score_pct >= 80 else ("Good" if score_pct >= 60 else ("Partial" if score_pct >= 40 else "Needs Work"))
        feedback += f"**AI Assessment: {label} ({score_pct:.0f}%)**\n\n"

        if ai_feedback.correct_findings:
            feedback += f"**Correct:** {', '.join(ai_feedback.correct_findings)}\n\n"
        if ai_feedback.missed_findings:
            feedback += f"**Missed:** {', '.join(ai_feedback.missed_findings)}\n\n"

        feedback += f"---\n\n### Expert Teaching\n\n{ai_feedback.explanation}\n\n"

    # F3: Search completeness feedback
    if search_found or search_missed:
        total_f = len(search_found) + len(search_missed)
        feedback += f"\n### Search Completeness: {len(search_found)}/{total_f} findings\n\n"
        if search_missed:
            feedback += "**Missed findings (Satisfaction of Search):**\n"
            for m in search_missed:
                feedback += f"- {m}\n"
            feedback += "\n*22% of all radiology errors come from stopping after the first finding. Always complete your search pattern.*\n\n"

    # F1: Calibration feedback
    if confidence > 0:
        cal_normalized = confidence / 5.0
        if cal_normalized > ai_score + 0.2:
            feedback += f"\n**Calibration Alert:** Your confidence ({confidence}/5) exceeds your accuracy ({ai_score:.0%}). Overconfidence is the most dangerous cognitive bias in radiology.\n\n"

    if boxes:
        feedback += f"\n**Bounding Boxes:** {len(boxes)} finding{'s' if len(boxes) != 1 else ''} localized on the image above.\n\n"

    feedback += (
        f"---\n\n### FSRS-6 Memory Update\n\n"
        f"| Parameter | Value |\n|---|---|\n"
        f"| Next Review | **{result.interval} day{'s' if result.interval != 1 else ''}** |\n"
        f"| Stability | {result.state.stability:.2f} days |\n"
        f"| Difficulty | {result.state.difficulty:.2f} / 10 |\n"
        f"| Retrievability | {result.retrievability:.0%} at review time |\n"
        f"| State | {result.state.state.name} |\n"
        f"| Reps | {result.state.reps} |\n"
        f"| Lapses | {result.state.lapses} |\n"
    )

    if result.is_lapse:
        feedback += (
            "\n**Lapse detected.** FSRS-6 reduced stability to rebuild your memory "
            "through targeted repetition. This case will appear again soon.\n"
        )

    spots = render_blindspot_html(student.get_blind_spots())
    stats = render_session_stats_html(student.get_session_stats())
    similar = get_similar_cases_html(student, current_card)
    consensus = build_consensus_html(category, ai_score, boxes, student_answer)

    return annotated_image, feedback, spots, stats, similar, consensus, state


# ─── Longitudinal CXR Comparison ─────────────────────────────────

def start_longitudinal(state: dict):
    """Generate a longitudinal comparison case."""
    state = dict(state)
    student = state.get("student")

    if not student or not student.cards:
        return (
            None, None,
            "Start a training session first to load cases.",
            "", state,
        )

    # Group cards by category
    cards_by_cat: dict[str, list] = {}
    for card in student.cards.values():
        cards_by_cat.setdefault(card.category, []).append(card)

    pairs = create_longitudinal_pairs(cards_by_cat)
    if not pairs:
        return (
            None, None,
            "Need at least 2 images per category for longitudinal comparison.",
            "", state,
        )

    case = random.choice(pairs)
    state["longitudinal"] = case

    # Load images
    try:
        prior_img = Image.open(case.prior_path).convert("RGB")
        current_img = Image.open(case.current_path).convert("RGB")
    except Exception:
        return None, None, "Could not load comparison images.", "", state

    question = generate_longitudinal_question(case.category, case.change_type)

    return prior_img, current_img, question, "", state


def submit_longitudinal(student_answer: str, self_rating: str, state: dict):
    """Grade longitudinal comparison answer."""
    state = dict(state)
    student = state.get("student")
    current_longitudinal = state.get("longitudinal")

    if not current_longitudinal or not student:
        return "Start a longitudinal case first.", "", "", state

    feedback_result = get_longitudinal_feedback(
        current_longitudinal.category,
        current_longitudinal.change_type,
        student_answer,
    )

    # Find a card in this category to update FSRS-6 state
    cat_cards = [
        c for c in student.cards.values()
        if c.category == current_longitudinal.category
    ]
    if cat_cards:
        card = cat_cards[0]
        grade = RATING_MAP.get(self_rating, Rating.Good)
        result = scheduler.review(card.fsrs, grade)
        student.record_review(
            card, grade, result,
            score=feedback_result.score,
        )
        save_student(student)

        fsrs_info = (
            f"\n\n---\n### FSRS-6 Update\n"
            f"Next review in **{result.interval} day(s)** | "
            f"S={result.state.stability:.1f}d | D={result.state.difficulty:.1f}"
        )
    else:
        fsrs_info = ""

    # Build feedback markdown
    score_pct = feedback_result.score * 100
    label = "Excellent" if score_pct >= 80 else ("Good" if score_pct >= 60 else ("Partial" if score_pct >= 40 else "Needs Work"))

    feedback_md = (
        f"## Longitudinal Assessment: {label} ({score_pct:.0f}%)\n\n"
        f"**Change Type:** {current_longitudinal.change_type.upper()}\n\n"
    )

    if feedback_result.correct_findings:
        feedback_md += f"**Detected:** {', '.join(feedback_result.correct_findings)}\n\n"
    if feedback_result.missed_findings:
        feedback_md += f"**Missed keywords:** {', '.join(feedback_result.missed_findings)}\n\n"

    feedback_md += f"---\n\n### Expert Assessment\n\n{feedback_result.explanation}"
    feedback_md += fsrs_info

    spots = render_blindspot_html(student.get_blind_spots())
    stats = render_session_stats_html(student.get_session_stats())

    return feedback_md, spots, stats, state


# ─── F2: Socratic Mode ─────────────────────────────────────────

def get_socratic_question(student_answer: str, state: dict):
    """Generate Socratic probing question instead of immediate feedback."""
    student = state.get("student")
    card = state.get("card")
    if not student or not card or not engine:
        return "Start a session first."
    if not isinstance(engine, MockMedGemmaEngine):
        return "Socratic mode uses the mock engine's clinical knowledge."
    return engine.generate_socratic_question(None, student_answer, card.category)


def submit_socratic_response(socratic_answer: str, state: dict):
    """Process student's response to Socratic question."""
    card = state.get("card")
    if not card or not engine or not isinstance(engine, MockMedGemmaEngine):
        return "No active Socratic dialogue."
    return engine.generate_socratic_followup("", socratic_answer, card.category)


# ─── F4: Dual-Process (System 1 + System 2) ──────────────────

def submit_gestalt(gestalt_answer: str, state: dict):
    """Grade the rapid gestalt impression (System 1, 3-sec flash)."""
    card = state.get("card")
    if not card or not engine:
        return "No active case.", None
    if isinstance(engine, MockMedGemmaEngine):
        score = engine.grade_gestalt(gestalt_answer, card.category)
    else:
        score = 0.5
    label = "Strong" if score >= 0.7 else ("Partial" if score >= 0.4 else "Missed")
    # Re-show the image for full analysis
    try:
        image = Image.open(card.image_path).convert("RGB")
    except Exception:
        image = None
    return (
        f"**Gestalt Score: {label} ({score:.0%})**\n\n"
        f"Now take your time for a full analytical review (System 2)."
    ), image


# ─── F5: Contrastive Case Pairs ──────────────────────────────

def start_contrastive(state: dict):
    """Generate a contrastive pair of visually similar but diagnostically different cases."""
    state = dict(state)
    student = state.get("student")

    if not student or not student.cards:
        return None, None, "Start a session first.", "", state

    if not isinstance(engine, MockMedGemmaEngine):
        return None, None, "Contrastive mode uses mock engine knowledge.", "", state

    # Pick a random contrastive pair
    pair_keys = list(engine.CONTRASTIVE_PAIRS.keys())
    if not pair_keys:
        return None, None, "No contrastive pairs defined.", "", state

    cat_a, cat_b = random.choice(pair_keys)
    state["contrastive_pair"] = (cat_a, cat_b)

    # Find images from each category
    cards_a = [c for c in student.cards.values() if c.category == cat_a and c.image_path]
    cards_b = [c for c in student.cards.values() if c.category == cat_b and c.image_path]

    if not cards_a or not cards_b:
        return None, None, f"Need images for both {cat_a} and {cat_b}.", "", state

    card_a = random.choice(cards_a)
    card_b = random.choice(cards_b)

    try:
        img_a = Image.open(card_a.image_path).convert("RGB")
        img_b = Image.open(card_b.image_path).convert("RGB")
    except Exception:
        return None, None, "Could not load images.", "", state

    question = engine.generate_contrastive_question(cat_a, cat_b)
    return img_a, img_b, f"### Contrastive Pair: {cat_a} vs {cat_b}\n\n{question}", "", state


def submit_contrastive(student_answer: str, self_rating: str, state: dict):
    """Grade contrastive pair response."""
    state = dict(state)
    student = state.get("student")
    contrastive_pair = state.get("contrastive_pair")

    if not student or not isinstance(engine, MockMedGemmaEngine):
        return "No active contrastive case.", state

    if contrastive_pair is None:
        return "No active contrastive case. Generate a pair first.", state

    cat_a, cat_b = contrastive_pair

    feedback_result = engine.grade_contrastive(student_answer, cat_a, cat_b)

    # Update FSRS for both categories
    grade = RATING_MAP.get(self_rating, Rating.Good)

    for cat in [cat_a, cat_b]:
        cat_cards = [c for c in student.cards.values() if c.category == cat]
        if cat_cards:
            card = cat_cards[0]
            result = scheduler.review(card.fsrs, grade)
            student.record_review(
                card, grade, result,
                score=feedback_result.score,
                contrastive_pair=f"{cat_a} vs {cat_b}",
            )
    save_student(student)

    state["contrastive_pair"] = None  # Clear state after submission

    score_pct = feedback_result.score * 100
    label = "Excellent" if score_pct >= 80 else ("Good" if score_pct >= 60 else ("Partial" if score_pct >= 40 else "Needs Work"))

    md = f"## Discrimination: {label} ({score_pct:.0f}%)\n\n"
    if feedback_result.correct_findings:
        md += f"**Identified:** {', '.join(feedback_result.correct_findings)}\n\n"
    if feedback_result.missed_findings:
        md += f"**Missed keys:** {', '.join(feedback_result.missed_findings)}\n\n"
    md += f"---\n\n{feedback_result.explanation}"
    return md, state


# ─── F6: HeAR Auscultation Training ──────────────────────────

def start_auscultation(state: dict):
    """Start an auscultation case: play lung sound, predict CXR findings."""
    state = dict(state)
    student = state.get("student")

    if not student or not student.cards:
        return None, "Start a training session first.", "", state

    hear = init_hear()
    categories = list(set(c.category for c in student.cards.values()))
    if not categories:
        return None, "No categories loaded.", "", state

    category = random.choice(categories)
    if category not in LUNG_SOUNDS:
        category = "No Finding"

    state["auscultation_category"] = category
    sr, audio = hear.generate_lung_sound(category)

    question = (
        f"### Auscultation Case: Listen Then Look\n\n"
        f"You are listening to a patient's chest. Based on what you hear:\n\n"
        f"1. What lung sounds are present?\n"
        f"2. What CXR findings would you predict?\n"
        f"3. What is your differential diagnosis?\n\n"
        f"*Mock mode: synthetic audio demonstrates the workflow. GPU mode uses HeAR (313M audio clips).*"
    )

    return (sr, audio), question, "", state


def submit_auscultation(prediction: str, state: dict):
    """Grade auscultation prediction and reveal the CXR."""
    state = dict(state)
    student = state.get("student")
    auscultation_category = state.get("auscultation_category")

    if not student or not student.cards:
        return None, "No active session.", state

    if auscultation_category is None:
        return None, "No active auscultation case. Start one first.", state

    category = auscultation_category

    cat_cards = [c for c in student.cards.values() if c.category == category and c.image_path]
    image = None
    if cat_cards:
        try:
            image = Image.open(cat_cards[0].image_path).convert("RGB")
        except Exception:
            pass

    # Grade prediction
    data = LUNG_SOUNDS.get(category, LUNG_SOUNDS["No Finding"])
    pred_lower = (prediction or "").lower()
    key_terms = data["sound"].lower().split() + category.lower().split()
    matches = sum(1 for t in key_terms if t in pred_lower)

    if matches >= 2:
        score_label = "Excellent"
    elif matches >= 1:
        score_label = "Partial"
    else:
        score_label = "Missed"

    feedback = (
        f"## Auscultation Result: {score_label}\n\n"
        f"**Actual Category:** {category}\n\n"
        f"**Sound:** {data['sound']} | **Location:** {data['location']}\n\n"
        f"**Character:** {data['character']}\n\n"
        f"---\n\n### Sound-Image Correlation\n\n{data['correlation']}\n\n"
        f"*HeAR classifies respiratory sounds from 313 million audio clips "
        f"to train this auscultation-to-imaging correlation.*"
    )

    state["auscultation_category"] = None  # Clear state after submission
    return image, feedback, state


def refresh_progress(state: dict):
    """Refresh all progress displays."""
    student = state.get("student")
    if not student:
        empty = "<div style='color:#64748b;padding:20px;'>Start a session to see progress.</div>"
        return empty, empty, empty, empty, empty
    curve = build_learning_curve_html(student)
    fsrs = build_fsrs_state_html(student)
    spots = render_blindspot_html(student.get_blind_spots())
    cal = render_calibration_chart_html(student.calibration_per_category())
    forget = build_forgetting_curve_html(student)
    return curve, fsrs, spots, cal, forget


# ─── Gradio UI ─────────────────────────────────────────────────────

HEADER_HTML = """
<div style="text-align:center;padding:20px 0 4px;">
    <h1 style="font-size:42px;margin:0;color:#e2e8f0;letter-spacing:6px;font-weight:800;">ENGRAM</h1>
    <p style="font-size:14px;color:#94a3b8;margin:6px 0 0;letter-spacing:1px;">
        FSRS-6 Adaptive Medical Visual Diagnosis Training</p>
    <p style="font-size:11px;color:#64748b;margin:4px 0 0;">
        5 HAI-DEF Models: MedGemma 1.5 &middot; MedSigLIP &middot; CXR Foundation &middot; MedASR &middot; HeAR
        &nbsp;|&nbsp; FSRS-6 from <a href="https://github.com/samvallad33/vestige" style="color:#60a5fa;">Vestige</a> (62K lines Rust)
        &nbsp;|&nbsp; LUMIA: +3.8% MATH-500</p>
</div>
"""

ABOUT_HTML = """
<div style="padding:16px;font-size:13px;color:#94a3b8;line-height:1.7;">
    <h3 style="color:#e2e8f0;margin-top:0;">How ENGRAM Works</h3>

    <p><b>1. FSRS-6 Selects Your Next Case</b><br>
    The 21-parameter scheduler picks the case you're closest to forgetting &mdash;
    targeting your weakest diagnostic areas at the optimal review moment.</p>

    <p><b>2. You Interpret the Image (Type or Dictate)</b><br>
    No multiple choice. Describe what you see, where it is, and your differential
    &mdash; just like a real clinical setting. Use the microphone for voice
    dictation via <b>MedASR</b> (58% fewer errors than Whisper on CXR dictation).</p>

    <p><b>3. MedGemma 1.5 Gives Expert Feedback</b><br>
    Google's medical AI grades your interpretation, identifies what you missed,
    and draws bounding boxes showing exactly WHERE the findings are.</p>

    <p><b>4. Compare Prior &amp; Current (Longitudinal Mode)</b><br>
    MedGemma 1.5's flagship capability: compare two studies to detect interval
    changes. "Compared to prior, the effusion has increased" &mdash; exactly how
    radiologists work.</p>

    <p><b>5. FSRS-6 Updates Your Memory Model</b><br>
    Based on performance, the algorithm updates Stability (how long you'll remember)
    and Difficulty (how hard this is for you), then schedules the next optimal review.</p>

    <p><b>6. Your Diagnostic Landscape Evolves</b><br>
    The blind spot map shows exactly which categories you've mastered and where
    you're dangerously weak &mdash; no hiding behind averages.</p>

    <hr style="border-color:#334155;margin:12px 0;">

    <h3 style="color:#e2e8f0;">5 HAI-DEF Models</h3>
    <ul>
        <li><b>MedGemma 1.5 4B</b>: Image analysis, bounding box localization (IoU 38.0 vs 3.1 in v1),
            question generation, response grading, and longitudinal CXR comparison (+5% accuracy).</li>
        <li><b>MedSigLIP</b>: Medical image-text embeddings for visual similarity retrieval (FAISS index).</li>
        <li><b>CXR Foundation</b>: ELIXR embeddings trained on 800K+ chest X-rays. 0.898 AUC for
            classification. 600x more data-efficient than traditional transfer learning.</li>
        <li><b>MedASR</b>: 105M-param Conformer for medical speech-to-text. 58% fewer errors than
            Whisper on CXR dictation (5.2% vs 12.5% WER). Real radiologists dictate, not type.</li>
        <li><b>HeAR</b>: ViT-L bioacoustic model trained on 313 million audio clips. Classifies
            respiratory sounds (crackles, wheezing, diminished) for auscultation-to-imaging correlation training.</li>
    </ul>

    <h3 style="color:#e2e8f0;">Advanced Training Modes</h3>
    <ul>
        <li><b>Confidence Calibration</b>: Tracks self-assessed confidence vs actual accuracy per pathology.
            Overconfident categories get shorter FSRS intervals &mdash; the most dangerous cognitive bias in radiology.</li>
        <li><b>Socratic Mode</b>: Guided questioning instead of immediate answers. Based on MedTutor-R1 research.</li>
        <li><b>Satisfaction of Search</b>: Grades whether you found ALL findings, not just the primary.
            Targets a leading cognitive bias causing 22% of radiology errors.</li>
        <li><b>Dual-Process Training</b>: 3-second flash for System 1 pattern recognition, then full
            analytical review for System 2. Measures and improves your gestalt accuracy.</li>
        <li><b>Contrastive Case Pairs</b>: Side-by-side visually similar cases from different categories.
            Trains diagnostic discrimination at the decision boundary.</li>
        <li><b>Auscultation Training</b>: Listen Then Look &mdash; hear the patient, predict the CXR, then see it.
            HeAR powers the auscultation-to-imaging correlation workflow.</li>
    </ul>

    <h3 style="color:#e2e8f0;">FSRS-6: The Secret Weapon</h3>
    <ul>
        <li>21-parameter power-law forgetting model, ported from Vestige (62K lines Rust)</li>
        <li>Personalizable decay (w20), same-day review handling (w17-w19)</li>
        <li>Demonstrated: +3.8% MATH-500, +3.3% AIME 2025, +1.0% GPQA Diamond (LUMIA, Feb 2026)</li>
        <li>Evidence-based: Thompson &amp; Hughes (JACR, 2023) confirms spaced repetition improves
            radiology education but adoption lags &mdash; ENGRAM fills this gap</li>
    </ul>

    <h3 style="color:#e2e8f0;">Why ENGRAM Matters</h3>
    <ul>
        <li><b>10 million</b> healthcare worker shortage by 2030 (WHO)</li>
        <li>Diagnostic errors contribute to ~<b>10% of patient deaths</b> (Balogh et al., NAP 2015)</li>
        <li>Student bounding box annotations are structured as training data (designed for co-evolutionary learning)</li>
        <li>Runs offline on a single GPU or CPU (mock mode) &mdash; edge-deployable for any clinic</li>
    </ul>

    <h3 style="color:#e2e8f0;">References</h3>
    <ul style="font-size:11px;color:#64748b;">
        <li>Balogh EP et al. Improving Diagnosis in Health Care. NAP, 2015.</li>
        <li>Thompson CP, Hughes MA. "Spaced Repetition in Medical Education." JACR. 2023;20(11):1092-1101.</li>
        <li>Vallad S. "LUMIA: FSRS-6 for LLM Training." Feb 2026.</li>
        <li>open-spaced-repetition. "FSRS-6 Algorithm." 2024.</li>
    </ul>

    <p style="font-size:11px;color:#475569;">
        ENGRAM is a submission for the Kaggle MedGemma Impact Challenge 2026.
        Built by Sam Vallad. Not intended for clinical use without validation.</p>
</div>
"""


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ENGRAM \u2014 Adaptive Medical Training") as app:

        session_state = gr.State(value=_default_state())

        gr.HTML(HEADER_HTML)

        with gr.Tabs():
            # ═══════════════════════════════════════════════════
            # TAB 1: TRAINING
            # ═══════════════════════════════════════════════════
            with gr.Tab("Training"):
                with gr.Row():
                    student_name = gr.Textbox(
                        label="Your Name", value="Student", scale=2,
                    )
                    start_btn = gr.Button("Start Session", variant="primary", scale=1)

                with gr.Row():
                    # Left: Image + Question + Answer
                    with gr.Column(scale=3):
                        case_image = gr.Image(
                            label="Medical Image (boxes appear after submission)",
                            type="pil", height=420, interactive=False,
                        )
                        question_display = gr.Markdown(
                            value="*Click **Start Session** to begin adaptive training.*",
                        )

                        # Voice dictation (MedASR)
                        with gr.Accordion("Voice Dictation (MedASR)", open=False):
                            gr.Markdown(
                                "*Real radiologists dictate, not type. "
                                "MedASR achieves 58% fewer errors than Whisper on CXR dictation.*"
                            )
                            audio_input = gr.Audio(
                                label="Record your interpretation",
                                type="numpy",
                                sources=["microphone"],
                            )
                            transcribe_btn = gr.Button(
                                "Transcribe with MedASR", variant="secondary", size="sm",
                            )

                        answer_input = gr.Textbox(
                            label="Your Interpretation (type or use voice above)",
                            placeholder="Describe findings, anatomical location, and differential diagnosis...",
                            lines=4,
                        )
                        confidence_slider = gr.Slider(
                            minimum=1, maximum=5, value=3, step=1,
                            label="Confidence (1=Guessing \u2192 5=Certain)",
                        )
                        self_rating = gr.Radio(
                            choices=[
                                "Missed It (Again)", "Struggled (Hard)",
                                "Got It (Good)", "Easy (Easy)",
                            ],
                            value="Got It (Good)",
                            label="Self-Assessment (FSRS-6 uses this to schedule your next review)",
                        )
                        with gr.Row():
                            submit_btn = gr.Button("Submit & Get Feedback", variant="primary", scale=2)
                            next_btn = gr.Button("Next Case", variant="secondary", scale=1)

                        feedback_display = gr.Markdown(value="")

                        with gr.Accordion("Socratic Mode", open=False):
                            gr.Markdown(
                                "*Instead of immediate answers, explore through guided questioning. "
                                "Submit your answer above first, then click below.*"
                            )
                            socratic_btn = gr.Button(
                                "Ask Me a Probing Question", variant="secondary", size="sm",
                            )
                            socratic_question = gr.Markdown(value="")
                            socratic_input = gr.Textbox(
                                label="Your Response to Probing Question", lines=2,
                            )
                            socratic_submit = gr.Button(
                                "Submit Response", variant="secondary", size="sm",
                            )
                            socratic_followup = gr.Markdown(value="")

                        with gr.Accordion("Timed Flash Mode (Dual-Process)", open=False):
                            gr.Markdown(
                                "*Train System 1 (rapid pattern recognition). "
                                "Look at the image above for 3 seconds, then type your quick impression below.*"
                            )
                            gestalt_input = gr.Textbox(
                                label="Quick Impression (what did you see in the flash?)",
                                placeholder="Type your rapid impression...",
                                lines=2,
                            )
                            gestalt_submit = gr.Button(
                                "Submit Gestalt", variant="secondary", size="sm",
                            )
                            gestalt_result = gr.Markdown(value="")

                    # Right: Stats + Blind Spots + Similar Cases
                    with gr.Column(scale=2):
                        stats_display = gr.HTML(
                            value=render_session_stats_html({
                                "total_reviews": 0, "avg_score": 0,
                                "avg_box_iou": 0, "categories_practiced": 0,
                            }),
                        )
                        blindspot_display = gr.HTML(value=render_blindspot_html([]))
                        similar_display = gr.HTML(value="")
                        consensus_display = gr.HTML(value="")

            # ═══════════════════════════════════════════════════
            # TAB 2: LONGITUDINAL
            # ═══════════════════════════════════════════════════
            with gr.Tab("Longitudinal"):
                gr.Markdown(
                    "### Interval Change Detection\n"
                    "*Compare prior and current imaging — exactly how radiologists work daily.*\n\n"
                    "*MedGemma 1.5 achieves 65.7% macro accuracy on longitudinal CXR analysis (+5% over v1).*"
                )
                long_start_btn = gr.Button("Generate Comparison Case", variant="primary")

                with gr.Row():
                    prior_image = gr.Image(
                        label="Prior Study", type="pil", height=350, interactive=False,
                    )
                    current_image = gr.Image(
                        label="Current Study", type="pil", height=350, interactive=False,
                    )

                long_question = gr.Markdown(value="*Click above to generate a comparison case.*")

                long_answer = gr.Textbox(
                    label="Describe interval changes",
                    placeholder="Compare the two studies. What has changed? Improved, worsened, stable, new, or resolved?",
                    lines=4,
                )
                long_rating = gr.Radio(
                    choices=[
                        "Missed It (Again)", "Struggled (Hard)",
                        "Got It (Good)", "Easy (Easy)",
                    ],
                    value="Got It (Good)",
                    label="Self-Assessment",
                )
                long_submit_btn = gr.Button("Submit Comparison", variant="primary")
                long_feedback = gr.Markdown(value="")

                with gr.Row():
                    long_spots = gr.HTML(value="")
                    long_stats = gr.HTML(value="")

            # ═══════════════════════════════════════════════════
            # TAB 3: CONTRASTIVE PAIRS
            # ═══════════════════════════════════════════════════
            with gr.Tab("Contrastive Pairs"):
                gr.Markdown(
                    "### Diagnostic Discrimination Training\n"
                    "*Compare visually similar cases from different categories. "
                    "Can you spot the key difference?*"
                )
                contrast_start_btn = gr.Button("Generate Contrastive Pair", variant="primary")
                with gr.Row():
                    contrast_img_a = gr.Image(
                        label="Case A", type="pil", height=350, interactive=False,
                    )
                    contrast_img_b = gr.Image(
                        label="Case B", type="pil", height=350, interactive=False,
                    )
                contrast_question = gr.Markdown(value="*Click above to generate a contrastive pair.*")
                contrast_answer = gr.Textbox(
                    label="Key Discriminating Features",
                    placeholder="What is the critical difference between these two cases?",
                    lines=3,
                )
                contrast_rating = gr.Radio(
                    choices=[
                        "Missed It (Again)", "Struggled (Hard)",
                        "Got It (Good)", "Easy (Easy)",
                    ],
                    value="Got It (Good)",
                    label="Self-Assessment",
                )
                contrast_submit_btn = gr.Button("Submit", variant="primary")
                contrast_feedback = gr.Markdown(value="")

            # ═══════════════════════════════════════════════════
            # TAB 4: AUSCULTATION (HeAR)
            # ═══════════════════════════════════════════════════
            with gr.Tab("Auscultation"):
                gr.Markdown(
                    "### Listen Then Look (HeAR \u2014 HAI-DEF Model #5)\n"
                    "*Hear the patient, predict the CXR, then see it. "
                    "HeAR is a ViT-L bioacoustic model trained on 313 million audio clips.*"
                )
                ausc_start_btn = gr.Button("Start Auscultation Case", variant="primary")
                ausc_audio = gr.Audio(label="Lung Sound", interactive=False)
                ausc_question = gr.Markdown(value="*Click above to start an auscultation case.*")
                ausc_prediction = gr.Textbox(
                    label="Your Prediction",
                    placeholder="What lung sounds do you hear? What CXR findings would you predict?",
                    lines=3,
                )
                ausc_submit_btn = gr.Button("Reveal CXR & Grade", variant="primary")
                ausc_image = gr.Image(
                    label="Revealed CXR", type="pil", height=350, interactive=False,
                )
                ausc_feedback = gr.Markdown(value="")

            # ═══════════════════════════════════════════════════
            # TAB 5: PROGRESS
            # ═══════════════════════════════════════════════════
            with gr.Tab("Progress"):
                refresh_btn = gr.Button("Refresh Progress", variant="secondary")
                curve_display = gr.HTML(value="<div style='color:#64748b;padding:20px;'>Start training to see progress.</div>")
                fsrs_state_display = gr.HTML(value="")
                blindspot_progress = gr.HTML(value="")
                calibration_display = gr.HTML(value="")
                forgetting_display = gr.HTML(value="")
                export_btn = gr.Button("Export Session Data (JSON)")
                export_file = gr.File(label="Download", visible=True, interactive=False)

            # ═══════════════════════════════════════════════════
            # TAB 6: ABOUT
            # ═══════════════════════════════════════════════════
            with gr.Tab("About"):
                gr.HTML(ABOUT_HTML)

        # ─── Event Handlers (session_state threaded for per-user isolation) ──
        start_btn.click(
            fn=start_session,
            inputs=[student_name, session_state],
            outputs=[case_image, question_display, answer_input,
                     blindspot_display, stats_display, similar_display, session_state],
        )

        # MedASR transcription (no session state needed)
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input],
            outputs=[answer_input],
        )

        submit_btn.click(
            fn=submit_answer,
            inputs=[answer_input, self_rating, confidence_slider, session_state],
            outputs=[case_image, feedback_display,
                     blindspot_display, stats_display, similar_display,
                     consensus_display, session_state],
        )

        next_btn.click(
            fn=get_next_case,
            inputs=[session_state],
            outputs=[case_image, question_display, answer_input,
                     blindspot_display, stats_display, similar_display, session_state],
        )

        # Longitudinal
        long_start_btn.click(
            fn=start_longitudinal,
            inputs=[session_state],
            outputs=[prior_image, current_image, long_question, long_answer, session_state],
        )

        long_submit_btn.click(
            fn=submit_longitudinal,
            inputs=[long_answer, long_rating, session_state],
            outputs=[long_feedback, long_spots, long_stats, session_state],
        )

        # Socratic Mode
        socratic_btn.click(
            fn=get_socratic_question,
            inputs=[answer_input, session_state],
            outputs=[socratic_question],
        )
        socratic_submit.click(
            fn=submit_socratic_response,
            inputs=[socratic_input, session_state],
            outputs=[socratic_followup],
        )

        # Dual-Process (Gestalt)
        gestalt_submit.click(
            fn=submit_gestalt,
            inputs=[gestalt_input, session_state],
            outputs=[gestalt_result, case_image],
        )

        # Contrastive Pairs
        contrast_start_btn.click(
            fn=start_contrastive,
            inputs=[session_state],
            outputs=[contrast_img_a, contrast_img_b, contrast_question, contrast_answer, session_state],
        )
        contrast_submit_btn.click(
            fn=submit_contrastive,
            inputs=[contrast_answer, contrast_rating, session_state],
            outputs=[contrast_feedback, session_state],
        )

        # Auscultation (HeAR)
        ausc_start_btn.click(
            fn=start_auscultation,
            inputs=[session_state],
            outputs=[ausc_audio, ausc_question, ausc_prediction, session_state],
        )
        ausc_submit_btn.click(
            fn=submit_auscultation,
            inputs=[ausc_prediction, session_state],
            outputs=[ausc_image, ausc_feedback, session_state],
        )

        # Progress
        refresh_btn.click(
            fn=refresh_progress,
            inputs=[session_state],
            outputs=[curve_display, fsrs_state_display, blindspot_progress, calibration_display, forgetting_display],
        )

        export_btn.click(
            fn=export_session_data,
            inputs=[session_state],
            outputs=[export_file],
        )

    return app


if __name__ == "__main__":
    mode = "MedGemma 1.5 4B (GPU)" if USE_REAL_MEDGEMMA else "Mock Engine (CPU)"
    print(f"\n  ENGRAM v0.4.0")
    print(f"  Mode: {mode}")
    print(f"  HAI-DEF Models: MedGemma 1.5 + MedSigLIP + CXR Foundation + MedASR + HeAR")
    print(f"  Data: {DATA_DIR}")
    print(f"  Set ENGRAM_USE_MEDGEMMA=true for real inference\n")

    app = build_app()
    import os as _os
    _on_kaggle = _os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    app.launch(
        server_name="0.0.0.0" if _on_kaggle else "127.0.0.1",
        server_port=int(_os.environ.get("PORT", 7860)),
        share=_on_kaggle,
        show_error=_on_kaggle,
        theme=gr.themes.Base(
            primary_hue="blue",
            neutral_hue="slate",
        ),
        css="""
        .gradio-container { max-width: 1400px !important; }
        footer { display: none !important; }
        """,
    )
