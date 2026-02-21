# ENGRAM: FSRS-6 Adaptive Medical Visual Diagnosis Training

**Kaggle MedGemma Impact Challenge 2026**
**Sam Vallad | February 2026**

---

## 1. Problem: The Diagnostic Training Gap

The WHO projects a global shortage of **10 million healthcare workers by 2030**. Meanwhile, diagnostic errors remain a leading cause of patient harm, contributing to an estimated **10% of patient deaths** (Balogh et al., NAP 2015). Radiology residency programs face a fundamental constraint: learning to read medical images requires thousands of deliberate practice encounters with expert feedback, but attending radiologist time is the bottleneck.

A 2023 JACR review (Thompson & Hughes) confirmed that **spaced repetition significantly improves radiology education outcomes** — yet adoption in medical imaging training remains critically low. To our knowledge, no existing tool combines optimal memory scheduling with AI-powered medical image feedback.

Current training tools suffer from four critical gaps:

1. **No memory optimization.** Flash card apps present cases randomly, ignoring individual forgetting patterns. Students waste time re-reviewing known cases while dangerous blind spots go undetected.

2. **No spatial training.** Medical image interpretation is a spatial task — radiologists must learn *where* to look. Existing tools rarely train localization.

3. **No adaptive feedback.** Static answer keys cannot explain *why* findings look the way they do or adapt to student misconceptions.

4. **No multimodal clinical workflow.** Real radiologists dictate findings, compare with prior studies, and use specialized CXR-trained models — training tools should match this workflow.

**ENGRAM** addresses all four gaps by combining **5 HAI-DEF models** — MedGemma 1.5, MedSigLIP, CXR Foundation, MedASR, and HeAR — with **FSRS-6**, a state-of-the-art 21-parameter spaced repetition algorithm.

---

## 2. Approach: Four Innovations

### 2.1 FSRS-6: Optimal Memory Scheduling

FSRS-6 (Free Spaced Repetition Scheduler, version 6) is a 21-parameter algorithm modeling human memory via power-law forgetting curves. Ported from our Vestige project (62,000+ lines of Rust).

**Key properties:**
- **Per-concept tracking**: Each diagnostic category has independent Stability (S) and Difficulty (D).
- **Power-law decay**: R(t) = (1 + factor * t/S)^(-w20), matching empirical human forgetting data better than exponential models (Wixted & Ebbesen, 1991).
- **Same-day reviews**: Parameters w17-w19 (new in FSRS-6) handle intra-session reviews — critical for clinical training.
- **Personalizable decay**: w20 adapts to individual learner forgetting rates.

**Demonstrated performance:** In our LUMIA project (February 2026), FSRS-6 improved LLM training outcomes by +3.8% on MATH-500, +3.3% on AIME 2025, and +1.0% on GPQA Diamond.

### 2.2 MedGemma 1.5: Expert Visual Feedback

ENGRAM uses MedGemma 1.5 4B for five tasks:

1. **Image Analysis**: Systematic interpretation with findings, locations, and differentials.
2. **Bounding Box Localization**: Normalized [y0, x0, y1, x1] coordinates — the "See What I See" feature. MedGemma 1.5 achieves IoU 38.0 (vs 3.1 in v1).
3. **Question Generation**: Clinical vignettes adapted to category and difficulty.
4. **Response Grading**: Structured evaluation with scores, missed findings, and teaching.
5. **Longitudinal CXR Comparison**: MedGemma 1.5's flagship capability — comparing prior and current studies to detect interval changes (65.7% macro accuracy, +5% over v1).

### 2.3 MedASR: Voice Dictation Pipeline

Real radiologists dictate, not type. ENGRAM integrates Google's **MedASR** (HAI-DEF model #3) — a 105M-parameter Conformer model trained on ~5,000 hours of physician dictations. It achieves **58% fewer errors** than Whisper large-v3 on chest X-ray dictation (5.2% vs 12.5% WER).

Students can record their interpretation via microphone, and MedASR transcribes it with medical terminology accuracy — matching the clinical workflow that radiologists actually use.

### 2.4 CXR Foundation: Specialized Retrieval

**CXR Foundation** (HAI-DEF model #4) provides ELIXR embeddings trained on 800,000+ chest X-rays — far more specialized than generic medical image models.

- **0.898 AUC** for data-efficient classification (5 CheXpert findings)
- **0.846 AUC** zero-shot classification using textual prompts
- **Up to 600x less data** needed compared to traditional transfer learning

ENGRAM uses CXR Foundation for similar case retrieval via FAISS index, replacing generic MedSigLIP embeddings with CXR-specialized vectors for superior chest X-ray matching.

### 2.5 HeAR: Auscultation-to-Imaging Correlation

**HeAR** (Health Acoustic Representations, HAI-DEF model #5) is a ViT-L bioacoustic foundation model trained on **313 million two-second audio clips**. ENGRAM uses HeAR for the "**Listen Then Look**" workflow — students hear lung sounds, predict CXR findings, then see the image.

This trains the auscultation-to-imaging correlation that defines real clinical practice: crackles predict consolidation, absent sounds predict pneumothorax, bilateral basilar crackles predict edema. ENGRAM maps all 11 pathology categories to their expected auscultation findings with clinical descriptions.

### 2.6 Advanced Training Modes (v0.4.0)

Six cognitive training modes attack specific diagnostic failure patterns:

1. **Confidence Calibration**: Tracks confidence (1-5) vs accuracy per pathology. Shortens FSRS intervals for overconfident categories.
2. **Socratic Mode**: Probing questions instead of immediate answers — forces deeper reasoning before reveal.
3. **Satisfaction of Search**: Grades whether students found ALL findings, not just the primary. Directly targets a leading cognitive bias in radiology (22% of errors per Kim & Mansfield, 2014).
4. **Dual-Process Training**: 3-second flash for System 1 (gestalt), then full analytical review for System 2.
5. **Contrastive Case Pairs**: Side-by-side visually similar cases from different categories to train discrimination.
6. **HeAR Auscultation**: Listen-then-look workflow correlating lung sounds with CXR findings.

### 2.7 Forgetting Landscape: Blind Spot Detection

The **Diagnostic Landscape** visualizes a student's strengths and weaknesses across all pathology categories in real-time:

- **Retention**: Current probability of recall (power-law curve)
- **Stability**: Days until retention drops below target (0.9)
- **Difficulty**: How challenging this category is for the student
- **Mastery Level**: danger / weak / developing / strong / mastered

This transforms vague intuitions into precise, quantified action items.

---

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                           ENGRAM v0.4.0                          │
│                                                                  │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │  MedGemma  │  │  MedASR  │  │   CXR    │  │ MedSigLIP│  │  HeAR    │ │
│  │  1.5 4B    │  │  Voice   │  │ Foundtn  │  │ Retrieval│  │ Lung     │ │
│  │ Analysis + │  │ Dictation│  │ ELIXR    │  │ (FAISS)  │  │ Sounds   │ │
│  │ Localize + │  │  (105M)  │  │ (800K)   │  │          │  │ (313M)   │ │
│  │ Longitudnl │  │          │  │          │  │          │  │          │ │
│  └─────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│        │              │             │              │              │       │
│        ▼              ▼             ▼              ▼              ▼       │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              FSRS-6 Scheduler (21 params)                  │  │
│  │   Per-concept S & D · Power-law decay · Same-day reviews  │  │
│  └────────────────────────────┬───────────────────────────────┘  │
│                               │                                  │
│        ┌──────────────────────┼──────────────────────┐          │
│        ▼                      ▼                      ▼          │
│  ┌──────────┐    ┌────────────────┐    ┌──────────────────┐     │
│  │ Student  │    │    Bounding    │    │   Diagnostic     │     │
│  │ State    │    │    Box Overlay │    │   Landscape      │     │
│  │ (JSON)   │    │  "See What I  │    │  (Blind Spots)   │     │
│  │          │    │   See"        │    │                   │     │
│  └──────────┘    └────────────────┘    └──────────────────┘     │
└──────────────────────────────────────────────────────────────────┘
```

**Technology stack:**
- Python 3.10+, Gradio (web interface)
- MedGemma 1.5 4B via HuggingFace Transformers (float16, single GPU)
- MedASR (105M Conformer) for medical speech-to-text
- CXR Foundation (ELIXR embeddings) + MedSigLIP for retrieval (FAISS)
- FSRS-6 ported from Vestige (pure Python, no external ML dependencies)
- Student state persisted as JSON (offline-first, no cloud required)

**Deployment modes:**
- **GPU (Kaggle/Cloud)**: Real inference with all 5 HAI-DEF models
- **CPU (Offline)**: Mock engine with clinically accurate knowledge base for 11 pathology categories

---

## 4. Results & Impact

### Quantitative

| Metric | Value |
|---|---|
| HAI-DEF models used | 5 (MedGemma 1.5 + MedSigLIP + CXR Foundation + MedASR + HeAR) |
| FSRS-6 parameters | 21 (w0-w20) |
| Pathology categories | 11 (CheXpert taxonomy) |
| Longitudinal change types | 5 (worsened, improved, stable, new, resolved) |
| Advanced training modes | 6 (confidence calibration, Socratic, satisfaction of search, dual-process, contrastive pairs, auscultation) |
| Total codebase | ~9,300 lines |
| Tests passing | 82 |
| Runs offline | Yes — single GPU or CPU-only (mock mode) |
| Student data privacy | Local JSON storage, no cloud dependency |

### Impact Potential

**For medical education:**
- Transforms passive image review into active diagnostic practice
- FSRS-6 ensures every study minute is optimally allocated
- Voice dictation matches real clinical workflow (MedASR)
- Longitudinal comparison trains the most common radiology task
- Bounding box training develops spatial diagnostic skills
- Forgetting Landscape prevents dangerous blind spots

**For the HAI-DEF ecosystem:**
- Demonstrates **5 HAI-DEF models** working in concert
- MedGemma 1.5's new longitudinal CXR comparison capability (flagship)
- MedASR's voice dictation for realistic clinical workflow
- CXR Foundation's specialized embeddings for chest X-ray retrieval
- HeAR's bioacoustic respiratory sound classification (auscultation training)
- Shows offline/edge deployment viability for clinical environments

**Evidence-based approach:**
- Thompson & Hughes (JACR, 2023): spaced repetition improves radiology education outcomes, but adoption lags — ENGRAM fills this gap
- FSRS-6 demonstrated in LUMIA (Feb 2026): +3.8% MATH-500, +3.3% AIME 2025
- Simulated effectiveness study (50 students × 20 reviews): FSRS-6 achieves significantly higher diagnostic retention than random scheduling

**Data flywheel (designed for co-evolutionary learning):**
- Student bounding box annotations are structured as labeled training data, ready for future model fine-tuning
- FSRS-6 is designed to aggregate blind spots across student populations, identifying collective weak areas
- Cases that students consistently fail can be prioritized for targeted fine-tuning
- Architecture supports a co-evolutionary loop where the system and models improve together over time

### Alignment with Competition Criteria

1. **Effective HAI-DEF model use**: 5 models — MedGemma (analysis, localization, longitudinal) + MedSigLIP (retrieval) + CXR Foundation (specialized CXR retrieval) + MedASR (voice dictation) + HeAR (auscultation)
2. **Problem importance**: Global physician shortage, diagnostic errors, JACR evidence gap
3. **Real-world impact**: Runs offline in any clinical education setting, matches real workflow
4. **Technical feasibility**: Complete working system, ~9,300 lines, 82 tests passing
5. **Execution quality**: Clean architecture, security-audited, 82 tests, 5 HAI-DEF models, 6 advanced training modes

---

## 5. Conclusion

To our knowledge, ENGRAM is the first system to apply a state-of-the-art spaced repetition algorithm to medical visual diagnosis training with a full clinical workflow. By combining **FSRS-6's** demonstrated memory optimization with **5 HAI-DEF models** — MedGemma 1.5 for expert analysis and longitudinal comparison, MedASR for voice dictation, CXR Foundation for specialized retrieval, and HeAR for auscultation training — we create an adaptive training system that teaches medical students not just *what* to see, but *where* to look, *when* to review, *how* to dictate findings, and *what* to listen for like a real radiologist.

The same algorithm that improved LLM training by 3.8% on MATH-500 now teaches doctors to see.

---

## References

1. Balogh EP, Miller BT, Ball JR. Improving Diagnosis in Health Care. National Academies Press, 2015.
2. Thompson CP, Hughes MA. "The Effectiveness of Spaced Learning, Interleaving, and Retrieval Practice in Radiology Education: A Systematic Review." Journal of the American College of Radiology (JACR). 2023;20(11):1092-1101.
3. Vallad S. "LUMIA: FSRS-6 for LLM Training." February 2026.
4. open-spaced-repetition. "FSRS-6 Algorithm." 2024.
5. Wixted JT, Ebbesen EB. "On the Form of Forgetting." Psychological Science, 1991.

**Source code:** Reproducible, security-audited, clean architecture.
**License:** Apache 2.0 (code) | HAI-DEF terms (models)
