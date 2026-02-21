# ENGRAM

**FSRS-6 Adaptive Medical Visual Diagnosis Training**

Kaggle MedGemma Impact Challenge 2026 | Sam Vallad

---

ENGRAM combines a 21-parameter spaced repetition algorithm (FSRS-6) with 5 HAI-DEF medical AI foundation models to create an adaptive diagnostic training system for medical students.

## Models

| Model | Role |
|-------|------|
| **MedGemma 1.5 4B** | Image analysis, bounding box localization, question generation, response grading, longitudinal CXR comparison |
| **MedASR** | Voice dictation (58% fewer errors than Whisper on CXR dictation) |
| **CXR Foundation** | Similar case retrieval (ELIXR embeddings, 800K+ CXRs) |
| **MedSigLIP** | Visual similarity search via FAISS |
| **HeAR** | "Listen Then Look" auscultation-to-imaging correlation |

## Training Modes

1. **Confidence Calibration** -- tracks confidence vs accuracy per pathology
2. **Satisfaction of Search** -- find ALL findings, not just the obvious one
3. **Dual-Process Training** -- 3-second flash (System 1) then full review (System 2)
4. **Socratic Mode** -- questions before answers
5. **Contrastive Pairs** -- visually similar pathologies side by side
6. **HeAR Auscultation** -- correlate lung sounds with CXR findings

## Architecture

```
MedGemma 1.5  +  MedASR  +  CXR Foundation  +  MedSigLIP  +  HeAR
       \            |             |                |           /
        └───────────┴─────────────┴────────────────┴──────────┘
                          FSRS-6 Scheduler
                        (21 params, power-law)
                               │
              ┌────────────────┼────────────────┐
         Student State    Bounding Boxes    Diagnostic Landscape
```

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run (requires GPU with 5 HAI-DEF models)
python app.py

# Run in dev/test mode (CPU, no models required)
ENGRAM_USE_MEDGEMMA=false python app.py
```

## Project Structure

```
app.py                  # Gradio application (~1600 lines)
engram/
  fsrs6.py              # FSRS-6 scheduler (21 params, power-law decay)
  medgemma.py           # MedGemma 1.5 4B inference
  medasr.py             # MedASR voice dictation
  cxr_foundation.py     # CXR Foundation retrieval
  retrieval.py          # MedSigLIP FAISS retrieval
  hear.py               # HeAR bioacoustic engine
  longitudinal.py       # Longitudinal CXR comparison
  student.py            # Student state (JSON persistence)
  blindspot.py          # Diagnostic landscape visualization
  dataset.py            # CheXpert dataset loader
  mock_engine.py        # CPU-only mock with clinical knowledge base
notebooks/
  engram_kaggle.ipynb    # Main Kaggle notebook (45 cells)
  engram_lora_finetune.ipynb  # LoRA fine-tuning demo
tests/
  test_engram.py        # 69 unit tests
  test_e2e.py           # 13 end-to-end tests
```

## Stats

- ~9,300 lines of Python
- 82 tests passing
- 5 HAI-DEF models
- 6 training modes
- 11 pathology categories (CheXpert taxonomy)
- Runs offline on CPU or single GPU

## License

Apache 2.0 (code) | HAI-DEF terms (models)
