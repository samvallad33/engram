# ENGRAM — Novel Task Special Award Submission

**Kaggle MedGemma Impact Challenge 2026**
**Sam Vallad | February 2026**

---

## Novel Task: Memory-Optimized Medical Diagnosis Training

### What makes this task novel?

ENGRAM applies MedGemma to a task it was never designed for: **adaptive spaced repetition for medical visual diagnosis training**. This is not image classification, not report generation, not clinical decision support — it is a fundamentally new application: using a medical vision-language model as a personalized tutor within a scientifically optimized memory scheduling framework.

### The Novel Combination

To our knowledge, no prior system combines these three elements:

1. **FSRS-6 spaced repetition** (21-parameter power-law forgetting model, demonstrated +3.8% MATH-500 in LUMIA) — determines *when* each student reviews each pathology based on their individual forgetting curve.

2. **MedGemma 1.5 as adaptive expert tutor** — not just analyzing images, but generating clinical questions, grading free-text responses, localizing findings with bounding boxes, and comparing longitudinal studies. Five distinct MedGemma tasks orchestrated per student interaction.

3. **Multi-model clinical workflow** — Five HAI-DEF models working in concert: MedGemma teaches, MedASR transcribes dictation, CXR Foundation retrieves similar cases, MedSigLIP provides visual matching, and HeAR bridges auscultation and imaging. Each model targets a specific cognitive skill that real radiologists use daily.

### Why This Task Matters

Medical education faces a structural bottleneck: attending radiologist time. A student needs thousands of deliberate practice encounters to develop diagnostic competency, but feedback is scarce and inconsistent. ENGRAM makes expert-level feedback infinitely scalable by combining AI assessment with optimal memory scheduling.

The cognitive science is specific and evidence-based:
- **Satisfaction of Search** training targets a leading radiology cognitive bias (22% of errors per Kim & Mansfield, 2014)
- **Confidence Calibration** tracks and corrects overconfidence per pathology category
- **Dual-Process Training** develops both System 1 (gestalt) and System 2 (analytical) reasoning
- **Contrastive Pairs** train discrimination between visually similar pathologies

### Technical Novelty

**LoRA fine-tuning for diagnostic grading** — We demonstrate that MedGemma can be parameter-efficiently adapted (LoRA, rank-16, ~4.2M trainable parameters / 0.1% of 4B model) to grade student diagnostic responses using FSRS-6 review history as training signal. The student's learning data becomes the model's training data — a novel co-evolutionary loop where the tutor improves from teaching.

**Simulated effectiveness study** — Our notebook demonstrates measurable learning improvement: FSRS-6 adaptive scheduling achieves significantly higher retention than random review across 50 simulated students, mirroring real spaced repetition research outcomes.

### Competitive Uniqueness

Among 124+ teams in this competition, ENGRAM occupies a unique position in the medical education space. While most teams apply MedGemma to clinical workflows (diagnosis, triage, report generation), ENGRAM uses MedGemma to *teach humans* — making the model's knowledge transferable to the next generation of physicians.

### Summary

| Aspect | ENGRAM's Novel Contribution |
|---|---|
| **Task** | Adaptive medical diagnosis training (not classification/diagnosis) |
| **Algorithm** | FSRS-6 spaced repetition (demonstrated in LUMIA, now applied to medicine) |
| **Models** | 5 HAI-DEF models in a single clinical training workflow |
| **Cognitive Science** | 6 training modes targeting specific diagnostic failure patterns |
| **Data Loop** | Student annotations → model fine-tuning (LoRA) → better teaching |
| **Unique Positioning** | Medical education focus among 124+ teams |

ENGRAM doesn't just use MedGemma for a new task — it creates an entirely new category of medical AI application: **the AI-powered, memory-optimized clinical tutor**.
