# ENGRAM — 3-Minute Presentation Script

**Kaggle MedGemma Impact Challenge 2026**
**Sam Vallad | February 2026**

---

## [0:00 — THE HOOK]

*[Screen: Dark background. Personal photo fades in.]*

I built a memory system to solve a personal problem — how do you remember what matters? That led to Vestige: sixty-two thousand lines of Rust implementing FSRS-6, a twenty-one-parameter spaced repetition algorithm. Last month, FSRS-6 improved LLM training by 3.8 percent on MATH-500.

Then I asked: what if the same algorithm that makes AI smarter could make medical students better diagnosticians?

*[Screen: WHO stat fades in.]*

The WHO projects a shortage of ten million healthcare workers by 2030. Diagnostic errors cause an estimated ten percent of patient deaths. Learning to read medical images takes thousands of practice cases — but current tools show them randomly, with no spatial training, no voice dictation, and no memory science. Until now.

---

## [0:40 — FIVE MODELS, ONE ALGORITHM]

*[Screen: ENGRAM logo + architecture diagram]*

ENGRAM combines FSRS-6 with five HAI-DEF foundation models to teach students not just *what* to see, but *where* to look, *when* to review, and *how* to think.

**MedGemma 1.5** grades responses, draws bounding boxes on findings, and compares current images against prior studies — its flagship longitudinal capability.

**MedASR** enables voice dictation — because real radiologists don't type. Fifty-eight percent fewer errors than Whisper on medical speech.

**CXR Foundation** retrieves similar cases from embeddings trained on 800,000 chest X-rays.

**MedSigLIP** provides complementary visual similarity search.

And **HeAR** enables "Listen Then Look" — students hear lung sounds, predict what the X-ray shows, then see the image. Crackles predict consolidation. Absent breath sounds predict pneumothorax. This is how real clinical reasoning works.

---

## [1:30 — COGNITIVE TRAINING]

*[Screen: Six training mode icons]*

Five models aren't enough. Diagnostic errors are cognitive failures. So ENGRAM includes six evidence-based training modes.

**Confidence Calibration** catches overconfidence per pathology. **Satisfaction of Search** forces students to find *all* findings — targeting the bias behind 22 percent of radiology errors. **Dual-Process Training** flashes images for three seconds to build System 1 pattern recognition. **Socratic Mode** asks questions instead of giving answers. **Contrastive Pairs** present look-alike pathologies side by side. And **HeAR Auscultation** bridges hearing and seeing.

Everything feeds into the **Diagnostic Landscape** — a real-time map of each student's strengths, blind spots, and forgetting curves across eleven pathology categories.

---

## [2:15 — EVIDENCE & CLOSE]

*[Screen: Effectiveness chart + stats]*

We simulated fifty students across twenty reviews. FSRS-6 scheduling achieved significantly higher retention than random review — the difference between cramming and real learning.

ENGRAM runs offline on a single GPU or CPU-only. No cloud. No data leaves the device. It works anywhere a medical student has a laptop.

*[Screen: Dark background. Final line fades in.]*

I built a memory system to help myself remember. Then I showed it makes AI smarter. Now I'm showing it makes doctors better.

ENGRAM. Five models. Twenty-one parameters. One mission: close the diagnostic training gap before it costs more lives.

*[Screen: ENGRAM logo]*

---

**[END — 3:00]**

*~415 spoken words. Target pace: 140 wpm = 2:58.*
