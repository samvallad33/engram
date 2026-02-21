# ENGRAM — 3-Minute Demo Video Script

**Kaggle MedGemma Impact Challenge 2026**
**Sam Vallad | February 2026**

---

## [0:00 — THE PERSONAL HOOK] *(20 seconds)*

*[Screen: Black. Typing cursor blinks. Code scrolls slowly in background.]*

A year ago I couldn't remember anything that mattered. So I built a memory system — sixty-two thousand lines of Rust, a twenty-one-parameter algorithm that models exactly how humans forget. I called it Vestige.

Last month, that same algorithm improved LLM training by 3.8 percent on MATH-500.

Then I read a statistic that changed what I was building.

---

## [0:20 — THE PROBLEM] *(25 seconds)*

*[Screen: Black. White text fades in, one line at a time.]*

> Ten million healthcare workers short by 2030.
>
> Diagnostic errors in 10% of patient deaths.
>
> A radiology student needs thousands of practice cases to become competent.
>
> They get a handful — shown randomly, graded inconsistently, forgotten within weeks.

*[Beat. Text clears.]*

What if the algorithm that models human forgetting could fix how medical students learn to see?

*[ENGRAM logo appears.]*

---

## [0:45 — THE LIVE DEMO] *(90 seconds)*

*[Screen: ENGRAM running live on RunPod RTX 5090. Gradio interface.]*

This is ENGRAM. Five Google foundation models. One memory algorithm. Running live.

*[Click "Start Session." CXR appears with clinical question.]*

**FSRS-6 picks your next case** — not randomly, but at the exact moment your memory is about to fail. This student hasn't seen Atelectasis in six days. Retention has dropped to 68%. Now is the optimal moment to review.

*[Type a student answer. Click Submit.]*

**MedGemma 1.5 grades your response in real time** — what you got right, what you missed, and why. Then it draws bounding boxes directly on the image: *here* is the finding you overlooked.

*[Scroll down to show FSRS-6 memory update table, search completeness, consensus panel.]*

But grading isn't enough. Diagnostic errors aren't knowledge failures — they're cognitive failures. So ENGRAM has six training modes built on radiology education research.

*[Quick cuts through each mode — 8 seconds each:]*

**Satisfaction of Search** — did you find ALL the findings, or did you stop after the first one? Twenty-two percent of radiology errors come from stopping too early.

**Confidence Calibration** — your confidence was 4 out of 5, but your accuracy was 45%. FSRS-6 shortens your review interval. Overconfidence kills.

**Contrastive Pairs** — consolidation and atelectasis look identical. Both show white. One has volume loss. Can you tell which?

**Socratic Mode** — no answers. Just a question that makes you think harder.

**Dual-Process** — three-second flash. What did you see? That's your System 1. Now take your time. That's System 2. ENGRAM trains both.

**Listen Then Look** — HeAR plays lung sounds from 313 million audio clips. You hear crackles. You predict consolidation. Then you see the X-ray. This is how real clinical reasoning works.

*[Show the Diagnostic Landscape — blind spot heatmap, forgetting curves, learning velocity.]*

Every review updates your personal forgetting curve across eleven pathology categories. ENGRAM knows exactly where you're strong and exactly where you're dangerously weak.

---

## [2:15 — THE FLYWHEEL] *(25 seconds)*

*[Screen: Architecture diagram — the co-evolutionary data flywheel.]*

Now the part nobody else has.

Students review cases. FSRS-6 measures which cases are hardest. Those difficulty signals become **fine-tuning weights for MedGemma itself** — a LoRA adapter trained on human memory data.

The cases students struggle with most get the highest training emphasis. The model gets better at explaining exactly what's hard. Students learn faster. New difficulty data flows back. The flywheel spins.

*[Show notebook: QLoRA, 1000 FSRS-weighted examples, curriculum ordering.]*

Every existing curriculum learning system uses model-internal signals — loss, gradients. ENGRAM is the first to use **human memory parameters** as VLM fine-tuning weights.

---

## [2:40 — THE CLOSE] *(20 seconds)*

*[Screen fades to black. Pause.]*

*[White text, one line at a time:]*

> 9,300 lines of Python. 82 tests. Runs offline on a single GPU.
>
> Five models. Twenty-one parameters. Six training modes.
>
> No cloud. No data leaves the device. Works anywhere a medical student has a laptop.

*[Final beat. Sam speaking directly to camera.]*

I built Vestige to help myself remember. Then I showed it makes AI smarter. Now I'm showing it makes doctors better — before the shortage costs more lives.

*[ENGRAM logo. Fade to black.]*

---

**[END — 3:00]**

*~460 spoken words. Target pace: 155 wpm = 2:58.*
*Live demo is the centerpiece — 90 seconds of real ENGRAM on RunPod.*

---

## Production Notes

### Screen Recording Checklist
- [ ] RunPod RTX 5090 with ENGRAM running (`ENGRAM_USE_MEDGEMMA=true`)
- [ ] Demo CXR images loaded (11 categories, 33 images in `data/demo/`)
- [ ] Record at 1080p, dark Gradio theme
- [ ] Show terminal briefly at start: `ENGRAM v0.4.0 | Mode: 5 HAI-DEF Models (GPU)`
- [ ] Have pre-typed student answers ready (paste, don't type live)
- [ ] For Contrastive Pairs: use Consolidation vs Atelectasis (most dramatic)
- [ ] For Auscultation: use Pneumonia category (crackles → consolidation correlation)
- [ ] Show FSRS-6 memory state table with real S/D/R values
- [ ] Show forgetting curves with actual power-law decay visualization

### Key Numbers (all verifiable in codebase)
- 62,000 lines Rust (Vestige)
- 9,300 lines Python (ENGRAM)
- 82 tests passing
- 5 HAI-DEF models
- 6 training modes
- 21 FSRS-6 parameters
- 11 CheXpert categories
- 4.2M trainable LoRA params (0.1% of 4B)
- 22% of radiology errors from satisfaction of search (Kim & Mansfield, 2014)
- 10M healthcare worker shortage by 2030 (WHO)
- 10% of patient deaths involve diagnostic error (Balogh et al., NAP 2015)
- +3.8% MATH-500 (LUMIA, Feb 2026)
