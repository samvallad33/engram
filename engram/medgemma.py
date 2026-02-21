"""
ENGRAM MedGemma Pipeline
Handles medical image interpretation, bounding box localization,
question generation, and clinical feedback using MedGemma 1.5 4B.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np
from PIL import Image

# torch is lazy-imported only when MedGemmaEngine.load() is called
# This allows mock_engine and type imports to work without GPU


@dataclass
class BoundingBox:
    """Bounding box in normalized [0, 1000] coordinates (y0, x0, y1, x1)."""
    y0: float
    x0: float
    y1: float
    x1: float
    label: str

    def to_pixel(self, img_h: int, img_w: int) -> tuple[int, int, int, int]:
        return (
            int(self.y0 / 1000 * img_h),
            int(self.x0 / 1000 * img_w),
            int(self.y1 / 1000 * img_h),
            int(self.x1 / 1000 * img_w),
        )

    def iou(self, other: BoundingBox) -> float:
        """Intersection over Union between two boxes."""
        y0 = max(self.y0, other.y0)
        x0 = max(self.x0, other.x0)
        y1 = min(self.y1, other.y1)
        x1 = min(self.x1, other.x1)

        if y1 <= y0 or x1 <= x0:
            return 0.0

        intersection = (y1 - y0) * (x1 - x0)
        area_self = (self.y1 - self.y0) * (self.x1 - self.x0)
        area_other = (other.y1 - other.y0) * (other.x1 - other.x0)
        union = area_self + area_other - intersection

        return intersection / union if union > 0 else 0.0


@dataclass
class FeedbackResult:
    """MedGemma's assessment of a student's response."""
    score: float                 # 0.0 to 1.0
    correct_findings: list[str]
    missed_findings: list[str]
    false_positives: list[str]
    explanation: str             # Teaching explanation
    box_iou: float              # Spatial accuracy (0-1)


def pad_image_to_square(image: Image.Image) -> Image.Image:
    """Pad image to square — REQUIRED for accurate bounding boxes."""
    img_array = np.array(image)

    # Handle grayscale
    if len(img_array.shape) < 3:
        img_array = np.stack([img_array] * 3, axis=-1)
    # Handle RGBA
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]

    h, w = img_array.shape[:2]

    if h < w:
        dh = w - h
        img_array = np.pad(
            img_array, ((dh // 2, dh - dh // 2), (0, 0), (0, 0)),
            mode="constant",
        )
    elif w < h:
        dw = h - w
        img_array = np.pad(
            img_array, ((0, 0), (dw // 2, dw - dw // 2), (0, 0)),
            mode="constant",
        )

    return Image.fromarray(img_array)


def _strip_thinking_tokens(text: str) -> str:
    """Remove MedGemma thinking traces (<unused95> tokens).
    Defensive: only triggers on 27B thinking variant, not 4B."""
    if "<unused95>" in text:
        text = text.split("<unused95>", 1)[1].lstrip()
    return text


def _parse_boxes(text: str) -> list[BoundingBox]:
    """Parse bounding boxes from MedGemma JSON output."""
    text = _strip_thinking_tokens(text)

    # Try JSON code block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        raw = json_match.group(1)
    elif "Final Answer:" in text:
        raw = text.split("Final Answer:")[-1].strip()
    else:
        # Try parsing entire response
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        raw = bracket_match.group(0) if bracket_match else "[]"

    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        return []

    boxes = []
    for item in items:
        if "box_2d" in item and "label" in item:
            coords = item["box_2d"]
            if len(coords) == 4:
                boxes.append(BoundingBox(
                    y0=coords[0], x0=coords[1],
                    y1=coords[2], x1=coords[3],
                    label=item["label"],
                ))
    return boxes


class MedGemmaEngine:
    """
    MedGemma 1.5 4B inference engine for ENGRAM.
    Handles image analysis, localization, question generation, and feedback.
    Supports optional LoRA adapter via ENGRAM_LORA_PATH env var.
    """

    def __init__(
        self,
        model_id: str = "google/medgemma-1.5-4b-it",
        device: str = "auto",
        lora_path: str | None = None,
    ):
        self.model_id = model_id
        self.device = device
        self.lora_path = lora_path
        self.pipe = None
        self._loaded = False

    def load(self):
        """Load the model (call once at startup). Applies LoRA adapter if path set."""
        if self._loaded:
            return

        import os
        import torch
        from transformers import pipeline

        # Resolve LoRA path from constructor arg or env var
        adapter_path = self.lora_path or os.environ.get("ENGRAM_LORA_PATH")

        if adapter_path and os.path.isdir(adapter_path):
            # Validate adapter directory
            config_file = os.path.join(adapter_path, "adapter_config.json")
            if not os.path.isfile(config_file):
                raise FileNotFoundError(
                    f"ENGRAM_LORA_PATH={adapter_path} is not a valid PEFT adapter "
                    f"(missing adapter_config.json)"
                )

            # Load base model + LoRA adapter directly
            from transformers import AutoModelForImageTextToText, AutoProcessor
            from peft import PeftModel

            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            base_model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                torch_dtype=compute_dtype,
                device_map=self.device,
            )
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
            processor = AutoProcessor.from_pretrained(self.model_id)
            # NOTE: Do NOT pass device_map to pipeline — model is already dispatched
            self.pipe = pipeline(
                "image-text-to-text",
                model=model,
                processor=processor,
                torch_dtype=compute_dtype,
            )
        else:
            compute_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
            self.pipe = pipeline(
                "image-text-to-text",
                model=self.model_id,
                torch_dtype=compute_dtype,
                device_map=self.device,
            )
        self._loaded = True

    def _infer(self, image: Image.Image, prompt: str, max_tokens: int = 2000) -> str:
        """Run a single inference."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        output = self.pipe(text=messages, max_new_tokens=max_tokens, do_sample=False)
        return output[0]["generated_text"][-1]["content"]

    def analyze_image(self, image: Image.Image, category: str = "") -> str:
        """Get a full clinical interpretation of a medical image."""
        image = pad_image_to_square(image)
        prompt = (
            "You are an expert radiologist. Analyze this chest X-ray and provide:\n"
            "1. All significant findings\n"
            "2. Anatomical location of each finding\n"
            "3. Clinical significance\n"
            "4. Differential diagnosis\n"
            "Be thorough and systematic."
        )
        return _strip_thinking_tokens(self._infer(image, prompt))

    def localize_findings(self, image: Image.Image, category: str = "") -> list[BoundingBox]:
        """Get bounding boxes for all findings in the image."""
        image = pad_image_to_square(image)
        prompt = (
            "Locate all anatomical structures and abnormal findings in this chest X-ray.\n\n"
            "Format: bounding box coordinates as [y0, x0, y1, x1] where (y0, x0) is "
            "top-left and (y1, x1) is bottom-right, normalized to range [0, 1000].\n\n"
            "Output as JSON: [{\"box_2d\": [y0, x0, y1, x1], \"label\": \"finding_name\"}]\n\n"
            "Include both normal anatomy and any abnormalities."
        )
        response = self._infer(image, prompt, max_tokens=1000)
        return _parse_boxes(response)

    def generate_question(
        self,
        image: Image.Image,
        category: str,
        difficulty: str = "intermediate",
    ) -> str:
        """Generate a diagnostic question for the student."""
        image = pad_image_to_square(image)
        prompt = (
            f"You are a radiology professor creating a {difficulty}-level "
            f"teaching case about {category}.\n\n"
            "Look at this chest X-ray and generate a clinical question that tests "
            "the student's ability to:\n"
            "1. Identify the key finding(s)\n"
            "2. Describe their anatomical location\n"
            "3. Provide a differential diagnosis\n\n"
            "Write ONLY the question, as you would present it to a medical student. "
            "Include a brief clinical vignette (age, symptoms) to make it realistic."
        )
        return _strip_thinking_tokens(self._infer(image, prompt, max_tokens=500))

    def grade_response(
        self,
        image: Image.Image,
        student_answer: str,
        category: str,
        ground_truth: str = "",
    ) -> FeedbackResult:
        """Grade a student's diagnostic response and provide teaching feedback."""
        image = pad_image_to_square(image)
        prompt = (
            "You are an attending radiologist evaluating a medical student's interpretation.\n\n"
            f"**Known findings:** {ground_truth}\n"
            f"**Student's answer:** {student_answer}\n\n"
            "Evaluate the student's response. Output ONLY valid JSON:\n"
            "```json\n"
            "{\n"
            '  "score": 0.0-1.0,\n'
            '  "correct_findings": ["list of things they got right"],\n'
            '  "missed_findings": ["list of things they missed"],\n'
            '  "false_positives": ["list of incorrect claims"],\n'
            '  "explanation": "Teaching explanation as if speaking to the student. '
            "Be encouraging but thorough. Explain WHY the findings look the way they do.\"\n"
            "}\n"
            "```"
        )
        response = _strip_thinking_tokens(self._infer(image, prompt, max_tokens=1500))

        # Parse JSON
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            raw = json_match.group(1)
        else:
            raw = response

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return FeedbackResult(
                score=0.5,
                correct_findings=[],
                missed_findings=[],
                false_positives=[],
                explanation=response,
                box_iou=0.0,
            )

        return FeedbackResult(
            score=float(data.get("score", 0.5)),
            correct_findings=data.get("correct_findings", []),
            missed_findings=data.get("missed_findings", []),
            false_positives=data.get("false_positives", []),
            explanation=data.get("explanation", ""),
            box_iou=0.0,
        )

    # ─── F2: Socratic Mode ────────────────────────────────────────

    def generate_socratic_question(
        self, image: Image.Image | None, student_answer: str, category: str,
    ) -> str:
        """Generate a Socratic probing question using MedGemma inference."""
        if image is not None:
            prompt = (
                "You are a radiology professor using the Socratic method. "
                f"A student described this chest X-ray (category: {category}).\n\n"
                f"Student's answer: {student_answer}\n\n"
                "Ask ONE probing question that guides them toward any findings "
                "they may have missed. Do not reveal the answer directly.\n\n"
                "Format your response as:\n"
                "**Socratic Question:**\n\n[your question]\n\n"
                "*Think about this before seeing the full answer.*"
            )
            return _strip_thinking_tokens(self._infer(image, prompt, max_tokens=300))
        return (
            f"**Socratic Question:**\n\nWhat other findings associated with "
            f"{category} should you consider in this region?\n\n"
            f"*Think about this before seeing the full answer.*"
        )

    def generate_socratic_followup(
        self, student_answer: str, socratic_response: str, category: str,
    ) -> str:
        """Evaluate student's Socratic response."""
        response_text = (socratic_response or "").strip()
        if len(response_text) > 30:
            return (
                "**Good thinking!** You're engaging with the question critically. "
                "Let's see the full expert analysis."
            )
        return (
            "**Consider more carefully.** Take a moment to think about the key "
            f"features of {category}. Let me show you the complete picture."
        )

    # ─── F3: Satisfaction of Search ────────────────────────────────

    def grade_search_completeness(
        self, student_answer: str, category: str,
    ) -> tuple[list[str], list[str], float]:
        """Grade search completeness. For real engine, this is derived from
        grade_response() results in app.py. Kept for interface compatibility."""
        return [], [], 0.0

    # ─── F4: Dual-Process Gestalt Grading ──────────────────────────

    GESTALT_KEYWORDS = {
        "Cardiomegaly": ["heart", "enlarged", "big", "cardio", "large"],
        "Pneumothorax": ["pneumo", "air", "collapsed", "lung"],
        "Pleural Effusion": ["fluid", "effusion", "white", "base"],
        "Lung Opacity": ["opacity", "white", "hazy", "shadow"],
        "Consolidation": ["consolidation", "solid", "dense", "white"],
        "Atelectasis": ["collapse", "atelectasis", "volume"],
        "Edema": ["edema", "fluid", "hazy", "bilateral"],
        "Fracture": ["fracture", "broken", "rib", "break"],
        "No Finding": ["normal", "clear", "nothing", "unremarkable"],
        "Pneumonia": ["pneumonia", "infection", "consolidation"],
        "Support Devices": ["tube", "line", "device", "wire"],
    }

    def grade_gestalt(self, student_answer: str, category: str) -> float:
        """Grade a rapid gestalt impression (System 1). Fast keyword matching
        to avoid inference delay — gestalt is about instant recognition."""
        answer_lower = (student_answer or "").lower()
        if not answer_lower.strip():
            return 0.0
        keywords = self.GESTALT_KEYWORDS.get(category, [category.lower()])
        matches = sum(1 for kw in keywords if kw in answer_lower)
        if matches >= 2:
            return 0.85
        elif matches >= 1:
            return 0.6
        return 0.15

    # ─── F5: Contrastive Case Pairs ────────────────────────────────

    CONTRASTIVE_PAIRS = {
        ("Consolidation", "Atelectasis"): {
            "question": "Both images show opacification. One is consolidation, the other atelectasis. What is the KEY distinguishing feature?",
            "key_difference": "Volume loss. Atelectasis has volume loss (elevated diaphragm, mediastinal shift toward opacity, fissure displacement). Consolidation does not.",
            "keywords": ["volume loss", "shift", "collapse", "fissure", "diaphragm"],
        },
        ("Pleural Effusion", "Lung Opacity"): {
            "question": "Both show whiteness at the base. One is effusion, the other parenchymal opacity. How do you tell them apart?",
            "key_difference": "Meniscus sign. Effusions are gravity-dependent with a curved upper border (meniscus). Parenchymal opacities don't layer with position change.",
            "keywords": ["meniscus", "gravity", "layer", "decubitus", "costophrenic"],
        },
        ("Cardiomegaly", "Edema"): {
            "question": "Both cases involve the heart and lungs. One shows cardiomegaly alone, the other pulmonary edema. What features distinguish them?",
            "key_difference": "Cardiomegaly is heart-size only (CTR > 0.5). Edema shows lung findings: cephalization, Kerley B lines, bilateral haziness, peribronchial cuffing.",
            "keywords": ["cephalization", "kerley", "lung", "ctr", "haziness"],
        },
        ("Pneumothorax", "No Finding"): {
            "question": "One of these films has a subtle pneumothorax. The other is normal. Can you identify which is which and why?",
            "key_difference": "Look for the visceral pleural line — a thin white line parallel to the chest wall with absent lung markings beyond it. Normal films have vascular markings extending to the periphery.",
            "keywords": ["pleural line", "lung markings", "peripheral", "visceral"],
        },
        ("Pneumonia", "Consolidation"): {
            "question": "Both show dense opacification. One is typical bacterial pneumonia, the other a non-infectious consolidation. What clinical and imaging clues help?",
            "key_difference": "Imaging alone often cannot distinguish them. Clinical context is key: fever + acute onset suggests infection. Air bronchograms appear in both. Follow-up imaging showing resolution with antibiotics confirms pneumonia.",
            "keywords": ["clinical", "fever", "follow-up", "resolution", "antibiotics"],
        },
        ("Edema", "Pneumonia"): {
            "question": "Both show bilateral opacities. One is pulmonary edema, the other bilateral pneumonia. What patterns help distinguish them?",
            "key_difference": "Edema is typically symmetric, perihilar (bat-wing), with cephalization and often cardiomegaly. Pneumonia tends to be asymmetric, lobar, with air bronchograms and no cephalization.",
            "keywords": ["symmetric", "perihilar", "asymmetric", "cardiomegaly", "cephalization"],
        },
    }

    def generate_contrastive_question(self, category_a: str, category_b: str) -> str:
        """Generate a contrastive comparison question."""
        pair = self.CONTRASTIVE_PAIRS.get(
            (category_a, category_b),
            self.CONTRASTIVE_PAIRS.get((category_b, category_a)),
        )
        if pair:
            return pair["question"]
        return (
            f"Compare these two cases. One is {category_a}, the other is "
            f"{category_b}. What is the key distinguishing feature?"
        )

    def grade_contrastive(
        self, student_answer: str, category_a: str, category_b: str,
    ) -> FeedbackResult:
        """Grade contrastive pair discrimination."""
        pair = self.CONTRASTIVE_PAIRS.get(
            (category_a, category_b),
            self.CONTRASTIVE_PAIRS.get((category_b, category_a)),
        )
        answer_lower = (student_answer or "").lower()

        if pair:
            keywords = pair["keywords"]
            matches = sum(1 for kw in keywords if kw in answer_lower)
            if matches >= 2:
                score = 0.85
            elif matches >= 1:
                score = 0.6
            else:
                score = 0.2
            explanation = f"**Key Difference:** {pair['key_difference']}"
            correct = [kw for kw in keywords if kw in answer_lower]
            missed = [kw for kw in keywords if kw not in answer_lower]
        else:
            score = 0.5
            explanation = (
                f"Compare the characteristic features of {category_a} "
                f"versus {category_b}."
            )
            correct, missed = [], []

        return FeedbackResult(
            score=min(1.0, score),
            correct_findings=correct,
            missed_findings=missed[:3],
            false_positives=[],
            explanation=explanation,
            box_iou=0.0,
        )
