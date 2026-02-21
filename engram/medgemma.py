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
    """

    def __init__(self, model_id: str = "google/medgemma-1.5-4b-it", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.pipe = None
        self._loaded = False

    def load(self):
        """Load the model (call once at startup)."""
        if self._loaded:
            return

        import torch
        from transformers import pipeline

        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            torch_dtype=torch.float16,
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
