"""
ENGRAM MedGemma Pipeline
Handles medical image interpretation, bounding box localization, and feedback.
"""
from __future__ import annotations
import json, re
from dataclasses import dataclass
from PIL import Image, ImageOps

@dataclass
class BoundingBox:
    y0: float; x0: float; y1: float; x1: float; label: str
    def to_pixel(self, img_h: int, img_w: int) -> tuple[int, int, int, int]:
        return (int(self.y0 / 1000 * img_h), int(self.x0 / 1000 * img_w), int(self.y1 / 1000 * img_h), int(self.x1 / 1000 * img_w))

@dataclass
class FeedbackResult:
    score: float; correct_findings: list[str]; missed_findings: list[str]; false_positives: list[str]; explanation: str; box_iou: float

def pad_image_to_square(img: Image.Image) -> Image.Image:
    """Pad image to square (REQUIRED for MedGemma box accuracy)."""
    s = max(img.size)
    return ImageOps.pad(img.convert("RGB"), (s, s), method=Image.Resampling.LANCZOS, color=(0, 0, 0))

def _parse_boxes(text: str) -> list[BoundingBox]:
    text = text.split("<unused95>", 1)[-1].lstrip() if "<unused95>" in text else text
    raw = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    raw_str = raw.group(1) if raw else (re.search(r"\[.*\]", text, re.DOTALL).group(0) if re.search(r"\[.*\]", text, re.DOTALL) else "[]")
    try:
        return [BoundingBox(*i["box_2d"], i["label"]) for i in json.loads(raw_str) if "box_2d" in i and len(i["box_2d"]) == 4]
    except Exception: return []

class MedGemmaEngine:
    def __init__(self, model_id="google/medgemma-1.5-4b-it", device="auto", lora_path=None):
        self.model_id, self.device, self.lora_path, self.pipe, self._loaded = model_id, device, lora_path, None, False

    def load(self):
        if self._loaded: return
        import os, torch
        from transformers import pipeline, AutoModelForImageTextToText, AutoProcessor
        
        adapter = self.lora_path or os.environ.get("ENGRAM_LORA_PATH")
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        
        if adapter and os.path.isdir(adapter):
            from peft import PeftModel
            base = AutoModelForImageTextToText.from_pretrained(self.model_id, torch_dtype=dtype, device_map=self.device)
            model = PeftModel.from_pretrained(base, adapter).eval()
            self.pipe = pipeline("image-text-to-text", model=model, processor=AutoProcessor.from_pretrained(self.model_id), torch_dtype=dtype)
        else:
            self.pipe = pipeline("image-text-to-text", model=self.model_id, torch_dtype=dtype, device_map=self.device)
        self._loaded = True

    def _infer(self, img: Image.Image, prompt: str, tokens: int = 1500) -> str:
        out = self.pipe(text=[{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}], max_new_tokens=tokens, do_sample=False)[0]["generated_text"][-1]["content"]
        return out.split("<unused95>", 1)[-1].lstrip() if "<unused95>" in out else out

    def localize_findings(self, img: Image.Image, cat: str = "") -> list[BoundingBox]:
        return _parse_boxes(self._infer(pad_image_to_square(img), "Locate findings. Output JSON: [{\"box_2d\": [y0, x0, y1, x1], \"label\": \"name\"}]", 1000))

    def generate_question(self, img: Image.Image, cat: str) -> str:
        return self._infer(pad_image_to_square(img), f"Create a clinical teaching vignette asking the student to interpret {cat} findings.", 500)

    def grade_response(self, img: Image.Image, ans: str, cat: str, truth: str = "") -> FeedbackResult:
        res = self._infer(pad_image_to_square(img), f"Grade student. Truth: {truth}\nStudent: {ans}\nOutput JSON: score (0-1), correct_findings, missed_findings, false_positives, explanation")
        raw = re.search(r"```json\s*(.*?)\s*```", res, re.DOTALL)
        data = json.loads(raw.group(1)) if raw else {}
        return FeedbackResult(float(data.get("score", 0.5)), data.get("correct_findings", []), data.get("missed_findings", []), data.get("false_positives", []), data.get("explanation", res), 0.0)

    def generate_socratic_question(self, img: Image.Image | None, ans: str, cat: str) -> str:
        if img: return self._infer(pad_image_to_square(img), f"Student said: {ans}. Ask ONE Socratic probing question about {cat} findings they missed. Do not reveal answer.", 300)
        return f"**Socratic Question:** What other features of {cat} should you look for?"