"""ENGRAM MedASR Voice Dictation Pipeline (HAI-DEF #3)"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class TranscriptionResult:
    text: str; confidence: float

class MedASREngine:
    def __init__(self, model_id="google/medasr"):
        self.model_id, self.model, self.processor, self._loaded = model_id, None, None, False

    def load(self):
        if self._loaded: return
        import torch
        from transformers import AutoProcessor, AutoModelForCTC
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(self.model_id, torch_dtype=torch.float32).to(self.device)
        self._loaded = True

    def transcribe_array(self, audio_array, sr=16000) -> TranscriptionResult:
        import torch
        if isinstance(audio_array, tuple): sr, audio_array = audio_array
        audio = np.array(audio_array, dtype=np.float32)
        if len(audio.shape) > 1: audio = audio.mean(axis=1)
        if np.abs(audio).max() > 0: audio = audio / np.abs(audio).max() # Normalize
        
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt").to(self.device)
        with torch.no_grad():
            ids = torch.argmax(self.model(**inputs).logits, dim=-1)
        return TranscriptionResult(self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip(), 0.95)

class MockMedASREngine:
    def __init__(self): self._loaded = True
    def load(self): pass
    def transcribe_array(self, audio_array, sr=16000) -> TranscriptionResult:
        if isinstance(audio_array, tuple): sr, audio_array = audio_array
        if len(np.array(audio_array)) < 1000: return TranscriptionResult("", 0.0)
        return TranscriptionResult(f"[MedASR mock: {len(audio_array)/max(sr,1):.1f}s dictation captured]", 0.90)