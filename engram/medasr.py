"""
ENGRAM MedASR Voice Dictation Pipeline
Transcribes medical speech using Google's MedASR (HAI-DEF model #3).

Real radiologists don't type — they dictate. MedASR achieves 58% fewer
errors than Whisper large-v3 on chest X-ray dictations (5.2% vs 12.5% WER).

Pipeline: Student speaks → MedASR transcribes → MedGemma grades
"""

from __future__ import annotations

from dataclasses import dataclass

# torch + transformers lazy-imported in load()


@dataclass
class TranscriptionResult:
    """Result of MedASR transcription."""
    text: str
    confidence: float  # 0.0-1.0


class MedASREngine:
    """
    MedASR speech-to-text engine for medical dictation.
    105M parameter Conformer model trained on ~5000 hours of physician dictations.
    """

    def __init__(self, model_id: str = "google/medasr", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.processor = None
        self.model = None
        self._loaded = False

    def load(self):
        """Load MedASR model."""
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForCTC

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForCTC.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
        ).to(device)
        self.device = device
        self._loaded = True

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Transcribe audio file to text using MedASR."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        import torch
        import numpy as np

        # Load audio
        try:
            import librosa
            waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        except ImportError:
            # Fallback: try soundfile
            import soundfile as sf
            waveform, sr = sf.read(audio_path)
            if sr != 16000:
                # Simple resampling
                import scipy.signal
                waveform = scipy.signal.resample(
                    waveform, int(len(waveform) * 16000 / sr)
                )

        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return TranscriptionResult(
            text=transcription.strip(),
            confidence=0.95,  # MedASR typically high confidence
        )

    def transcribe_array(self, audio_array, sample_rate: int = 16000) -> TranscriptionResult:
        """Transcribe from numpy array (for Gradio audio input)."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        import torch
        import numpy as np

        # Ensure mono float32
        if isinstance(audio_array, tuple):
            sample_rate, audio_array = audio_array
        audio_array = np.array(audio_array, dtype=np.float32)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Normalize to [-1, 1]
        max_val = np.abs(audio_array).max()
        if max_val > 0:
            audio_array = audio_array / max_val

        inputs = self.processor(
            audio_array, sampling_rate=sample_rate, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        transcription = self.processor.batch_decode(
            predicted_ids, skip_special_tokens=True
        )[0]

        return TranscriptionResult(
            text=transcription.strip(),
            confidence=0.95,
        )


class MockMedASREngine:
    """
    Mock MedASR for CPU demo mode.
    Returns the text as-is (since we can't actually record audio in mock mode,
    the Gradio audio input will provide the waveform which we 'transcribe').
    """

    def __init__(self):
        self._loaded = True

    def load(self):
        pass

    def transcribe(self, audio_path: str) -> TranscriptionResult:
        """Mock transcription — returns placeholder."""
        return TranscriptionResult(
            text="[Voice dictation captured — MedASR transcription would appear here in GPU mode]",
            confidence=0.90,
        )

    def transcribe_array(self, audio_array, sample_rate: int = 16000) -> TranscriptionResult:
        """Mock transcription from audio array."""
        import numpy as np

        if isinstance(audio_array, tuple):
            sample_rate, audio_array = audio_array
        audio_array = np.array(audio_array, dtype=np.float32)

        # Check if there's actual audio data
        if len(audio_array) < 1000:
            return TranscriptionResult(text="", confidence=0.0)

        duration = len(audio_array) / max(sample_rate, 1)
        return TranscriptionResult(
            text=(
                f"[MedASR mock: {duration:.1f}s audio captured. "
                f"In GPU mode, MedASR (105M params, 58% fewer errors than Whisper) "
                f"would transcribe your radiological dictation here.]"
            ),
            confidence=0.90,
        )
