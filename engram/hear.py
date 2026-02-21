"""
ENGRAM HeAR Integration (HAI-DEF Model #5)
Classifies respiratory sounds using HeAR embeddings.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class LungSoundCase:
    category: str; sound_type: str; description: str; correlation: str

LUNG_SOUNDS = {
    "Cardiomegaly": {"sound": "S3 gallop", "location": "apex", "character": "Low-pitched third heart sound", "correlation": "S3 gallop indicates pulmonary congestion seen on CXR."},
    "Pneumothorax": {"sound": "absent", "location": "affected side", "character": "No breath sounds", "correlation": "Absent sounds correlate with visceral pleural line on CXR."},
    "Consolidation": {"sound": "bronchial", "location": "over consolidation", "character": "Harsh, tubular", "correlation": "Bronchial sounds indicate solid lung tissue (air bronchograms)."},
    "Edema": {"sound": "crackles", "location": "bilateral bases", "character": "Fine inspiratory crackles", "correlation": "Crackles correlate with perihilar haziness and Kerley B lines."},
    "No Finding": {"sound": "vesicular", "location": "all fields", "character": "Normal rustling", "correlation": "Normal sounds correlate with clear lung fields."}
}

class HeAREngine:
    def __init__(self, model_id="google/hear-pytorch"):
        self.model_id, self.model, self._loaded, self._refs = model_id, None, False, {}

    def load(self):
        if self._loaded: return
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.model_id).eval()
            self._loaded = True
        except Exception: pass

    def get_embedding(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray | None:
        if not self.model: return None
        try:
            import torch, torchaudio
            audio = np.pad(audio, (0, max(0, sr*2 - len(audio))))[:sr*2] # Force 2 seconds
            mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=400, hop_length=160, n_mels=128)(torch.tensor(audio).unsqueeze(0).float())
            with torch.no_grad():
                out = self.model(mel)
                return out.pooler_output[0].numpy() if hasattr(out, 'pooler_output') and out.pooler_output is not None else out.last_hidden_state.mean(dim=1)[0].numpy()
        except Exception: return None


class MockHeAREngine:
    """Mock HeAR for local dev without GPU."""
    def __init__(self): self._loaded = False
    def load(self): self._loaded = True

    def generate_lung_sound(self, category: str) -> tuple[int, np.ndarray]:
        data = LUNG_SOUNDS.get(category, LUNG_SOUNDS["No Finding"])
        freq = {"S3 gallop": 60, "absent": 0, "bronchial": 500, "crackles": 800, "vesicular": 150}.get(data["sound"], 200)
        t = np.linspace(0, 2, 32000, dtype=np.float32)
        audio = (0.3 * np.sin(2 * np.pi * freq * t) + 0.05 * np.random.randn(32000)).astype(np.float32) if freq > 0 else np.zeros(32000, dtype=np.float32)
        return 16000, audio

    def classify_lung_sound(self, audio: np.ndarray) -> dict[str, float]:
        labels = ["crackles", "wheezing", "normal", "stridor"]
        vals = np.random.dirichlet(np.ones(len(labels)))
        return dict(zip(labels, vals.tolist()))

    def get_lung_sound_case(self, category: str) -> LungSoundCase:
        data = LUNG_SOUNDS.get(category, LUNG_SOUNDS["No Finding"])
        return LungSoundCase(category, data["sound"], data["character"], data["correlation"])