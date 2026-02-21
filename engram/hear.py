"""
ENGRAM HeAR Integration — Health Acoustic Representations (HAI-DEF Model #5)
Correlates lung sounds with chest X-ray findings.

HeAR is a ViT-L bioacoustic foundation model trained on 313 million
two-second audio clips. It classifies respiratory sounds (crackles,
wheezing, diminished breath sounds) to train students on the
auscultation-to-imaging correlation — the real clinical workflow.

"Listen then Look" — hear the patient, predict the CXR, then see it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LungSoundCase:
    """A lung sound training case."""
    category: str           # CXR pathology category
    sound_type: str         # e.g. "crackles", "wheezing", "diminished"
    description: str        # Clinical description
    correlation: str        # How sound correlates with CXR findings


# Maps CXR categories to expected auscultation findings
LUNG_SOUNDS = {
    "Cardiomegaly": {
        "sound": "S3 gallop",
        "location": "apex, left lateral decubitus",
        "character": "Low-pitched third heart sound with possible bilateral basilar crackles",
        "correlation": "Cardiomegaly often accompanies heart failure. Listen for S3 gallop and bilateral basilar crackles indicating pulmonary congestion.",
    },
    "Pneumothorax": {
        "sound": "absent",
        "location": "affected hemithorax",
        "character": "No breath sounds on the affected side; hyperresonant percussion",
        "correlation": "Absent breath sounds on one side with hyperresonance strongly suggests pneumothorax. The CXR confirms with a visible visceral pleural line.",
    },
    "Pleural Effusion": {
        "sound": "diminished",
        "location": "bilateral bases",
        "character": "Decreased breath sounds at bases; dullness to percussion; egophony at fluid level",
        "correlation": "Dullness to percussion and diminished breath sounds at the base correlate with the homogeneous opacity and meniscus sign on CXR.",
    },
    "Lung Opacity": {
        "sound": "crackles",
        "location": "over opacity region",
        "character": "Coarse crackles or bronchial breath sounds over the opacity",
        "correlation": "Crackles over a lung opacity suggest fluid-filled airspaces. The CXR shows the corresponding area of increased density.",
    },
    "Consolidation": {
        "sound": "bronchial",
        "location": "over consolidation",
        "character": "Bronchial breath sounds, increased tactile fremitus, egophony (E-to-A change)",
        "correlation": "Bronchial breath sounds heard peripherally indicate consolidation — sound transmits through the solid lung. CXR shows dense opacity with air bronchograms.",
    },
    "Atelectasis": {
        "sound": "diminished",
        "location": "over collapsed segment",
        "character": "Decreased breath sounds with possible tracheal deviation toward the affected side",
        "correlation": "Diminished sounds with tracheal deviation toward the opacity = atelectasis (volume loss). CXR shows elevated diaphragm and mediastinal shift.",
    },
    "Edema": {
        "sound": "crackles",
        "location": "bilateral bases, progressing upward",
        "character": "Fine, bibasilar, inspiratory crackles; may have wheezing (cardiac asthma)",
        "correlation": "Bilateral basilar crackles correlate with the bilateral perihilar haziness on CXR. Crackles progress from bases upward as edema worsens.",
    },
    "Fracture": {
        "sound": "crepitus",
        "location": "over fracture site",
        "character": "Point tenderness with palpable crepitus; breath sounds may be normal unless complicated",
        "correlation": "Auscultation is often normal with simple rib fractures. Crepitus on palpation localizes the fracture site seen on CXR.",
    },
    "No Finding": {
        "sound": "vesicular",
        "location": "bilateral, all fields",
        "character": "Normal vesicular breath sounds; soft, low-pitched, heard throughout inspiration",
        "correlation": "Normal bilateral vesicular breath sounds correlate with clear lung fields on CXR. This is the baseline you must know.",
    },
    "Pneumonia": {
        "sound": "crackles",
        "location": "over affected lobe",
        "character": "Coarse inspiratory crackles; bronchial breath sounds if consolidated; possible pleural rub",
        "correlation": "Focal crackles with bronchial breathing predict lobar pneumonia on CXR. Bilateral crackles suggest multilobar or atypical pneumonia.",
    },
    "Support Devices": {
        "sound": "variable",
        "location": "depends on device position",
        "character": "Check for bilateral and equal breath sounds post-intubation; absent sounds on one side suggests mainstem intubation",
        "correlation": "After ETT placement, auscultate bilaterally. Unilateral breath sounds = mainstem intubation. CXR confirms tube position.",
    },
}

# Synthetic sound generation — spectral parameters derived from ICBHI 2017
# Respiratory Sound Database literature (Rocha et al., 2019).
_SAMPLE_RATE = 16000
_BREATH_RATE = 0.25  # Hz ≈ 15 breaths/min


def _breathing_envelope(n: int, duty: float = 0.4) -> np.ndarray:
    """Breathing envelope: inspiration (duty) then expiration (1-duty)."""
    t = np.linspace(0, 1, n, dtype=np.float32)
    cycle = np.sin(2 * np.pi * _BREATH_RATE * t * (n / _SAMPLE_RATE))
    insp = np.clip(cycle, 0, 1) ** 0.7  # sharper inspiration
    exp_ = np.clip(-cycle, 0, 1) ** 1.2  # gentler expiration
    return (insp * duty + exp_ * (1 - duty)).astype(np.float32)


def _generate_crackles(duration: float = 3.0) -> np.ndarray:
    """Clinically accurate crackles (ICBHI-style).

    Fine crackles: 650-1000 Hz, <10ms duration, inspiratory.
    Coarse crackles: 200-500 Hz, 10-25ms, early inspiratory.
    Superimposed on vesicular baseline with proper breathing envelope.
    """
    n = int(_SAMPLE_RATE * duration)
    envelope = _breathing_envelope(n)
    # Vesicular baseline
    baseline = 0.06 * envelope * np.random.normal(0, 1, n).astype(np.float32)
    t = np.linspace(0, duration, n, dtype=np.float32)
    baseline += 0.04 * envelope * np.sin(2 * np.pi * 120 * t).astype(np.float32)

    # Fine crackles (inspiratory phase — where envelope > 0.3)
    n_fine = int(duration * 12)
    for _ in range(n_fine):
        pos = np.random.randint(0, max(1, n - 160))
        if envelope[pos] < 0.3:
            continue
        freq = np.random.uniform(650, 1000)
        width = int(_SAMPLE_RATE * np.random.uniform(0.002, 0.008))  # 2-8ms
        amp = np.random.uniform(0.3, 0.6)
        click_t = np.arange(width, dtype=np.float32) / _SAMPLE_RATE
        click = amp * np.sin(2 * np.pi * freq * click_t) * np.exp(-click_t * 800)
        end = min(pos + width, n)
        baseline[pos:end] += click[:end - pos]

    # Coarse crackles (fewer, louder)
    n_coarse = int(duration * 4)
    for _ in range(n_coarse):
        pos = np.random.randint(0, max(1, n - 400))
        if envelope[pos] < 0.2:
            continue
        freq = np.random.uniform(200, 500)
        width = int(_SAMPLE_RATE * np.random.uniform(0.010, 0.025))
        amp = np.random.uniform(0.4, 0.8)
        click_t = np.arange(width, dtype=np.float32) / _SAMPLE_RATE
        click = amp * np.sin(2 * np.pi * freq * click_t) * np.exp(-click_t * 300)
        end = min(pos + width, n)
        baseline[pos:end] += click[:end - pos]

    return baseline


def _generate_wheezing(duration: float = 3.0) -> np.ndarray:
    """Clinically accurate wheezing (ICBHI-style).

    Polyphonic: multiple simultaneous tones (100-1600 Hz).
    Predominantly expiratory. Duration >100ms (typically 250ms+).
    """
    n = int(_SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    envelope = _breathing_envelope(n, duty=0.3)  # expiratory emphasis
    exp_mask = np.clip(1.0 - envelope, 0.2, 1.0)  # stronger during expiration

    # 2-4 simultaneous tonal components (polyphonic wheeze)
    n_tones = np.random.randint(2, 5)
    signal = np.zeros(n, dtype=np.float32)
    for _ in range(n_tones):
        freq = np.random.uniform(200, 1200)
        amp = np.random.uniform(0.08, 0.20)
        # Slight frequency modulation (natural variation)
        fm = 1.0 + 0.02 * np.sin(2 * np.pi * np.random.uniform(1, 4) * t)
        signal += amp * np.sin(2 * np.pi * freq * fm * t).astype(np.float32)

    signal *= exp_mask
    # Add breath noise baseline
    signal += 0.04 * envelope * np.random.normal(0, 1, n).astype(np.float32)
    return signal


def _generate_diminished(duration: float = 3.0) -> np.ndarray:
    """Diminished breath sounds: very low amplitude vesicular with reduced high frequencies."""
    n = int(_SAMPLE_RATE * duration)
    envelope = _breathing_envelope(n) * 0.15  # much quieter
    t = np.linspace(0, duration, n, dtype=np.float32)
    # Low-frequency only (high frequencies attenuated by fluid/tissue)
    breath = envelope * (
        0.06 * np.sin(2 * np.pi * 80 * t)
        + 0.03 * np.sin(2 * np.pi * 150 * t)
        + 0.02 * np.random.normal(0, 1, n)
    ).astype(np.float32)
    return breath


def _generate_absent(duration: float = 3.0) -> np.ndarray:
    """Absent breath sounds: near-silence with minimal ambient noise."""
    n = int(_SAMPLE_RATE * duration)
    return (0.003 * np.random.normal(0, 1, n)).astype(np.float32)


def _generate_bronchial(duration: float = 3.0) -> np.ndarray:
    """Clinically accurate bronchial breath sounds.

    Harsh, tubular quality. Louder than vesicular. Equal or louder expiratory phase.
    Higher frequency content (200-800 Hz) with prominent overtones.
    """
    n = int(_SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    envelope = _breathing_envelope(n, duty=0.45)

    # Tubular harmonics — fundamental + overtones
    signal = np.zeros(n, dtype=np.float32)
    fundamental = np.random.uniform(150, 250)
    for harmonic in range(1, 5):
        freq = fundamental * harmonic
        amp = 0.2 / harmonic  # natural harmonic rolloff
        signal += amp * np.sin(2 * np.pi * freq * t).astype(np.float32)

    signal *= envelope * 1.5  # louder than vesicular
    signal += 0.06 * np.random.normal(0, 1, n).astype(np.float32)
    return signal


def _generate_vesicular(duration: float = 3.0) -> np.ndarray:
    """Normal vesicular breath sounds.

    Soft, low-pitched, inspiration > expiration.
    Frequency range 60-600 Hz, gentle rustling quality.
    """
    n = int(_SAMPLE_RATE * duration)
    t = np.linspace(0, duration, n, dtype=np.float32)
    envelope = _breathing_envelope(n, duty=0.6)  # inspiration dominant

    breath = envelope * (
        0.08 * np.sin(2 * np.pi * 120 * t)
        + 0.04 * np.sin(2 * np.pi * 250 * t)
        + 0.03 * np.random.normal(0, 1, n)
    ).astype(np.float32)
    return breath


def _generate_s3_gallop(duration: float = 3.0) -> np.ndarray:
    """S3 gallop: low-pitched third heart sound after S2 in cardiac cycle.

    ~0.04s duration, 25-50 Hz, occurring 0.12-0.18s after S2.
    S1-S2-S3 rhythm at ~75 bpm.
    """
    n = int(_SAMPLE_RATE * duration)
    signal = np.zeros(n, dtype=np.float32)
    beat_interval = int(_SAMPLE_RATE * 0.8)  # ~75 bpm

    for beat_start in range(0, n - beat_interval, beat_interval):
        for s_offset, s_freq, s_amp, s_dur in [
            (0, 60, 0.5, 0.04),         # S1
            (0.32, 80, 0.4, 0.03),      # S2
            (0.48, 35, 0.25, 0.04),     # S3 (pathological)
        ]:
            pos = beat_start + int(s_offset * _SAMPLE_RATE)
            w = int(s_dur * _SAMPLE_RATE)
            if pos + w > n:
                continue
            ht = np.arange(w, dtype=np.float32) / _SAMPLE_RATE
            heart = s_amp * np.sin(2 * np.pi * s_freq * ht) * np.exp(-ht * 60)
            signal[pos:pos + w] += heart

    # Add faint bilateral basilar crackles (CHF association)
    crackle_base = _generate_crackles(duration) * 0.3
    signal += crackle_base
    return signal

_SOUND_GENERATORS = {
    "crackles": _generate_crackles,
    "wheezing": _generate_wheezing,
    "diminished": _generate_diminished,
    "absent": _generate_absent,
    "bronchial": _generate_bronchial,
    "vesicular": _generate_vesicular,
    "crepitus": _generate_crackles,
    "S3 gallop": _generate_s3_gallop,
    "variable": _generate_vesicular,
}


class MockHeAREngine:
    """Mock HeAR engine for CPU mode. Generates synthetic lung sounds."""

    def __init__(self):
        self._loaded = False

    def load(self):
        self._loaded = True

    def generate_lung_sound(
        self, category: str, duration: float = 3.0,
    ) -> tuple[int, np.ndarray]:
        """Generate synthetic lung sound for a CXR category.
        Returns (sample_rate, audio_array).
        """
        data = LUNG_SOUNDS.get(category, LUNG_SOUNDS["No Finding"])
        sound_type = data["sound"]
        generator = _SOUND_GENERATORS.get(sound_type, _generate_vesicular)
        audio = generator(duration)
        return _SAMPLE_RATE, audio

    def classify_lung_sound(self, audio: np.ndarray) -> dict[str, float]:
        """Mock classification of lung sound into categories."""
        # Mock: return random probabilities
        categories = list(LUNG_SOUNDS.keys())
        probs = np.random.dirichlet(np.ones(len(categories)))
        return dict(zip(categories, probs.tolist()))

    def get_lung_sound_case(self, category: str) -> LungSoundCase:
        """Get a complete lung sound case for training."""
        data = LUNG_SOUNDS.get(category, LUNG_SOUNDS["No Finding"])
        return LungSoundCase(
            category=category,
            sound_type=data["sound"],
            description=f"**Sound:** {data['sound']} | **Location:** {data['location']}\n\n**Character:** {data['character']}",
            correlation=data["correlation"],
        )


class HeAREngine:
    """
    Real HeAR engine for GPU mode (google/hear-pytorch).
    HeAR is a ViT-L bioacoustic foundation model trained on 313M two-second
    audio clips. Produces 512-dim embeddings for respiratory sound classification.
    Falls back to MockHeAREngine if the model is unavailable.

    Classification uses a reference-embedding approach: we encode one synthetic
    example per category at load time, then classify query audio by cosine
    similarity against these reference embeddings (nearest-neighbor).
    """

    def __init__(self, model_id: str = "google/hear-pytorch"):
        self.model_id = model_id
        self.model = None
        self._loaded = False
        self._real = False
        self._mock = MockHeAREngine()
        self._reference_embeddings: dict[str, np.ndarray] = {}

    def load(self):
        if self._loaded:
            return
        try:
            from transformers import AutoModel
            self.model = AutoModel.from_pretrained(self.model_id)
            self._real = True
            self._build_reference_embeddings()
        except Exception:
            self._real = False
        self._loaded = True

    def _build_reference_embeddings(self):
        """Build reference embeddings for each lung sound category.

        Encodes one synthetic example per category so we can classify
        query audio by cosine similarity (nearest-neighbor in embedding space).
        """
        if not self._real or self.model is None:
            return
        for category in LUNG_SOUNDS:
            _, audio = self._mock.generate_lung_sound(category, duration=2.0)
            emb = self.get_embedding(audio)
            if emb is not None:
                norm = np.linalg.norm(emb)
                if norm > 0:
                    self._reference_embeddings[category] = emb / norm

    def get_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray | None:
        """Get HeAR embedding from audio. Returns embedding vector or None."""
        if not self._real or self.model is None:
            return None
        try:
            import torch
            # HeAR expects 2-second clips at 16kHz (32000 samples)
            target_len = sample_rate * 2
            if len(audio) > target_len:
                audio = audio[:target_len]
            elif len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
            tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                outputs = self.model(tensor)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output[0].cpu().numpy()
                else:
                    embedding = outputs.last_hidden_state[:, 0, :][0].cpu().numpy()
            return embedding
        except Exception:
            return None

    def generate_lung_sound(
        self, category: str, duration: float = 3.0,
    ) -> tuple[int, np.ndarray]:
        """Generate synthetic lung sound. Uses mock generators (HeAR classifies, not generates)."""
        return self._mock.generate_lung_sound(category, duration)

    def classify_lung_sound(self, audio: np.ndarray) -> dict[str, float]:
        """Classify lung sound using HeAR embeddings with nearest-neighbor matching.

        Computes cosine similarity between the query audio embedding and
        reference embeddings built at load time (one per category).
        Falls back to mock if HeAR is unavailable or no references built.
        """
        if self._real and self.model is not None and self._reference_embeddings:
            query_emb = self.get_embedding(audio)
            if query_emb is not None:
                query_norm = np.linalg.norm(query_emb)
                if query_norm > 0:
                    query_emb = query_emb / query_norm
                categories = list(self._reference_embeddings.keys())
                similarities = np.array([
                    float(np.dot(query_emb, self._reference_embeddings[cat]))
                    for cat in categories
                ])
                # Shift to positive range and softmax for probabilities
                similarities = similarities - similarities.min() + 1e-6
                exp_sim = np.exp(similarities * 5.0)  # Temperature scaling
                probs = exp_sim / exp_sim.sum()
                return dict(zip(categories, probs.tolist()))
        return self._mock.classify_lung_sound(audio)
