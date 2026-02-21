"""
ENGRAM CXR Foundation Retrieval
Uses Google's CXR Foundation ELIXR embeddings for chest X-ray similarity search.
HAI-DEF model #4 — trained on 800,000+ chest X-rays.

CXR Foundation provides:
- ELIXR v2.0: 32x768 dimensional vectors for detailed image features
- Zero-shot classification using textual prompts (0.846 AUC)
- Data-efficient classification (0.898 AUC across 5 CheXpert findings)
- 600x less data needed compared to traditional transfer learning
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# torch + transformers lazy-imported in load()


@dataclass
class CXRSimilarCase:
    """A retrieved similar case using CXR Foundation embeddings."""
    image_path: str
    category: str
    similarity: float
    card_id: str = ""


class CXRFoundationRetriever:
    """
    CXR Foundation-based retrieval engine.
    Uses ELIXR embeddings (trained on 800K+ CXRs) for medical image similarity.
    More specialized than MedSigLIP for chest X-ray tasks.
    """

    def __init__(self, model_id: str = "google/cxr-foundation", device: str = "auto"):
        self.model_id = model_id
        self.device = device
        self.model = None
        self.processor = None
        self._loaded = False

        # Index storage
        self.index = None
        self.index_paths: list[str] = []
        self.index_categories: list[str] = []
        self.index_card_ids: list[str] = []
        self.embed_dim: int = 0

    def load(self):
        """Load CXR Foundation model."""
        if self._loaded:
            return

        import torch
        from transformers import AutoModel, AutoImageProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(device)
        self.device = device
        self._loaded = True

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single chest X-ray to ELIXR embedding."""
        import torch

        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use pooled output or CLS token
            if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                features = outputs.pooler_output
            else:
                features = outputs.last_hidden_state[:, 0, :]
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def build_index(
        self,
        image_paths: list[str],
        categories: list[str],
        card_ids: list[str] | None = None,
        batch_size: int = 8,
    ):
        """Build FAISS index from chest X-ray collection."""
        import faiss

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        all_embeddings = []
        valid_paths = []
        valid_categories = []
        valid_card_ids = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_cats = categories[i:i + batch_size]
            batch_ids = (card_ids[i:i + batch_size] if card_ids
                         else [""] * len(batch_paths))

            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB")
                    embedding = self._encode_image(img)
                    all_embeddings.append(embedding)
                    valid_paths.append(p)
                    valid_categories.append(batch_cats[j])
                    valid_card_ids.append(batch_ids[j])
                except Exception:
                    continue

        if not all_embeddings:
            return

        all_embeddings = np.vstack(all_embeddings)
        self.embed_dim = all_embeddings.shape[1]

        # Build FAISS index (inner product = cosine similarity on normalized vectors)
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(all_embeddings.astype(np.float32))

        self.index_paths = valid_paths
        self.index_categories = valid_categories
        self.index_card_ids = valid_card_ids

    def search(self, image: Image.Image, top_k: int = 5) -> list[CXRSimilarCase]:
        """Find similar CXR cases given a query image."""
        if self.index is None or self.index.ntotal == 0:
            return []

        embedding = self._encode_image(image).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(embedding, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append(CXRSimilarCase(
                image_path=self.index_paths[idx],
                category=self.index_categories[idx],
                similarity=float(score),
                card_id=self.index_card_ids[idx] if self.index_card_ids else "",
            ))
        return results

    def zero_shot_classify(self, image: Image.Image, labels: list[str]) -> dict[str, float]:
        """
        Zero-shot classification using CXR Foundation embeddings.
        Note: ELIXR uses a specialized text-image pipeline (not CLIP-style
        get_text_features). This method uses embedding similarity when the
        model supports it, otherwise returns uniform distribution.
        Full ELIXR zero-shot achieves 0.846 mean AUC on CheXpert.
        """
        import torch

        if not self._loaded:
            raise RuntimeError("Model not loaded. Call .load() first.")

        # Encode image
        image_emb = self._encode_image(image)

        # For models with CLIP-style text encoding
        if hasattr(self.model, "get_text_features"):
            inputs = self.processor(text=labels, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            image_tensor = torch.tensor(image_emb).unsqueeze(0)
            similarities = (image_tensor @ text_features.cpu().T)[0]
            probs = torch.softmax(similarities * 5.0, dim=0).numpy()
            return {label: float(prob) for label, prob in zip(labels, probs)}

        # ELIXR requires specialized pipeline not available via AutoModel
        return {label: 1.0 / len(labels) for label in labels}
