"""
ENGRAM Visual RAG with MedSigLIP
Retrieves similar medical cases using MedSigLIP embeddings + FAISS index.
"Show me 3 similar cases" feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# torch is lazy-imported only when MedSigLIPRetriever.load() is called
# This allows type imports to work without GPU


@dataclass
class SimilarCase:
    """A retrieved similar case."""
    image_path: str
    category: str
    similarity: float
    card_id: str = ""


class MedSigLIPRetriever:
    """
    MedSigLIP-based case retrieval engine.
    Builds a FAISS index over medical image embeddings for fast similarity search.
    """

    def __init__(self, model_id: str = "google/medsiglip-448", device: str = "auto"):
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
        """Load MedSigLIP model."""
        if self._loaded:
            return

        import torch
        from transformers import AutoProcessor, AutoModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(self.model_id).to(device)
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.device = device
        self._loaded = True

    def _encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single image to embedding."""
        import torch
        image = image.convert("RGB").resize((448, 448), Image.Resampling.LANCZOS)
        inputs = self.processor(images=[image], return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def build_index(
        self,
        image_paths: list[str],
        categories: list[str],
        card_ids: list[str] | None = None,
        batch_size: int = 16,
    ):
        """Build FAISS index from a collection of images."""
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

            images = []
            good_indices = []
            for j, p in enumerate(batch_paths):
                try:
                    img = Image.open(p).convert("RGB").resize((448, 448), Image.Resampling.LANCZOS)
                    images.append(img)
                    good_indices.append(j)
                except Exception:
                    continue

            if not images:
                continue

            import torch
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                features = features / features.norm(dim=-1, keepdim=True)

            embeddings = features.cpu().numpy()
            all_embeddings.append(embeddings)

            for j in good_indices:
                valid_paths.append(batch_paths[j])
                valid_categories.append(batch_cats[j])
                valid_card_ids.append(batch_ids[j])

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

    def search_by_image(self, image: Image.Image, top_k: int = 5) -> list[SimilarCase]:
        """Find similar cases given a query image."""
        if self.index is None or self.index.ntotal == 0:
            return []

        embedding = self._encode_image(image).reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(embedding, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append(SimilarCase(
                image_path=self.index_paths[idx],
                category=self.index_categories[idx],
                similarity=float(score),
                card_id=self.index_card_ids[idx] if self.index_card_ids else "",
            ))
        return results

    def search_by_text(self, query: str, top_k: int = 5) -> list[SimilarCase]:
        """Find cases matching a text description."""
        import torch
        if self.index is None or self.index.ntotal == 0:
            return []

        inputs = self.processor(text=[query], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        embedding = text_features.cpu().numpy().astype(np.float32)
        scores, indices = self.index.search(embedding, min(top_k, self.index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append(SimilarCase(
                image_path=self.index_paths[idx],
                category=self.index_categories[idx],
                similarity=float(score),
                card_id=self.index_card_ids[idx] if self.index_card_ids else "",
            ))
        return results
