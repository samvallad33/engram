"""ENGRAM Visual RAG with MedSigLIP"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image

@dataclass
class SimilarCase:
    image_path: str; category: str; similarity: float; card_id: str = ""

class MedSigLIPRetriever:
    def __init__(self, model_id="google/medsiglip-448"):
        self.model_id, self.model, self.processor, self._loaded, self.index = model_id, None, None, False, None
        self.paths, self.cats, self.ids = [], [], []

    def load(self):
        if self._loaded: return
        import torch
        from transformers import AutoProcessor, AutoModel
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self._loaded = True

    def _encode(self, img: Image.Image) -> np.ndarray:
        import torch
        inputs = self.processor(images=[img.convert("RGB")], return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
            return (feat / feat.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

    def build_index(self, paths: list[str], cats: list[str], ids: list[str] = None):
        import faiss
        embs = []
        for p in paths:
            try: embs.append(self._encode(Image.open(p)))
            except Exception: pass
        if not embs: return
        
        self.index = faiss.IndexFlatIP(len(embs[0]))
        self.index.add(np.vstack(embs).astype(np.float32))
        self.paths, self.cats, self.ids = paths, cats, ids or [""]*len(paths)

    def search_by_image(self, img: Image.Image, top_k=5) -> list[SimilarCase]:
        if not self.index or self.index.ntotal == 0: return []
        scores, idxs = self.index.search(self._encode(img).reshape(1, -1).astype(np.float32), min(top_k, self.index.ntotal))
        return [SimilarCase(self.paths[i], self.cats[i], float(s), self.ids[i]) for s, i in zip(scores[0], idxs[0]) if i >= 0]