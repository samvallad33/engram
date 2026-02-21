"""ENGRAM CXR Foundation Retrieval (HAI-DEF #4)"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from PIL import Image

@dataclass
class CXRSimilarCase:
    image_path: str; category: str; similarity: float; card_id: str = ""

class CXRFoundationRetriever:
    def __init__(self, model_id="google/cxr-foundation"):
        self.model_id, self.model, self.processor, self._loaded, self.index = model_id, None, None, False, None
        self.paths, self.cats, self.ids = [], [], []

    def load(self):
        if self._loaded: return
        import torch
        from transformers import AutoModel, AutoImageProcessor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id).to(self.device)
        self._loaded = True

    def _encode(self, img: Image.Image) -> np.ndarray:
        import torch
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model(**inputs)
            feat = out.pooler_output if hasattr(out, "pooler_output") and out.pooler_output is not None else out.last_hidden_state[:, 0, :]
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

    def search(self, img: Image.Image, top_k=5) -> list[CXRSimilarCase]:
        if not self.index or self.index.ntotal == 0: return []
        scores, idxs = self.index.search(self._encode(img).reshape(1, -1).astype(np.float32), min(top_k, self.index.ntotal))
        return [CXRSimilarCase(self.paths[i], self.cats[i], float(s), self.ids[i]) for s, i in zip(scores[0], idxs[0]) if i >= 0]