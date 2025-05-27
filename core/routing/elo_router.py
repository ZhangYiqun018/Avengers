"""elo_router.py

A router that chooses the best experts for a query by combining
  • distance between the query embedding and pre‑computed K‑Means cluster centres
  • pre‑computed TrueSkill (or Elo) ratings for each model inside each cluster

Training / scoring is **not** included – all heavy lifting is assumed
already done offline. The router simply loads static artefacts at start‑up.

Expected artefacts
-----------------
centres.npy
    np.ndarray of shape (K, D) – cluster centres in the same embedding space.
ratings.json
    {
      "0": { "model_A": {"mu": 25.1, "sigma": 0.71}, ... },
      "1": { ... }
    }

Config example (pass in `router_config["elo_router"]`):
```
{
  "centres_path": "data/centres.npy",
  "ratings_path": "data/ratings.json",
  "mapping_path": "data/map.json",
  "top_n": 3,
  "beta": 3.0,
  "k_sigma": 3,
  "default_mu": 25.0,
  "embedding_model": "text-embedding-3-small"
}
```

"""
from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Any

from core.experts.load_experts import Expert
from core.routing.base_router import BaseRouter, RouterOutput
from diversity.embedding_cache import EmbeddingCache
from sklearn.preprocessing import Normalizer
import joblib

class EloRouter(BaseRouter):
    """Cluster‑Aware Elo Router."""

    def __init__(
        self,
        normal_experts: List[Expert],
        thinking_experts: List[Expert],
        router_config: Dict[str, Any],
    ) -> None:
        super().__init__(normal_experts, thinking_experts)

        cfg = router_config["elo_router"]
        # 1. load artefacts --------------------------------------------------
        self.centres = np.load(Path(cfg["centres_path"]))  # (K, D)
        self.ratings = self._load_ratings(cfg["ratings_path"])  # dict[int, dict[str, (mu,sigma)]]
        self.mapping = self._load_mapping(cfg["mapping_path"])  # expert id -> expert name , dict[str, str]
        self.normalizer = self._load_normalizer(cfg["normalizer_path"])  # Normalizer

        # 2. set hyperparameters --------------------------------------------
        self.top_k: int = cfg.get("top_k", 11)
        self.top_n: int = cfg.get("top_n", 3)
        self.beta: float = cfg.get("beta", 3.0)        # softmax temperature
        self.k_sigma: int = cfg.get("k_sigma", 3)
        self.default_mu: float = cfg.get("default_mu", 25.0)
        self.available_models = cfg.get("available_models")
        print(self.available_models)
        # 3. Build quick lookup: expert ID -> Expert object
        self.expert_map = {}
        for e in self.normal_experts:
            # Find the expert ID (e.g. M01) that maps to this expert's model name
            for expert_id, model_name in self.mapping.items():
                if model_name == e.model_name:
                    self.expert_map[expert_id] = e
                    break
        self.availabel_models_id = []
        for id, model in self.mapping.items():
            if model in self.available_models:
                self.availabel_models_id.append(id)
        assert len(self.availabel_models_id) == len(self.available_models), f"Length of available models ({len(self.available_models)}) does not match length of available models id ({len(self.availabel_models_id)})"
        
        # Embedding helper – uses local cache automatically
        self.embedder = EmbeddingCache(
            base_url="http://172.30.12.113:8000/v1",
            api_key="sk-1234567890",
            model_name=cfg.get("embedding_model", "text-embedding-3-small"))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def route(self, question: str) -> RouterOutput:
        """Select Top‑N experts for *question* based on cluster probability × Elo."""
        q_vec = np.array(self.embedder.get(question))  # (D,)
        q_vec = self.normalizer.transform([q_vec])[0]
        # 1. distance to each cluster centre
        dists = 1 - self.centres @ q_vec
        idx = np.argsort(dists)[:self.top_k]
        dists = dists[idx]
        # 2. softmax to probability (the smaller the dist, the larger P)
        
        logits = -self.beta * dists
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()  # (K,)

        # 3. fuse with ratings
        scores: Dict[str, float] = {}
        for cid, p in zip(idx, probs):
            table = self.ratings.get(cid, {})
            for model, (mu, sigma) in table.items():
                if model in self.availabel_models_id:
                    conservative = mu - self.k_sigma * sigma
                    scores[model] = scores.get(model, 0.0) + p * conservative

        # 4. fill missing models with default mu
        for model in self.expert_map:
            scores.setdefault(model, self.default_mu)

        # 5. rank & pick
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        chosen_models = [m for m, _ in ranked if m in self.expert_map][: self.top_n]
        chosen_experts = [self.expert_map[m] for m in chosen_models]

        return RouterOutput(
            normal_experts=chosen_experts,
            thinking_experts=self.thinking_experts,
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_ratings(path: str | Path) -> Dict[int, Dict[str, tuple[float, float]]]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        parsed: Dict[int, Dict[str, tuple[float, float]]] = {}
        for cid_str, models in raw.items():
            cid = int(cid_str)
            parsed[cid] = {m: (vals["mu"], vals["sigma"]) for m, vals in models.items()}
        return parsed
    
    @staticmethod
    def _load_mapping(path: str | Path) -> Dict[str, str]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
        
    @staticmethod
    def _load_normalizer(path: str | Path) -> Normalizer:
        return joblib.load(path)