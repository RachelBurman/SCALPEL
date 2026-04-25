"""
Disk-backed cache for (query, action) → (embedding, reward, eval_score).

Extends the original StepCache to also store query strings and embeddings,
which the similarity index needs to build recommendations.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


class RetrievalCache:
    """
    Persists every (query, action, score) triple to disk.

    Cache format per entry:
        {
            "query":      str,
            "action":     [int, int, int, int],
            "embedding":  [float, ...],   # 768-dim, stored as list
            "reward":     float,
            "eval_score": float,
        }
    """

    def __init__(self, path: Path = Path("models/retrieval/step_cache.json")):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _key(self, query: str, action: list[int]) -> str:
        raw = json.dumps({"q": query, "a": action}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, action: list[int]) -> tuple[float, float] | None:
        entry = self._data.get(self._key(query, action))
        if entry:
            return entry["reward"], entry["eval_score"]
        return None

    def put(
        self,
        query: str,
        action: list[int],
        embedding: np.ndarray,
        reward: float,
        eval_score: float,
    ) -> None:
        self._data[self._key(query, action)] = {
            "query": query,
            "action": action,
            "embedding": embedding.tolist(),
            "reward": reward,
            "eval_score": eval_score,
        }
        self.path.write_text(json.dumps(self._data, indent=2), encoding="utf-8")

    def all_entries(self) -> list[dict]:
        return list(self._data.values())

    def best_per_query(self) -> list[dict]:
        """Return only the highest-scoring action for each unique query."""
        best: dict[str, dict] = {}
        for entry in self._data.values():
            q = entry["query"]
            if q not in best or entry["eval_score"] > best[q]["eval_score"]:
                best[q] = entry
        return list(best.values())

    def __len__(self) -> int:
        return len(self._data)
