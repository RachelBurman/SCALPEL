"""
Vectorisation-based retrieval parameter optimiser.

Replaces the PPO agent with a cosine similarity k-NN approach:
  1. Build an index of (embedding, best_action) pairs from collected data
  2. For a new query, embed it and find the k most similar past queries
  3. Weight each past query's best action by similarity score
  4. Return the winning action index for each of the 4 discrete dimensions
  5. Fall back to defaults when the index is empty
"""

from __future__ import annotations

import numpy as np

from scalpel.retrieval.cache import RetrievalCache
from scalpel.retrieval.params import ACTION_DIMS, DEFAULT_ACTION, RetrievalParams


class RetrievalOptimizer:
    """
    k-NN retrieval parameter selector using cosine similarity.

    Usage:
        optimizer = RetrievalOptimizer(cache)
        optimizer.build_index()
        params = optimizer.get_params(query_embedding, k=5)
    """

    def __init__(self, cache: RetrievalCache):
        self.cache = cache
        self._embeddings: np.ndarray | None = None   # (N, 768)
        self._actions: np.ndarray | None = None       # (N, 4)

    def build_index(self) -> int:
        """
        (Re)build the similarity index from the best entry per unique query.

        Returns the number of index entries.
        """
        entries = self.cache.best_per_query()
        if not entries:
            self._embeddings = None
            self._actions = None
            return 0

        self._embeddings = np.array(
            [e["embedding"] for e in entries], dtype=np.float32
        )
        self._actions = np.array(
            [e["action"] for e in entries], dtype=np.int32
        )
        return len(entries)

    def get_params(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> RetrievalParams:
        """
        Return the best RetrievalParams for this query embedding.

        When the index has fewer than k entries it uses all available ones.
        Falls back to defaults if the index is empty.
        """
        if self._embeddings is None or len(self._embeddings) == 0:
            return RetrievalParams.default()

        k = min(k, len(self._embeddings))

        # Cosine similarity: normalise then dot product
        q = query_embedding.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True) + 1e-8
        normed = self._embeddings / norms
        similarities = normed @ q_norm  # (N,)

        top_k_idx = np.argpartition(similarities, -k)[-k:]
        top_k_sims = similarities[top_k_idx]
        top_k_actions = self._actions[top_k_idx]  # (k, 4)

        # Softmax-weight the similarities so they sum to 1
        exp_sims = np.exp(top_k_sims - top_k_sims.max())
        weights = exp_sims / exp_sims.sum()  # (k,)

        # For each action dimension, weighted vote across the k neighbours
        best_action = []
        for dim_idx, n_choices in enumerate(ACTION_DIMS):
            votes = np.zeros(n_choices, dtype=np.float32)
            for i, action_row in enumerate(top_k_actions):
                votes[action_row[dim_idx]] += weights[i]
            best_action.append(int(np.argmax(votes)))

        return RetrievalParams.from_action(best_action)

    @property
    def index_size(self) -> int:
        return len(self._embeddings) if self._embeddings is not None else 0
