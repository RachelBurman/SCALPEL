"""
Retrieval parameter definitions for SCALPEL.

Defines the discrete action space for retrieval optimisation.
"""

import random
from dataclasses import dataclass

from scalpel.config import settings

N_CHUNKS_OPTIONS:    list[int]   = [3, 5, 8, 10]
CHUNK_TRUNC_OPTIONS: list[int]   = [256, 512, 1024, 2048]
THRESHOLD_OPTIONS:   list[float] = [0.0, 0.3, 0.5, 0.7]
RERANK_OPTIONS:      list[bool]  = [False, True]

ACTION_DIMS = [
    len(N_CHUNKS_OPTIONS),
    len(CHUNK_TRUNC_OPTIONS),
    len(THRESHOLD_OPTIONS),
    len(RERANK_OPTIONS),
]

STATE_DIM = settings.embedding_dimensions  # 768

DEFAULT_ACTION = [1, 1, 0, 0]  # n=5, trunc=512, thresh=0.0, rerank=False


@dataclass
class RetrievalParams:
    """Concrete retrieval parameters decoded from an action index vector."""

    n_chunks: int
    chunk_trunc: int
    similarity_threshold: float
    rerank: bool

    @classmethod
    def from_action(cls, action: list[int]) -> "RetrievalParams":
        return cls(
            n_chunks=N_CHUNKS_OPTIONS[action[0]],
            chunk_trunc=CHUNK_TRUNC_OPTIONS[action[1]],
            similarity_threshold=THRESHOLD_OPTIONS[action[2]],
            rerank=bool(RERANK_OPTIONS[action[3]]),
        )

    @classmethod
    def default(cls) -> "RetrievalParams":
        return cls.from_action(DEFAULT_ACTION)

    @classmethod
    def random(cls) -> "RetrievalParams":
        return cls(
            n_chunks=random.choice(N_CHUNKS_OPTIONS),
            chunk_trunc=random.choice(CHUNK_TRUNC_OPTIONS),
            similarity_threshold=random.choice(THRESHOLD_OPTIONS),
            rerank=random.choice(RERANK_OPTIONS),
        )

    def to_action(self) -> list[int]:
        return [
            N_CHUNKS_OPTIONS.index(self.n_chunks),
            CHUNK_TRUNC_OPTIONS.index(self.chunk_trunc),
            THRESHOLD_OPTIONS.index(self.similarity_threshold),
            RERANK_OPTIONS.index(self.rerank),
        ]

    def __str__(self) -> str:
        return (
            f"n={self.n_chunks} trunc={self.chunk_trunc} "
            f"thresh={self.similarity_threshold} rerank={self.rerank}"
        )
