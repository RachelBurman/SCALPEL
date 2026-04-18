"""
RAG Environment for SCALPEL RL agent.

State:  query embedding (768-dim float vector from nomic-embed-text)
Action: retrieval parameters — 4 discrete heads
Reward: EvaluationScore.overall normalised to [-1, 1]
"""

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from scalpel.config import settings

# ── Result cache ─────────────────────────────────────────────────────────────

class StepCache:
    """
    Disk-backed cache for (query, action) → (reward, eval_score).

    Each unique combination costs 2 LLM calls. Caching means repeat
    combinations during training are free, which matters a lot on the
    free-tier daily request budget.
    """

    def __init__(self, path: Path = Path("models/rl/step_cache.json")):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
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

    def put(self, query: str, action: list[int], reward: float, eval_score: float):
        self._data[self._key(query, action)] = {"reward": reward, "eval_score": eval_score}
        self.path.write_text(json.dumps(self._data, indent=2))

    def __len__(self) -> int:
        return len(self._data)

# ── Action space ────────────────────────────────────────────────────────────

N_CHUNKS_OPTIONS:    list[int]   = [3, 5, 8, 10]
CHUNK_TRUNC_OPTIONS: list[int]   = [256, 512, 1024, 2048]   # chars to keep per chunk
THRESHOLD_OPTIONS:   list[float] = [0.0, 0.3, 0.5, 0.7]    # min similarity score
RERANK_OPTIONS:      list[bool]  = [False, True]

ACTION_DIMS = [
    len(N_CHUNKS_OPTIONS),
    len(CHUNK_TRUNC_OPTIONS),
    len(THRESHOLD_OPTIONS),
    len(RERANK_OPTIONS),
]

STATE_DIM = settings.embedding_dimensions  # 768


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class RetrievalParams:
    """Concrete retrieval parameters decoded from an action."""

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

    def to_action(self) -> list[int]:
        return [
            N_CHUNKS_OPTIONS.index(self.n_chunks),
            CHUNK_TRUNC_OPTIONS.index(self.chunk_trunc),
            THRESHOLD_OPTIONS.index(self.similarity_threshold),
            RERANK_OPTIONS.index(self.rerank),
        ]

    @classmethod
    def random(cls) -> "RetrievalParams":
        return cls(
            n_chunks=random.choice(N_CHUNKS_OPTIONS),
            chunk_trunc=random.choice(CHUNK_TRUNC_OPTIONS),
            similarity_threshold=random.choice(THRESHOLD_OPTIONS),
            rerank=random.choice(RERANK_OPTIONS),
        )

    def __str__(self) -> str:
        return (
            f"n={self.n_chunks} trunc={self.chunk_trunc} "
            f"thresh={self.similarity_threshold} rerank={self.rerank}"
        )


class StepResult(NamedTuple):
    state:      np.ndarray    # query embedding [STATE_DIM]
    action:     list[int]     # action indices
    reward:     float         # normalised to [-1, 1]
    params:     RetrievalParams
    eval_score: float         # raw 0–10


# ── Environment ───────────────────────────────────────────────────────────────

class DailyLimitError(RuntimeError):
    """Raised when the OpenRouter daily free-tier limit is hit."""


class RAGEnvironment:
    """
    SCALPEL RAG pipeline wrapped as an RL environment.

    Each step:
      1. Check cache — return stored result if seen before
      2. Embed the query
      3. Retrieve chunks with the chosen params
      4. Generate a response via LLM
      5. Score with the LLM judge
      6. Cache and return the result
    """

    def __init__(self, verbose: bool = False, cache: StepCache | None = None):
        self.verbose = verbose
        self.cache = cache or StepCache()
        self._llm_client = None

    # ── Helpers ────────────────────────────────────────────────────────────

    def embed_query(self, query: str) -> np.ndarray:
        import ollama as _ollama
        response = _ollama.embed(model=settings.embedding_model, input=query)
        return np.array(response["embeddings"][0], dtype=np.float32)

    def _retrieve(self, query: str, params: RetrievalParams) -> list[str]:
        from scalpel.embeddings import search as vector_search

        results = vector_search(query, n_results=params.n_chunks)

        if params.similarity_threshold > 0.0:
            results = [r for r in results if r.score >= params.similarity_threshold]

        if not results:
            return []

        if params.rerank:
            priority = {"Methods", "Results", "Methodology", "Experimental"}
            results.sort(key=lambda r: (0 if r.section in priority else 1, -r.score))

        return [r.text[: params.chunk_trunc] for r in results]

    def _generate(self, query: str, chunks: list[str], max_retries: int = 3) -> str:
        import time
        from scalpel.analysis.llm_client import get_client
        from scalpel.analysis.prompts import get_template

        if self._llm_client is None:
            self._llm_client = get_client()

        context = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
        )
        template = get_template("rag_query")
        system_prompt, user_prompt = template.format(question=query, context=context)

        for attempt in range(max_retries):
            try:
                return self._llm_client.generate(prompt=user_prompt, system=system_prompt).content
            except RuntimeError as e:
                msg = str(e)
                if "per-day" in msg or "per_day" in msg:
                    raise DailyLimitError("Daily free-tier request limit reached.") from e
                transient = any(code in msg for code in ("502", "503", "429", "524"))
                if transient and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5  # 5s, 10s, 20s
                    time.sleep(wait)
                    continue
                raise

    # ── Core step ─────────────────────────────────────────────────────────

    def step(self, query: str, action: list[int]) -> StepResult:
        """
        Execute one environment step.

        Args:
            query:  Natural-language query for this episode.
            action: Action indices [n_chunks, chunk_trunc, threshold, rerank].

        Returns:
            StepResult with state, action, normalised reward, params, raw score.
        """
        from scalpel.evaluation import evaluate as eval_response

        params = RetrievalParams.from_action(action)
        state = self.embed_query(query)

        # Return cached result if available — saves LLM calls
        cached = self.cache.get(query, action)
        if cached is not None:
            reward, eval_score = cached
            return StepResult(state=state, action=action, reward=reward,
                              params=params, eval_score=eval_score)

        chunks = self._retrieve(query, params)
        if not chunks:
            return StepResult(state=state, action=action, reward=-1.0,
                              params=params, eval_score=0.0)

        response = self._generate(query, chunks)
        score = eval_response(query, chunks, response, verbose=self.verbose)

        reward = (score.overall / 10.0) * 2.0 - 1.0  # [0,10] → [-1,1]

        self.cache.put(query, action, reward, score.overall)

        return StepResult(state=state, action=action, reward=reward,
                          params=params, eval_score=score.overall)

    def random_action(self) -> list[int]:
        return [random.randint(0, dim - 1) for dim in ACTION_DIMS]
