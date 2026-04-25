"""
Data collector for SCALPEL retrieval optimisation.

Replaces the PPO training loop. Instead of updating a neural network,
each collection run:
  1. Samples queries from a list
  2. Tries random retrieval parameter combinations
  3. Scores each with the LLM judge
  4. Caches results to disk

The collected (query, embedding, action, score) data is used by
RetrievalOptimizer to recommend params for new queries.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from scalpel.config import settings
from scalpel.console import console
from scalpel.retrieval.cache import RetrievalCache
from scalpel.retrieval.params import ACTION_DIMS, RetrievalParams


class DailyLimitError(RuntimeError):
    """Raised when the OpenRouter daily free-tier limit is hit."""


@dataclass
class CollectionStats:
    total_steps: int = 0
    cache_hits: int = 0
    new_evaluations: int = 0
    daily_limit_hit: bool = False
    scores: list[float] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        return round(sum(self.scores) / len(self.scores), 2) if self.scores else 0.0

    @property
    def best_score(self) -> float:
        return round(max(self.scores), 2) if self.scores else 0.0


class RAGCollector:
    """
    Collects (query, params, score) data for the similarity index.

    Each call to collect() runs one round of random exploration over
    the provided queries, building up the cache for RetrievalOptimizer.
    """

    def __init__(
        self,
        cache: RetrievalCache | None = None,
        verbose: bool = False,
    ):
        self.cache = cache or RetrievalCache()
        self.verbose = verbose
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
                return self._llm_client.generate(
                    prompt=user_prompt, system=system_prompt
                ).content
            except RuntimeError as e:
                msg = str(e)
                if "per-day" in msg or "per_day" in msg:
                    raise DailyLimitError(
                        "Daily free-tier request limit reached."
                    ) from e
                transient = any(code in msg for code in ("502", "503", "429", "524"))
                if transient and attempt < max_retries - 1:
                    wait = 2 ** attempt * 5
                    time.sleep(wait)
                    continue
                raise

    def _step(
        self,
        query: str,
        action: list[int],
        embedding: np.ndarray,
    ) -> tuple[float, float]:
        """
        Run one (query, action) step, using cache when available.

        Returns (reward, eval_score).
        """
        from scalpel.evaluation import evaluate as eval_response

        cached = self.cache.get(query, action)
        if cached is not None:
            return cached

        params = RetrievalParams.from_action(action)
        chunks = self._retrieve(query, params)
        if not chunks:
            reward, eval_score = -1.0, 0.0
            self.cache.put(query, action, embedding, reward, eval_score)
            return reward, eval_score

        response = self._generate(query, chunks)
        score = eval_response(query, chunks, response, verbose=self.verbose)

        reward = (score.overall / 10.0) * 2.0 - 1.0
        self.cache.put(query, action, embedding, reward, score.overall)
        return reward, score.overall

    # ── Public API ─────────────────────────────────────────────────────────

    def collect(
        self,
        queries: list[str],
        steps_per_query: int = 4,
        exploration: str = "random",
    ) -> CollectionStats:
        """
        Run one collection round over the provided queries.

        Args:
            queries:          List of query strings to explore.
            steps_per_query:  How many random param combinations to try per query.
            exploration:      "random" (uniform) or "greedy" (exploit best so far).

        Returns:
            CollectionStats with summary metrics.
        """
        stats = CollectionStats()

        for query in queries:
            if self.verbose:
                console.print(f"\n[bold cyan]Query:[/bold cyan] {query}")

            try:
                embedding = self.embed_query(query)
            except Exception as e:
                console.print(f"[red]Embed failed:[/red] {e}")
                continue

            actions = self._sample_actions(steps_per_query, exploration)

            for action in actions:
                stats.total_steps += 1
                cached = self.cache.get(query, action)

                if cached is not None:
                    stats.cache_hits += 1
                    reward, eval_score = cached
                else:
                    try:
                        reward, eval_score = self._step(query, action, embedding)
                        stats.new_evaluations += 1
                    except DailyLimitError:
                        stats.daily_limit_hit = True
                        console.print(
                            "[bold red]Daily API limit reached — stopping.[/bold red]"
                        )
                        return stats
                    except Exception as e:
                        console.print(f"[red]Step failed:[/red] {e}")
                        continue

                stats.scores.append(eval_score)
                params = RetrievalParams.from_action(action)

                if self.verbose:
                    console.print(
                        f"  {params}  →  score={eval_score:.1f}  "
                        f"reward={reward:+.2f}"
                        + (" [dim](cached)[/dim]" if cached else "")
                    )

        return stats

    def _sample_actions(
        self, n: int, exploration: str
    ) -> list[list[int]]:
        if exploration == "random":
            seen: set[tuple] = set()
            actions = []
            attempts = 0
            while len(actions) < n and attempts < n * 10:
                a = [random.randint(0, dim - 1) for dim in ACTION_DIMS]
                key = tuple(a)
                if key not in seen:
                    seen.add(key)
                    actions.append(a)
                attempts += 1
            return actions
        return [[random.randint(0, dim - 1) for dim in ACTION_DIMS] for _ in range(n)]
