"""
SCALPEL LLM Evaluation Framework.

Scores RAG responses via LLM-as-judge on groundedness, relevance,
and confidence calibration.
"""

from scalpel.evaluation.evaluator import (
    EvaluationScore,
    ResponseEvaluator,
    evaluate,
    get_evaluator,
)

__all__ = [
    "EvaluationScore",
    "ResponseEvaluator",
    "evaluate",
    "get_evaluator",
]
