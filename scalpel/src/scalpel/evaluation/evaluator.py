"""
LLM Evaluation Framework for SCALPEL.

Implements the LLM-as-judge pattern to score RAG responses on three dimensions:
  - Groundedness: is the response supported by retrieved chunks?
  - Relevance: did the retrieved chunks answer the query?
  - Confidence calibration: is the model appropriately uncertain when evidence is weak?
"""

import re
from dataclasses import dataclass, field

from rich.panel import Panel
from rich.table import Table

from scalpel.analysis.llm_client import LLMClient, get_client
from scalpel.console import console
from scalpel.evaluation.prompts import JUDGE_SYSTEM, JUDGE_USER_TEMPLATE


@dataclass
class EvaluationScore:
    """Structured output from the LLM judge."""

    query: str
    groundedness: float       # 0-10, higher is better
    relevance: float          # 0-10, higher is better
    confidence_calibration: float  # 0-10, higher is better
    unsupported_claims: list[str] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str = ""
    judge_model: str = ""

    @property
    def overall(self) -> float:
        return round((self.groundedness + self.relevance + self.confidence_calibration) / 3, 2)

    @property
    def grade(self) -> str:
        s = self.overall
        if s >= 8:
            return "Excellent"
        elif s >= 6:
            return "Good"
        elif s >= 4:
            return "Concerning"
        else:
            return "Poor"

    def display(self) -> None:
        color = "green" if self.overall >= 7 else "yellow" if self.overall >= 4 else "red"

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="dim", width=26)
        table.add_column(style="bold")

        def _bar(score: float) -> str:
            filled = round(score)
            return "█" * filled + "░" * (10 - filled) + f"  {score:.1f}/10"

        table.add_row("Groundedness",          _bar(self.groundedness))
        table.add_row("Relevance",             _bar(self.relevance))
        table.add_row("Confidence Calibration", _bar(self.confidence_calibration))
        table.add_row("", "")
        table.add_row(f"[{color}]Overall[/{color}]", f"[bold {color}]{self.overall}/10 — {self.grade}[/bold {color}]")

        if self.unsupported_claims:
            claims_text = "\n[bold]Unsupported Claims:[/bold]\n"
            claims_text += "\n".join(f"  • {c}" for c in self.unsupported_claims)
            console.print(Panel(
                table,
                title="[bold]🔬 RAG Evaluation[/bold]",
                subtitle=f"[dim]Query: {self.query[:60]}{'...' if len(self.query) > 60 else ''}[/dim]",
                border_style=color,
            ))
            if self.unsupported_claims:
                console.print(Panel(
                    "\n".join(f"• {c}" for c in self.unsupported_claims),
                    title="[bold yellow]Unsupported Claims[/bold yellow]",
                    border_style="yellow",
                ))
            if self.reasoning:
                console.print(Panel(
                    self.reasoning,
                    title="[dim]Judge Reasoning[/dim]",
                    border_style="dim",
                ))
            return

        console.print(Panel(
            table,
            title="[bold]🔬 RAG Evaluation[/bold]",
            subtitle=f"[dim]Query: {self.query[:60]}{'...' if len(self.query) > 60 else ''}[/dim]",
            border_style=color,
        ))
        if self.reasoning:
            console.print(Panel(
                self.reasoning,
                title="[dim]Judge Reasoning[/dim]",
                border_style="dim",
            ))

    def __repr__(self) -> str:
        return f"EvaluationScore(overall={self.overall}/10, grade={self.grade!r})"


class ResponseEvaluator:
    """
    Evaluates RAG responses using a second LLM call as judge.

    Takes a query, the retrieved chunks that informed the response,
    and the generated response itself. Returns a structured EvaluationScore.
    """

    def __init__(self, client: LLMClient | None = None):
        self.client = client or get_client()

    def evaluate(
        self,
        query: str,
        chunks: list[str],
        response: str,
        verbose: bool = True,
    ) -> EvaluationScore:
        """
        Score a RAG response across three dimensions.

        Args:
            query: The original user query.
            chunks: Retrieved text chunks used to generate the response.
            response: The generated response to evaluate.
            verbose: Show progress indicator.

        Returns:
            EvaluationScore with per-dimension scores and reasoning.
        """
        if verbose:
            console.print("[cyan]🔬 Evaluating response...[/cyan]")

        chunks_text = "\n\n---\n\n".join(
            f"[Chunk {i + 1}]\n{chunk}" for i, chunk in enumerate(chunks)
        )

        user_prompt = JUDGE_USER_TEMPLATE.format(
            query=query,
            chunks=chunks_text,
            response=response,
        )

        judge_response = self.client.generate(
            prompt=user_prompt,
            system=JUDGE_SYSTEM,
        )

        score = self._parse(query, judge_response.content)
        score.judge_model = self.client.model

        if verbose:
            console.print("[green]✓[/green] Evaluation complete")

        return score

    def _parse(self, query: str, raw: str) -> EvaluationScore:
        def _extract(label: str) -> float:
            m = re.search(
                rf"{label}:\s*\[?(\d+(?:\.\d+)?)\s*(?:/\s*10)?\]?",
                raw,
                re.IGNORECASE,
            )
            return float(m.group(1)) if m else 5.0

        groundedness = _extract("GROUNDEDNESS")
        relevance = _extract("RELEVANCE")
        confidence = _extract("CONFIDENCE_CALIBRATION")

        unsupported: list[str] = []
        claims_match = re.search(
            r"UNSUPPORTED_CLAIMS:\s*\n(.*?)(?=\nREASONING:|\Z)",
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        if claims_match:
            block = claims_match.group(1).strip()
            if block.lower() != "none":
                for line in block.splitlines():
                    line = line.strip().lstrip("•-*").strip()
                    if line and line.lower() != "none":
                        unsupported.append(line)

        reasoning = ""
        reasoning_match = re.search(
            r"REASONING:\s*\n?(.*?)$",
            raw,
            re.IGNORECASE | re.DOTALL,
        )
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()

        return EvaluationScore(
            query=query,
            groundedness=groundedness,
            relevance=relevance,
            confidence_calibration=confidence,
            unsupported_claims=unsupported,
            reasoning=reasoning,
            raw_response=raw,
        )


_default_evaluator: ResponseEvaluator | None = None


def get_evaluator() -> ResponseEvaluator:
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = ResponseEvaluator()
    return _default_evaluator


def evaluate(
    query: str,
    chunks: list[str],
    response: str,
    verbose: bool = True,
) -> EvaluationScore:
    """Evaluate a RAG response using the default evaluator."""
    return get_evaluator().evaluate(query, chunks, response, verbose=verbose)
