"""
Paper Analyzer for SCALPEL.

The brain of the operation. Analyzes papers using LLM-powered critique.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown

from scalpel.analysis.llm_client import LLMClient, get_client
from scalpel.analysis.prompts import get_template, AnalysisType
from scalpel.console import console
from scalpel.ingestion import ExtractedPaper, TextChunk, chunk_paper, chunk_for_analysis


@dataclass
class AnalysisResult:
    """Result of a paper analysis."""
    
    paper_title: str
    analysis_type: str
    content: str
    sections_analyzed: list[str] = field(default_factory=list)
    chunks_processed: int = 0
    model_used: str = ""
    
    def display(self) -> None:
        """Display the analysis result in the console."""
        console.print(Panel(
            Markdown(self.content),
            title=f"[bold]{self.analysis_type}[/bold] — {self.paper_title}",
            border_style="cyan",
        ))
    
    def __repr__(self) -> str:
        return f"AnalysisResult({self.analysis_type}, {len(self.content)} chars)"


@dataclass
class BullshitScore:
    """Structured bullshit score result."""
    
    overall_score: float
    methodology_score: float | None = None
    statistics_score: float | None = None
    claims_score: float | None = None
    transparency_score: float | None = None
    red_flags: list[str] = field(default_factory=list)
    summary: str = ""
    raw_response: str = ""
    
    @property
    def rating(self) -> str:
        """Get a text rating based on score."""
        if self.overall_score <= 2:
            return "Excellent"
        elif self.overall_score <= 4:
            return "Good"
        elif self.overall_score <= 6:
            return "Concerning"
        elif self.overall_score <= 8:
            return "Poor"
        else:
            return "Severe Issues"
    
    def display(self) -> None:
        """Display the bullshit score in the console."""
        color = "green" if self.overall_score <= 4 else "yellow" if self.overall_score <= 6 else "red"
        
        content = f"[bold {color}]{self.overall_score}/10[/bold {color}] — {self.rating}\n\n"
        content += f"{self.summary}\n\n"
        
        if self.red_flags:
            content += "[bold]Red Flags:[/bold]\n"
            content += "\n".join(f"• {flag}" for flag in self.red_flags)
        
        console.print(Panel(
            content,
            title="[bold]🔪 Bullshit Score[/bold]",
            border_style=color,
        ))
    
    def __repr__(self) -> str:
        return f"BullshitScore({self.overall_score}/10 - {self.rating})"


class Analyzer:
    """
    Paper analyzer using LLM-powered critique.
    
    Provides methods for summarization, methodology critique,
    claim extraction, and the infamous bullshit score.
    """
    
    def __init__(self, client: LLMClient | None = None):
        """
        Initialize the analyzer.
        
        Args:
            client: LLM client to use (default: creates new one)
        """
        self.client = client or get_client()
    
    def _get_content_for_analysis(
        self,
        paper: ExtractedPaper,
        sections: list[str] | None = None,
        max_tokens: int = 8000,
    ) -> tuple[str, list[str]]:
        """
        Extract content from paper for analysis.
        
        Args:
            paper: Paper to extract from
            sections: Specific sections to include (None = all)
            max_tokens: Approximate max tokens to include
            
        Returns:
            Tuple of (content_string, list_of_sections_included)
        """
        if sections:
            chunks = chunk_for_analysis(paper, target_sections=sections)
        else:
            chunks = chunk_paper(paper, mode="sections")
        
        # Accumulate chunks up to token limit
        content_parts = []
        sections_included = set()
        total_tokens = 0
        
        for chunk in chunks:
            if total_tokens + chunk.token_count > max_tokens:
                break
            content_parts.append(chunk.text)
            if chunk.source_section:
                sections_included.add(chunk.source_section)
            total_tokens += chunk.token_count
        
        content = "\n\n---\n\n".join(content_parts)
        return content, list(sections_included)
    
    def _run_analysis(
        self,
        template_name: AnalysisType,
        paper: ExtractedPaper,
        sections: list[str] | None = None,
        max_tokens: int = 8000,
        extra_kwargs: dict | None = None,
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Run an analysis using a prompt template.
        
        Args:
            template_name: Name of the prompt template to use
            paper: Paper to analyze
            sections: Specific sections to analyze
            max_tokens: Max tokens of content to include
            extra_kwargs: Additional kwargs for the prompt template
            verbose: Show progress
            
        Returns:
            AnalysisResult
        """
        template = get_template(template_name)
        
        if verbose:
            console.print(f"[cyan]{template.description}[/cyan] — {paper.title}...")
        
        # Get content
        content, analyzed_sections = self._get_content_for_analysis(
            paper, sections=sections, max_tokens=max_tokens
        )
        
        if not content.strip():
            # Fallback to full paper if targeted sections not found
            content, analyzed_sections = self._get_content_for_analysis(
                paper, max_tokens=max_tokens
            )
        
        # Format prompt
        kwargs = {"context": content}
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        
        system_prompt, user_prompt = template.format(**kwargs)
        
        # Generate response
        response = self.client.generate(
            prompt=user_prompt,
            system=system_prompt,
        )
        
        if verbose:
            console.print(f"[green]✓[/green] Complete")
        
        return AnalysisResult(
            paper_title=paper.title,
            analysis_type=template.description,
            content=response.content,
            sections_analyzed=analyzed_sections,
            chunks_processed=len(analyzed_sections),
            model_used=self.client.model,
        )
    
    def summarize(
        self,
        paper: ExtractedPaper,
        mode: Literal["full", "brief"] = "full",
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Summarize a paper.
        
        Args:
            paper: Paper to summarize
            mode: "full" for detailed summary, "brief" for abstract-level
            verbose: Show progress
            
        Returns:
            AnalysisResult with summary
        """
        if mode == "brief" and paper.abstract:
            template = get_template("summarize_brief")
            
            if verbose:
                console.print(f"[cyan]{template.description}[/cyan] — {paper.title}...")
            
            content, sections = self._get_content_for_analysis(
                paper,
                sections=["Introduction", "Results", "Conclusion"],
                max_tokens=4000,
            )
            
            system_prompt, user_prompt = template.format(
                abstract=paper.abstract,
                context=content,
            )
            
            response = self.client.generate(prompt=user_prompt, system=system_prompt)
            
            if verbose:
                console.print(f"[green]✓[/green] Complete")
            
            return AnalysisResult(
                paper_title=paper.title,
                analysis_type="Brief Summary",
                content=response.content,
                sections_analyzed=sections,
                model_used=self.client.model,
            )
        else:
            return self._run_analysis(
                "summarize_paper",
                paper,
                max_tokens=8000,
                verbose=verbose,
            )
    
    def critique_methodology(
        self,
        paper: ExtractedPaper,
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Critique the methodology of a paper.
        
        Args:
            paper: Paper to critique
            verbose: Show progress
            
        Returns:
            AnalysisResult with methodology critique
        """
        return self._run_analysis(
            "critique_methods",
            paper,
            sections=["Methods", "Methodology", "Materials And Methods", "Experimental", "Results"],
            max_tokens=6000,
            verbose=verbose,
        )
    
    def critique_statistics(
        self,
        paper: ExtractedPaper,
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Critique the statistical analysis of a paper.
        
        Args:
            paper: Paper to critique
            verbose: Show progress
            
        Returns:
            AnalysisResult with statistical critique
        """
        return self._run_analysis(
            "critique_statistics",
            paper,
            sections=["Methods", "Results", "Statistics", "Analysis"],
            max_tokens=6000,
            verbose=verbose,
        )
    
    def critique_full(
        self,
        paper: ExtractedPaper,
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Comprehensive critique of a paper.
        
        Args:
            paper: Paper to critique
            verbose: Show progress
            
        Returns:
            AnalysisResult with full critique
        """
        return self._run_analysis(
            "critique_full",
            paper,
            max_tokens=8000,
            verbose=verbose,
        )
    
    def extract_claims(
        self,
        paper: ExtractedPaper,
        sections: list[str] | None = None,
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Extract key claims from a paper.
        
        Args:
            paper: Paper to analyze
            sections: Specific sections to extract from
            verbose: Show progress
            
        Returns:
            AnalysisResult with extracted claims
        """
        target_sections = sections or ["Abstract", "Results", "Discussion", "Conclusion"]
        return self._run_analysis(
            "extract_claims",
            paper,
            sections=target_sections,
            max_tokens=6000,
            verbose=verbose,
        )
    
    def identify_limitations(
        self,
        paper: ExtractedPaper,
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Identify stated and unstated limitations.
        
        Args:
            paper: Paper to analyze
            verbose: Show progress
            
        Returns:
            AnalysisResult with limitations analysis
        """
        return self._run_analysis(
            "identify_limitations",
            paper,
            sections=["Methods", "Results", "Discussion", "Limitations", "Conclusion"],
            max_tokens=6000,
            verbose=verbose,
        )
    
    def bullshit_score(
        self,
        paper: ExtractedPaper,
        verbose: bool = True,
    ) -> BullshitScore:
        """
        Calculate the bullshit score for a paper.
        
        The crown jewel of SCALPEL. Evaluates scientific rigor
        and assigns a score from 0-10 (lower is better).
        
        Args:
            paper: Paper to evaluate
            verbose: Show progress
            
        Returns:
            BullshitScore with breakdown and red flags
        """
        template = get_template("bullshit_score")
        
        if verbose:
            console.print(f"[cyan]🔪 Calculating bullshit score[/cyan] — {paper.title}...")
        
        content, sections = self._get_content_for_analysis(paper, max_tokens=8000)
        
        system_prompt, user_prompt = template.format(context=content)
        
        response = self.client.generate(prompt=user_prompt, system=system_prompt)
        
        # Parse the response
        score = self._parse_bullshit_score(response.content)
        
        if verbose:
            console.print(f"[green]✓[/green] Bullshit score calculated")
        
        return score
    
    def _parse_bullshit_score(self, response: str) -> BullshitScore:
        """Parse the LLM response into a structured BullshitScore."""
        
        # Try to extract overall score
        overall_match = re.search(
            r"BULLSHIT SCORE:\s*\[?(\d+(?:\.\d+)?)\s*(?:/\s*10)?\]?",
            response,
            re.IGNORECASE
        )
        overall_score = float(overall_match.group(1)) if overall_match else 5.0
        
        # Try to extract component scores
        methodology_match = re.search(
            r"Methodology.*?:\s*\[?(\d+(?:\.\d+)?)\s*(?:/\s*10)?\]?",
            response,
            re.IGNORECASE
        )
        statistics_match = re.search(
            r"Statistics.*?:\s*\[?(\d+(?:\.\d+)?)\s*(?:/\s*10)?\]?",
            response,
            re.IGNORECASE
        )
        claims_match = re.search(
            r"Claims.*?:\s*\[?(\d+(?:\.\d+)?)\s*(?:/\s*10)?\]?",
            response,
            re.IGNORECASE
        )
        transparency_match = re.search(
            r"Transparency.*?:\s*\[?(\d+(?:\.\d+)?)\s*(?:/\s*10)?\]?",
            response,
            re.IGNORECASE
        )
        
        # Extract red flags section
        red_flags = []
        flags_match = re.search(
            r"RED FLAGS:\s*\n(.*?)(?=\n\n|SUMMARY:|Final|$)",
            response,
            re.IGNORECASE | re.DOTALL
        )
        if flags_match:
            flags_text = flags_match.group(1)
            for line in flags_text.split("\n"):
                line = line.strip().lstrip("•-*").strip()
                # Filter out empty lines, template text, and "none" variations
                if (line 
                    and len(line) > 3
                    and not line.lower().startswith("none")
                    and "list any serious" not in line.lower()
                    and "[list" not in line.lower()
                    and "if clean" not in line.lower()):
                    red_flags.append(line)
        
        # Extract summary
        summary = ""
        summary_match = re.search(
            r"SUMMARY:\s*\n?(.*?)$",
            response,
            re.IGNORECASE | re.DOTALL
        )
        if summary_match:
            summary = summary_match.group(1).strip()
        
        return BullshitScore(
            overall_score=overall_score,
            methodology_score=float(methodology_match.group(1)) if methodology_match else None,
            statistics_score=float(statistics_match.group(1)) if statistics_match else None,
            claims_score=float(claims_match.group(1)) if claims_match else None,
            transparency_score=float(transparency_match.group(1)) if transparency_match else None,
            red_flags=red_flags,
            summary=summary,
            raw_response=response,
        )
    
    def quick_assess(
        self,
        paper: ExtractedPaper,
        verbose: bool = True,
    ) -> AnalysisResult:
        """
        Quick red flag assessment.
        
        Args:
            paper: Paper to assess
            verbose: Show progress
            
        Returns:
            AnalysisResult with quick assessment
        """
        return self._run_analysis(
            "quick_assess",
            paper,
            max_tokens=4000,
            verbose=verbose,
        )
    
    def full_analysis(
        self,
        paper: ExtractedPaper,
        verbose: bool = True,
    ) -> dict[str, AnalysisResult | BullshitScore]:
        """
        Run a complete analysis of a paper.
        
        Includes: summary, methodology critique, claims, limitations, and bullshit score.
        
        Args:
            paper: Paper to analyze
            verbose: Show progress
            
        Returns:
            Dict with all analysis results
        """
        if verbose:
            console.print(f"\n[bold]🔪 Full SCALPEL Analysis:[/bold] {paper.title}\n")
        
        results = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            
            task = progress.add_task("Summarizing...", total=None)
            results["summary"] = self.summarize(paper, verbose=False)
            
            progress.update(task, description="Critiquing methodology...")
            results["methodology"] = self.critique_methodology(paper, verbose=False)
            
            progress.update(task, description="Critiquing statistics...")
            results["statistics"] = self.critique_statistics(paper, verbose=False)
            
            progress.update(task, description="Extracting claims...")
            results["claims"] = self.extract_claims(paper, verbose=False)
            
            progress.update(task, description="Identifying limitations...")
            results["limitations"] = self.identify_limitations(paper, verbose=False)
            
            progress.update(task, description="Calculating bullshit score...")
            results["bullshit_score"] = self.bullshit_score(paper, verbose=False)
        
        if verbose:
            console.print(f"[green]✓[/green] Full analysis complete\n")
        
        return results


# Convenience functions
_default_analyzer: Analyzer | None = None


def get_analyzer() -> Analyzer:
    """Get or create the default analyzer."""
    global _default_analyzer
    if _default_analyzer is None:
        _default_analyzer = Analyzer()
    return _default_analyzer


def summarize(paper: ExtractedPaper, **kwargs) -> AnalysisResult:
    """Summarize a paper using the default analyzer."""
    return get_analyzer().summarize(paper, **kwargs)


def critique_methodology(paper: ExtractedPaper, **kwargs) -> AnalysisResult:
    """Critique methodology using the default analyzer."""
    return get_analyzer().critique_methodology(paper, **kwargs)


def critique_statistics(paper: ExtractedPaper, **kwargs) -> AnalysisResult:
    """Critique statistics using the default analyzer."""
    return get_analyzer().critique_statistics(paper, **kwargs)


def critique_full(paper: ExtractedPaper, **kwargs) -> AnalysisResult:
    """Full critique using the default analyzer."""
    return get_analyzer().critique_full(paper, **kwargs)


def extract_claims(paper: ExtractedPaper, **kwargs) -> AnalysisResult:
    """Extract claims using the default analyzer."""
    return get_analyzer().extract_claims(paper, **kwargs)


def identify_limitations(paper: ExtractedPaper, **kwargs) -> AnalysisResult:
    """Identify limitations using the default analyzer."""
    return get_analyzer().identify_limitations(paper, **kwargs)


def bullshit_score(paper: ExtractedPaper, **kwargs) -> BullshitScore:
    """Calculate bullshit score using the default analyzer."""
    return get_analyzer().bullshit_score(paper, **kwargs)


def full_analysis(paper: ExtractedPaper, **kwargs) -> dict:
    """Run full analysis using the default analyzer."""
    return get_analyzer().full_analysis(paper, **kwargs)


if __name__ == "__main__":
    console.print("[bold]🔪 SCALPEL Analyzer[/bold]\n")
    console.print("[dim]Available analysis functions:[/dim]")
    console.print("  • summarize(paper)")
    console.print("  • critique_methodology(paper)")
    console.print("  • critique_statistics(paper)")
    console.print("  • critique_full(paper)")
    console.print("  • extract_claims(paper)")
    console.print("  • identify_limitations(paper)")
    console.print("  • bullshit_score(paper)  [bold]← The crown jewel[/bold]")
    console.print("  • full_analysis(paper)")
    console.print("\n[dim]Example:[/dim]")
    console.print("  from scalpel.ingestion import extract_pdf")
    console.print("  from scalpel.analysis import bullshit_score")
    console.print("  paper = extract_pdf('paper.pdf')")
    console.print("  bs = bullshit_score(paper)")
    console.print("  bs.display()")