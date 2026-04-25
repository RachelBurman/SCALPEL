"""
Cross-reference module for BEAR.

Queries both the 'papers' and 'markets' LanceDB collections simultaneously
to find scientific literature that supports or contradicts company claims.
"""

from dataclasses import dataclass

from rich.markdown import Markdown
from rich.panel import Panel

from scalpel.analysis.llm_client import LLMClient, get_client
from scalpel.bear.ingestion import MarketSearchResult, MarketStore, get_market_store
from scalpel.console import console
from scalpel.embeddings.vector_store import SearchResult, VectorStore, get_store


_SYSTEM = (
    "You are a financial analyst who synthesises scientific research with company claims. "
    "Identify exactly where research supports, contradicts, or is silent on corporate narratives."
)

_PROMPT = """\
Cross-reference the following company claims against the relevant scientific literature.

{ticker} COMPANY DATA:
{market_context}

RELEVANT SCIENTIFIC LITERATURE:
{papers_context}

Provide:
1. SUPPORTED CLAIMS: Where the literature backs what the company says (cite paper titles)
2. CONTRADICTED CLAIMS: Where research undermines or contradicts company claims (cite paper titles)
3. UNSUPPORTED CLAIMS: Major company claims with no scientific backing in the library
4. MISSING CONTEXT: Important research the company appears to be ignoring

Be specific and cite paper sources where possible.\
"""


@dataclass
class CrossReferenceResult:
    """Result of cross-referencing a company against the scientific paper library."""

    ticker: str
    company_name: str
    market_results: list[MarketSearchResult]
    paper_results: list[SearchResult]
    analysis: str

    def display(self) -> None:
        market_lines = "\n".join(
            f"  • [{r.ticker}] {r.section} (relevance: {r.score:.3f})"
            for r in self.market_results[:5]
        )
        paper_lines = "\n".join(
            f"  • {r.paper_title}{f' [{r.section}]' if r.section else ''} "
            f"(relevance: {r.score:.3f})"
            for r in self.paper_results[:5]
        )

        console.print(Panel(
            f"[bold]Market data chunks used:[/bold]\n{market_lines or '  None'}\n\n"
            f"[bold]Scientific papers used:[/bold]\n{paper_lines or '  None'}",
            title=f"[bold cyan]Sources — {self.ticker}[/bold cyan]",
            border_style="dim",
        ))

        console.print(Panel(
            Markdown(self.analysis),
            title=f"[bold]Cross-Reference Analysis — {self.company_name} ({self.ticker})[/bold]",
            border_style="cyan",
        ))


class CrossReferencer:
    """
    Cross-references market data against scientific literature.

    Queries both the 'markets' LanceDB collection (company data) and the
    'papers' LanceDB collection (scientific literature) simultaneously,
    then uses the LLM to synthesise conflicts and alignments.
    """

    def __init__(
        self,
        client: LLMClient | None = None,
        paper_store: VectorStore | None = None,
        market_store: MarketStore | None = None,
    ):
        self.client = client or get_client()
        self.paper_store = paper_store or get_store()
        self.market_store = market_store or get_market_store()

    def cross_reference(
        self,
        ticker: str,
        n_market_results: int = 10,
        n_paper_results: int = 10,
        verbose: bool = True,
    ) -> CrossReferenceResult:
        """
        Cross-reference a company against the scientific paper library.

        Retrieves relevant chunks from both LanceDB collections, then uses
        the LLM to identify where research supports or contradicts company claims.
        """
        ticker = ticker.upper()
        if verbose:
            console.print(
                f"\n[cyan]Cross-referencing[/cyan] [bold]{ticker}[/bold] "
                f"against paper library..."
            )

        market_context = self.market_store.get_company_context(ticker)
        if not market_context:
            raise ValueError(
                f"No market data for '{ticker}'. "
                f"Run [cyan]scalpel bear add {ticker}[/cyan] first."
            )

        company_name = ticker
        for line in market_context.splitlines():
            if line.startswith("Company:"):
                company_name = line.split(":", 1)[1].split("(")[0].strip()
                break

        # Query the markets collection for the most relevant chunks
        if verbose:
            console.print("[dim]Querying markets collection...[/dim]")
        market_results = self.market_store.search(
            f"{company_name} technology claims evidence",
            n_results=n_market_results,
            ticker_filter=ticker,
        )

        # Query the papers collection — try company name first, fall back to sector/overview
        if verbose:
            console.print("[dim]Querying papers collection...[/dim]")

        first_word = company_name.split()[0] if company_name else ticker
        paper_results = self.paper_store.search(
            f"{first_word} {company_name}",
            n_results=n_paper_results,
        )

        # If no strong signal on company name, broaden to sector/technology description
        if not paper_results or all(r.score < 0.25 for r in paper_results):
            overview_snippet = " ".join(market_context.split()[:60])
            paper_results = self.paper_store.search(overview_snippet, n_results=n_paper_results)

        if verbose:
            strong = len([r for r in paper_results if r.score > 0.25])
            console.print(
                f"[dim]Retrieved {strong} relevant paper chunks, "
                f"{len(market_results)} market data chunks[/dim]"
            )
            console.print("[dim]Generating analysis...[/dim]")

        papers_context = (
            "\n\n---\n\n".join(
                f"[{r.paper_title}{f' | {r.section}' if r.section else ''}]\n{r.text}"
                for r in paper_results
            )
            if paper_results
            else "No relevant scientific papers found in library."
        )

        prompt = _PROMPT.format(
            ticker=ticker,
            market_context=market_context[:8000],
            papers_context=papers_context[:8000],
        )
        response = self.client.generate(prompt=prompt, system=_SYSTEM)

        if verbose:
            console.print(f"[green]✓[/green] Cross-reference complete")

        return CrossReferenceResult(
            ticker=ticker,
            company_name=company_name,
            market_results=market_results,
            paper_results=paper_results,
            analysis=response.content,
        )


_default_cross_referencer: CrossReferencer | None = None


def get_cross_referencer() -> CrossReferencer:
    """Get or create the default CrossReferencer."""
    global _default_cross_referencer
    if _default_cross_referencer is None:
        _default_cross_referencer = CrossReferencer()
    return _default_cross_referencer


def cross_reference(ticker: str, **kwargs) -> CrossReferenceResult:
    """Cross-reference a ticker using the default CrossReferencer."""
    return get_cross_referencer().cross_reference(ticker, **kwargs)
