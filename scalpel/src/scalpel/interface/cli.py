"""
SCALPEL Command Line Interface.

Scientific Critique & Analysis Pipeline for Evidence Literature.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from scalpel import __version__

console = Console()
app = typer.Typer(
    name="scalpel",
    help="🔪 SCALPEL - Scientific Critique & Analysis Pipeline for Evidence Literature",
    add_completion=False,
    rich_markup_mode="rich",
)


def version_callback(value: bool):
    """Show version and exit."""
    if value:
        console.print(f"[bold cyan]SCALPEL[/bold cyan] version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
):
    """
    🔪 SCALPEL - Scientific Critique & Analysis Pipeline for Evidence Literature
    
    Analyze, critique, and search academic papers with AI-powered insights.
    """
    pass


# =============================================================================
# INGESTION COMMANDS
# =============================================================================

@app.command()
def add(
    filepath: Path = typer.Argument(
        ...,
        help="Path to PDF file to add to the library.",
        exists=True,
        readable=True,
    ),
):
    """
    Add a PDF paper to the library.
    
    Extracts text, chunks it, generates embeddings, and indexes for search.
    """
    from scalpel.ingestion import extract_pdf
    from scalpel.embeddings import add_paper
    
    console.print(f"\n[bold]Adding paper to library...[/bold]\n")
    
    try:
        paper = extract_pdf(filepath)
        count = add_paper(paper)
        
        if count > 0:
            console.print(f"\n[green]✓[/green] Added [bold]{paper.title}[/bold] ({count} chunks indexed)")
        else:
            console.print(f"\n[yellow]Paper already in library.[/yellow] Use [cyan]scalpel remove[/cyan] first to re-index.")
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("add-arxiv")
def add_arxiv(
    identifier: str = typer.Argument(
        ...,
        help="arXiv ID or URL (e.g., '1706.03762' or 'https://arxiv.org/abs/1706.03762')",
    ),
):
    """
    Fetch a paper from arXiv and add it to the library.
    
    Accepts arXiv IDs (1706.03762) or full URLs.
    """
    from scalpel.ingestion import fetch_arxiv
    from scalpel.embeddings import add_paper
    
    console.print(f"\n[bold]Fetching from arXiv...[/bold]\n")
    
    try:
        paper = fetch_arxiv(identifier)
        count = add_paper(paper)
        
        if count > 0:
            console.print(f"\n[green]✓[/green] Added [bold]{paper.title}[/bold] ({count} chunks indexed)")
        else:
            console.print(f"\n[yellow]Paper already in library.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("list")
def list_papers():
    """
    List all papers in the library.
    """
    from scalpel.embeddings import list_papers as get_papers
    
    papers = get_papers()
    
    if not papers:
        console.print("\n[dim]No papers in library yet.[/dim]")
        console.print("[dim]Use [cyan]scalpel add <pdf>[/cyan] or [cyan]scalpel add-arxiv <id>[/cyan] to add papers.[/dim]\n")
        return
    
    table = Table(title=f"📚 Paper Library ({len(papers)} papers)")
    table.add_column("Title", style="cyan", max_width=60)
    table.add_column("Chunks", style="green", justify="right")
    
    for paper in papers:
        title = paper["title"]
        if len(title) > 57:
            title = title[:57] + "..."
        table.add_row(title, str(paper["chunk_count"]))
    
    console.print()
    console.print(table)
    console.print()


@app.command()
def remove(
    title: str = typer.Argument(
        ...,
        help="Title (or partial title) of the paper to remove.",
    ),
):
    """
    Remove a paper from the library.
    """
    from scalpel.embeddings import get_store
    
    store = get_store()
    papers = store.list_papers()
    
    # Find matching paper
    matches = [p for p in papers if title.lower() in p["title"].lower()]
    
    if not matches:
        console.print(f"\n[yellow]No paper found matching:[/yellow] {title}")
        return
    
    if len(matches) > 1:
        console.print(f"\n[yellow]Multiple matches found:[/yellow]")
        for p in matches:
            console.print(f"  • {p['title']}")
        console.print("\n[dim]Please be more specific.[/dim]")
        return
    
    paper = matches[0]
    
    if typer.confirm(f"Remove '{paper['title']}'?"):
        count = store.delete_paper(paper_title=paper["title"])
        console.print(f"\n[green]✓[/green] Removed {count} chunks")
    else:
        console.print("\n[dim]Cancelled.[/dim]")


# =============================================================================
# SEARCH COMMANDS
# =============================================================================

@app.command()
def search(
    query: str = typer.Argument(
        ...,
        help="Search query (natural language).",
    ),
    n: int = typer.Option(
        5,
        "--results", "-n",
        help="Number of results to return.",
    ),
    section: Optional[str] = typer.Option(
        None,
        "--section", "-s",
        help="Filter by section (e.g., 'Methods', 'Results').",
    ),
):
    """
    Search across all papers in the library.
    
    Uses semantic search to find relevant content.
    """
    from scalpel.embeddings import search as do_search
    
    results = do_search(query, n_results=n, section_filter=section)
    
    if not results:
        console.print(f"\n[yellow]No results found for:[/yellow] {query}")
        console.print("[dim]Make sure you have papers indexed with [cyan]scalpel add[/cyan][/dim]\n")
        return
    
    console.print(f"\n[bold]Search results for:[/bold] {query}\n")
    
    for i, result in enumerate(results, 1):
        section_info = f" [{result.section}]" if result.section else ""
        
        # Truncate text preview
        preview = result.text[:200].replace("\n", " ")
        if len(result.text) > 200:
            preview += "..."
        
        console.print(Panel(
            f"[dim]{preview}[/dim]",
            title=f"[cyan]{i}. {result.paper_title}{section_info}[/cyan]",
            subtitle=f"[green]Score: {result.score:.3f}[/green]",
            border_style="dim",
        ))
    
    console.print()


# =============================================================================
# ANALYSIS COMMANDS
# =============================================================================

@app.command()
def analyze(
    filepath: Path = typer.Argument(
        ...,
        help="Path to PDF file to analyze.",
        exists=True,
        readable=True,
    ),
    full: bool = typer.Option(
        False,
        "--full", "-f",
        help="Run full analysis (slower but comprehensive).",
    ),
):
    """
    Analyze a paper with AI-powered critique.
    
    By default, runs a quick assessment. Use --full for comprehensive analysis.
    """
    from scalpel.ingestion import extract_pdf
    from scalpel.analysis import full_analysis, bullshit_score, summarize
    
    console.print(f"\n[bold]🔪 Analyzing paper...[/bold]\n")
    
    try:
        paper = extract_pdf(filepath)
        
        if full:
            results = full_analysis(paper)
            
            console.print("\n")
            results["summary"].display()
            results["methodology"].display()
            results["bullshit_score"].display()
        else:
            # Quick analysis: summary + bullshit score
            summary = summarize(paper, mode="brief")
            bs = bullshit_score(paper)
            
            console.print("\n")
            summary.display()
            bs.display()
            
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def summarize(
    filepath: Path = typer.Argument(
        ...,
        help="Path to PDF file to summarize.",
        exists=True,
        readable=True,
    ),
    brief: bool = typer.Option(
        False,
        "--brief", "-b",
        help="Generate a brief summary (faster).",
    ),
):
    """
    Generate a summary of a paper.
    """
    from scalpel.ingestion import extract_pdf
    from scalpel.analysis import summarize as do_summarize
    
    try:
        paper = extract_pdf(filepath)
        mode = "brief" if brief else "full"
        result = do_summarize(paper, mode=mode)
        
        console.print("\n")
        result.display()
        
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@app.command("bs")
def bullshit_score_cmd(
    filepath: Path = typer.Argument(
        ...,
        help="Path to PDF file to evaluate.",
        exists=True,
        readable=True,
    ),
):
    """
    Calculate the Bullshit Score for a paper.
    
    The crown jewel of SCALPEL. Evaluates scientific rigor on a 0-10 scale.
    """
    from scalpel.ingestion import extract_pdf
    from scalpel.analysis import bullshit_score
    
    try:
        paper = extract_pdf(filepath)
        bs = bullshit_score(paper)
        
        console.print("\n")
        bs.display()
        
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def critique(
    filepath: Path = typer.Argument(
        ...,
        help="Path to PDF file to critique.",
        exists=True,
        readable=True,
    ),
    focus: str = typer.Option(
        "full",
        "--focus", "-f",
        help="Focus area: 'methods', 'statistics', 'claims', or 'full'.",
    ),
):
    """
    Generate a detailed critique of a paper.
    """
    from scalpel.ingestion import extract_pdf
    from scalpel.analysis import critique_methodology, critique_statistics, critique_full
    from scalpel.analysis import extract_claims
    
    try:
        paper = extract_pdf(filepath)
        
        if focus == "methods":
            result = critique_methodology(paper)
        elif focus == "statistics":
            result = critique_statistics(paper)
        elif focus == "claims":
            result = extract_claims(paper)
        else:
            result = critique_full(paper)
        
        console.print("\n")
        result.display()
        
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# INFO COMMANDS
# =============================================================================

@app.command()
def stats():
    """
    Show library statistics.
    """
    from scalpel.embeddings import get_store
    from scalpel.config import settings
    
    store = get_store()
    info = store.get_stats()
    
    console.print(Panel(
        f"""[bold]Papers indexed:[/bold] {info['total_papers']}
[bold]Total chunks:[/bold] {info['total_chunks']}

[bold]Embedding model:[/bold] {info['embedding_model']}
[bold]LLM model:[/bold] {settings.ollama_model}
[bold]Database:[/bold] {info['db_path']}""",
        title="[bold cyan]🔪 SCALPEL Stats[/bold cyan]",
        border_style="cyan",
    ))


@app.command()
def config():
    """
    Show current configuration.
    """
    from scalpel.config import settings
    
    console.print(Panel(
        f"""[bold]Ollama Host:[/bold] {settings.ollama_host}
[bold]LLM Model:[/bold] {settings.ollama_model}
[bold]Embedding Model:[/bold] {settings.embedding_model}
[bold]Temperature:[/bold] {settings.model_temperature}
[bold]Context Length:[/bold] {settings.model_context_length:,}

[bold]Chunk Size:[/bold] {settings.chunk_size} tokens
[bold]Chunk Overlap:[/bold] {settings.chunk_overlap} tokens

[bold]Papers Directory:[/bold] {settings.papers_dir}
[bold]Vector DB:[/bold] {settings.lancedb_path}""",
        title="[bold cyan]⚙️ Configuration[/bold cyan]",
        border_style="cyan",
    ))


def run():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()