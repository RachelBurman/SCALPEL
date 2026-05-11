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


def _save_report(content: str, output: Optional[Path], default_name: str) -> None:
    """Save a markdown report. Uses output path if given, otherwise auto-saves to reports_dir."""
    from datetime import datetime
    from scalpel.config import settings

    if output is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        output = settings.reports_dir / f"{default_name}_{timestamp}.md"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    console.print(f"\n[green]✓[/green] Report saved to [cyan]{output}[/cyan]")


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
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
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
            _save_report(
                "\n\n".join(r.to_markdown() for r in results.values()),
                output,
                f"analyse_{filepath.stem}",
            )
        else:
            summary = summarize(paper, mode="brief")
            bs = bullshit_score(paper)

            console.print("\n")
            summary.display()
            bs.display()
            _save_report(summary.to_markdown() + "\n\n" + bs.to_markdown(), output, f"analyse_{filepath.stem}")

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
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
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
        _save_report(result.to_markdown(), output, f"summarize_{filepath.stem}")

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
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
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
        _save_report(bs.to_markdown(), output, f"bs_{filepath.stem}")

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
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
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
        _save_report(result.to_markdown(), output, f"critique_{focus}_{filepath.stem}")

    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


# =============================================================================
# EVAL COMMAND
# =============================================================================

@app.command()
def eval(
    query: str = typer.Argument(
        ...,
        help="Question to ask against the indexed papers.",
    ),
    n: int = typer.Option(
        5,
        "--results", "-n",
        help="Number of chunks to retrieve.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
    ),
):
    """
    Ask a question, generate a RAG response, then score it.

    Retrieves relevant chunks, generates an answer, and runs the LLM judge
    to score groundedness, relevance, and confidence calibration.
    """
    from scalpel.embeddings import search as do_search
    from scalpel.analysis.llm_client import get_client
    from scalpel.analysis.prompts import get_template
    from scalpel.evaluation import evaluate as eval_response

    console.print(f"\n[dim]Searching for relevant context...[/dim]")
    results = do_search(query, n_results=n)

    if not results:
        console.print("[yellow]No indexed papers found. Run [cyan]scalpel add[/cyan] first.[/yellow]")
        raise typer.Exit(1)

    chunk_texts = [r.text for r in results]
    chunks_display = "\n\n---\n\n".join(
        f"[{r.paper_title}]\n{r.text}" for r in results
    )

    console.print("[dim]Generating response...[/dim]")
    client = get_client()
    template = get_template("rag_query")
    system_prompt, user_prompt = template.format(question=query, context=chunks_display)
    llm_response = client.generate(prompt=user_prompt, system=system_prompt)

    console.print(Panel(
        Markdown(llm_response.content),
        title=f"[bold]Response[/bold]",
        subtitle=f"[dim]{len(results)} chunks retrieved[/dim]",
        border_style="cyan",
    ))

    console.print()
    score = eval_response(query, chunk_texts, llm_response.content)
    score.display()
    console.print()

    _save_report(
        f"# Query: {query}\n\n## Response\n\n{llm_response.content}\n\n{score.to_markdown()}",
        output,
        "eval",
    )


# =============================================================================
# AGENT COMMAND
# =============================================================================

@app.command()
def agent(
    goal: str = typer.Argument(
        ...,
        help="Research goal or question for the agent to investigate autonomously.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save the agent report to this file (markdown).",
    ),
):
    """
    Run the SCALPEL autonomous research agent.

    The agent decides which papers to retrieve, which analyses to run,
    and how to sequence its reasoning — no per-step instruction needed.

    Requires a tool-calling capable model (Qwen 2.5 7B+ or any OpenRouter model).

    Example:
        scalpel agent "critique the statistical methods across all indexed papers"
    """
    from langchain_core.messages import AIMessage, ToolMessage
    from scalpel.agent.runner import stream_run

    console.print(f"\n[bold cyan]🔪 SCALPEL Agent[/bold cyan]\n")
    console.print(Panel(
        f"[bold]Goal:[/bold] {goal}",
        border_style="dim",
    ))
    console.print()

    report = ""
    step = 0

    try:
        for event in stream_run(goal):

            if "planner" in event:
                step += 1
                msgs = event["planner"].get("messages", [])
                if msgs:
                    last = msgs[-1]
                    tool_calls = getattr(last, "tool_calls", None)
                    if tool_calls:
                        for tc in tool_calls:
                            args = tc.get("args", {})
                            # Show first arg value as a short preview
                            preview = next(iter(args.values()), "") if args else ""
                            if isinstance(preview, str) and len(preview) > 50:
                                preview = preview[:50] + "..."
                            preview_str = f"({preview!r})" if preview else "()"
                            console.print(
                                f"[dim]Step {step}[/dim] [cyan]→ {tc['name']}{preview_str}[/cyan]"
                            )
                    else:
                        console.print(f"[dim]Step {step}[/dim] [green]Writing report...[/green]")

            elif "tools" in event:
                msgs = event["tools"].get("messages", [])
                for msg in msgs:
                    if isinstance(msg, ToolMessage):
                        first_line = msg.content.split("\n")[0][:90]
                        console.print(f"[dim]  ↳ {first_line}[/dim]")

            elif "synthesise" in event:
                report = event["synthesise"].get("report", "")

    except Exception as e:
        console.print(f"\n[red]✗ Agent error:[/red] {e}")
        console.print(
            "[dim]Check that your model supports tool calling. "
            "Qwen 2.5 7B+ or any OpenRouter model recommended.[/dim]"
        )
        raise typer.Exit(1)

    if not report:
        console.print("\n[yellow]Agent finished but produced no report.[/yellow]")
        raise typer.Exit(1)

    console.print()
    console.print(Panel(
        Markdown(report),
        title="[bold]🔪 Agent Report[/bold]",
        border_style="cyan",
    ))

    _save_report(report, output, "agent")


# =============================================================================
# STREAMLIT UI COMMAND
# =============================================================================

@app.command()
def ui():
    """
    Launch the SCALPEL Streamlit web interface.

    Opens a browser with the full UI: query, chunk viewer, evaluation scores,
    and RL training history charts.
    Requires: pip install scalpel[ui]
    """
    import subprocess
    import sys
    from pathlib import Path

    app_path = Path(__file__).parent / "app.py"

    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            check=True,
        )
    except FileNotFoundError:
        console.print("[red]✗ Streamlit not installed.[/red]")
        console.print("[dim]Install with: pip install scalpel[ui][/dim]")
        raise typer.Exit(1)


# =============================================================================
# GUI COMMAND
# =============================================================================

@app.command()
def gui():
    """
    Launch the SCALPEL terminal user interface.
    
    Opens a Textual-based TUI in your terminal.
    Requires: pip install scalpel[gui]
    """
    try:
        from scalpel.interface.gui_textual import main as run_gui
    except ImportError:
        console.print("[red]✗ Textual not installed![/red]")
        console.print("[dim]Install with: pip install scalpel[gui][/dim]")
        raise typer.Exit(1)
    
    run_gui()


# =============================================================================
# COLLECT COMMAND
# =============================================================================

DEFAULT_QUERIES = [
    "What statistical methods were used to validate the results?",
    "What are the main limitations acknowledged by the authors?",
    "How was the sample size determined and is it adequate?",
    "What is the effect size and is it practically significant?",
    "Are the conclusions supported by the data presented?",
    "What potential confounds were not addressed?",
    "How does this study compare to prior work in the field?",
    "What are the key assumptions underlying the methodology?",
]


@app.command()
def collect(
    steps: int = typer.Option(
        4, "--steps", "-s",
        help="Random parameter combinations to try per query.",
    ),
    queries_file: Optional[Path] = typer.Option(
        None, "--queries", "-q",
        help="Text file with one query per line. Uses built-in defaults if omitted.",
        exists=True, readable=True,
    ),
    cache_path: Path = typer.Option(
        Path("models/retrieval/step_cache.json"),
        "--cache", "-c",
        help="Path to the cache file.",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show per-step details."),
):
    """
    Collect retrieval parameter data to improve future searches.

    Tries random retrieval parameter combinations, scores responses with the
    LLM judge, and caches results. Run repeatedly to build up the similarity
    index — more data means smarter parameter selection.

    Requires papers to be indexed (run scalpel add / add-arxiv first).
    """
    from scalpel.retrieval.cache import RetrievalCache
    from scalpel.retrieval.collector import RAGCollector
    from scalpel.retrieval.optimizer import RetrievalOptimizer

    if queries_file:
        queries = [
            line.strip()
            for line in queries_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        console.print(f"\n[dim]Loaded {len(queries)} queries from {queries_file}[/dim]")
    else:
        queries = DEFAULT_QUERIES
        console.print(f"\n[dim]Using {len(queries)} built-in queries[/dim]")

    cache = RetrievalCache(path=cache_path)
    collector = RAGCollector(cache=cache, verbose=verbose)

    console.print(
        f"[bold]Starting collection:[/bold] {len(queries)} queries × {steps} steps  "
        f"(cache: {len(cache)} entries)\n"
    )

    try:
        stats = collector.collect(queries=queries, steps_per_query=steps)
    except Exception as e:
        console.print(f"\n[red]✗ Collection failed:[/red] {e}")
        raise typer.Exit(1)

    console.print(f"\n[bold green]Collection complete[/bold green]")
    console.print(f"  Total steps:      {stats.total_steps}")
    console.print(f"  Cache hits:       {stats.cache_hits}")
    console.print(f"  New evaluations:  {stats.new_evaluations}")
    if stats.scores:
        console.print(f"  Mean score:       {stats.mean_score:.2f} / 10")
        console.print(f"  Best score:       {stats.best_score:.2f} / 10")
    if stats.daily_limit_hit:
        console.print("\n[yellow]Daily API limit reached — run again tomorrow to collect more.[/yellow]")

    # Show index size after rebuild
    optimizer = RetrievalOptimizer(cache)
    n = optimizer.build_index()
    console.print(f"\n[dim]Similarity index: {n} unique queries indexed[/dim]\n")


# =============================================================================
# SETUP COMMAND
# =============================================================================

OPENROUTER_FREE_MODELS_FALLBACK = [
    "google/gemma-2-9b-it:free",
    "microsoft/phi-3-mini-128k-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "other (enter manually)",
]


def _fetch_free_models(api_key: str) -> list[str]:
    """Fetch currently available free models from OpenRouter."""
    try:
        import httpx
        with httpx.Client(timeout=10, follow_redirects=True) as client:
            r = client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            r.raise_for_status()
            models = r.json().get("data", [])
            free = sorted(
                m["id"] for m in models
                if m["id"].endswith(":free")
            )
            return free + ["other (enter manually)"] if free else OPENROUTER_FREE_MODELS_FALLBACK
    except Exception:
        return OPENROUTER_FREE_MODELS_FALLBACK

OLLAMA_SUGGESTED_MODELS = [
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "llama3.2:1b",
    "llama3.2:3b",
    "phi3.5:mini",
    "gemma2:2b",
    "other (enter manually)",
]


@app.command()
def setup():
    """
    Interactive setup wizard.

    Configures your LLM provider and writes a .env file.
    Run this first if you're new to SCALPEL.
    """
    console.print(Panel(
        "[bold]Welcome to SCALPEL setup![/bold]\n\n"
        "This wizard will configure your LLM provider.\n"
        "Settings are saved to [cyan].env[/cyan] in your current directory.",
        title="[bold cyan]🔪 SCALPEL Setup[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # Choose provider
    console.print("[bold]LLM Provider[/bold]")
    console.print("  [cyan]1[/cyan] - Ollama (local, private, free — needs Ollama installed)")
    console.print("  [cyan]2[/cyan] - OpenRouter (cloud, free tier available — needs API key)")
    console.print()

    provider_choice = typer.prompt("Choose provider", default="1")
    while provider_choice not in ("1", "2"):
        provider_choice = typer.prompt("Please enter 1 or 2", default="1")

    env_lines: list[str] = []

    if provider_choice == "1":
        provider = "ollama"
        env_lines.append("LLM_PROVIDER=ollama")

        console.print("\n[bold]Ollama model[/bold]")
        for i, m in enumerate(OLLAMA_SUGGESTED_MODELS, 1):
            console.print(f"  [cyan]{i}[/cyan] - {m}")
        console.print()

        model_choice = typer.prompt("Choose model number", default="1")
        try:
            idx = int(model_choice) - 1
            if 0 <= idx < len(OLLAMA_SUGGESTED_MODELS) - 1:
                model = OLLAMA_SUGGESTED_MODELS[idx]
            else:
                model = typer.prompt("Enter model name")
        except ValueError:
            model = typer.prompt("Enter model name")

        env_lines.append(f"OLLAMA_MODEL={model}")

        ollama_host = typer.prompt("Ollama host", default="http://localhost:11434")
        if ollama_host != "http://localhost:11434":
            env_lines.append(f"OLLAMA_HOST={ollama_host}")

        console.print(f"\n[dim]After setup, pull the model with:[/dim]")
        console.print(f"[cyan]  ollama pull {model}[/cyan]")
        console.print(f"[cyan]  ollama pull nomic-embed-text[/cyan]  [dim](for search)[/dim]")

    else:
        provider = "openrouter"
        env_lines.append("LLM_PROVIDER=openrouter")

        console.print()
        console.print(
            "[dim]Get a free API key at [link=https://openrouter.ai]openrouter.ai[/link][/dim]\n"
        )
        api_key = typer.prompt("OpenRouter API key", hide_input=True)
        env_lines.append(f"OPENROUTER_API_KEY={api_key}")

        console.print("\n[dim]Fetching available free models...[/dim]")
        free_models = _fetch_free_models(api_key)
        console.print("\n[bold]Free models[/bold]")
        for i, m in enumerate(free_models, 1):
            console.print(f"  [cyan]{i}[/cyan] - {m}")
        console.print()

        model_choice = typer.prompt("Choose model number", default="1")
        try:
            idx = int(model_choice) - 1
            if 0 <= idx < len(free_models) - 1:
                model = free_models[idx]
            else:
                model = typer.prompt("Enter model name (e.g. google/gemma-2-9b-it:free)")
        except ValueError:
            model = typer.prompt("Enter model name")

        env_lines.append(f"OPENROUTER_MODEL={model}")

        console.print(
            "\n[dim]Note: search/embeddings still use Ollama (nomic-embed-text).\n"
            "Install Ollama and run [cyan]ollama pull nomic-embed-text[/cyan] to enable search.[/dim]"
        )

    # Write .env
    env_path = Path(".env")
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    for line in env_lines:
        k, _, v = line.partition("=")
        existing[k.strip()] = v.strip()

    env_content = "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n"
    env_path.write_text(env_content, encoding="utf-8")

    console.print(f"\n[green]✓[/green] Saved config to [cyan].env[/cyan]")
    console.print(f"[green]✓[/green] Provider: [bold]{provider}[/bold]  Model: [bold]{model}[/bold]")
    console.print("\n[dim]Run [cyan]scalpel config[/cyan] to verify, or [cyan]scalpel --help[/cyan] to get started.[/dim]\n")


# =============================================================================
# MODEL COMMAND
# =============================================================================

@app.command()
def model():
    """
    Switch the active LLM model.

    Fetches live free models from OpenRouter (if configured) and lets you
    pick a new one, then saves it to .env.
    """
    from scalpel.config import settings

    if settings.llm_provider != "openrouter":
        console.print("[yellow]Active provider is Ollama, not OpenRouter.[/yellow]")
        console.print("[dim]Run [cyan]scalpel setup[/cyan] to switch providers or change the Ollama model.[/dim]")
        raise typer.Exit()

    if not settings.openrouter_api_key:
        console.print("[red]✗ OPENROUTER_API_KEY not set.[/red]")
        console.print("[dim]Run [cyan]scalpel setup[/cyan] first.[/dim]")
        raise typer.Exit(1)

    console.print("\n[dim]Fetching available free models...[/dim]")
    free_models = _fetch_free_models(settings.openrouter_api_key)

    console.print(f"\n[bold]Free models[/bold]  (current: [cyan]{settings.openrouter_model}[/cyan])\n")
    for i, m in enumerate(free_models, 1):
        marker = "  [green]←[/green]" if m == settings.openrouter_model else ""
        console.print(f"  [cyan]{i}[/cyan] - {m}{marker}")
    console.print()

    model_choice = typer.prompt("Choose model number")
    try:
        idx = int(model_choice) - 1
        if 0 <= idx < len(free_models) - 1:
            new_model = free_models[idx]
        else:
            new_model = typer.prompt("Enter model name (e.g. google/gemma-3-27b-it:free)")
    except ValueError:
        new_model = typer.prompt("Enter model name")

    # Update .env
    env_path = Path(".env")
    existing: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    existing["OPENROUTER_MODEL"] = new_model
    env_path.write_text(
        "\n".join(f"{k}={v}" for k, v in existing.items()) + "\n",
        encoding="utf-8",
    )

    console.print(f"\n[green]✓[/green] Model set to [bold]{new_model}[/bold]\n")


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
[bold]LLM model:[/bold] {settings.active_model} ({settings.llm_provider})
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
    
    if settings.llm_provider == "openrouter":
        provider_line = f"[bold]Provider:[/bold] [green]OpenRouter[/green] (cloud)"
        model_line = f"[bold]LLM Model:[/bold] {settings.openrouter_model}"
        key_status = "[green]set[/green]" if settings.openrouter_api_key else "[red]missing![/red]"
        extra = f"[bold]API Key:[/bold] {key_status}"
    else:
        provider_line = f"[bold]Provider:[/bold] [cyan]Ollama[/cyan] (local)"
        model_line = f"[bold]LLM Model:[/bold] {settings.ollama_model}"
        extra = f"[bold]Ollama Host:[/bold] {settings.ollama_host}"

    console.print(Panel(
        f"""{provider_line}
{model_line}
{extra}
[bold]Embedding Model:[/bold] {settings.embedding_model} (Ollama)
[bold]Temperature:[/bold] {settings.model_temperature}

[bold]Chunk Size:[/bold] {settings.chunk_size} tokens
[bold]Chunk Overlap:[/bold] {settings.chunk_overlap} tokens

[bold]Papers Directory:[/bold] {settings.papers_dir}
[bold]Vector DB:[/bold] {settings.lancedb_path}""",
        title="[bold cyan]⚙️ Configuration[/bold cyan]",
        border_style="cyan",
    ))


# =============================================================================
# BEAR SUB-APP — Bias Evaluation and Analysis of Research
# =============================================================================

bear_app = typer.Typer(
    name="bear",
    help="BEAR - Bias Evaluation and Analysis of Research. AI-powered investment analysis.",
    add_completion=False,
    rich_markup_mode="rich",
)
app.add_typer(bear_app)


@bear_app.command("add")
def bear_add(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker symbol to ingest (e.g. AAPL, TSLA).",
    ),
):
    """
    Ingest a company's market data into the library.

    Fetches financials, earnings, news, key metrics, and analyst recommendations
    from Yahoo Finance, then chunks and embeds everything into the 'markets' collection.
    """
    from scalpel.bear.fetcher import fetch_ticker
    from scalpel.bear.ingestion import get_market_store

    console.print(f"\n[bold]Adding {ticker.upper()} to market library...[/bold]\n")

    try:
        data = fetch_ticker(ticker)
        store = get_market_store()
        count = store.add_company(data)

        if count > 0:
            console.print(
                f"\n[green]✓[/green] Added [bold]{data.company_name}[/bold] "
                f"({count} chunks indexed)"
            )
        else:
            console.print(
                f"\n[yellow]{ticker.upper()} already in library.[/yellow] "
                f"Use [cyan]scalpel bear remove {ticker.upper()}[/cyan] first to re-index."
            )
    except ImportError as e:
        console.print(f"\n[red]✗[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@bear_app.command("analyse")
def bear_analyse(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker symbol to analyse (e.g. AAPL).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
    ),
):
    """
    Full investment report: bull case, bear case, key assumptions, and bullshit scores.

    Requires the company to have been ingested first with [cyan]scalpel bear add[/cyan].
    """
    from scalpel.bear.analyst import get_analyst

    try:
        report = get_analyst().full_report(ticker)
        console.print()
        report.display()
        _save_report(report.to_markdown(), output, f"bear_analyse_{ticker.upper()}")
    except ValueError as e:
        console.print(f"\n[red]✗[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@bear_app.command("bs")
def bear_bs(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker symbol to evaluate (e.g. TSLA).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
    ),
):
    """
    Bullshit score only — how credible are this company's claims?

    Evaluates each major company claim or narrative on a 0-10 scale
    (0 = iron-clad evidence, 10 = pure spin).
    """
    from scalpel.bear.analyst import get_analyst

    try:
        result = get_analyst().bullshit_score(ticker)
        console.print()
        result.display()
        _save_report(result.to_markdown(), output, f"bear_bs_{ticker.upper()}")
    except ValueError as e:
        console.print(f"\n[red]✗[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@bear_app.command("cross")
def bear_cross(
    ticker: str = typer.Argument(
        ...,
        help="Stock ticker symbol to cross-reference (e.g. NVDA).",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
    ),
):
    """
    Cross-reference company claims against the scientific paper library.

    Queries both the 'markets' and 'papers' LanceDB collections simultaneously
    to identify where research supports, contradicts, or is silent on company claims.
    """
    from scalpel.bear.cross_reference import get_cross_referencer

    try:
        result = get_cross_referencer().cross_reference(ticker)
        console.print()
        result.display()
        _save_report(result.to_markdown(), output, f"bear_cross_{ticker.upper()}")
    except ValueError as e:
        console.print(f"\n[red]✗[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@bear_app.command("compare")
def bear_compare(
    ticker1: str = typer.Argument(..., help="First ticker symbol."),
    ticker2: str = typer.Argument(..., help="Second ticker symbol."),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Save report to this file (markdown).",
    ),
):
    """
    Comparative investment analysis of two companies.

    Both companies must be ingested first with [cyan]scalpel bear add[/cyan].
    """
    from scalpel.bear.analyst import get_analyst

    try:
        comparison = get_analyst().compare(ticker1, ticker2)
        console.print(Panel(
            Markdown(comparison),
            title=(
                f"[bold cyan]Comparison: "
                f"{ticker1.upper()} vs {ticker2.upper()}[/bold cyan]"
            ),
            border_style="cyan",
        ))
        _save_report(
            f"# Comparison: {ticker1.upper()} vs {ticker2.upper()}\n\n{comparison}",
            output,
            f"bear_compare_{ticker1.upper()}_vs_{ticker2.upper()}",
        )
    except ValueError as e:
        console.print(f"\n[red]✗[/red] {e}")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"\n[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)


@bear_app.command("list")
def bear_list():
    """
    List all companies in the market library.
    """
    from scalpel.bear.ingestion import get_market_store

    companies = get_market_store().list_companies()

    if not companies:
        console.print("\n[dim]No companies in market library yet.[/dim]")
        console.print("[dim]Use [cyan]scalpel bear add <ticker>[/cyan] to add a company.[/dim]\n")
        return

    table = Table(title=f"📈 Market Library ({len(companies)} companies)")
    table.add_column("Ticker", style="cyan", width=10)
    table.add_column("Company", style="white", max_width=40)
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Fetched", style="dim", width=12)

    for co in sorted(companies, key=lambda x: x["ticker"]):
        fetched = co.get("fetched_at", "")[:10] if co.get("fetched_at") else "N/A"
        table.add_row(co["ticker"], co["company_name"], str(co["chunk_count"]), fetched)

    console.print()
    console.print(table)
    console.print()


@bear_app.command("remove")
def bear_remove(
    ticker: str = typer.Argument(..., help="Ticker symbol to remove."),
):
    """
    Remove a company from the market library.
    """
    from scalpel.bear.ingestion import get_market_store

    if typer.confirm(f"Remove all data for '{ticker.upper()}'?"):
        count = get_market_store().delete_company(ticker)
        if count > 0:
            console.print(f"\n[green]✓[/green] Removed {count} chunks for {ticker.upper()}")
        else:
            console.print(f"\n[yellow]No data found for {ticker.upper()}[/yellow]")
    else:
        console.print("\n[dim]Cancelled.[/dim]")


def run():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    run()