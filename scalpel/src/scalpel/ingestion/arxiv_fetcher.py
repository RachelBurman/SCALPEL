"""
arXiv Paper Fetcher

Downloads papers from arXiv and returns ExtractedPaper objects.
Uses httpx for cleaner HTTP handling and future async support.
"""

import re
from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn

from scalpel.config import settings
from scalpel.ingestion.pdf_reader import ExtractedPaper, extract_pdf

console = Console()

# Timeout settings
DOWNLOAD_TIMEOUT = httpx.Timeout(30.0, connect=10.0)

# Pattern to extract arXiv ID from various formats
ARXIV_PATTERNS = [
    r"arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)",  # arxiv.org/abs/2301.07041
    r"arxiv\.org/pdf/(\d{4}\.\d{4,5}(?:v\d+)?)",  # arxiv.org/pdf/2301.07041
    r"^(\d{4}\.\d{4,5}(?:v\d+)?)$",  # Just the ID: 2301.07041
    r"arxiv:(\d{4}\.\d{4,5}(?:v\d+)?)",  # arxiv:2301.07041
    # Legacy format (pre-2007): hep-th/9901001
    r"arxiv\.org/abs/([a-z-]+/\d{7}(?:v\d+)?)",
    r"arxiv\.org/pdf/([a-z-]+/\d{7}(?:v\d+)?)",
    r"^([a-z-]+/\d{7}(?:v\d+)?)$",
]


def parse_arxiv_id(input_str: str) -> str | None:
    """
    Extract arXiv ID from URL or raw ID string.
    
    Accepts:
        - https://arxiv.org/abs/2301.07041
        - https://arxiv.org/pdf/2301.07041.pdf
        - arxiv.org/abs/2301.07041
        - 2301.07041
        - 2301.07041v2
        - arxiv:2301.07041
        - hep-th/9901001 (legacy format)
    
    Returns:
        The arXiv ID (e.g., "2301.07041") or None if not recognized.
    """
    input_str = input_str.strip()
    
    # Remove .pdf extension if present
    input_str = re.sub(r"\.pdf$", "", input_str, flags=re.IGNORECASE)
    
    for pattern in ARXIV_PATTERNS:
        match = re.search(pattern, input_str, re.IGNORECASE)
        if match:
            return match.group(1)
    
    return None


def get_pdf_url(arxiv_id: str) -> str:
    """Get the direct PDF download URL for an arXiv ID."""
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


def download_pdf(url: str, output_path: Path, show_progress: bool = True) -> None:
    """
    Download a PDF file with optional progress bar.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        show_progress: Whether to show a progress bar
    """
    with httpx.stream("GET", url, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True) as response:
        response.raise_for_status()
        
        total_size = int(response.headers.get("content-length", 0))
        
        if show_progress and total_size > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading", total=total_size)
                
                with open(output_path, "wb") as f:
                    for chunk in response.iter_bytes(chunk_size=8192):
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))
        else:
            # No content-length header or progress disabled — just download
            output_path.write_bytes(response.content)


def fetch_arxiv(
    identifier: str,
    output_dir: Path | None = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> ExtractedPaper:
    """
    Download a paper from arXiv and extract its content.
    
    Args:
        identifier: arXiv URL or ID (e.g., "2301.07041" or "arxiv.org/abs/2301.07041")
        output_dir: Directory to save PDF (defaults to settings.papers_dir)
        verbose: Whether to print status messages
        show_progress: Whether to show download progress bar
        
    Returns:
        ExtractedPaper ready for analysis
        
    Raises:
        ValueError: If the identifier cannot be parsed as an arXiv reference
        FileNotFoundError: If the arXiv paper doesn't exist
        ConnectionError: If download fails
    """
    arxiv_id = parse_arxiv_id(identifier)
    if not arxiv_id:
        raise ValueError(
            f"Could not parse arXiv ID from: {identifier}\n"
            "Expected formats: 2301.07041, arxiv.org/abs/2301.07041, arxiv.org/pdf/2301.07041"
        )
    
    output_dir = output_dir or settings.papers_dir
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sanitize legacy IDs for filename (hep-th/9901001 -> hep-th_9901001)
    safe_id = arxiv_id.replace("/", "_")
    pdf_path = output_dir / f"arxiv_{safe_id}.pdf"
    pdf_url = get_pdf_url(arxiv_id)
    
    # Check if already downloaded
    if pdf_path.exists():
        if verbose:
            console.print(f"[dim]Using cached:[/dim] {pdf_path.name}")
    else:
        if verbose:
            console.print(f"[cyan]Fetching[/cyan] arXiv:{arxiv_id}")
        
        try:
            download_pdf(pdf_url, pdf_path, show_progress=show_progress)
            if verbose:
                console.print(f"[green]✓[/green] Saved to {pdf_path.name}")
        except httpx.HTTPStatusError as e:
            # Clean up partial download if it exists
            pdf_path.unlink(missing_ok=True)
            if e.response.status_code == 404:
                raise FileNotFoundError(f"arXiv paper not found: {arxiv_id}") from e
            raise ConnectionError(f"HTTP {e.response.status_code} downloading {arxiv_id}") from e
        except httpx.RequestError as e:
            pdf_path.unlink(missing_ok=True)
            raise ConnectionError(f"Network error downloading {arxiv_id}: {e}") from e
    
    return extract_pdf(pdf_path, verbose=verbose)


def fetch_multiple_arxiv(
    identifiers: list[str],
    output_dir: Path | None = None,
    verbose: bool = True,
    show_progress: bool = True,
) -> list[ExtractedPaper]:
    """
    Download multiple papers from arXiv.
    
    Args:
        identifiers: List of arXiv URLs or IDs
        output_dir: Directory to save PDFs (defaults to settings.papers_dir)
        verbose: Whether to print status messages
        show_progress: Whether to show download progress bars
        
    Returns:
        List of ExtractedPaper objects (failed downloads are skipped)
    """
    papers = []
    
    if verbose:
        console.print(f"\n[bold]Fetching {len(identifiers)} papers from arXiv[/bold]\n")
    
    for i, identifier in enumerate(identifiers, 1):
        if verbose:
            console.print(f"[dim]({i}/{len(identifiers)})[/dim]", end=" ")
        
        try:
            paper = fetch_arxiv(
                identifier,
                output_dir,
                verbose=verbose,
                show_progress=show_progress,
            )
            papers.append(paper)
        except Exception as e:
            console.print(f"[red]✗ Failed[/red] {identifier}: {e}")
    
    if verbose:
        console.print(f"\n[bold green]Successfully fetched {len(papers)}/{len(identifiers)} papers[/bold green]\n")
    
    return papers


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        paper = fetch_arxiv(sys.argv[1])
        console.print(f"\n[bold]Title:[/bold] {paper.title}")
        console.print(f"[bold]Pages:[/bold] {paper.page_count}")
        console.print(f"[bold]Words:[/bold] {paper.word_count}")
        
        if paper.abstract:
            console.print(f"\n[bold]Abstract:[/bold]\n{paper.abstract[:500]}...")
        
        if paper.sections:
            console.print(f"\n[bold]Sections:[/bold]")
            for section in paper.sections:
                console.print(f"  • {section.name} ({len(section.content)} chars)")
    else:
        console.print("[yellow]Usage: python arxiv_fetcher.py <arxiv_id_or_url>[/yellow]")
        console.print("[dim]Examples:[/dim]")
        console.print("  python arxiv_fetcher.py 2301.07041")
        console.print("  python arxiv_fetcher.py https://arxiv.org/abs/2301.07041")
        console.print("  python arxiv_fetcher.py hep-th/9901001  [dim](legacy format)[/dim]")