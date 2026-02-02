"""
PDF Text Extraction using PyMuPDF

Handles extraction, cleaning, and structural parsing of academic PDFs.
"""

import re
from pathlib import Path
from dataclasses import dataclass, field

import pymupdf
from rich.console import Console

console = Console()


# Common section headers in academic papers
SECTION_PATTERNS = [
    r"^abstract\s*$",
    r"^introduction\s*$",
    r"^background\s*$",
    r"^related\s+work\s*$",
    r"^literature\s+review\s*$",
    r"^methods?\s*$",
    r"^methodology\s*$",
    r"^materials?\s+and\s+methods?\s*$",
    r"^experimental?\s+(?:setup|design|methods?)?\s*$",
    r"^results?\s*$",
    r"^findings?\s*$",
    r"^results?\s+and\s+discussion\s*$",
    r"^discussion\s*$",
    r"^analysis\s*$",
    r"^conclusion\s*$",
    r"^conclusions?\s+and\s+future\s+work\s*$",
    r"^summary\s*$",
    r"^limitations?\s*$",
    r"^future\s+work\s*$",
    r"^acknowledg[e]?ments?\s*$",
    r"^references?\s*$",
    r"^bibliography\s*$",
    r"^appendix\s*",
    r"^supplementary\s+materials?\s*$",
]

SECTION_REGEX = re.compile(
    "|".join(SECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE
)


@dataclass
class PaperSection:
    """A detected section within the paper."""
    
    name: str
    content: str
    start_pos: int
    end_pos: int


@dataclass
class ExtractedPaper:
    """Container for extracted PDF content."""
    
    filepath: Path
    title: str
    text: str
    page_count: int
    metadata: dict
    abstract: str | None = None
    sections: list[PaperSection] = field(default_factory=list)
    
    @property
    def has_abstract(self) -> bool:
        return self.abstract is not None and len(self.abstract) > 50
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    def get_section(self, name: str) -> PaperSection | None:
        """Get a section by name (case-insensitive)."""
        name_lower = name.lower()
        for section in self.sections:
            if section.name.lower() == name_lower:
                return section
        return None


def clean_text(text: str) -> str:
    """Clean extracted text by removing artifacts and normalizing whitespace."""
    # Remove hyphenation at line breaks
    text = re.sub(r"-\n", "", text)
    
    # Normalize whitespace (but preserve paragraph breaks)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove common PDF artifacts
    text = re.sub(r"\x00", "", text)  # Null bytes
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f]", "", text)  # Control chars
    
    # Clean up ligatures that might not render properly
    ligatures = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for lig, replacement in ligatures.items():
        text = text.replace(lig, replacement)
    
    return text.strip()


def extract_abstract(text: str) -> str | None:
    """
    Attempt to extract the abstract from paper text.
    
    Tries multiple strategies since abstract formatting varies wildly.
    """
    # Strategy 1: Look for explicit "Abstract" header
    abstract_pattern = re.compile(
        r"(?:^|\n)\s*abstract[:\s]*\n(.*?)(?=\n\s*(?:introduction|keywords?|1[\.\s]|background)|\n\n\n)",
        re.IGNORECASE | re.DOTALL
    )
    match = abstract_pattern.search(text)
    if match:
        abstract = clean_text(match.group(1))
        if 50 < len(abstract) < 3000:  # Sanity check
            return abstract
    
    # Strategy 2: Look for "Abstract:" or "Abstract." inline
    inline_pattern = re.compile(
        r"abstract[:\.\s]+(.{100,2000}?)(?=\n\s*(?:introduction|keywords?|1[\.\s]))",
        re.IGNORECASE | re.DOTALL
    )
    match = inline_pattern.search(text[:5000])  # Only check first part
    if match:
        abstract = clean_text(match.group(1))
        if 50 < len(abstract) < 3000:
            return abstract
    
    # Strategy 3: arXiv style - abstract is often the first substantial paragraph
    lines = text[:4000].split("\n\n")
    for para in lines[1:5]:  # Skip title, check next few paragraphs
        para = para.strip()
        if 150 < len(para) < 2500 and not para.lower().startswith(("keyword", "index", "copyright")):
            # Likely the abstract
            return clean_text(para)
    
    return None


def detect_sections(text: str) -> list[PaperSection]:
    """
    Detect and extract sections from the paper.
    
    Returns list of PaperSection objects with name, content, and positions.
    """
    sections = []
    
    # Find all section headers
    matches = list(SECTION_REGEX.finditer(text))
    
    if not matches:
        return sections
    
    for i, match in enumerate(matches):
        section_name = match.group().strip().title()
        start_pos = match.end()
        
        # End position is start of next section, or end of text
        if i + 1 < len(matches):
            end_pos = matches[i + 1].start()
        else:
            end_pos = len(text)
        
        content = clean_text(text[start_pos:end_pos])
        
        if content:  # Only add if there's actual content
            sections.append(PaperSection(
                name=section_name,
                content=content,
                start_pos=start_pos,
                end_pos=end_pos,
            ))
    
    return sections


def extract_pdf(filepath: Path, verbose: bool = True) -> ExtractedPaper:
    """
    Extract text and metadata from a PDF file.
    
    Args:
        filepath: Path to the PDF file
        verbose: Whether to print progress messages
        
    Returns:
        ExtractedPaper with cleaned text, abstract, sections, and metadata
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"PDF not found: {filepath}")
    
    if not filepath.suffix.lower() == ".pdf":
        raise ValueError(f"Not a PDF file: {filepath}")
    
    with pymupdf.open(filepath) as doc:
        # Extract metadata
        metadata = doc.metadata or {}
        title = metadata.get("title", "") or filepath.stem
        
        # Extract text from all pages
        pages_text = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages_text.append(text)
        
        page_count = len(pages_text)
    
    # Combine and clean
    full_text = "\n\n".join(pages_text)
    cleaned_text = clean_text(full_text)
    
    # Extract structure
    abstract = extract_abstract(cleaned_text)
    sections = detect_sections(cleaned_text)
    
    if verbose:
        console.print(f"[green]✓[/green] Extracted [bold]{title}[/bold]")
        console.print(f"  └─ {page_count} pages, {len(cleaned_text.split())} words, {len(sections)} sections detected")
        if abstract:
            console.print(f"  └─ [dim]Abstract found ({len(abstract)} chars)[/dim]")
    
    return ExtractedPaper(
        filepath=filepath,
        title=title,
        text=cleaned_text,
        page_count=page_count,
        metadata=metadata,
        abstract=abstract,
        sections=sections,
    )


def extract_all_pdfs(directory: Path, verbose: bool = True) -> list[ExtractedPaper]:
    """
    Extract text from all PDFs in a directory.
    
    Args:
        directory: Path to directory containing PDFs
        verbose: Whether to print progress messages
        
    Returns:
        List of ExtractedPaper objects
    """
    directory = Path(directory)
    papers = []
    pdf_files = list(directory.glob("*.pdf"))
    
    if verbose:
        console.print(f"\n[bold]Processing {len(pdf_files)} PDFs from {directory}[/bold]\n")
    
    for pdf_path in pdf_files:
        try:
            paper = extract_pdf(pdf_path, verbose=verbose)
            papers.append(paper)
        except Exception as e:
            console.print(f"[red]✗ Failed to extract {pdf_path.name}[/red]: {e}")
    
    if verbose:
        console.print(f"\n[bold green]Successfully extracted {len(papers)}/{len(pdf_files)} papers[/bold green]\n")
    
    return papers


if __name__ == "__main__":
    # Quick test
    import sys
    
    if len(sys.argv) > 1:
        paper = extract_pdf(Path(sys.argv[1]))
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
        console.print("[yellow]Usage: python pdf_reader.py <path_to_pdf>[/yellow]")