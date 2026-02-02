"""
Text Chunking for SCALPEL.

Combines accurate token counting (tiktoken) with semantic boundary awareness.
Splits on paragraphs and sentences, never mid-word or mid-abbreviation.
"""

import re
from dataclasses import dataclass
from typing import Literal

import tiktoken
from rich.console import Console

from scalpel.config import settings
from scalpel.ingestion.pdf_reader import ExtractedPaper

console = Console()

# Use cl100k_base encoding (GPT-4, ChatGPT models) as reasonable default
# Works well for most modern LLMs including Qwen
ENCODER = tiktoken.get_encoding("cl100k_base")

# Common abbreviations that shouldn't trigger sentence splits
ABBREVIATIONS = [
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sr.", "Jr.",
    "vs.", "etc.", "i.e.", "e.g.", "cf.", "al.", "Fig.",
    "Eq.", "Ref.", "Vol.", "No.", "pp.", "et al.", "Ph.D.",
    "Inc.", "Ltd.", "Corp.", "Co.", "St.", "Mt.", "Ft.",
]


@dataclass
class TextChunk:
    """A chunk of text from a paper."""
    
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    source_section: str | None = None
    chunk_index: int = 0
    total_chunks: int = 0
    
    @property
    def word_count(self) -> int:
        return len(self.text.split())
    
    @property
    def metadata(self) -> dict:
        """Return chunk metadata for embedding/retrieval."""
        return {
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "token_count": self.token_count,
            "word_count": self.word_count,
            "source_section": self.source_section,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }
    
    def __repr__(self) -> str:
        section_info = f" [{self.source_section}]" if self.source_section else ""
        return f"TextChunk({self.chunk_index}{section_info}, {self.token_count} tokens)"


def count_tokens(text: str) -> int:
    """Count tokens in text using tiktoken."""
    return len(ENCODER.encode(text))


def _protect_abbreviations(text: str) -> str:
    """Replace periods in abbreviations with placeholder."""
    protected = text
    for abbr in ABBREVIATIONS:
        protected = protected.replace(abbr, abbr.replace(".", "<<DOT>>"))
    return protected


def _restore_abbreviations(text: str) -> str:
    """Restore periods in abbreviations."""
    return text.replace("<<DOT>>", ".")


def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences, respecting abbreviations.
    
    Handles common academic writing patterns without splitting on
    "et al.", "Fig.", "Eq.", etc.
    """
    protected = _protect_abbreviations(text)
    
    # Split on sentence boundaries (. ! ?) followed by space and capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    
    # Restore and clean
    sentences = [_restore_abbreviations(s).strip() for s in sentences]
    return [s for s in sentences if s]


def split_into_paragraphs(text: str) -> list[str]:
    """Split text on paragraph boundaries (double newline)."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _find_semantic_break(text: str, target_pos: int, window: int = 200) -> int:
    """
    Find the best semantic break point near target_pos.
    
    Looks for (in order of preference):
    1. Paragraph break
    2. Sentence break
    3. Word break
    
    Args:
        text: The text to search in
        target_pos: Ideal character position to break at
        window: How far to search before/after target
        
    Returns:
        Character position of best break point
    """
    start = max(0, target_pos - window)
    end = min(len(text), target_pos + window)
    search_region = text[start:end]
    
    # Look for paragraph break (double newline)
    para_break = search_region.rfind("\n\n", 0, window + (target_pos - start))
    if para_break > window // 2:
        return start + para_break + 2
    
    # Look for sentence break
    protected = _protect_abbreviations(search_region)
    
    # Find sentence endings in the region before target
    best_sentence_break = -1
    for match in re.finditer(r'[.!?]\s+', protected[:window + (target_pos - start)]):
        pos = match.end()
        if pos > best_sentence_break:
            best_sentence_break = pos
    
    if best_sentence_break > window // 2:
        return start + best_sentence_break
    
    # Fall back to word break
    word_break = search_region.rfind(" ", 0, window + (target_pos - start))
    if word_break > 0:
        return start + word_break + 1
    
    # Last resort: just use target
    return target_pos


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    source_section: str | None = None,
) -> list[TextChunk]:
    """
    Split text into overlapping chunks with semantic boundary awareness.
    
    Uses tiktoken for accurate token counting, but splits at paragraph/sentence
    boundaries rather than arbitrary token positions.
    
    Args:
        text: The text to chunk
        chunk_size: Max tokens per chunk (default from settings)
        chunk_overlap: Token overlap between chunks (default from settings)
        source_section: Optional section name for metadata
        
    Returns:
        List of TextChunk objects
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    if not text.strip():
        return []
    
    total_tokens = count_tokens(text)
    
    # If text fits in one chunk, just return it
    if total_tokens <= chunk_size:
        return [TextChunk(
            text=text.strip(),
            start_pos=0,
            end_pos=len(text),
            token_count=total_tokens,
            source_section=source_section,
            chunk_index=0,
            total_chunks=1,
        )]
    
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # Estimate where chunk should end based on token ratio
        remaining_text = text[current_pos:]
        remaining_tokens = count_tokens(remaining_text)
        
        if remaining_tokens <= chunk_size:
            # Last chunk - take everything
            chunk_text_content = remaining_text.strip()
            chunks.append(TextChunk(
                text=chunk_text_content,
                start_pos=current_pos,
                end_pos=text_length,
                token_count=count_tokens(chunk_text_content),
                source_section=source_section,
                chunk_index=len(chunks),
                total_chunks=0,  # Will update later
            ))
            break
        
        # Estimate character position for target tokens
        # Rough ratio: chars_per_token ≈ len(text) / tokens
        chars_per_token = len(remaining_text) / remaining_tokens
        estimated_end = current_pos + int(chunk_size * chars_per_token)
        
        # Find a good semantic break point
        break_pos = _find_semantic_break(text, estimated_end)
        
        # Ensure we're making progress
        if break_pos <= current_pos:
            break_pos = min(estimated_end, text_length)
        
        chunk_text_content = text[current_pos:break_pos].strip()
        actual_tokens = count_tokens(chunk_text_content)
        
        # If we overshot, try to find an earlier break
        attempts = 0
        while actual_tokens > chunk_size * 1.1 and attempts < 3:  # Allow 10% overage
            estimated_end = current_pos + int((break_pos - current_pos) * 0.8)
            break_pos = _find_semantic_break(text, estimated_end)
            chunk_text_content = text[current_pos:break_pos].strip()
            actual_tokens = count_tokens(chunk_text_content)
            attempts += 1
        
        if chunk_text_content:
            chunks.append(TextChunk(
                text=chunk_text_content,
                start_pos=current_pos,
                end_pos=break_pos,
                token_count=actual_tokens,
                source_section=source_section,
                chunk_index=len(chunks),
                total_chunks=0,  # Will update later
            ))
        
        # Calculate overlap start position
        if chunk_overlap > 0 and break_pos < text_length:
            # Find overlap in tokens, convert back to chars
            overlap_chars = int(chunk_overlap * chars_per_token)
            overlap_start = max(current_pos, break_pos - overlap_chars)
            # Find semantic break for overlap start
            current_pos = _find_semantic_break(text, overlap_start, window=100)
            if current_pos >= break_pos:
                current_pos = break_pos  # No overlap if break finding fails
        else:
            current_pos = break_pos
    
    # Update total_chunks
    total = len(chunks)
    for chunk in chunks:
        chunk.total_chunks = total
    
    return chunks


def chunk_paper(
    paper: ExtractedPaper,
    mode: Literal["full", "sections"] = "sections",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    verbose: bool = False,
) -> list[TextChunk]:
    """
    Chunk an entire paper.
    
    Args:
        paper: ExtractedPaper to chunk
        mode: "full" chunks entire text, "sections" chunks each section separately
        chunk_size: Max tokens per chunk (default from settings)
        chunk_overlap: Token overlap between chunks (default from settings)
        verbose: Print progress messages
        
    Returns:
        List of TextChunk objects
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    
    if mode == "sections" and paper.sections:
        all_chunks = []
        
        for section in paper.sections:
            section_chunks = chunk_text(
                section.content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_section=section.name,
            )
            all_chunks.extend(section_chunks)
        
        # Re-index chunks sequentially
        for i, chunk in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = len(all_chunks)
        
        if verbose:
            total_tokens = sum(c.token_count for c in all_chunks)
            console.print(
                f"[green]✓[/green] Chunked [bold]{paper.title}[/bold]: "
                f"{len(all_chunks)} chunks from {len(paper.sections)} sections "
                f"({total_tokens:,} tokens)"
            )
        
        return all_chunks
    
    # Full text mode
    chunks = chunk_text(
        paper.text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    if verbose:
        total_tokens = sum(c.token_count for c in chunks)
        console.print(
            f"[green]✓[/green] Chunked [bold]{paper.title}[/bold]: "
            f"{len(chunks)} chunks ({total_tokens:,} tokens)"
        )
    
    return chunks


def chunk_for_analysis(
    paper: ExtractedPaper,
    target_sections: list[str],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    verbose: bool = False,
) -> list[TextChunk]:
    """
    Chunk only specific sections of a paper for targeted analysis.
    
    Useful when you only want to analyze Methods, Results, Discussion, etc.
    
    Args:
        paper: ExtractedPaper to chunk
        target_sections: List of section names to include (case-insensitive)
        chunk_size: Max tokens per chunk
        chunk_overlap: Token overlap between chunks
        verbose: Print progress messages
        
    Returns:
        List of TextChunk objects for the specified sections only
    """
    if not paper.sections:
        if verbose:
            console.print("[yellow]No sections detected, chunking full text[/yellow]")
        return chunk_paper(paper, mode="full", chunk_size=chunk_size, 
                          chunk_overlap=chunk_overlap, verbose=verbose)
    
    # Normalize target names for comparison
    target_lower = {s.lower() for s in target_sections}
    
    all_chunks = []
    matched_sections = []
    
    for section in paper.sections:
        if section.name.lower() in target_lower:
            matched_sections.append(section.name)
            section_chunks = chunk_text(
                section.content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                source_section=section.name,
            )
            all_chunks.extend(section_chunks)
    
    # Re-index
    for i, chunk in enumerate(all_chunks):
        chunk.chunk_index = i
        chunk.total_chunks = len(all_chunks)
    
    if verbose:
        total_tokens = sum(c.token_count for c in all_chunks)
        console.print(
            f"[green]✓[/green] Chunked sections {matched_sections}: "
            f"{len(all_chunks)} chunks ({total_tokens:,} tokens)"
        )
    
    return all_chunks


def chunk_papers(
    papers: list[ExtractedPaper],
    mode: Literal["full", "sections"] = "sections",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    verbose: bool = True,
) -> dict[str, list[TextChunk]]:
    """
    Chunk multiple papers.
    
    Args:
        papers: List of ExtractedPaper objects
        mode: "full" or "sections" chunking mode
        chunk_size: Max tokens per chunk
        chunk_overlap: Token overlap between chunks
        verbose: Print progress messages
        
    Returns:
        Dict mapping paper title to list of chunks
    """
    results = {}
    
    if verbose:
        console.print(f"\n[bold]Chunking {len(papers)} papers[/bold]\n")
    
    for paper in papers:
        chunks = chunk_paper(
            paper,
            mode=mode,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            verbose=verbose,
        )
        results[paper.title] = chunks
    
    if verbose:
        total_chunks = sum(len(c) for c in results.values())
        total_tokens = sum(
            sum(chunk.token_count for chunk in chunks) 
            for chunks in results.values()
        )
        console.print(
            f"\n[bold green]Created {total_chunks} chunks "
            f"({total_tokens:,} tokens) from {len(papers)} papers[/bold green]\n"
        )
    
    return results


if __name__ == "__main__":
    from rich.table import Table
    
    # Test with sample academic text
    sample_text = """
    Abstract

    This paper presents a novel approach to neural network optimization. We demonstrate
    that our method, et al. (2023), achieves state-of-the-art results on multiple benchmarks.
    The key contribution is a new loss function that improves convergence by 40%.

    Introduction

    Deep learning has revolutionized machine learning. As shown in Fig. 1, the performance
    of neural networks has improved dramatically over the past decade. Dr. Smith and
    Prof. Johnson first proposed the foundational architecture in their seminal work.

    Our contributions are threefold: First, we introduce a novel optimization technique.
    Second, we provide theoretical analysis of convergence properties. Third, we demonstrate
    empirical improvements across diverse tasks including image classification, natural
    language processing, and reinforcement learning benchmarks.

    Methods

    We propose a three-stage training pipeline. In Stage 1, we pretrain the model using
    self-supervised learning on a large corpus. The learning rate is set to 1e-4 with
    cosine annealing. In Stage 2, we fine-tune on task-specific data using our novel
    loss function. The batch size is 32 and we train for 100 epochs. In Stage 3, we
    apply knowledge distillation to compress the model for efficient inference.

    The architecture consists of 12 transformer layers with 768 hidden dimensions.
    We use dropout of 0.1 and layer normalization. Following Vaswani et al. (2017),
    we employ multi-head attention with 12 heads.

    Results

    Table 1 shows our main results. Our method achieves 94.2% accuracy on ImageNet,
    compared to 91.3% for the baseline. The improvement is statistically significant
    (p < 0.001). On NLP tasks, we observe similar gains: BLEU score improves from
    28.4 to 32.1 on WMT translation.

    Discussion

    These results suggest that our optimization approach is broadly applicable.
    However, there are limitations. The method requires 2x more compute during training.
    Future work should address this efficiency gap.
    """
    
    console.print("[bold]Testing merged chunker with semantic boundaries + tiktoken[/bold]\n")
    
    chunks = chunk_text(sample_text, chunk_size=200, chunk_overlap=30)
    
    table = Table(title=f"Created {len(chunks)} chunks")
    table.add_column("Idx", style="cyan", width=4)
    table.add_column("Tokens", style="green", width=8)
    table.add_column("Words", style="yellow", width=8)
    table.add_column("Preview", style="white")
    
    for chunk in chunks:
        preview = chunk.text[:70].replace("\n", " ") + "..."
        table.add_row(
            str(chunk.chunk_index),
            str(chunk.token_count),
            str(chunk.word_count),
            preview
        )
    
    console.print(table)
    
    # Verify no mid-sentence cuts
    console.print("\n[bold]Checking chunk boundaries:[/bold]")
    for i, chunk in enumerate(chunks[:-1]):  # Skip last
        ending = chunk.text[-50:].replace("\n", " ")
        console.print(f"  Chunk {i} ends: [dim]...{ending}[/dim]")