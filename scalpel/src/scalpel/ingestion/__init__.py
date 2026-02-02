"""
Ingestion module for SCALPEL.

Handles PDF extraction, arXiv fetching, and text processing.

# PDF Extraction
from scalpel.ingestion import extract_pdf, extract_all_pdfs

paper = extract_pdf("path/to/paper.pdf")
print(paper.title, paper.abstract, paper.sections)

# arXiv Fetching
from scalpel.ingestion import fetch_arxiv

# All these work:
paper = fetch_arxiv("2301.07041")
paper = fetch_arxiv("https://arxiv.org/abs/2301.07041")
paper = fetch_arxiv("arxiv.org/pdf/2301.07041.pdf")

# Text Chunking (tiktoken + semantic boundaries)
from scalpel.ingestion import chunk_paper, chunk_for_analysis

chunks = chunk_paper(paper)  # Respects section boundaries
chunks = chunk_for_analysis(paper, target_sections=["Methods", "Results"])

for chunk in chunks:
    print(chunk.token_count, chunk.source_section)
"""

from scalpel.ingestion.pdf_reader import (
    ExtractedPaper,
    PaperSection,
    extract_pdf,
    extract_all_pdfs,
)
from scalpel.ingestion.arxiv_fetcher import (
    fetch_arxiv,
    fetch_multiple_arxiv,
    parse_arxiv_id,
)
from scalpel.ingestion.chunker import (
    TextChunk,
    chunk_text,
    chunk_paper,
    chunk_papers,
    chunk_for_analysis,
    count_tokens,
)

__all__ = [
    # PDF extraction
    "ExtractedPaper",
    "PaperSection", 
    "extract_pdf",
    "extract_all_pdfs",
    # arXiv
    "fetch_arxiv",
    "fetch_multiple_arxiv",
    "parse_arxiv_id",
    # Chunking
    "TextChunk",
    "chunk_text",
    "chunk_paper",
    "chunk_papers",
    "chunk_for_analysis",
    "count_tokens",
]