"""
SCALPEL - Scientific Critique & Analysis Pipeline for Evidence Literature

A tool for analyzing, critiquing, and cross-referencing academic papers.
"""

__version__ = "0.1.0"
__author__ = "Rachel"

from scalpel.config import settings

# Convenience imports for common operations
from scalpel.ingestion import extract_pdf, chunk_paper, ExtractedPaper, TextChunk
from scalpel.embeddings import add_pdf, search, list_papers, get_store, VectorStore
from scalpel.analysis import (
    summarize,
    critique_methodology,
    critique_statistics,
    critique_full,
    extract_claims,
    identify_limitations,
    bullshit_score,
    full_analysis,
    get_analyzer,
    Analyzer,
)

__all__ = [
    "settings",
    "__version__",
    # Ingestion
    "extract_pdf",
    "chunk_paper",
    "ExtractedPaper",
    "TextChunk",
    # Embeddings
    "add_pdf",
    "search",
    "list_papers",
    "get_store",
    "VectorStore",
    # Analysis
    "summarize",
    "critique_methodology",
    "critique_statistics",
    "critique_full",
    "extract_claims",
    "identify_limitations",
    "bullshit_score",
    "full_analysis",
    "get_analyzer",
    "Analyzer",
]