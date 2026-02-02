"""Embeddings module - Phase 3.
Embeddings module for SCALPEL.

Handles vector embeddings and semantic search across papers.

# Quick start
from scalpel.embeddings import add_pdf, search, list_papers

# Index a paper
add_pdf("data/papers/my_paper.pdf")

# Search across all papers
results = search("attention mechanism transformer")
for r in results:
    print(f"{r.paper_title} [{r.section}]: {r.score:.3f}")

# Advanced usage with VectorStore class
from scalpel.embeddings import VectorStore

store = VectorStore()
store.add_paper(paper)
results = store.search("query", section_filter="Methods")
"""

from scalpel.embeddings.vector_store import (
    VectorStore,
    SearchResult,
    get_store,
    add_paper,
    add_pdf,
    search,
    list_papers,
)

__all__ = [
    "VectorStore",
    "SearchResult",
    "get_store",
    "add_paper",
    "add_pdf",
    "search",
    "list_papers",
]