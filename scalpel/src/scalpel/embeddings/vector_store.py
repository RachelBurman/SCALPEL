"""
Vector Store for SCALPEL using LanceDB.

Uses Ollama for local embeddings and LanceDB for storage/retrieval.
"""

from dataclasses import dataclass
from pathlib import Path

import lancedb
import ollama
import pyarrow as pa
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from scalpel.config import settings
from scalpel.ingestion import ExtractedPaper, TextChunk, chunk_paper, extract_pdf

console = Console()


@dataclass
class SearchResult:
    """A search result with chunk content and metadata."""
    
    text: str
    score: float  # Similarity score (higher = more similar)
    paper_title: str
    section: str | None
    chunk_index: int
    metadata: dict
    
    def __repr__(self) -> str:
        section_info = f" [{self.section}]" if self.section else ""
        return f"SearchResult({self.paper_title}{section_info}, score={self.score:.3f})"


class VectorStore:
    """
    Vector database for paper embeddings.
    
    Uses LanceDB for storage and Ollama for embedding generation.
    """
    
    def __init__(
        self,
        db_path: Path | None = None,
        embedding_model: str | None = None,
    ):
        """
        Initialize the vector store.
        
        Args:
            db_path: Directory for LanceDB persistence (default from settings)
            embedding_model: Ollama embedding model (default from settings)
        """
        self.db_path = Path(db_path or settings.lancedb_path)
        self.embedding_model = embedding_model or settings.embedding_model
        self.table_name = settings.lancedb_table
        
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self._db = lancedb.connect(str(self.db_path))
        self._table = None
        
        if self.table_name in self._db.table_names():
            self._table = self._db.open_table(self.table_name)
    
    def _get_schema(self) -> pa.Schema:
        """Get the Arrow schema for the papers table."""
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), settings.embedding_dimensions)),
            pa.field("paper_title", pa.string()),
            pa.field("paper_path", pa.string()),
            pa.field("section", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("total_chunks", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("start_pos", pa.int32()),
            pa.field("end_pos", pa.int32()),
        ])
    
    def _embed_text(self, text: str) -> list[float]:
        """Generate embedding for a single text using Ollama."""
        response = ollama.embed(
            model=self.embedding_model,
            input=text,
        )
        return response["embeddings"][0]
    
    def _embed_batch(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        embeddings = []
        batch_size = settings.embedding_batch_size
        
        if show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[cyan]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Embedding", total=len(texts))
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = ollama.embed(
                        model=self.embedding_model,
                        input=batch,
                    )
                    embeddings.extend(response["embeddings"])
                    progress.update(task, advance=len(batch))
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = ollama.embed(
                    model=self.embedding_model,
                    input=batch,
                )
                embeddings.extend(response["embeddings"])
        
        return embeddings
    
    def add_paper(
        self,
        paper: ExtractedPaper,
        chunks: list[TextChunk] | None = None,
        verbose: bool = True,
    ) -> int:
        """
        Add a paper to the vector store.
        
        Args:
            paper: ExtractedPaper to add
            chunks: Pre-computed chunks (if None, will chunk the paper)
            verbose: Whether to print progress
            
        Returns:
            Number of chunks added
        """
        if self._table is not None:
            existing = self._table.search().where(
                f"paper_path = '{paper.filepath}'", prefilter=True
            ).limit(1).to_list()
            
            if existing:
                if verbose:
                    console.print(f"[yellow]Paper already indexed:[/yellow] {paper.title}")
                    console.print("[dim]Use delete_paper() first to re-index[/dim]")
                return 0
        
        if chunks is None:
            chunks = chunk_paper(paper, mode="sections", verbose=verbose)
        
        if not chunks:
            if verbose:
                console.print(f"[yellow]No chunks to index for:[/yellow] {paper.title}")
            return 0
        
        if verbose:
            console.print(f"[cyan]Indexing[/cyan] {paper.title}...")
        
        texts = [chunk.text for chunk in chunks]
        embeddings = self._embed_batch(texts, show_progress=verbose)
        
        records = []
        for i, chunk in enumerate(chunks):
            records.append({
                "id": f"{paper.filepath.stem}_chunk_{i}",
                "text": chunk.text,
                "vector": embeddings[i],
                "paper_title": paper.title,
                "paper_path": str(paper.filepath),
                "section": chunk.source_section or "",
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "token_count": chunk.token_count,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
            })
        
        if self._table is None:
            self._table = self._db.create_table(
                self.table_name,
                data=records,
                schema=self._get_schema(),
            )
        else:
            self._table.add(records)
        
        if verbose:
            console.print(
                f"[green]✓[/green] Indexed [bold]{paper.title}[/bold]: "
                f"{len(chunks)} chunks"
            )
        
        return len(chunks)
    
    def add_papers(
        self,
        papers: list[ExtractedPaper],
        verbose: bool = True,
    ) -> int:
        """Add multiple papers to the vector store."""
        total_chunks = 0
        
        if verbose:
            console.print(f"\n[bold]Indexing {len(papers)} papers[/bold]\n")
        
        for paper in papers:
            count = self.add_paper(paper, verbose=verbose)
            total_chunks += count
        
        if verbose:
            console.print(
                f"\n[bold green]Indexed {total_chunks} chunks "
                f"from {len(papers)} papers[/bold green]\n"
            )
        
        return total_chunks
    
    def add_pdf(
        self,
        filepath: Path | str,
        verbose: bool = True,
    ) -> int:
        """Extract and add a PDF in one step."""
        paper = extract_pdf(filepath)
        return self.add_paper(paper, verbose=verbose)
    
    def delete_paper(
        self,
        paper_path: Path | str | None = None,
        paper_title: str | None = None,
        verbose: bool = True,
    ) -> int:
        """Remove a paper from the vector store."""
        if self._table is None:
            if verbose:
                console.print("[yellow]No table exists yet[/yellow]")
            return 0
        
        if paper_path:
            condition = f"paper_path = '{paper_path}'"
        elif paper_title:
            condition = f"paper_title = '{paper_title}'"
        else:
            raise ValueError("Must provide either paper_path or paper_title")
        
        count_before = self._table.count_rows()
        self._table.delete(condition)
        count_after = self._table.count_rows()
        deleted = count_before - count_after
        
        if verbose:
            if deleted > 0:
                console.print(f"[green]✓[/green] Deleted {deleted} chunks")
            else:
                console.print("[yellow]No matching paper found[/yellow]")
        
        return deleted
    
    def list_papers(self) -> list[dict]:
        """List all papers in the vector store."""
        if self._table is None:
            return []
        
        df = self._table.to_pandas()
        
        if df.empty:
            return []
        
        papers = {}
        for _, row in df.iterrows():
            path = row.get("paper_path", "unknown")
            if path not in papers:
                papers[path] = {
                    "title": row.get("paper_title", "Unknown"),
                    "path": path,
                    "chunk_count": 0,
                }
            papers[path]["chunk_count"] += 1
        
        return list(papers.values())
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        paper_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Semantic search across all indexed papers.
        
        Args:
            query: Search query (natural language)
            n_results: Number of results to return
            paper_filter: Filter to specific paper (by title)
            section_filter: Filter to specific section type (e.g., "Methods")
            
        Returns:
            List of SearchResult objects, ranked by similarity
        """
        if self._table is None:
            return []
        
        query_embedding = self._embed_text(query)
        
        search_query = self._table.search(query_embedding).limit(n_results)
        
        conditions = []
        if paper_filter:
            conditions.append(f"paper_title = '{paper_filter}'")
        if section_filter:
            conditions.append(f"section = '{section_filter}'")
        
        if conditions:
            search_query = search_query.where(" AND ".join(conditions), prefilter=True)
        
        results = search_query.to_list()
        
        search_results = []
        for row in results:
            distance = row.get("_distance", 0)
            similarity = 1 / (1 + distance)
            
            search_results.append(SearchResult(
                text=row["text"],
                score=similarity,
                paper_title=row.get("paper_title", "Unknown"),
                section=row.get("section") or None,
                chunk_index=row.get("chunk_index", 0),
                metadata={
                    "paper_path": row.get("paper_path"),
                    "token_count": row.get("token_count"),
                    "total_chunks": row.get("total_chunks"),
                },
            ))
        
        return search_results
    
    def search_similar(
        self,
        chunk_text: str,
        n_results: int = 5,
        exclude_same_paper: bool = True,
    ) -> list[SearchResult]:
        """Find chunks similar to a given chunk (for cross-referencing)."""
        fetch_n = n_results * 3 if exclude_same_paper else n_results
        
        results = self.search(chunk_text, n_results=fetch_n)
        
        if exclude_same_paper and results:
            query_embedding = self._embed_text(chunk_text)
            exact_match = self._table.search(query_embedding).limit(1).to_list()
            
            if exact_match:
                source_paper = exact_match[0].get("paper_path")
                results = [r for r in results if r.metadata.get("paper_path") != source_paper]
        
        return results[:n_results]
    
    def get_stats(self) -> dict:
        """Get statistics about the vector store."""
        papers = self.list_papers()
        total_chunks = sum(p["chunk_count"] for p in papers)
        
        return {
            "total_papers": len(papers),
            "total_chunks": total_chunks,
            "embedding_model": self.embedding_model,
            "db_path": str(self.db_path),
        }


_default_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Get or create the default vector store."""
    global _default_store
    if _default_store is None:
        _default_store = VectorStore()
    return _default_store


def add_paper(paper: ExtractedPaper, verbose: bool = True) -> int:
    """Add a paper to the default vector store."""
    return get_store().add_paper(paper, verbose=verbose)


def add_pdf(filepath: Path | str, verbose: bool = True) -> int:
    """Extract and add a PDF to the default vector store."""
    return get_store().add_pdf(filepath, verbose=verbose)


def search(
    query: str,
    n_results: int = 5,
    paper_filter: str | None = None,
    section_filter: str | None = None,
) -> list[SearchResult]:
    """Search the default vector store."""
    return get_store().search(
        query=query,
        n_results=n_results,
        paper_filter=paper_filter,
        section_filter=section_filter,
    )


def list_papers() -> list[dict]:
    """List all papers in the default vector store."""
    return get_store().list_papers()


if __name__ == "__main__":
    from rich.table import Table
    
    console.print("[bold]SCALPEL Vector Store Test[/bold]\n")
    
    console.print(f"Embedding model: [cyan]{settings.embedding_model}[/cyan]")
    console.print(f"LanceDB path: [cyan]{settings.lancedb_path}[/cyan]\n")
    
    store = VectorStore()
    
    stats = store.get_stats()
    console.print(f"Papers indexed: {stats['total_papers']}")
    console.print(f"Total chunks: {stats['total_chunks']}")
    
    papers = store.list_papers()
    if papers:
        console.print("\n[bold]Indexed papers:[/bold]")
        table = Table()
        table.add_column("Title", style="cyan")
        table.add_column("Chunks", style="green")
        
        for paper in papers:
            table.add_row(paper["title"], str(paper["chunk_count"]))
        
        console.print(table)
    else:
        console.print("\n[dim]No papers indexed yet.[/dim]")
        console.print("[dim]Use: store.add_pdf('data/papers/your_paper.pdf')[/dim]")
