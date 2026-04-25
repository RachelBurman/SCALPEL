"""
Market data ingestion for BEAR module.

Chunks and embeds company market data into the 'markets' LanceDB collection,
following the same patterns as src/scalpel/embeddings/vector_store.py.
"""

from dataclasses import dataclass
from pathlib import Path

import lancedb
import ollama
import pyarrow as pa
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from scalpel.config import settings
from scalpel.console import console
from scalpel.ingestion.chunker import TextChunk, chunk_text
from scalpel.bear.fetcher import CompanyData

MARKETS_TABLE = "markets"


@dataclass
class MarketSearchResult:
    """A search result from the markets vector store."""

    text: str
    score: float
    ticker: str
    company_name: str
    section: str
    chunk_index: int
    metadata: dict

    def __repr__(self) -> str:
        return f"MarketSearchResult({self.ticker} [{self.section}], score={self.score:.3f})"


class MarketStore:
    """
    Vector database for market data embeddings.

    Uses LanceDB for storage and Ollama for embedding generation.
    Follows the same patterns as VectorStore for the 'papers' collection,
    but targets the 'markets' table.
    """

    def __init__(
        self,
        db_path: Path | None = None,
        embedding_model: str | None = None,
    ):
        self.db_path = Path(db_path or settings.lancedb_path)
        self.embedding_model = embedding_model or settings.embedding_model

        self.db_path.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(str(self.db_path))
        self._table = None

        if MARKETS_TABLE in self._db.table_names():
            self._table = self._db.open_table(MARKETS_TABLE)

    def _get_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), settings.embedding_dimensions)),
            pa.field("ticker", pa.string()),
            pa.field("company_name", pa.string()),
            pa.field("section", pa.string()),
            pa.field("chunk_index", pa.int32()),
            pa.field("total_chunks", pa.int32()),
            pa.field("token_count", pa.int32()),
            pa.field("fetched_at", pa.string()),
        ])

    def _embed_text(self, text: str) -> list[float]:
        response = ollama.embed(model=self.embedding_model, input=text, truncate=True)
        return response["embeddings"][0]

    def _embed_batch(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
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
                task = progress.add_task("Embedding market data", total=len(texts))
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    response = ollama.embed(model=self.embedding_model, input=batch, truncate=True)
                    embeddings.extend(response["embeddings"])
                    progress.update(task, advance=len(batch))
        else:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = ollama.embed(model=self.embedding_model, input=batch, truncate=True)
                embeddings.extend(response["embeddings"])

        return embeddings

    def add_company(
        self,
        data: CompanyData,
        verbose: bool = True,
    ) -> int:
        """
        Add a company's market data to the vector store.

        Skips if the ticker is already indexed. Use delete_company() to re-index.

        Returns:
            Number of chunks added
        """
        if self._table is not None:
            existing = (
                self._table.search()
                .where(f"ticker = '{data.ticker}'", prefilter=True)
                .limit(1)
                .to_list()
            )
            if existing:
                if verbose:
                    console.print(f"[yellow]Already indexed:[/yellow] {data.ticker}")
                    console.print("[dim]Use delete_company() first to re-index[/dim]")
                return 0

        if verbose:
            console.print(f"[cyan]Chunking[/cyan] {data.company_name} ({data.ticker})...")

        # Chunk each section separately — mirrors chunk_paper(mode="sections")
        all_chunks: list[tuple[TextChunk, str]] = []
        for section in data.sections:
            for chunk in chunk_text(section.content, source_section=section.name):
                all_chunks.append((chunk, section.name))

        if not all_chunks:
            if verbose:
                console.print(f"[yellow]No chunks produced for[/yellow] {data.ticker}")
            return 0

        total = len(all_chunks)
        for i, (chunk, _) in enumerate(all_chunks):
            chunk.chunk_index = i
            chunk.total_chunks = total

        texts = [chunk.text for chunk, _ in all_chunks]
        embeddings = self._embed_batch(texts, show_progress=verbose)

        records = []
        for i, ((chunk, section_name), embedding) in enumerate(zip(all_chunks, embeddings)):
            records.append({
                "id": f"{data.ticker}_{section_name}_{i}",
                "text": chunk.text,
                "vector": embedding,
                "ticker": data.ticker,
                "company_name": data.company_name,
                "section": section_name,
                "chunk_index": chunk.chunk_index,
                "total_chunks": total,
                "token_count": chunk.token_count,
                "fetched_at": data.fetched_at,
            })

        if self._table is None:
            self._table = self._db.create_table(
                MARKETS_TABLE,
                data=records,
                schema=self._get_schema(),
            )
        else:
            self._table.add(records)

        if verbose:
            console.print(
                f"[green]✓[/green] Indexed [bold]{data.company_name}[/bold] ({data.ticker}): "
                f"{total} chunks"
            )

        return total

    def delete_company(self, ticker: str, verbose: bool = True) -> int:
        """Remove all indexed data for a ticker."""
        if self._table is None:
            if verbose:
                console.print("[yellow]No markets table exists yet[/yellow]")
            return 0

        count_before = self._table.count_rows()
        self._table.delete(f"ticker = '{ticker.upper()}'")
        count_after = self._table.count_rows()
        deleted = count_before - count_after

        if verbose:
            if deleted > 0:
                console.print(f"[green]✓[/green] Removed {deleted} chunks for {ticker.upper()}")
            else:
                console.print(f"[yellow]No data found for ticker:[/yellow] {ticker.upper()}")

        return deleted

    def list_companies(self) -> list[dict]:
        """List all companies in the market store."""
        if self._table is None:
            return []

        df = self._table.to_pandas()
        if df.empty:
            return []

        companies: dict[str, dict] = {}
        for _, row in df.iterrows():
            ticker = row.get("ticker", "UNKNOWN")
            if ticker not in companies:
                companies[ticker] = {
                    "ticker": ticker,
                    "company_name": row.get("company_name", "Unknown"),
                    "chunk_count": 0,
                    "fetched_at": row.get("fetched_at", ""),
                }
            companies[ticker]["chunk_count"] += 1

        return list(companies.values())

    def search(
        self,
        query: str,
        n_results: int = 5,
        ticker_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[MarketSearchResult]:
        """Semantic search across indexed market data."""
        if self._table is None:
            return []

        query_embedding = self._embed_text(query)
        search_query = self._table.search(query_embedding).limit(n_results)

        conditions = []
        if ticker_filter:
            conditions.append(f"ticker = '{ticker_filter.upper()}'")
        if section_filter:
            conditions.append(f"section = '{section_filter}'")

        if conditions:
            search_query = search_query.where(" AND ".join(conditions), prefilter=True)

        results = search_query.to_list()

        return [
            MarketSearchResult(
                text=row["text"],
                score=1 / (1 + row.get("_distance", 0)),
                ticker=row.get("ticker", ""),
                company_name=row.get("company_name", ""),
                section=row.get("section", ""),
                chunk_index=row.get("chunk_index", 0),
                metadata={
                    "token_count": row.get("token_count"),
                    "total_chunks": row.get("total_chunks"),
                    "fetched_at": row.get("fetched_at"),
                },
            )
            for row in results
        ]

    def get_company_context(self, ticker: str, max_chunks: int = 30) -> str:
        """
        Retrieve all indexed text for a ticker as a single context string,
        organised by section.
        """
        if self._table is None:
            return ""

        rows = (
            self._table.search()
            .where(f"ticker = '{ticker.upper()}'", prefilter=True)
            .limit(max_chunks)
            .to_list()
        )

        if not rows:
            return ""

        rows.sort(key=lambda x: x.get("chunk_index", 0))

        by_section: dict[str, list[str]] = {}
        for row in rows:
            section = row.get("section") or "General"
            by_section.setdefault(section, []).append(row["text"])

        parts = []
        for section, texts in by_section.items():
            parts.append(f"=== {section.upper()} ===")
            parts.extend(texts)

        return "\n\n".join(parts)

    def get_stats(self) -> dict:
        companies = self.list_companies()
        return {
            "total_companies": len(companies),
            "total_chunks": sum(c["chunk_count"] for c in companies),
            "embedding_model": self.embedding_model,
            "db_path": str(self.db_path),
        }


_default_market_store: MarketStore | None = None


def get_market_store() -> MarketStore:
    """Get or create the default market store."""
    global _default_market_store
    if _default_market_store is None:
        _default_market_store = MarketStore()
    return _default_market_store
